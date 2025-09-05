"""Anthropic LLM client implementation."""

import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import BaseLLMClient, LLMResponse
from deep_agent_system.config.models import ModelConfig

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic LLM client implementation."""
    
    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
        
        try:
            self._client = AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            logger.info("Anthropic client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using Anthropic API."""
        try:
            params = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **kwargs
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            response = await self._client.messages.create(**params)
            
            return LLMResponse(
                content=response.content[0].text if response.content else "",
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                } if response.usage else None,
                metadata={
                    "stop_reason": response.stop_reason,
                    "provider": "anthropic"
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Anthropic API."""
        try:
            params = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
                **kwargs
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            stream = await self._client.messages.create(**params)
            
            async for chunk in stream:
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                    yield chunk.delta.text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise