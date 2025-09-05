"""OpenRouter LLM client implementation."""

import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

from openai import AsyncOpenAI

from .base import BaseLLMClient, LLMResponse
from deep_agent_system.config.models import ModelConfig

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseLLMClient):
    """OpenRouter LLM client implementation using OpenAI-compatible API."""
    
    async def initialize(self) -> None:
        """Initialize the OpenRouter client."""
        try:
            # OpenRouter uses OpenAI-compatible API
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                **self.config.extra_headers
            }
            
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://openrouter.ai/api/v1",
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=headers,
            )
            logger.info("OpenRouter client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenRouter API."""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # Merge config parameters with kwargs
            params = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **self.config.extra_params,
                **kwargs
            }
            
            response = await self._client.chat.completions.create(**params)
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage=response.usage.model_dump() if response.usage else None,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "provider": "openrouter"
                }
            )
            
        except Exception as e:
            logger.error(f"OpenRouter generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using OpenRouter API."""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # Merge config parameters with kwargs
            params = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
                **self.config.extra_params,
                **kwargs
            }
            
            stream = await self._client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenRouter streaming failed: {e}")
            raise