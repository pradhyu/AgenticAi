"""OpenAI LLM client implementation."""

import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from .base import BaseLLMClient, LLMResponse
from deep_agent_system.config.models import ModelConfig

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client implementation."""
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenAI API."""
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
                **kwargs
            }
            
            response: ChatCompletion = await self._client.chat.completions.create(**params)
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage=response.usage.model_dump() if response.usage else None,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "provider": "openai"
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using OpenAI API."""
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
                **kwargs
            }
            
            stream = await self._client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise