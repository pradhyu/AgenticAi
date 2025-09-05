"""Hugging Face LLM client implementation."""

import logging
import json
from typing import Any, Dict, List, Optional, AsyncGenerator

import aiohttp

from .base import BaseLLMClient, LLMResponse
from deep_agent_system.config.models import ModelConfig

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseLLMClient):
    """Hugging Face LLM client implementation."""
    
    async def initialize(self) -> None:
        """Initialize the Hugging Face client."""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                }
            )
            logger.info("Hugging Face client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face client: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using Hugging Face API."""
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            
            url = f"{self.config.base_url}/models/{self.config.model_name}"
            
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "temperature": self.config.temperature,
                    "max_new_tokens": self.config.max_tokens,
                    "return_full_text": False,
                    **kwargs
                }
            }
            
            async with self._session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].get("generated_text", "")
                else:
                    content = str(result)
                
                return LLMResponse(
                    content=content,
                    model=self.config.model_name,
                    usage=None,  # HF doesn't provide usage info in this format
                    metadata={
                        "provider": "huggingface"
                    }
                )
                
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Hugging Face API."""
        # Note: Hugging Face Inference API doesn't support streaming by default
        # This is a fallback that yields the complete response
        try:
            response = await self.generate(prompt, system_prompt, **kwargs)
            yield response.content
        except Exception as e:
            logger.error(f"Hugging Face streaming failed: {e}")
            raise
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if hasattr(self, '_session') and self._session:
            await self._session.close()