"""Ollama LLM client implementation."""

import logging
import json
from typing import Any, Dict, List, Optional, AsyncGenerator

import aiohttp

from .base import BaseLLMClient, LLMResponse
from deep_agent_system.config.models import ModelConfig

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """Ollama LLM client implementation."""
    
    async def initialize(self) -> None:
        """Initialize the Ollama client."""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
            # Test connection to Ollama
            base_url = self.config.ollama_host or "http://localhost:11434"
            async with self._session.get(f"{base_url}/api/tags") as response:
                if response.status != 200:
                    raise ConnectionError(f"Cannot connect to Ollama at {base_url}")
            
            logger.info("Ollama client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using Ollama API."""
        try:
            base_url = self.config.ollama_host or "http://localhost:11434"
            url = f"{base_url}/api/generate"
            
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    **kwargs
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with self._session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                return LLMResponse(
                    content=result.get("response", ""),
                    model=self.config.model_name,
                    usage={
                        "prompt_eval_count": result.get("prompt_eval_count"),
                        "eval_count": result.get("eval_count"),
                    } if "prompt_eval_count" in result else None,
                    metadata={
                        "done": result.get("done", False),
                        "provider": "ollama"
                    }
                )
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Ollama API."""
        try:
            base_url = self.config.ollama_host or "http://localhost:11434"
            url = f"{base_url}/api/generate"
            
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    **kwargs
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with self._session.post(url, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if "response" in chunk:
                                yield chunk["response"]
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if hasattr(self, '_session') and self._session:
            await self._session.close()