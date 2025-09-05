"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel

from deep_agent_system.config.models import ModelConfig


class LLMResponse(BaseModel):
    """Response from LLM client."""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the LLM client.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self._client = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the client connection."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            LLM response
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Response chunks
        """
        pass
    
    async def close(self) -> None:
        """Close the client connection."""
        if hasattr(self._client, 'close'):
            await self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'close'):
            import asyncio
            asyncio.create_task(self.close())
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()