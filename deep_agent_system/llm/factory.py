"""LLM client factory for creating provider-specific clients."""

import logging
from typing import Type

from deep_agent_system.config.models import ModelConfig, LLMProvider
from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .openrouter_client import OpenRouterClient
from .lambda_ai_client import LambdaAIClient
from .together_ai_client import TogetherAIClient
from .huggingface_client import HuggingFaceClient
from .ollama_client import OllamaClient
from .custom_client import CustomClient

logger = logging.getLogger(__name__)


class LLMClientFactory:
    """Factory for creating LLM clients based on provider configuration."""
    
    # Registry of provider to client class mappings
    _client_registry = {
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.OPENROUTER: OpenRouterClient,
        LLMProvider.LAMBDA_AI: LambdaAIClient,
        LLMProvider.TOGETHER_AI: TogetherAIClient,
        LLMProvider.HUGGINGFACE: HuggingFaceClient,
        LLMProvider.OLLAMA: OllamaClient,
        LLMProvider.CUSTOM: CustomClient,
    }
    
    @classmethod
    def create_client(cls, config: ModelConfig) -> BaseLLMClient:
        """Create an LLM client based on the provider configuration.
        
        Args:
            config: Model configuration containing provider information
            
        Returns:
            Initialized LLM client instance
            
        Raises:
            ValueError: If provider is not supported
            ImportError: If required dependencies are missing
        """
        provider = config.provider
        
        if provider not in cls._client_registry:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        client_class = cls._client_registry[provider]
        
        try:
            client = client_class(config)
            logger.info(f"Created {provider} client for model {config.model_name}")
            return client
        except ImportError as e:
            logger.error(f"Missing dependencies for {provider}: {e}")
            raise ImportError(
                f"Missing dependencies for {provider}. "
                f"Please install the required packages: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to create {provider} client: {e}")
            raise
    
    @classmethod
    async def create_and_initialize_client(cls, config: ModelConfig) -> BaseLLMClient:
        """Create and initialize an LLM client.
        
        Args:
            config: Model configuration
            
        Returns:
            Initialized and ready-to-use LLM client
        """
        client = cls.create_client(config)
        await client.initialize()
        return client
    
    @classmethod
    def register_client(
        cls, 
        provider: LLMProvider, 
        client_class: Type[BaseLLMClient]
    ) -> None:
        """Register a custom client class for a provider.
        
        Args:
            provider: LLM provider enum
            client_class: Client class that implements BaseLLMClient
        """
        if not issubclass(client_class, BaseLLMClient):
            raise ValueError("Client class must inherit from BaseLLMClient")
        
        cls._client_registry[provider] = client_class
        logger.info(f"Registered custom client for provider: {provider}")
    
    @classmethod
    def get_supported_providers(cls) -> list[LLMProvider]:
        """Get list of supported LLM providers.
        
        Returns:
            List of supported provider enums
        """
        return list(cls._client_registry.keys())
    
    @classmethod
    def is_provider_supported(cls, provider: LLMProvider) -> bool:
        """Check if a provider is supported.
        
        Args:
            provider: LLM provider to check
            
        Returns:
            True if provider is supported
        """
        return provider in cls._client_registry