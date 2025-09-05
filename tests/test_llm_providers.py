"""Tests for LLM provider implementations."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from deep_agent_system.config.models import ModelConfig, LLMProvider
from deep_agent_system.llm.factory import LLMClientFactory
from deep_agent_system.llm.base import BaseLLMClient, LLMResponse


class TestLLMClientFactory:
    """Test cases for LLM client factory."""
    
    def test_get_supported_providers(self):
        """Test getting supported providers."""
        providers = LLMClientFactory.get_supported_providers()
        
        expected_providers = [
            LLMProvider.OPENAI,
            LLMProvider.ANTHROPIC,
            LLMProvider.OPENROUTER,
            LLMProvider.LAMBDA_AI,
            LLMProvider.TOGETHER_AI,
            LLMProvider.HUGGINGFACE,
            LLMProvider.OLLAMA,
            LLMProvider.CUSTOM,
        ]
        
        for provider in expected_providers:
            assert provider in providers
    
    def test_is_provider_supported(self):
        """Test provider support checking."""
        assert LLMClientFactory.is_provider_supported(LLMProvider.OPENAI)
        assert LLMClientFactory.is_provider_supported(LLMProvider.LAMBDA_AI)
        assert LLMClientFactory.is_provider_supported(LLMProvider.OLLAMA)
    
    def test_create_openai_client(self):
        """Test creating OpenAI client."""
        config = ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        
        client = LLMClientFactory.create_client(config)
        
        from deep_agent_system.llm.openai_client import OpenAIClient
        assert isinstance(client, OpenAIClient)
        assert client.config == config
    
    def test_create_lambda_ai_client(self):
        """Test creating Lambda AI client."""
        config = ModelConfig(
            provider=LLMProvider.LAMBDA_AI,
            model_name="hermes-2-pro-llama-3-8b",
            api_key="test-key"
        )
        
        client = LLMClientFactory.create_client(config)
        
        from deep_agent_system.llm.lambda_ai_client import LambdaAIClient
        assert isinstance(client, LambdaAIClient)
        assert client.config == config
    
    def test_create_openrouter_client(self):
        """Test creating OpenRouter client."""
        config = ModelConfig(
            provider=LLMProvider.OPENROUTER,
            model_name="openai/gpt-4",
            api_key="test-key",
            site_url="https://example.com",
            app_name="Test App"
        )
        
        client = LLMClientFactory.create_client(config)
        
        from deep_agent_system.llm.openrouter_client import OpenRouterClient
        assert isinstance(client, OpenRouterClient)
        assert client.config == config
    
    def test_create_ollama_client(self):
        """Test creating Ollama client."""
        config = ModelConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama2",
            api_key=None,  # Ollama doesn't require API key
            ollama_host="http://localhost:11434"
        )
        
        client = LLMClientFactory.create_client(config)
        
        from deep_agent_system.llm.ollama_client import OllamaClient
        assert isinstance(client, OllamaClient)
        assert client.config == config
    
    def test_create_custom_client(self):
        """Test creating custom client."""
        config = ModelConfig(
            provider=LLMProvider.CUSTOM,
            model_name="custom-model",
            api_key="test-key",
            base_url="https://api.example.com/v1",
            extra_headers={"Custom-Header": "value"},
            extra_params={"top_p": 0.9}
        )
        
        client = LLMClientFactory.create_client(config)
        
        from deep_agent_system.llm.custom_client import CustomClient
        assert isinstance(client, CustomClient)
        assert client.config == config
    
    def test_unsupported_provider(self):
        """Test error handling for unsupported provider."""
        # This would require adding a new enum value, so we'll mock it
        with patch('deep_agent_system.llm.factory.LLMClientFactory._client_registry', {}):
            config = ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="test",
                api_key="test"
            )
            
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                LLMClientFactory.create_client(config)
    
    def test_register_custom_client(self):
        """Test registering a custom client class."""
        class TestClient(BaseLLMClient):
            async def initialize(self):
                pass
            
            async def generate(self, prompt, system_prompt=None, **kwargs):
                return LLMResponse(content="test", model="test")
            
            async def generate_stream(self, prompt, system_prompt=None, **kwargs):
                yield "test"
        
        # Register the custom client
        LLMClientFactory.register_client(LLMProvider.CUSTOM, TestClient)
        
        config = ModelConfig(
            provider=LLMProvider.CUSTOM,
            model_name="test",
            api_key="test"
        )
        
        client = LLMClientFactory.create_client(config)
        assert isinstance(client, TestClient)
    
    def test_register_invalid_client(self):
        """Test error when registering invalid client class."""
        class InvalidClient:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseLLMClient"):
            LLMClientFactory.register_client(LLMProvider.CUSTOM, InvalidClient)


class TestModelConfig:
    """Test cases for ModelConfig validation."""
    
    def test_valid_openai_config(self):
        """Test valid OpenAI configuration."""
        config = ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="sk-test123",
            temperature=0.7,
            max_tokens=4000
        )
        
        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-4"
        assert config.api_key == "sk-test123"
        assert config.temperature == 0.7
        assert config.max_tokens == 4000
    
    def test_valid_lambda_ai_config(self):
        """Test valid Lambda AI configuration."""
        config = ModelConfig(
            provider=LLMProvider.LAMBDA_AI,
            model_name="hermes-2-pro-llama-3-8b",
            api_key="lambda-key-123",
            base_url="https://api.lambdalabs.com/v1",
            lambda_endpoint="https://custom-endpoint.com"
        )
        
        assert config.provider == LLMProvider.LAMBDA_AI
        assert config.lambda_endpoint == "https://custom-endpoint.com"
    
    def test_valid_openrouter_config(self):
        """Test valid OpenRouter configuration."""
        config = ModelConfig(
            provider=LLMProvider.OPENROUTER,
            model_name="openai/gpt-4",
            api_key="or-key-123",
            site_url="https://mysite.com",
            app_name="My App",
            extra_headers={"Custom": "header"}
        )
        
        assert config.provider == LLMProvider.OPENROUTER
        assert config.site_url == "https://mysite.com"
        assert config.app_name == "My App"
        assert config.extra_headers["Custom"] == "header"
    
    def test_valid_ollama_config(self):
        """Test valid Ollama configuration (no API key required)."""
        config = ModelConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama2",
            api_key=None,
            ollama_host="http://localhost:11434"
        )
        
        assert config.provider == LLMProvider.OLLAMA
        assert config.api_key is None
        assert config.ollama_host == "http://localhost:11434"
    
    def test_invalid_temperature(self):
        """Test invalid temperature validation."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                api_key="test",
                temperature=3.0  # Invalid
            )
    
    def test_missing_api_key_for_cloud_provider(self):
        """Test validation error for missing API key."""
        with pytest.raises(ValueError, match="API key is required"):
            ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                api_key=None  # Required for OpenAI
            )


@pytest.mark.asyncio
class TestLLMClientIntegration:
    """Integration tests for LLM clients."""
    
    async def test_openai_client_mock(self):
        """Test OpenAI client with mocked responses."""
        config = ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        
        from deep_agent_system.llm.openai_client import OpenAIClient
        
        client = OpenAIClient(config)
        
        # Mock the OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = Mock()
        mock_response.usage.model_dump.return_value = {"total_tokens": 100}
        
        with patch('deep_agent_system.llm.openai_client.AsyncOpenAI') as mock_openai:
            mock_client_instance = Mock()
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client_instance
            
            await client.initialize()
            response = await client.generate("Test prompt")
            
            assert response.content == "Test response"
            assert response.model == "gpt-4"
            assert response.metadata["provider"] == "openai"
    
    async def test_factory_create_and_initialize(self):
        """Test factory create and initialize method."""
        config = ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        
        with patch('deep_agent_system.llm.openai_client.AsyncOpenAI'):
            client = await LLMClientFactory.create_and_initialize_client(config)
            
            assert isinstance(client, BaseLLMClient)
            assert client.config == config