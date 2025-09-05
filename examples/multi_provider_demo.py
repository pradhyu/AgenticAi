#!/usr/bin/env python3
"""
Demonstration of multi-provider LLM support in Deep Agent System.

This script shows how to configure and use different LLM providers
including OpenAI, OpenRouter, Lambda AI, and others.
"""

import asyncio
import os
from typing import Dict, Any

from deep_agent_system.config.models import ModelConfig, LLMProvider
from deep_agent_system.llm.factory import LLMClientFactory


async def demo_provider(provider_name: str, config: ModelConfig) -> None:
    """Demonstrate a specific LLM provider.
    
    Args:
        provider_name: Human-readable provider name
        config: Model configuration for the provider
    """
    print(f"\n{'='*60}")
    print(f"Testing {provider_name}")
    print(f"{'='*60}")
    
    try:
        # Create and initialize client
        client = await LLMClientFactory.create_and_initialize_client(config)
        print(f"✅ {provider_name} client initialized successfully")
        print(f"   Model: {config.model_name}")
        print(f"   Base URL: {config.base_url or 'Default'}")
        
        # Test generation (with mock prompt)
        test_prompt = "What is artificial intelligence?"
        print(f"\n📝 Testing generation with prompt: '{test_prompt}'")
        
        # Note: This would require actual API keys to work
        # For demo purposes, we'll just show the configuration
        print(f"   Temperature: {config.temperature}")
        print(f"   Max Tokens: {config.max_tokens}")
        
        if config.extra_headers:
            print(f"   Extra Headers: {config.extra_headers}")
        
        if config.extra_params:
            print(f"   Extra Params: {config.extra_params}")
        
        await client.close()
        print(f"✅ {provider_name} client closed successfully")
        
    except ImportError as e:
        print(f"❌ {provider_name} dependencies missing: {e}")
    except Exception as e:
        print(f"❌ {provider_name} configuration error: {e}")


async def main():
    """Main demonstration function."""
    print("🚀 Deep Agent System - Multi-Provider LLM Demo")
    print("=" * 60)
    print("This demo shows how to configure different LLM providers.")
    print("Note: Actual API calls require valid API keys.")
    
    # Demo configurations for different providers
    provider_configs = {
        "OpenAI": ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4-turbo-preview",
            api_key=os.getenv("OPENAI_API_KEY", "demo-key"),
            temperature=0.7,
            max_tokens=4000
        ),
        
        "Anthropic": ModelConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY", "demo-key"),
            temperature=0.7,
            max_tokens=4000
        ),
        
        "OpenRouter": ModelConfig(
            provider=LLMProvider.OPENROUTER,
            model_name="openai/gpt-4-turbo-preview",
            api_key=os.getenv("OPENROUTER_API_KEY", "demo-key"),
            base_url="https://openrouter.ai/api/v1",
            site_url="https://example.com",
            app_name="Deep Agent System Demo",
            temperature=0.7,
            max_tokens=4000,
            extra_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "Deep Agent System Demo"
            }
        ),
        
        "Lambda AI": ModelConfig(
            provider=LLMProvider.LAMBDA_AI,
            model_name="hermes-2-pro-llama-3-8b",
            api_key=os.getenv("LAMBDA_AI_API_KEY", "demo-key"),
            base_url="https://api.lambdalabs.com/v1",
            temperature=0.7,
            max_tokens=4000
        ),
        
        "Together AI": ModelConfig(
            provider=LLMProvider.TOGETHER_AI,
            model_name="meta-llama/Llama-2-70b-chat-hf",
            api_key=os.getenv("TOGETHER_AI_API_KEY", "demo-key"),
            base_url="https://api.together.xyz/v1",
            temperature=0.7,
            max_tokens=4000
        ),
        
        "Hugging Face": ModelConfig(
            provider=LLMProvider.HUGGINGFACE,
            model_name="microsoft/DialoGPT-large",
            api_key=os.getenv("HUGGINGFACE_API_KEY", "demo-key"),
            base_url="https://api-inference.huggingface.co",
            temperature=0.7,
            max_tokens=4000
        ),
        
        "Ollama (Local)": ModelConfig(
            provider=LLMProvider.OLLAMA,
            model_name="llama2",
            api_key=None,  # No API key needed for Ollama
            ollama_host="http://localhost:11434",
            temperature=0.7,
            max_tokens=4000
        ),
        
        "Custom Provider": ModelConfig(
            provider=LLMProvider.CUSTOM,
            model_name="custom-model",
            api_key=os.getenv("CUSTOM_API_KEY", "demo-key"),
            base_url="https://api.example.com/v1",
            temperature=0.7,
            max_tokens=4000,
            extra_headers={
                "Authorization": "Bearer demo-key",
                "Custom-Header": "custom-value"
            },
            extra_params={
                "top_p": 0.9,
                "frequency_penalty": 0.1
            }
        )
    }
    
    # Test each provider configuration
    for provider_name, config in provider_configs.items():
        await demo_provider(provider_name, config)
    
    # Show supported providers
    print(f"\n{'='*60}")
    print("Supported Providers Summary")
    print(f"{'='*60}")
    
    supported_providers = LLMClientFactory.get_supported_providers()
    for i, provider in enumerate(supported_providers, 1):
        status = "✅" if LLMClientFactory.is_provider_supported(provider) else "❌"
        print(f"{i:2d}. {status} {provider.value}")
    
    print(f"\n📊 Total supported providers: {len(supported_providers)}")
    
    # Configuration examples
    print(f"\n{'='*60}")
    print("Environment Variable Examples")
    print(f"{'='*60}")
    
    print("""
# Set your preferred provider
export LLM_PROVIDER=openai

# OpenAI
export OPENAI_API_KEY=sk-your-key-here
export OPENAI_MODEL=gpt-4-turbo-preview

# OpenRouter  
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-your-key-here
export OPENROUTER_MODEL=openai/gpt-4-turbo-preview

# Lambda AI
export LLM_PROVIDER=lambda_ai
export LAMBDA_AI_API_KEY=your-lambda-key-here
export LAMBDA_AI_MODEL=hermes-2-pro-llama-3-8b

# Ollama (Local)
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama2
export OLLAMA_HOST=http://localhost:11434

# Custom Provider
export LLM_PROVIDER=custom
export CUSTOM_API_KEY=your-custom-key
export CUSTOM_BASE_URL=https://your-api.com/v1
export CUSTOM_MODEL=your-model-name
""")
    
    print("🎉 Multi-provider demo complete!")
    print("\nTo use with Deep Agent System:")
    print("1. Set your preferred LLM_PROVIDER in .env")
    print("2. Configure the corresponding API key and settings")
    print("3. Run: deep-agent ask 'Your question here'")


if __name__ == "__main__":
    asyncio.run(main())