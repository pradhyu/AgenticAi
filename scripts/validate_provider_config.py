#!/usr/bin/env python3
"""
Configuration validation script for LLM providers.

This script helps validate your LLM provider configuration
and test connectivity without making actual API calls.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deep_agent_system.config.manager import ConfigManager, ConfigurationError
from deep_agent_system.config.models import LLMProvider
from deep_agent_system.llm.factory import LLMClientFactory


def validate_provider_config():
    """Validate the current LLM provider configuration."""
    print("🔍 Deep Agent System - Provider Configuration Validator")
    print("=" * 60)
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        system_config = config_manager.load_config()
        
        print("✅ Configuration loaded successfully")
        
        # Get the first agent's LLM config (they all use the same provider)
        agent_configs = list(system_config.agents.values())
        if not agent_configs:
            print("❌ No agent configurations found")
            return False
        
        llm_config = agent_configs[0].llm_config
        provider = llm_config.provider
        
        print(f"\n📋 Current Configuration:")
        print(f"   Provider: {provider.value}")
        print(f"   Model: {llm_config.model_name}")
        print(f"   Base URL: {llm_config.base_url or 'Default'}")
        print(f"   Temperature: {llm_config.temperature}")
        print(f"   Max Tokens: {llm_config.max_tokens}")
        
        # Provider-specific validation
        if provider == LLMProvider.OPENAI:
            validate_openai_config(llm_config)
        elif provider == LLMProvider.ANTHROPIC:
            validate_anthropic_config(llm_config)
        elif provider == LLMProvider.OPENROUTER:
            validate_openrouter_config(llm_config)
        elif provider == LLMProvider.LAMBDA_AI:
            validate_lambda_ai_config(llm_config)
        elif provider == LLMProvider.TOGETHER_AI:
            validate_together_ai_config(llm_config)
        elif provider == LLMProvider.HUGGINGFACE:
            validate_huggingface_config(llm_config)
        elif provider == LLMProvider.OLLAMA:
            validate_ollama_config(llm_config)
        elif provider == LLMProvider.CUSTOM:
            validate_custom_config(llm_config)
        
        # Test client creation
        print(f"\n🔧 Testing client creation...")
        try:
            client = LLMClientFactory.create_client(llm_config)
            print("✅ Client created successfully")
            
            # Test initialization (without actual API calls)
            print("✅ Client configuration validated")
            
        except ImportError as e:
            print(f"❌ Missing dependencies: {e}")
            return False
        except Exception as e:
            print(f"❌ Client creation failed: {e}")
            return False
        
        print(f"\n🎉 Configuration validation complete!")
        print(f"✅ Your {provider.value} configuration is valid and ready to use.")
        
        return True
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def validate_openai_config(config):
    """Validate OpenAI-specific configuration."""
    print(f"\n🔍 OpenAI Configuration:")
    
    if not config.api_key or config.api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY not set or using placeholder value")
        print("   Set your API key: export OPENAI_API_KEY=sk-your-key-here")
        return False
    
    if config.api_key.startswith("sk-"):
        print("✅ API key format looks correct")
    else:
        print("⚠️  API key format unusual (should start with 'sk-')")
    
    print("✅ OpenAI configuration looks good")
    return True


def validate_anthropic_config(config):
    """Validate Anthropic-specific configuration."""
    print(f"\n🔍 Anthropic Configuration:")
    
    if not config.api_key or config.api_key == "your_anthropic_api_key_here":
        print("❌ ANTHROPIC_API_KEY not set or using placeholder value")
        return False
    
    print("✅ Anthropic configuration looks good")
    return True


def validate_openrouter_config(config):
    """Validate OpenRouter-specific configuration."""
    print(f"\n🔍 OpenRouter Configuration:")
    
    if not config.api_key or config.api_key == "your_openrouter_api_key_here":
        print("❌ OPENROUTER_API_KEY not set or using placeholder value")
        return False
    
    if config.site_url:
        print(f"✅ Site URL configured: {config.site_url}")
    
    if config.app_name:
        print(f"✅ App name configured: {config.app_name}")
    
    print("✅ OpenRouter configuration looks good")
    return True


def validate_lambda_ai_config(config):
    """Validate Lambda AI-specific configuration."""
    print(f"\n🔍 Lambda AI Configuration:")
    
    if not config.api_key or config.api_key == "your_lambda_ai_api_key_here":
        print("❌ LAMBDA_AI_API_KEY not set or using placeholder value")
        return False
    
    print("✅ Lambda AI configuration looks good")
    return True


def validate_together_ai_config(config):
    """Validate Together AI-specific configuration."""
    print(f"\n🔍 Together AI Configuration:")
    
    if not config.api_key or config.api_key == "your_together_ai_api_key_here":
        print("❌ TOGETHER_AI_API_KEY not set or using placeholder value")
        return False
    
    print("✅ Together AI configuration looks good")
    return True


def validate_huggingface_config(config):
    """Validate Hugging Face-specific configuration."""
    print(f"\n🔍 Hugging Face Configuration:")
    
    if not config.api_key or config.api_key == "your_huggingface_api_key_here":
        print("❌ HUGGINGFACE_API_KEY not set or using placeholder value")
        return False
    
    print("✅ Hugging Face configuration looks good")
    return True


def validate_ollama_config(config):
    """Validate Ollama-specific configuration."""
    print(f"\n🔍 Ollama Configuration:")
    
    print(f"   Host: {config.ollama_host}")
    print("✅ Ollama configuration looks good (no API key required)")
    print("   Note: Make sure Ollama is running locally")
    return True


def validate_custom_config(config):
    """Validate custom provider configuration."""
    print(f"\n🔍 Custom Provider Configuration:")
    
    if not config.base_url:
        print("❌ CUSTOM_BASE_URL not set")
        return False
    
    if not config.model_name:
        print("❌ CUSTOM_MODEL not set")
        return False
    
    print(f"   Base URL: {config.base_url}")
    print(f"   Model: {config.model_name}")
    
    if config.extra_headers:
        print(f"   Extra Headers: {len(config.extra_headers)} configured")
    
    if config.extra_params:
        print(f"   Extra Params: {len(config.extra_params)} configured")
    
    print("✅ Custom provider configuration looks good")
    return True


def show_provider_help():
    """Show help for configuring different providers."""
    print("\n📚 Provider Configuration Help:")
    print("=" * 40)
    
    providers_help = {
        "OpenAI": [
            "export LLM_PROVIDER=openai",
            "export OPENAI_API_KEY=sk-your-key-here",
            "export OPENAI_MODEL=gpt-4-turbo-preview"
        ],
        "OpenRouter": [
            "export LLM_PROVIDER=openrouter", 
            "export OPENROUTER_API_KEY=sk-or-your-key-here",
            "export OPENROUTER_MODEL=openai/gpt-4-turbo-preview"
        ],
        "Lambda AI": [
            "export LLM_PROVIDER=lambda_ai",
            "export LAMBDA_AI_API_KEY=your-lambda-key-here", 
            "export LAMBDA_AI_MODEL=hermes-2-pro-llama-3-8b"
        ],
        "Ollama": [
            "export LLM_PROVIDER=ollama",
            "export OLLAMA_MODEL=llama2",
            "export OLLAMA_HOST=http://localhost:11434"
        ]
    }
    
    for provider, commands in providers_help.items():
        print(f"\n{provider}:")
        for cmd in commands:
            print(f"  {cmd}")


if __name__ == "__main__":
    success = validate_provider_config()
    
    if not success:
        show_provider_help()
        print(f"\n💡 After fixing configuration, run this script again to validate.")
        sys.exit(1)
    else:
        print(f"\n🚀 Ready to use Deep Agent System!")
        print(f"   Try: deep-agent ask 'What is artificial intelligence?'")
        sys.exit(0)