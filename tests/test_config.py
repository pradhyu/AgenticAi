"""Tests for configuration management."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from deep_agent_system.config.manager import ConfigManager, ConfigurationError
from deep_agent_system.config.models import (
    SystemConfig,
    AgentType,
    LogLevel,
    LogFormat,
    ModelConfig,
    ChromaDBConfig,
    Neo4jConfig,
    RAGConfig,
)


class TestConfigModels:
    """Test configuration data models."""
    
    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Valid config
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000,
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        
        # Invalid temperature
        with pytest.raises(ValidationError):
            ModelConfig(
                provider="openai",
                model_name="gpt-4",
                api_key="test-key",
                temperature=3.0,
            )
    
    def test_chromadb_config_validation(self):
        """Test ChromaDBConfig validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ChromaDBConfig(
                host="localhost",
                port=8000,
                collection_name="test",
                persist_directory=temp_dir,
            )
            assert config.host == "localhost"
            assert config.port == 8000
            assert Path(config.persist_directory).exists()
    
    def test_neo4j_config_validation(self):
        """Test Neo4jConfig validation."""
        # Valid URI
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
        )
        assert config.uri == "bolt://localhost:7687"
        
        # Invalid URI
        with pytest.raises(ValueError, match="Neo4j URI must start with"):
            Neo4jConfig(
                uri="http://localhost:7687",
                username="neo4j",
                password="password",
            )
    
    def test_rag_config_validation(self):
        """Test RAGConfig validation."""
        # Valid config
        config = RAGConfig(
            chunk_size=1000,
            chunk_overlap=200,
        )
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        
        # Invalid overlap (greater than chunk size)
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            RAGConfig(
                chunk_size=1000,
                chunk_overlap=1500,
            )


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_env_vars = {
            "OPENAI_API_KEY": "test-api-key",
            "NEO4J_PASSWORD": "test-password",
            "PROMPT_DIR": "./test_prompts",
        }
    
    def test_init_without_env_file(self):
        """Test ConfigManager initialization without env file."""
        manager = ConfigManager()
        assert manager.env_file is None
    
    def test_init_with_env_file(self):
        """Test ConfigManager initialization with env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("TEST_VAR=test_value\n")
            env_file = f.name
        
        try:
            manager = ConfigManager(env_file=env_file)
            assert manager.env_file == env_file
        finally:
            os.unlink(env_file)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_env_var_with_default(self):
        """Test getting environment variable with default."""
        manager = ConfigManager()
        
        # Test with default
        value = manager._get_env_var("NONEXISTENT_VAR", "default_value")
        assert value == "default_value"
        
        # Test without default
        value = manager._get_env_var("NONEXISTENT_VAR")
        assert value is None
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_env_var_required_missing(self):
        """Test getting required environment variable that's missing."""
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError, match="Required environment variable"):
            manager._get_env_var("REQUIRED_VAR", required=True)
    
    @patch.dict(os.environ, {"TEST_BOOL": "true"}, clear=True)
    def test_get_bool_env(self):
        """Test getting boolean environment variable."""
        manager = ConfigManager()
        
        # Test true values
        for true_val in ["true", "1", "yes", "on", "TRUE", "True"]:
            with patch.dict(os.environ, {"TEST_BOOL": true_val}):
                assert manager._get_bool_env("TEST_BOOL") is True
        
        # Test false values
        for false_val in ["false", "0", "no", "off", "FALSE", "False"]:
            with patch.dict(os.environ, {"TEST_BOOL": false_val}):
                assert manager._get_bool_env("TEST_BOOL") is False
        
        # Test default
        assert manager._get_bool_env("NONEXISTENT_BOOL", True) is True
    
    @patch.dict(os.environ, {"TEST_INT": "42"}, clear=True)
    def test_get_int_env(self):
        """Test getting integer environment variable."""
        manager = ConfigManager()
        
        assert manager._get_int_env("TEST_INT") == 42
        assert manager._get_int_env("NONEXISTENT_INT", 10) == 10
        
        # Test invalid integer
        with patch.dict(os.environ, {"TEST_INT": "not_an_int"}):
            with pytest.raises(ConfigurationError, match="must be an integer"):
                manager._get_int_env("TEST_INT")
    
    @patch.dict(os.environ, {"TEST_FLOAT": "3.14"}, clear=True)
    def test_get_float_env(self):
        """Test getting float environment variable."""
        manager = ConfigManager()
        
        assert manager._get_float_env("TEST_FLOAT") == 3.14
        assert manager._get_float_env("NONEXISTENT_FLOAT", 2.5) == 2.5
        
        # Test invalid float
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            with pytest.raises(ConfigurationError, match="must be a float"):
                manager._get_float_env("TEST_FLOAT")
    
    @patch.dict(os.environ, clear=True)
    def test_create_model_config(self):
        """Test creating ModelConfig from environment."""
        with patch.dict(os.environ, self.test_env_vars):
            manager = ConfigManager()
            config = manager._create_model_config()
            
            assert config.provider == "openai"
            assert config.api_key == "test-api-key"
            assert config.temperature == 0.7  # default
    
    @patch.dict(os.environ, clear=True)
    def test_create_chromadb_config(self):
        """Test creating ChromaDBConfig from environment."""
        manager = ConfigManager()
        config = manager._create_chromadb_config()
        
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.collection_name == "deep_agent_knowledge"
    
    @patch.dict(os.environ, clear=True)
    def test_create_neo4j_config(self):
        """Test creating Neo4jConfig from environment."""
        with patch.dict(os.environ, self.test_env_vars):
            manager = ConfigManager()
            config = manager._create_neo4j_config()
            
            assert config.uri == "bolt://localhost:7687"
            assert config.username == "neo4j"
            assert config.password == "test-password"
    
    @patch.dict(os.environ, clear=True)
    def test_create_agent_configs(self):
        """Test creating agent configurations."""
        with patch.dict(os.environ, self.test_env_vars):
            manager = ConfigManager()
            configs = manager._create_agent_configs()
            
            # Should have all agent types
            assert len(configs) == len(AgentType)
            
            # Check specific agent
            analyst_config = None
            for config in configs.values():
                if config.agent_type == AgentType.ANALYST:
                    analyst_config = config
                    break
            
            assert analyst_config is not None
            assert analyst_config.agent_id == "analyst_agent"
            assert analyst_config.rag_enabled is True
    
    @patch.dict(os.environ, clear=True)
    def test_load_config_success(self):
        """Test successful configuration loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {
                **self.test_env_vars,
                "PROMPT_DIR": temp_dir,
            }
            
            with patch.dict(os.environ, env_vars):
                manager = ConfigManager()
                config = manager.load_config()
                
                assert isinstance(config, SystemConfig)
                assert len(config.agents) == len(AgentType)
                assert config.prompt_dir == temp_dir
    
    @patch.dict(os.environ, {}, clear=True)
    def test_load_config_missing_required(self):
        """Test configuration loading with missing required variables."""
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError, match="Required environment variable"):
            manager.load_config()
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {
                **self.test_env_vars,
                "PROMPT_DIR": temp_dir,
            }
            
            with patch.dict(os.environ, env_vars):
                manager = ConfigManager()
                config = manager.load_config()
                
                assert manager.validate_config(config) is True
    
    def test_validate_config_missing_prompt_dir(self):
        """Test configuration validation with missing prompt directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {
                **self.test_env_vars,
                "PROMPT_DIR": temp_dir,
            }
            
            with patch.dict(os.environ, env_vars):
                manager = ConfigManager()
                config = manager.load_config()
                
                # Mock Path.exists to return False for validation test
                with patch('pathlib.Path.exists', return_value=False):
                    with pytest.raises(ConfigurationError, match="Prompt directory does not exist"):
                        manager.validate_config(config)
    
    def test_get_agent_config(self):
        """Test getting specific agent configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_vars = {
                **self.test_env_vars,
                "PROMPT_DIR": temp_dir,
            }
            
            with patch.dict(os.environ, env_vars):
                manager = ConfigManager()
                config = manager.load_config()
                
                # Get analyst agent config
                analyst_config = manager.get_agent_config(config, AgentType.ANALYST)
                assert analyst_config is not None
                assert analyst_config.agent_type == AgentType.ANALYST
                
                # Try to get non-existent agent type (shouldn't happen in practice)
                # This tests the None return case
                with patch.object(config, 'agents', {}):
                    result = manager.get_agent_config(config, AgentType.ANALYST)
                    assert result is None


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_config_cycle(self):
        """Test complete configuration loading and validation cycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test .env file
            env_file = Path(temp_dir) / ".env"
            env_content = """
OPENAI_API_KEY=test-key-123
NEO4J_PASSWORD=test-neo4j-password
PROMPT_DIR=./test_prompts
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.8
RAG_CHUNK_SIZE=500
CLI_THEME=light
LOG_LEVEL=DEBUG
"""
            env_file.write_text(env_content)
            
            # Create prompt directory
            prompt_dir = Path(temp_dir) / "test_prompts"
            prompt_dir.mkdir()
            
            # Load configuration
            manager = ConfigManager(env_file=str(env_file))
            config = manager.load_config()
            
            # Validate configuration
            assert manager.validate_config(config) is True
            
            # Check specific values
            assert config.rag_config.chunk_size == 500
            assert config.cli_config.theme == "light"
            assert config.logging_config.level == LogLevel.DEBUG
            
            # Check agent configurations
            analyst_config = manager.get_agent_config(config, AgentType.ANALYST)
            assert analyst_config.llm_config.temperature == 0.8
            assert analyst_config.llm_config.model_name == "gpt-4"