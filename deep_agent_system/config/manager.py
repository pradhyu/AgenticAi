"""Configuration manager for loading and validating system configuration."""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic import ValidationError

from .models import (
    SystemConfig,
    AgentConfig,
    AgentType,
    ModelConfig,
    RAGConfig,
    DatabaseConfig,
    ChromaDBConfig,
    Neo4jConfig,
    CLIConfig,
    LoggingConfig,
    WorkflowConfig,
    SecurityConfig,
    DevelopmentConfig,
    LogLevel,
    LogFormat,
)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ConfigManager:
    """Manager for loading and validating system configuration from environment variables."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            env_file: Optional path to .env file. If None, looks for .env in current directory.
        """
        self.env_file = env_file
        self._load_environment()
    
    def _load_environment(self) -> None:
        """Load environment variables from .env file if it exists."""
        if self.env_file:
            env_path = Path(self.env_file)
        else:
            env_path = Path(".env")
        
        if env_path.exists():
            load_dotenv(env_path)
    
    def _get_env_var(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get environment variable with optional default and required validation.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Environment variable value or default
            
        Raises:
            ConfigurationError: If required variable is missing
        """
        value = os.getenv(key, default)
        if required and value is None:
            raise ConfigurationError(f"Required environment variable '{key}' is not set")
        return value
    
    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = self._get_env_var(key, str(default).lower())
        return str(value).lower() in ('true', '1', 'yes', 'on')
    
    def _get_int_env(self, key: str, default: int = 0) -> int:
        """Get integer environment variable."""
        value = self._get_env_var(key, str(default))
        try:
            return int(value)
        except ValueError:
            raise ConfigurationError(f"Environment variable '{key}' must be an integer, got: {value}")
    
    def _get_float_env(self, key: str, default: float = 0.0) -> float:
        """Get float environment variable."""
        value = self._get_env_var(key, str(default))
        try:
            return float(value)
        except ValueError:
            raise ConfigurationError(f"Environment variable '{key}' must be a float, got: {value}")
    
    def _create_model_config(self) -> ModelConfig:
        """Create ModelConfig from environment variables."""
        return ModelConfig(
            provider="openai",  # Currently only OpenAI supported
            model_name=self._get_env_var("OPENAI_MODEL", "gpt-4-turbo-preview"),
            api_key=self._get_env_var("OPENAI_API_KEY", required=True),
            temperature=self._get_float_env("OPENAI_TEMPERATURE", 0.7),
            max_tokens=self._get_int_env("OPENAI_MAX_TOKENS", 4000),
            timeout=self._get_int_env("AGENT_TIMEOUT", 300),
            max_retries=self._get_int_env("MAX_RETRIES", 3),
        )
    
    def _create_chromadb_config(self) -> ChromaDBConfig:
        """Create ChromaDBConfig from environment variables."""
        return ChromaDBConfig(
            host=self._get_env_var("CHROMADB_HOST", "localhost"),
            port=self._get_int_env("CHROMADB_PORT", 8000),
            collection_name=self._get_env_var("CHROMADB_COLLECTION_NAME", "deep_agent_knowledge"),
            persist_directory=self._get_env_var("CHROMADB_PERSIST_DIRECTORY", "./data/chromadb"),
        )
    
    def _create_neo4j_config(self) -> Neo4jConfig:
        """Create Neo4jConfig from environment variables."""
        return Neo4jConfig(
            uri=self._get_env_var("NEO4J_URI", "bolt://localhost:7687"),
            username=self._get_env_var("NEO4J_USERNAME", "neo4j"),
            password=self._get_env_var("NEO4J_PASSWORD", required=True),
            database=self._get_env_var("NEO4J_DATABASE", "neo4j"),
        )
    
    def _create_database_config(self) -> DatabaseConfig:
        """Create DatabaseConfig from environment variables."""
        return DatabaseConfig(
            chromadb=self._create_chromadb_config(),
            neo4j=self._create_neo4j_config(),
        )
    
    def _create_rag_config(self) -> RAGConfig:
        """Create RAGConfig from environment variables."""
        return RAGConfig(
            embedding_model=self._get_env_var("EMBEDDING_MODEL", "text-embedding-ada-002"),
            embedding_dimension=self._get_int_env("EMBEDDING_DIMENSION", 1536),
            chunk_size=self._get_int_env("RAG_CHUNK_SIZE", 1000),
            chunk_overlap=self._get_int_env("RAG_CHUNK_OVERLAP", 200),
            retrieval_k=self._get_int_env("RAG_RETRIEVAL_K", 5),
            similarity_threshold=self._get_float_env("RAG_SIMILARITY_THRESHOLD", 0.7),
            vector_store_enabled=self._get_bool_env("VECTOR_STORE_ENABLED", True),
            graph_store_enabled=self._get_bool_env("GRAPH_STORE_ENABLED", True),
        )
    
    def _create_cli_config(self) -> CLIConfig:
        """Create CLIConfig from environment variables."""
        return CLIConfig(
            theme=self._get_env_var("CLI_THEME", "dark"),
            syntax_highlighting=self._get_bool_env("CLI_SYNTAX_HIGHLIGHTING", True),
            show_progress=self._get_bool_env("CLI_SHOW_PROGRESS", True),
            max_width=self._get_int_env("CLI_MAX_WIDTH", 120),
        )
    
    def _create_logging_config(self) -> LoggingConfig:
        """Create LoggingConfig from environment variables."""
        log_level_str = self._get_env_var("LOG_LEVEL", "INFO").upper()
        log_format_str = self._get_env_var("LOG_FORMAT", "json").lower()
        
        try:
            log_level = LogLevel(log_level_str)
        except ValueError:
            raise ConfigurationError(f"Invalid log level: {log_level_str}")
        
        try:
            log_format = LogFormat(log_format_str)
        except ValueError:
            raise ConfigurationError(f"Invalid log format: {log_format_str}")
        
        return LoggingConfig(
            level=log_level,
            format=log_format,
            log_file=self._get_env_var("LOG_FILE", "./logs/deep_agent_system.log"),
            enable_debug_logging=self._get_bool_env("ENABLE_DEBUG_LOGGING", False),
        )
    
    def _create_workflow_config(self) -> WorkflowConfig:
        """Create WorkflowConfig from environment variables."""
        return WorkflowConfig(
            timeout=self._get_int_env("WORKFLOW_TIMEOUT", 600),
            max_steps=self._get_int_env("WORKFLOW_MAX_STEPS", 50),
            enable_checkpoints=self._get_bool_env("ENABLE_WORKFLOW_CHECKPOINTS", True),
            checkpoint_dir=self._get_env_var("WORKFLOW_CHECKPOINT_DIR", "./data/checkpoints"),
        )
    
    def _create_security_config(self) -> SecurityConfig:
        """Create SecurityConfig from environment variables."""
        return SecurityConfig(
            api_rate_limit=self._get_int_env("API_RATE_LIMIT", 100),
            api_rate_window=self._get_int_env("API_RATE_WINDOW", 60),
            enable_input_validation=self._get_bool_env("ENABLE_INPUT_VALIDATION", True),
            max_input_length=self._get_int_env("MAX_INPUT_LENGTH", 10000),
            sanitize_outputs=self._get_bool_env("SANITIZE_OUTPUTS", True),
        )
    
    def _create_development_config(self) -> DevelopmentConfig:
        """Create DevelopmentConfig from environment variables."""
        return DevelopmentConfig(
            debug_mode=self._get_bool_env("DEBUG_MODE", False),
            development_mode=self._get_bool_env("DEVELOPMENT_MODE", False),
            enable_hot_reload=self._get_bool_env("ENABLE_HOT_RELOAD", False),
            test_mode=self._get_bool_env("TEST_MODE", False),
            mock_external_apis=self._get_bool_env("MOCK_EXTERNAL_APIS", False),
        )
    
    def _create_agent_configs(self) -> Dict[str, AgentConfig]:
        """Create default agent configurations."""
        model_config = self._create_model_config()
        
        agents = {}
        for agent_type in AgentType:
            agent_id = f"{agent_type.value}_agent"
            agents[agent_id] = AgentConfig(
                agent_id=agent_id,
                agent_type=agent_type,
                llm_config=model_config,
                prompt_templates={},  # Will be populated by PromptManager
                capabilities=[],  # Will be defined per agent type
                rag_enabled=True,
                graph_rag_enabled=True,
                timeout=self._get_int_env("AGENT_TIMEOUT", 300),
            )
        
        return agents
    
    def load_config(self) -> SystemConfig:
        """
        Load and validate the complete system configuration.
        
        Returns:
            SystemConfig: Validated system configuration
            
        Raises:
            ConfigurationError: If configuration is invalid or required values are missing
        """
        try:
            config = SystemConfig(
                agents=self._create_agent_configs(),
                rag_config=self._create_rag_config(),
                database_config=self._create_database_config(),
                prompt_dir=self._get_env_var("PROMPT_DIR", "./prompts"),
                cli_config=self._create_cli_config(),
                logging_config=self._create_logging_config(),
                workflow_config=self._create_workflow_config(),
                security_config=self._create_security_config(),
                development_config=self._create_development_config(),
            )
            
            return config
            
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def validate_config(self, config: SystemConfig) -> bool:
        """
        Validate a SystemConfig instance.
        
        Args:
            config: SystemConfig to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Pydantic validation happens automatically during model creation
            # Additional custom validation can be added here
            
            # Validate prompt directory exists
            prompt_path = Path(config.prompt_dir)
            if not prompt_path.exists():
                raise ConfigurationError(f"Prompt directory does not exist: {config.prompt_dir}")
            
            # Validate that required agent types are present
            required_agents = {agent_type.value for agent_type in AgentType}
            configured_agents = {agent.agent_type.value for agent in config.agents.values()}
            
            missing_agents = required_agents - configured_agents
            if missing_agents:
                raise ConfigurationError(f"Missing required agent types: {missing_agents}")
            
            return True
            
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def get_agent_config(self, config: SystemConfig, agent_type: AgentType) -> Optional[AgentConfig]:
        """
        Get configuration for a specific agent type.
        
        Args:
            config: System configuration
            agent_type: Type of agent to get configuration for
            
        Returns:
            AgentConfig or None if not found
        """
        for agent_config in config.agents.values():
            if agent_config.agent_type == agent_type:
                return agent_config
        return None