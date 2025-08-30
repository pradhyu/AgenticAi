"""Configuration data models for the Deep Agent System."""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pathlib import Path


class AgentType(str, Enum):
    """Enumeration of available agent types."""
    ANALYST = "analyst"
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    CODE_REVIEWER = "code_reviewer"
    TESTER = "tester"


class RetrievalType(str, Enum):
    """Enumeration of RAG retrieval types."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class LogLevel(str, Enum):
    """Enumeration of log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Enumeration of log formats."""
    JSON = "json"
    TEXT = "text"


class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    provider: str = Field(default="openai", description="LLM provider")
    model_name: str = Field(default="gpt-4-turbo-preview", description="Model name")
    api_key: str = Field(description="API key for the model provider")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=4000, gt=0, description="Maximum tokens per response")
    timeout: int = Field(default=300, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries")

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v


class ChromaDBConfig(BaseModel):
    """Configuration for ChromaDB vector store."""
    host: str = Field(default="localhost", description="ChromaDB host")
    port: int = Field(default=8000, gt=0, le=65535, description="ChromaDB port")
    collection_name: str = Field(default="deep_agent_knowledge", description="Collection name")
    persist_directory: str = Field(default="./data/chromadb", description="Persistence directory")
    
    @field_validator('persist_directory')
    @classmethod
    def validate_persist_directory(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j graph database."""
    uri: str = Field(description="Neo4j connection URI")
    username: str = Field(description="Neo4j username")
    password: str = Field(description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")
    
    @field_validator('uri')
    @classmethod
    def validate_uri(cls, v):
        if not v.startswith(('bolt://', 'neo4j://', 'bolt+s://', 'neo4j+s://')):
            raise ValueError('Neo4j URI must start with bolt://, neo4j://, bolt+s://, or neo4j+s://')
        return v


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) system."""
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model name")
    embedding_dimension: int = Field(default=1536, gt=0, description="Embedding dimension")
    chunk_size: int = Field(default=1000, gt=0, description="Document chunk size")
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap size")
    retrieval_k: int = Field(default=5, gt=0, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    vector_store_enabled: bool = Field(default=True, description="Enable vector store")
    graph_store_enabled: bool = Field(default=True, description="Enable graph store")
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        if hasattr(info, 'data') and 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError('Chunk overlap must be less than chunk size')
        return v


class DatabaseConfig(BaseModel):
    """Configuration for database connections."""
    chromadb: ChromaDBConfig = Field(description="ChromaDB configuration")
    neo4j: Neo4jConfig = Field(description="Neo4j configuration")


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    agent_id: str = Field(description="Unique agent identifier")
    agent_type: AgentType = Field(description="Type of agent")
    llm_config: ModelConfig = Field(description="Model configuration for this agent")
    prompt_templates: Dict[str, str] = Field(default_factory=dict, description="Prompt template mappings")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    rag_enabled: bool = Field(default=True, description="Enable RAG for this agent")
    graph_rag_enabled: bool = Field(default=True, description="Enable graph RAG for this agent")
    timeout: int = Field(default=300, gt=0, description="Agent timeout in seconds")


class CLIConfig(BaseModel):
    """Configuration for CLI interface."""
    theme: str = Field(default="dark", description="CLI theme")
    syntax_highlighting: bool = Field(default=True, description="Enable syntax highlighting")
    show_progress: bool = Field(default=True, description="Show progress indicators")
    max_width: int = Field(default=120, gt=0, description="Maximum display width")


class LoggingConfig(BaseModel):
    """Configuration for logging system."""
    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    format: LogFormat = Field(default=LogFormat.JSON, description="Log format")
    log_file: Optional[str] = Field(default="./logs/deep_agent_system.log", description="Log file path")
    enable_debug_logging: bool = Field(default=False, description="Enable debug logging")
    
    @field_validator('log_file')
    @classmethod
    def validate_log_file(cls, v):
        if v:
            log_path = Path(v)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            return str(log_path.absolute())
        return v


class WorkflowConfig(BaseModel):
    """Configuration for LangGraph workflows."""
    timeout: int = Field(default=600, gt=0, description="Workflow timeout in seconds")
    max_steps: int = Field(default=50, gt=0, description="Maximum workflow steps")
    enable_checkpoints: bool = Field(default=True, description="Enable workflow checkpoints")
    checkpoint_dir: str = Field(default="./data/checkpoints", description="Checkpoint directory")
    
    @field_validator('checkpoint_dir')
    @classmethod
    def validate_checkpoint_dir(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    api_rate_limit: int = Field(default=100, gt=0, description="API rate limit per window")
    api_rate_window: int = Field(default=60, gt=0, description="Rate limit window in seconds")
    enable_input_validation: bool = Field(default=True, description="Enable input validation")
    max_input_length: int = Field(default=10000, gt=0, description="Maximum input length")
    sanitize_outputs: bool = Field(default=True, description="Sanitize outputs")


class DevelopmentConfig(BaseModel):
    """Configuration for development settings."""
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    development_mode: bool = Field(default=False, description="Enable development mode")
    enable_hot_reload: bool = Field(default=False, description="Enable hot reload")
    test_mode: bool = Field(default=False, description="Enable test mode")
    mock_external_apis: bool = Field(default=False, description="Mock external APIs")


class SystemConfig(BaseModel):
    """Main system configuration containing all subsystem configurations."""
    agents: Dict[str, AgentConfig] = Field(description="Agent configurations")
    rag_config: RAGConfig = Field(description="RAG system configuration")
    database_config: DatabaseConfig = Field(description="Database configurations")
    prompt_dir: str = Field(default="./prompts", description="Prompt directory path")
    cli_config: CLIConfig = Field(default_factory=CLIConfig, description="CLI configuration")
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    workflow_config: WorkflowConfig = Field(default_factory=WorkflowConfig, description="Workflow configuration")
    security_config: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    development_config: DevelopmentConfig = Field(default_factory=DevelopmentConfig, description="Development configuration")
    
    @field_validator('prompt_dir')
    @classmethod
    def validate_prompt_dir(cls, v):
        path = Path(v)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True
    )