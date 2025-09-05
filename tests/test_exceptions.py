"""Tests for custom exception classes."""

import pytest

from deep_agent_system.exceptions import (
    AgentCommunicationError,
    AgentInitializationError,
    AgentNotFoundError,
    ConfigurationError,
    DeepAgentException,
    GraphStoreError,
    InvalidConfigurationError,
    MessageQueueFullError,
    MessageTimeoutError,
    MissingConfigurationError,
    PromptLoadError,
    PromptNotFoundError,
    PromptValidationError,
    RAGRetrievalError,
    SystemShutdownError,
    VectorStoreError,
    WorkflowExecutionError,
    WorkflowValidationError,
    agent_not_found,
    message_timeout,
    missing_config,
    prompt_not_found,
    rag_retrieval_failed,
)


class TestDeepAgentException:
    """Test base DeepAgentException functionality."""
    
    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = DeepAgentException("Test error message")
        
        assert str(exc) == "Test error message"
        assert exc.message == "Test error message"
        assert exc.error_code is None
        assert exc.details == {}
        assert exc.cause is None
    
    def test_exception_with_all_fields(self):
        """Test exception with all fields."""
        cause = ValueError("underlying error")
        details = {"key": "value", "count": 42}
        
        exc = DeepAgentException(
            message="Test error",
            error_code="TEST_ERROR",
            details=details,
            cause=cause,
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == "TEST_ERROR"
        assert exc.details == details
        assert exc.cause == cause
    
    def test_to_dict(self):
        """Test exception to_dict conversion."""
        cause = ValueError("underlying error")
        details = {"key": "value"}
        
        exc = DeepAgentException(
            message="Test error",
            error_code="TEST_ERROR",
            details=details,
            cause=cause,
        )
        
        result = exc.to_dict()
        
        assert result["error_type"] == "DeepAgentException"
        assert result["message"] == "Test error"
        assert result["error_code"] == "TEST_ERROR"
        assert result["details"] == details
        assert result["cause"] == "underlying error"
    
    def test_str_representation(self):
        """Test string representation with all fields."""
        cause = ValueError("underlying error")
        details = {"key": "value"}
        
        exc = DeepAgentException(
            message="Test error",
            error_code="TEST_ERROR",
            details=details,
            cause=cause,
        )
        
        str_repr = str(exc)
        assert "Test error" in str_repr
        assert "Code: TEST_ERROR" in str_repr
        assert "Details: {'key': 'value'}" in str_repr
        assert "Caused by: underlying error" in str_repr


class TestAgentCommunicationError:
    """Test AgentCommunicationError and related exceptions."""
    
    def test_agent_communication_error(self):
        """Test basic AgentCommunicationError."""
        exc = AgentCommunicationError(
            message="Communication failed",
            sender_id="agent1",
            recipient_id="agent2",
            message_id="msg123",
        )
        
        assert exc.message == "Communication failed"
        assert exc.sender_id == "agent1"
        assert exc.recipient_id == "agent2"
        assert exc.message_id == "msg123"
        assert exc.details["sender_id"] == "agent1"
        assert exc.details["recipient_id"] == "agent2"
        assert exc.details["message_id"] == "msg123"
    
    def test_agent_not_found_error(self):
        """Test AgentNotFoundError."""
        exc = AgentNotFoundError("agent123")
        
        assert "Agent 'agent123' not found" in exc.message
        assert exc.error_code == "AGENT_NOT_FOUND"
        assert exc.agent_id == "agent123"
    
    def test_message_timeout_error(self):
        """Test MessageTimeoutError."""
        exc = MessageTimeoutError(
            timeout_seconds=30.0,
            sender_id="agent1",
            recipient_id="agent2",
        )
        
        assert "timed out after 30.0 seconds" in exc.message
        assert exc.error_code == "MESSAGE_TIMEOUT"
        assert exc.timeout_seconds == 30.0
        assert exc.sender_id == "agent1"
        assert exc.recipient_id == "agent2"
    
    def test_message_queue_full_error(self):
        """Test MessageQueueFullError."""
        exc = MessageQueueFullError("agent1", 1000)
        
        assert "queue for agent 'agent1' is full" in exc.message
        assert exc.error_code == "QUEUE_FULL"
        assert exc.agent_id == "agent1"
        assert exc.queue_size == 1000


class TestPromptLoadError:
    """Test PromptLoadError and related exceptions."""
    
    def test_prompt_load_error(self):
        """Test basic PromptLoadError."""
        exc = PromptLoadError(
            message="Failed to load prompt",
            prompt_name="test_prompt",
            agent_type="analyst",
            prompt_path="/path/to/prompt.txt",
        )
        
        assert exc.message == "Failed to load prompt"
        assert exc.prompt_name == "test_prompt"
        assert exc.agent_type == "analyst"
        assert exc.prompt_path == "/path/to/prompt.txt"
        assert exc.details["prompt_name"] == "test_prompt"
        assert exc.details["agent_type"] == "analyst"
        assert exc.details["prompt_path"] == "/path/to/prompt.txt"
    
    def test_prompt_not_found_error(self):
        """Test PromptNotFoundError."""
        exc = PromptNotFoundError("classification", "analyst")
        
        assert "Prompt 'classification' not found for agent type 'analyst'" in exc.message
        assert exc.error_code == "PROMPT_NOT_FOUND"
        assert exc.prompt_name == "classification"
        assert exc.agent_type == "analyst"
    
    def test_prompt_validation_error(self):
        """Test PromptValidationError."""
        validation_errors = ["Missing required field", "Invalid format"]
        exc = PromptValidationError("test_prompt", validation_errors)
        
        assert "Prompt 'test_prompt' failed validation" in exc.message
        assert exc.error_code == "PROMPT_VALIDATION_FAILED"
        assert exc.prompt_name == "test_prompt"
        assert exc.validation_errors == validation_errors
        assert exc.details["validation_errors"] == validation_errors


class TestRAGRetrievalError:
    """Test RAGRetrievalError and related exceptions."""
    
    def test_rag_retrieval_error(self):
        """Test basic RAGRetrievalError."""
        exc = RAGRetrievalError(
            message="Retrieval failed",
            query="test query",
            retrieval_type="vector",
            store_type="chromadb",
        )
        
        assert exc.message == "Retrieval failed"
        assert exc.query == "test query"
        assert exc.retrieval_type == "vector"
        assert exc.store_type == "chromadb"
        assert exc.details["query"] == "test query"
        assert exc.details["retrieval_type"] == "vector"
        assert exc.details["store_type"] == "chromadb"
    
    def test_vector_store_error(self):
        """Test VectorStoreError."""
        exc = VectorStoreError("Vector operation failed")
        
        assert exc.message == "Vector operation failed"
        assert exc.error_code == "VECTOR_STORE_ERROR"
        assert exc.store_type == "vector"
    
    def test_graph_store_error(self):
        """Test GraphStoreError."""
        exc = GraphStoreError("Graph operation failed")
        
        assert exc.message == "Graph operation failed"
        assert exc.error_code == "GRAPH_STORE_ERROR"
        assert exc.store_type == "graph"


class TestWorkflowExecutionError:
    """Test WorkflowExecutionError and related exceptions."""
    
    def test_workflow_execution_error(self):
        """Test basic WorkflowExecutionError."""
        exc = WorkflowExecutionError(
            message="Workflow failed",
            workflow_id="wf123",
            step_name="process_data",
            agent_id="agent1",
        )
        
        assert exc.message == "Workflow failed"
        assert exc.workflow_id == "wf123"
        assert exc.step_name == "process_data"
        assert exc.agent_id == "agent1"
        assert exc.details["workflow_id"] == "wf123"
        assert exc.details["step_name"] == "process_data"
        assert exc.details["agent_id"] == "agent1"
    
    def test_workflow_validation_error(self):
        """Test WorkflowValidationError."""
        validation_errors = ["Missing step", "Invalid transition"]
        exc = WorkflowValidationError("wf123", validation_errors)
        
        assert "Workflow 'wf123' failed validation" in exc.message
        assert exc.error_code == "WORKFLOW_VALIDATION_FAILED"
        assert exc.workflow_id == "wf123"
        assert exc.validation_errors == validation_errors
        assert exc.details["validation_errors"] == validation_errors


class TestConfigurationError:
    """Test ConfigurationError and related exceptions."""
    
    def test_configuration_error(self):
        """Test basic ConfigurationError."""
        exc = ConfigurationError(
            message="Config error",
            config_key="api_key",
            config_file=".env",
        )
        
        assert exc.message == "Config error"
        assert exc.config_key == "api_key"
        assert exc.config_file == ".env"
        assert exc.details["config_key"] == "api_key"
        assert exc.details["config_file"] == ".env"
    
    def test_missing_configuration_error(self):
        """Test MissingConfigurationError."""
        exc = MissingConfigurationError("API_KEY")
        
        assert "Required configuration 'API_KEY' is missing" in exc.message
        assert exc.error_code == "MISSING_CONFIGURATION"
        assert exc.config_key == "API_KEY"
    
    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        exc = InvalidConfigurationError(
            config_key="PORT",
            value="invalid_port",
            reason="must be a number",
        )
        
        assert "Invalid configuration 'PORT' = 'invalid_port': must be a number" in exc.message
        assert exc.error_code == "INVALID_CONFIGURATION"
        assert exc.config_key == "PORT"
        assert exc.value == "invalid_port"
        assert exc.reason == "must be a number"
        assert exc.details["invalid_value"] == "invalid_port"
        assert exc.details["reason"] == "must be a number"


class TestOtherExceptions:
    """Test other exception types."""
    
    def test_agent_initialization_error(self):
        """Test AgentInitializationError."""
        exc = AgentInitializationError(
            message="Failed to initialize agent",
            agent_id="agent1",
            agent_type="analyst",
        )
        
        assert exc.message == "Failed to initialize agent"
        assert exc.agent_id == "agent1"
        assert exc.agent_type == "analyst"
        assert exc.details["agent_id"] == "agent1"
        assert exc.details["agent_type"] == "analyst"
    
    def test_system_shutdown_error(self):
        """Test SystemShutdownError."""
        exc = SystemShutdownError(
            message="Shutdown failed",
            component="communication_manager",
        )
        
        assert exc.message == "Shutdown failed"
        assert exc.error_code == "SHUTDOWN_ERROR"
        assert exc.component == "communication_manager"
        assert exc.details["component"] == "communication_manager"


class TestConvenienceFunctions:
    """Test convenience functions for creating exceptions."""
    
    def test_agent_not_found_function(self):
        """Test agent_not_found convenience function."""
        exc = agent_not_found("agent123")
        
        assert isinstance(exc, AgentNotFoundError)
        assert exc.agent_id == "agent123"
        assert "Agent 'agent123' not found" in exc.message
    
    def test_message_timeout_function(self):
        """Test message_timeout convenience function."""
        exc = message_timeout(30.0, "sender", "recipient")
        
        assert isinstance(exc, MessageTimeoutError)
        assert exc.timeout_seconds == 30.0
        assert exc.sender_id == "sender"
        assert exc.recipient_id == "recipient"
    
    def test_prompt_not_found_function(self):
        """Test prompt_not_found convenience function."""
        exc = prompt_not_found("classification", "analyst")
        
        assert isinstance(exc, PromptNotFoundError)
        assert exc.prompt_name == "classification"
        assert exc.agent_type == "analyst"
    
    def test_rag_retrieval_failed_function(self):
        """Test rag_retrieval_failed convenience function."""
        cause = ConnectionError("connection failed")
        exc = rag_retrieval_failed("test query", "vector", cause)
        
        assert isinstance(exc, RAGRetrievalError)
        assert exc.query == "test query"
        assert exc.retrieval_type == "vector"
        assert exc.cause == cause
    
    def test_missing_config_function(self):
        """Test missing_config convenience function."""
        exc = missing_config("API_KEY")
        
        assert isinstance(exc, MissingConfigurationError)
        assert exc.config_key == "API_KEY"


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""
    
    def test_inheritance_hierarchy(self):
        """Test that all exceptions inherit from DeepAgentException."""
        exceptions = [
            AgentCommunicationError("test"),
            AgentNotFoundError("agent1"),
            MessageTimeoutError(30.0),
            MessageQueueFullError("agent1", 100),
            PromptLoadError("test"),
            PromptNotFoundError("prompt", "agent"),
            PromptValidationError("prompt", []),
            RAGRetrievalError("test"),
            VectorStoreError("test"),
            GraphStoreError("test"),
            WorkflowExecutionError("test"),
            WorkflowValidationError("wf1", []),
            ConfigurationError("test"),
            MissingConfigurationError("key"),
            InvalidConfigurationError("key", "value", "reason"),
            AgentInitializationError("test"),
            SystemShutdownError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, DeepAgentException)
            assert isinstance(exc, Exception)
    
    def test_specific_inheritance(self):
        """Test specific inheritance relationships."""
        # AgentCommunicationError hierarchy
        assert isinstance(AgentNotFoundError("test"), AgentCommunicationError)
        assert isinstance(MessageTimeoutError(30.0), AgentCommunicationError)
        assert isinstance(MessageQueueFullError("agent", 100), AgentCommunicationError)
        
        # PromptLoadError hierarchy
        assert isinstance(PromptNotFoundError("prompt", "agent"), PromptLoadError)
        assert isinstance(PromptValidationError("prompt", []), PromptLoadError)
        
        # RAGRetrievalError hierarchy
        assert isinstance(VectorStoreError("test"), RAGRetrievalError)
        assert isinstance(GraphStoreError("test"), RAGRetrievalError)
        
        # WorkflowExecutionError hierarchy
        assert isinstance(WorkflowValidationError("wf", []), WorkflowExecutionError)
        
        # ConfigurationError hierarchy
        assert isinstance(MissingConfigurationError("key"), ConfigurationError)
        assert isinstance(InvalidConfigurationError("key", "val", "reason"), ConfigurationError)