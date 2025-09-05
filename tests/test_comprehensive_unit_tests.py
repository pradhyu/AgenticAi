"""Comprehensive unit tests for core components."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from deep_agent_system.models.agents import (
    AgentConfig, 
    AgentType, 
    Capability, 
    LLMConfig,
    AgentState
)
from deep_agent_system.models.messages import (
    Message, 
    Response, 
    Context, 
    Document,
    GraphData,
    ConversationHistory,
    MessageType,
    RetrievalType
)
from deep_agent_system.config.models import (
    SystemConfig,
    RAGConfig,
    ChromaDBConfig,
    Neo4jConfig,
    ModelConfig,
    CLIConfig,
    LoggingConfig,
    WorkflowConfig,
    SecurityConfig,
    DevelopmentConfig,
    DatabaseConfig,
    LogLevel,
    LogFormat
)
from deep_agent_system.exceptions import (
    DeepAgentException,
    AgentNotFoundError,
    PromptNotFoundError,
    RAGRetrievalError,
    WorkflowExecutionError,
    ConfigurationError,
    AgentCommunicationError
)


class TestLLMConfig:
    """Test cases for LLMConfig model."""
    
    def test_llm_config_creation(self):
        """Test creating LLMConfig with valid parameters."""
        config = LLMConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            api_key_env_var="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1"
        )
        
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.2
        assert config.api_key_env_var == "OPENAI_API_KEY"
        assert config.base_url == "https://api.openai.com/v1"
    
    def test_llm_config_defaults(self):
        """Test LLMConfig with default values."""
        config = LLMConfig(model_name="gpt-3.5-turbo")
        
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.api_key_env_var == "OPENAI_API_KEY"
        assert config.base_url is None
    
    def test_llm_config_validation_temperature(self):
        """Test temperature validation."""
        # Valid temperatures
        LLMConfig(model_name="gpt-4", temperature=0.0)
        LLMConfig(model_name="gpt-4", temperature=1.0)
        LLMConfig(model_name="gpt-4", temperature=2.0)
        
        # Invalid temperatures
        with pytest.raises(ValueError):
            LLMConfig(model_name="gpt-4", temperature=-0.1)
        
        with pytest.raises(ValueError):
            LLMConfig(model_name="gpt-4", temperature=2.1)
    
    def test_llm_config_validation_model_name(self):
        """Test model name validation."""
        # Valid model names
        LLMConfig(model_name="gpt-4")
        LLMConfig(model_name="  gpt-4  ")  # Should be stripped
        
        # Invalid model names
        with pytest.raises(ValueError):
            LLMConfig(model_name="")
        
        with pytest.raises(ValueError):
            LLMConfig(model_name="   ")
    
    def test_llm_config_validation_max_tokens(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        LLMConfig(model_name="gpt-4", max_tokens=1)
        LLMConfig(model_name="gpt-4", max_tokens=4000)
        LLMConfig(model_name="gpt-4", max_tokens=None)
        
        # Invalid max_tokens
        with pytest.raises(ValueError):
            LLMConfig(model_name="gpt-4", max_tokens=0)
        
        with pytest.raises(ValueError):
            LLMConfig(model_name="gpt-4", max_tokens=-1)


class TestAgentConfig:
    """Test cases for AgentConfig model."""
    
    def test_agent_config_creation(self):
        """Test creating AgentConfig with valid parameters."""
        llm_config = LLMConfig(model_name="gpt-4")
        config = AgentConfig(
            agent_id="test_agent",
            agent_type=AgentType.ANALYST,
            llm_config=llm_config,
            capabilities=[Capability.QUESTION_ROUTING, Capability.CLASSIFICATION],
            rag_enabled=True,
            graph_rag_enabled=True,
            response_timeout=60
        )
        
        assert config.agent_id == "test_agent"
        assert config.agent_type == AgentType.ANALYST
        assert config.llm_config == llm_config
        assert Capability.QUESTION_ROUTING in config.capabilities
        assert Capability.CLASSIFICATION in config.capabilities
        assert config.rag_enabled is True
        assert config.graph_rag_enabled is True
        assert config.response_timeout == 60
    
    def test_agent_config_defaults(self):
        """Test AgentConfig with default values."""
        llm_config = LLMConfig(model_name="gpt-4")
        config = AgentConfig(
            agent_id="test_agent",
            agent_type=AgentType.DEVELOPER,
            llm_config=llm_config,
            capabilities=[Capability.CODE_GENERATION]  # Required for DEVELOPER type
        )
        
        assert Capability.CODE_GENERATION in config.capabilities
        assert config.rag_enabled is True
        assert config.graph_rag_enabled is True
        assert config.response_timeout == 30  # Default timeout
        assert config.prompt_templates == {}


class TestAgentState:
    """Test cases for AgentState model."""
    
    def test_agent_state_creation(self):
        """Test creating AgentState."""
        state = AgentState(agent_id="test_agent")
        
        assert state.agent_id == "test_agent"
        assert state.is_active is True
        assert state.current_task is None
        assert state.conversation_count == 0
        assert state.last_activity is None  # Initially None
    
    def test_agent_state_update_activity(self):
        """Test updating agent activity."""
        state = AgentState(agent_id="test_agent")
        initial_activity = state.last_activity
        
        # Wait a bit to ensure timestamp changes
        import time
        time.sleep(0.01)
        
        state.update_activity("Processing message")
        
        assert state.current_task == "Processing message"
        assert state.last_activity != initial_activity
    
    def test_agent_state_clear_task(self):
        """Test clearing current task."""
        state = AgentState(agent_id="test_agent")
        state.update_activity("Test task")
        
        assert state.current_task == "Test task"
        
        state.clear_task()
        
        assert state.current_task is None
    
    def test_agent_state_increment_conversation(self):
        """Test incrementing conversation count through update_activity."""
        state = AgentState(agent_id="test_agent")
        initial_count = state.conversation_count
        
        # update_activity increments conversation count
        state.update_activity("Test task")
        
        assert state.conversation_count == initial_count + 1


class TestMessage:
    """Test cases for Message model."""
    
    def test_message_creation_with_all_fields(self):
        """Test creating Message with all fields."""
        metadata = {"priority": "high", "source": "user"}
        message = Message(
            sender_id="user_123",
            recipient_id="agent_456",
            content="Hello, agent!",
            message_type=MessageType.USER_QUESTION,
            metadata=metadata
        )
        
        assert message.sender_id == "user_123"
        assert message.recipient_id == "agent_456"
        assert message.content == "Hello, agent!"
        assert message.message_type == MessageType.USER_QUESTION
        assert message.metadata == metadata
        assert isinstance(message.id, str)
        assert isinstance(message.timestamp, datetime)
    
    def test_message_content_stripping(self):
        """Test that message content is stripped."""
        message = Message(
            sender_id="user",
            recipient_id="agent",
            content="  Hello, world!  ",
            message_type=MessageType.USER_QUESTION
        )
        
        assert message.content == "Hello, world!"
    
    def test_message_empty_content_validation(self):
        """Test validation of empty content."""
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            Message(
                sender_id="user",
                recipient_id="agent",
                content="",
                message_type=MessageType.USER_QUESTION
            )
        
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            Message(
                sender_id="user",
                recipient_id="agent",
                content="   ",
                message_type=MessageType.USER_QUESTION
            )
    
    def test_message_serialization(self):
        """Test message serialization."""
        message = Message(
            sender_id="user",
            recipient_id="agent",
            content="Test message",
            message_type=MessageType.AGENT_REQUEST,
            metadata={"test": True}
        )
        
        data = message.to_dict()
        
        assert data["sender_id"] == "user"
        assert data["recipient_id"] == "agent"
        assert data["content"] == "Test message"
        assert data["message_type"] == "agent_request"
        assert data["metadata"] == {"test": True}
        assert "id" in data
        assert "timestamp" in data
    
    def test_message_deserialization(self):
        """Test message deserialization."""
        data = {
            "id": "test_id_123",
            "sender_id": "user",
            "recipient_id": "agent",
            "content": "Test message",
            "message_type": "user_question",
            "metadata": {"test": True},
            "timestamp": "2023-01-01T12:00:00"
        }
        
        message = Message.from_dict(data)
        
        assert message.id == "test_id_123"
        assert message.sender_id == "user"
        assert message.recipient_id == "agent"
        assert message.content == "Test message"
        assert message.message_type == MessageType.USER_QUESTION
        assert message.metadata == {"test": True}


class TestResponse:
    """Test cases for Response model."""
    
    def test_response_creation_minimal(self):
        """Test creating Response with minimal parameters."""
        response = Response(
            message_id="msg_123",
            agent_id="agent_456",
            content="This is a response"
        )
        
        assert response.message_id == "msg_123"
        assert response.agent_id == "agent_456"
        assert response.content == "This is a response"
        assert response.confidence_score == 1.0
        assert response.context_used == []
        assert response.workflow_id is None
        assert response.metadata == {}
        assert isinstance(response.timestamp, datetime)
    
    def test_response_creation_full(self):
        """Test creating Response with all parameters."""
        response = Response(
            message_id="msg_123",
            agent_id="agent_456",
            content="Detailed response",
            confidence_score=0.85,
            context_used=["doc1", "doc2"],
            workflow_id="wf_789",
            metadata={"processing_time": 1.5}
        )
        
        assert response.confidence_score == 0.85
        assert response.context_used == ["doc1", "doc2"]
        assert response.workflow_id == "wf_789"
        assert response.metadata == {"processing_time": 1.5}
    
    def test_response_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        Response(message_id="msg", agent_id="agent", content="test", confidence_score=0.0)
        Response(message_id="msg", agent_id="agent", content="test", confidence_score=0.5)
        Response(message_id="msg", agent_id="agent", content="test", confidence_score=1.0)
        
        # Invalid confidence scores
        with pytest.raises(ValueError):
            Response(message_id="msg", agent_id="agent", content="test", confidence_score=-0.1)
        
        with pytest.raises(ValueError):
            Response(message_id="msg", agent_id="agent", content="test", confidence_score=1.1)
    
    def test_response_empty_content_validation(self):
        """Test validation of empty content."""
        with pytest.raises(ValueError, match="Response content cannot be empty"):
            Response(
                message_id="msg",
                agent_id="agent",
                content=""
            )


class TestDocument:
    """Test cases for Document model."""
    
    def test_document_creation_minimal(self):
        """Test creating Document with minimal parameters."""
        doc = Document(content="This is document content")
        
        assert doc.content == "This is document content"
        assert doc.metadata == {}
        assert doc.source is None
        assert doc.chunk_id is None
    
    def test_document_creation_full(self):
        """Test creating Document with all parameters."""
        metadata = {"author": "John Doe", "date": "2023-01-01"}
        doc = Document(
            content="Full document content",
            metadata=metadata,
            source="document.txt",
            chunk_id="chunk_1"
        )
        
        assert doc.content == "Full document content"
        assert doc.metadata == metadata
        assert doc.source == "document.txt"
        assert doc.chunk_id == "chunk_1"
    
    def test_document_empty_content_validation(self):
        """Test validation of empty content."""
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            Document(content="")


class TestContext:
    """Test cases for Context model."""
    
    def test_context_creation_minimal(self):
        """Test creating Context with minimal parameters."""
        context = Context(query="test query")
        
        assert context.query == "test query"
        assert context.documents == []
        assert context.graph_data is None
        assert context.relevance_scores == []
        assert context.retrieval_method == RetrievalType.NONE
    
    def test_context_creation_with_documents(self):
        """Test creating Context with documents."""
        docs = [
            Document(content="Doc 1"),
            Document(content="Doc 2")
        ]
        scores = [0.9, 0.7]
        
        context = Context(
            query="test query",
            documents=docs,
            relevance_scores=scores,
            retrieval_method=RetrievalType.VECTOR
        )
        
        assert len(context.documents) == 2
        assert context.relevance_scores == [0.9, 0.7]
        assert context.retrieval_method == RetrievalType.VECTOR
    
    def test_context_score_validation(self):
        """Test relevance score validation."""
        docs = [Document(content="Doc 1")]
        scores = [0.9, 0.7]  # More scores than documents
        
        with pytest.raises(ValueError, match="Number of relevance scores must match"):
            Context(
                query="test query",
                documents=docs,
                relevance_scores=scores
            )
    
    def test_get_top_documents(self):
        """Test getting top documents by relevance."""
        docs = [
            Document(content="Doc 1"),
            Document(content="Doc 2"),
            Document(content="Doc 3")
        ]
        scores = [0.7, 0.9, 0.5]
        
        context = Context(
            query="test query",
            documents=docs,
            relevance_scores=scores
        )
        
        top_docs = context.get_top_documents(k=2)
        
        assert len(top_docs) == 2
        assert top_docs[0].content == "Doc 2"  # Highest score (0.9)
        assert top_docs[1].content == "Doc 1"  # Second highest (0.7)
    
    def test_get_top_documents_without_scores(self):
        """Test getting top documents without relevance scores."""
        docs = [
            Document(content="Doc 1"),
            Document(content="Doc 2"),
            Document(content="Doc 3")
        ]
        
        context = Context(query="test query", documents=docs)
        top_docs = context.get_top_documents(k=2)
        
        assert len(top_docs) == 2
        assert top_docs[0].content == "Doc 1"
        assert top_docs[1].content == "Doc 2"


class TestConversationHistory:
    """Test cases for ConversationHistory model."""
    
    def test_conversation_history_creation(self):
        """Test creating ConversationHistory."""
        history = ConversationHistory()
        
        assert history.messages == []
        assert history.responses == []
        assert isinstance(history.session_id, str)
        assert isinstance(history.created_at, datetime)
    
    def test_add_message(self):
        """Test adding messages to history."""
        history = ConversationHistory()
        message = Message(
            sender_id="user",
            recipient_id="agent",
            content="Hello",
            message_type=MessageType.USER_QUESTION
        )
        
        history.add_message(message)
        
        assert len(history.messages) == 1
        assert history.messages[0] == message
    
    def test_add_response(self):
        """Test adding responses to history."""
        history = ConversationHistory()
        response = Response(
            message_id="msg123",
            agent_id="agent1",
            content="Hello back"
        )
        
        history.add_response(response)
        
        assert len(history.responses) == 1
        assert history.responses[0] == response
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        history = ConversationHistory()
        
        # Add multiple messages
        for i in range(5):
            message = Message(
                sender_id="user",
                recipient_id="agent",
                content=f"Message {i}",
                message_type=MessageType.USER_QUESTION
            )
            history.add_message(message)
        
        recent = history.get_recent_messages(count=3)
        
        assert len(recent) == 3
        assert recent[0].content == "Message 2"
        assert recent[1].content == "Message 3"
        assert recent[2].content == "Message 4"
    
    def test_clear_history(self):
        """Test clearing conversation history."""
        history = ConversationHistory()
        
        # Add some data
        message = Message(
            sender_id="user",
            recipient_id="agent",
            content="Hello",
            message_type=MessageType.USER_QUESTION
        )
        response = Response(
            message_id="msg123",
            agent_id="agent1",
            content="Hello back"
        )
        
        history.add_message(message)
        history.add_response(response)
        
        assert len(history.messages) == 1
        assert len(history.responses) == 1
        
        history.clear_history()
        
        assert len(history.messages) == 0
        assert len(history.responses) == 0


class TestExceptions:
    """Test cases for custom exceptions."""
    
    def test_deep_agent_exception(self):
        """Test base DeepAgentException."""
        exc = DeepAgentException("Test error", error_code="TEST_ERROR")
        assert "Test error" in str(exc)
        assert exc.error_code == "TEST_ERROR"
        assert exc.details == {}
    
    def test_agent_not_found_error(self):
        """Test AgentNotFoundError."""
        exc = AgentNotFoundError("test_agent")
        assert "test_agent" in str(exc)
        # Check that it's the right exception type
        assert isinstance(exc, AgentNotFoundError)
    
    def test_prompt_not_found_error(self):
        """Test PromptNotFoundError."""
        exc = PromptNotFoundError("test_prompt", "analyst")
        assert "test_prompt" in str(exc)
        assert isinstance(exc, PromptNotFoundError)
    
    def test_rag_retrieval_error(self):
        """Test RAGRetrievalError."""
        exc = RAGRetrievalError("RAG retrieval failed", query="test query")
        assert "RAG retrieval failed" in str(exc)
        assert isinstance(exc, RAGRetrievalError)
    
    def test_workflow_execution_error(self):
        """Test WorkflowExecutionError."""
        exc = WorkflowExecutionError("Workflow failed", workflow_id="wf_123")
        assert "Workflow failed" in str(exc)
        assert isinstance(exc, WorkflowExecutionError)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        exc = ConfigurationError("Config error")
        assert "Config error" in str(exc)
        assert isinstance(exc, ConfigurationError)
    
    def test_agent_communication_error(self):
        """Test AgentCommunicationError."""
        exc = AgentCommunicationError("Communication failed")
        assert "Communication failed" in str(exc)
        assert isinstance(exc, AgentCommunicationError)


class TestConfigModels:
    """Test cases for configuration models."""
    
    def test_rag_config_creation(self):
        """Test creating RAGConfig."""
        config = RAGConfig(
            embedding_model="text-embedding-ada-002",
            embedding_dimension=1536,
            chunk_size=500,
            chunk_overlap=50,
            retrieval_k=3,
            similarity_threshold=0.7,
            vector_store_enabled=True,
            graph_store_enabled=True
        )
        
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.embedding_dimension == 1536
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.retrieval_k == 3
        assert config.similarity_threshold == 0.7
        assert config.vector_store_enabled is True
        assert config.graph_store_enabled is True
    
    def test_rag_config_validation(self):
        """Test RAGConfig validation."""
        # Valid config
        RAGConfig(chunk_size=1000, chunk_overlap=200)
        
        # Invalid overlap (greater than chunk size)
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            RAGConfig(chunk_size=1000, chunk_overlap=1500)
    
    def test_chromadb_config_creation(self):
        """Test creating ChromaDBConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ChromaDBConfig(
                host="localhost",
                port=8000,
                collection_name="test_collection",
                persist_directory=temp_dir
            )
            
            assert config.host == "localhost"
            assert config.port == 8000
            assert config.collection_name == "test_collection"
            assert config.persist_directory == temp_dir
    
    def test_neo4j_config_creation(self):
        """Test creating Neo4jConfig."""
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="neo4j"
        )
        
        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.password == "password"
        assert config.database == "neo4j"
    
    def test_neo4j_config_validation(self):
        """Test Neo4jConfig URI validation."""
        # Valid URIs
        Neo4jConfig(uri="bolt://localhost:7687", username="neo4j", password="pass")
        Neo4jConfig(uri="neo4j://localhost:7687", username="neo4j", password="pass")
        
        # Invalid URI
        with pytest.raises(ValueError, match="Neo4j URI must start with"):
            Neo4jConfig(uri="http://localhost:7687", username="neo4j", password="pass")
    
    def test_model_config_creation(self):
        """Test creating ModelConfig."""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000,
            timeout=300,
            max_retries=3
        )
        
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.timeout == 300
        assert config.max_retries == 3
    
    def test_cli_config_creation(self):
        """Test creating CLIConfig."""
        config = CLIConfig(
            theme="dark",
            syntax_highlighting=True,
            show_progress=True,
            max_width=120
        )
        
        assert config.theme == "dark"
        assert config.syntax_highlighting is True
        assert config.show_progress is True
        assert config.max_width == 120
    
    def test_logging_config_creation(self):
        """Test creating LoggingConfig."""
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            format=LogFormat.JSON,
            log_file="/tmp/test.log",
            enable_debug_logging=True
        )
        
        assert config.level == LogLevel.DEBUG
        assert config.format == LogFormat.JSON
        assert config.log_file == "/tmp/test.log"
        assert config.enable_debug_logging is True
    
    def test_workflow_config_creation(self):
        """Test creating WorkflowConfig."""
        config = WorkflowConfig(
            timeout=600,
            max_steps=50,
            enable_checkpoints=True,
            checkpoint_dir="/tmp/checkpoints"
        )
        
        assert config.timeout == 600
        assert config.max_steps == 50
        assert config.enable_checkpoints is True
        assert config.checkpoint_dir == "/tmp/checkpoints"
    
    def test_security_config_creation(self):
        """Test creating SecurityConfig."""
        config = SecurityConfig(
            api_rate_limit=100,
            api_rate_window=60,
            enable_input_validation=True,
            max_input_length=10000,
            sanitize_outputs=True
        )
        
        assert config.api_rate_limit == 100
        assert config.api_rate_window == 60
        assert config.enable_input_validation is True
        assert config.max_input_length == 10000
        assert config.sanitize_outputs is True
    
    def test_development_config_creation(self):
        """Test creating DevelopmentConfig."""
        config = DevelopmentConfig(
            debug_mode=True,
            development_mode=True,
            enable_hot_reload=True,
            test_mode=False,
            mock_external_apis=True
        )
        
        assert config.debug_mode is True
        assert config.development_mode is True
        assert config.enable_hot_reload is True
        assert config.test_mode is False
        assert config.mock_external_apis is True


if __name__ == "__main__":
    pytest.main([__file__])