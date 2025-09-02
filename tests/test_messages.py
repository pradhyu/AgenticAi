"""Unit tests for message and response data models."""

import pytest
from datetime import datetime
from uuid import uuid4

from deep_agent_system.models.messages import (
    Message,
    Response,
    Context,
    Document,
    GraphData,
    ConversationHistory,
    MessageType,
    RetrievalType,
)


class TestMessage:
    """Test cases for Message model."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            content="Hello, world!",
            message_type=MessageType.USER_QUESTION
        )
        
        assert message.sender_id == "agent1"
        assert message.recipient_id == "agent2"
        assert message.content == "Hello, world!"
        assert message.message_type == MessageType.USER_QUESTION
        assert isinstance(message.id, str)
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}
    
    def test_message_with_metadata(self):
        """Test message creation with metadata."""
        metadata = {"priority": "high", "category": "urgent"}
        message = Message(
            sender_id="user",
            recipient_id="analyst",
            content="Urgent question",
            message_type=MessageType.USER_QUESTION,
            metadata=metadata
        )
        
        assert message.metadata == metadata
    
    def test_message_content_validation(self):
        """Test message content validation."""
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            Message(
                sender_id="agent1",
                recipient_id="agent2",
                content="",
                message_type=MessageType.USER_QUESTION
            )
        
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            Message(
                sender_id="agent1",
                recipient_id="agent2",
                content="   ",
                message_type=MessageType.USER_QUESTION
            )
    
    def test_message_content_stripping(self):
        """Test that message content is stripped of whitespace."""
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            content="  Hello, world!  ",
            message_type=MessageType.USER_QUESTION
        )
        
        assert message.content == "Hello, world!"
    
    def test_message_serialization(self):
        """Test message serialization to dictionary."""
        message = Message(
            sender_id="agent1",
            recipient_id="agent2",
            content="Test message",
            message_type=MessageType.AGENT_REQUEST,
            metadata={"test": "value"}
        )
        
        data = message.to_dict()
        
        assert data["sender_id"] == "agent1"
        assert data["recipient_id"] == "agent2"
        assert data["content"] == "Test message"
        assert data["message_type"] == "agent_request"
        assert data["metadata"] == {"test": "value"}
        assert "timestamp" in data
        assert "id" in data
    
    def test_message_deserialization(self):
        """Test message deserialization from dictionary."""
        data = {
            "id": str(uuid4()),
            "sender_id": "agent1",
            "recipient_id": "agent2",
            "content": "Test message",
            "message_type": "user_question",
            "metadata": {"test": "value"},
            "timestamp": "2023-01-01T12:00:00"
        }
        
        message = Message.from_dict(data)
        
        assert message.id == data["id"]
        assert message.sender_id == "agent1"
        assert message.recipient_id == "agent2"
        assert message.content == "Test message"
        assert message.message_type == MessageType.USER_QUESTION
        assert message.metadata == {"test": "value"}


class TestResponse:
    """Test cases for Response model."""
    
    def test_response_creation(self):
        """Test basic response creation."""
        response = Response(
            message_id="msg123",
            agent_id="agent1",
            content="This is a response"
        )
        
        assert response.message_id == "msg123"
        assert response.agent_id == "agent1"
        assert response.content == "This is a response"
        assert response.confidence_score == 1.0
        assert response.context_used == []
        assert response.workflow_id is None
        assert response.metadata == {}
        assert isinstance(response.timestamp, datetime)
    
    def test_response_with_confidence(self):
        """Test response with confidence score."""
        response = Response(
            message_id="msg123",
            agent_id="agent1",
            content="Response with confidence",
            confidence_score=0.85
        )
        
        assert response.confidence_score == 0.85
    
    def test_response_confidence_validation(self):
        """Test response confidence score validation."""
        with pytest.raises(ValueError):
            Response(
                message_id="msg123",
                agent_id="agent1",
                content="Test",
                confidence_score=-0.1
            )
        
        with pytest.raises(ValueError):
            Response(
                message_id="msg123",
                agent_id="agent1",
                content="Test",
                confidence_score=1.1
            )
    
    def test_response_content_validation(self):
        """Test response content validation."""
        with pytest.raises(ValueError, match="Response content cannot be empty"):
            Response(
                message_id="msg123",
                agent_id="agent1",
                content=""
            )
    
    def test_response_serialization(self):
        """Test response serialization."""
        response = Response(
            message_id="msg123",
            agent_id="agent1",
            content="Test response",
            confidence_score=0.9,
            context_used=["doc1", "doc2"],
            workflow_id="wf123"
        )
        
        data = response.to_dict()
        
        assert data["message_id"] == "msg123"
        assert data["agent_id"] == "agent1"
        assert data["content"] == "Test response"
        assert data["confidence_score"] == 0.9
        assert data["context_used"] == ["doc1", "doc2"]
        assert data["workflow_id"] == "wf123"


class TestDocument:
    """Test cases for Document model."""
    
    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(content="This is document content")
        
        assert doc.content == "This is document content"
        assert doc.metadata == {}
        assert doc.source is None
        assert doc.chunk_id is None
    
    def test_document_with_metadata(self):
        """Test document with metadata."""
        doc = Document(
            content="Document content",
            metadata={"author": "John Doe", "date": "2023-01-01"},
            source="file.txt",
            chunk_id="chunk_1"
        )
        
        assert doc.metadata == {"author": "John Doe", "date": "2023-01-01"}
        assert doc.source == "file.txt"
        assert doc.chunk_id == "chunk_1"
    
    def test_document_content_validation(self):
        """Test document content validation."""
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            Document(content="")


class TestGraphData:
    """Test cases for GraphData model."""
    
    def test_graph_data_creation(self):
        """Test basic graph data creation."""
        graph_data = GraphData()
        
        assert graph_data.entities == []
        assert graph_data.relationships == []
        assert graph_data.query is None
    
    def test_graph_data_with_data(self):
        """Test graph data with entities and relationships."""
        entities = [{"id": "1", "type": "Person", "name": "John"}]
        relationships = [{"from": "1", "to": "2", "type": "KNOWS"}]
        
        graph_data = GraphData(
            entities=entities,
            relationships=relationships,
            query="MATCH (p:Person) RETURN p"
        )
        
        assert graph_data.entities == entities
        assert graph_data.relationships == relationships
        assert graph_data.query == "MATCH (p:Person) RETURN p"
    
    def test_graph_data_counts(self):
        """Test graph data count methods."""
        graph_data = GraphData(
            entities=[{"id": "1"}, {"id": "2"}],
            relationships=[{"from": "1", "to": "2"}]
        )
        
        assert graph_data.get_entity_count() == 2
        assert graph_data.get_relationship_count() == 1


class TestContext:
    """Test cases for Context model."""
    
    def test_context_creation(self):
        """Test basic context creation."""
        context = Context(query="test query")
        
        assert context.query == "test query"
        assert context.documents == []
        assert context.graph_data is None
        assert context.relevance_scores == []
        assert context.retrieval_method == RetrievalType.NONE
    
    def test_context_with_documents(self):
        """Test context with documents and scores."""
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
        """Test context relevance score validation."""
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
        """Test basic conversation history creation."""
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