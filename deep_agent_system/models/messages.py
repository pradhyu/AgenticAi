"""Message and response data models for agent communication."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""
    USER_QUESTION = "user_question"
    AGENT_REQUEST = "agent_request"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_MESSAGE = "system_message"
    ERROR_MESSAGE = "error_message"


class RetrievalType(str, Enum):
    """Types of context retrieval methods."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    NONE = "none"


class Message(BaseModel):
    """Core message model for agent communication."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    sender_id: str = Field(..., description="ID of the sending agent")
    recipient_id: str = Field(..., description="ID of the receiving agent")
    content: str = Field(..., description="Message content")
    message_type: MessageType = Field(..., description="Type of message")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message content cannot be empty')
        return v.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "message_type": self.message_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize message from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if "message_type" in data and isinstance(data["message_type"], str):
            data["message_type"] = MessageType(data["message_type"])
        return cls(**data)


class Document(BaseModel):
    """Document model for RAG context."""
    
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = Field(None, description="Document source")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Document content cannot be empty')
        return v.strip()


class GraphData(BaseModel):
    """Graph-based context data from Neo4j."""
    
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    query: Optional[str] = Field(None, description="Cypher query used")
    
    def get_entity_count(self) -> int:
        """Get the number of entities in the graph data."""
        return len(self.entities)
    
    def get_relationship_count(self) -> int:
        """Get the number of relationships in the graph data."""
        return len(self.relationships)


class Context(BaseModel):
    """Context model for RAG retrieval results."""
    
    documents: List[Document] = Field(default_factory=list)
    graph_data: Optional[GraphData] = Field(None)
    relevance_scores: List[float] = Field(default_factory=list)
    retrieval_method: RetrievalType = Field(default=RetrievalType.NONE)
    query: str = Field(..., description="Original query used for retrieval")
    
    @field_validator('relevance_scores')
    @classmethod
    def scores_match_documents(cls, v, info):
        if info.data.get('documents') and len(v) > 0:
            if len(v) != len(info.data['documents']):
                raise ValueError('Number of relevance scores must match number of documents')
        return v
    
    def get_top_documents(self, k: int = 3) -> List[Document]:
        """Get top k documents by relevance score."""
        if not self.relevance_scores:
            return self.documents[:k]
        
        # Sort documents by relevance score (descending)
        doc_score_pairs = list(zip(self.documents, self.relevance_scores))
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_pairs[:k]]


class Response(BaseModel):
    """Response model for agent outputs."""
    
    message_id: str = Field(..., description="ID of the message being responded to")
    agent_id: str = Field(..., description="ID of the responding agent")
    content: str = Field(..., description="Response content")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    context_used: List[str] = Field(default_factory=list)
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Response content cannot be empty')
        return v.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize response to dictionary."""
        return {
            "message_id": self.message_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "confidence_score": self.confidence_score,
            "context_used": self.context_used,
            "workflow_id": self.workflow_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Response":
        """Deserialize response from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class ConversationHistory(BaseModel):
    """Model for tracking conversation history."""
    
    messages: List[Message] = Field(default_factory=list)
    responses: List[Response] = Field(default_factory=list)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)
    
    def add_response(self, response: Response) -> None:
        """Add a response to the conversation history."""
        self.responses.append(response)
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:] if count > 0 else self.messages
    
    def get_recent_responses(self, count: int = 10) -> List[Response]:
        """Get the most recent responses."""
        return self.responses[-count:] if count > 0 else self.responses
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.messages.clear()
        self.responses.clear()