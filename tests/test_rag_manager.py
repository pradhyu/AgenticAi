"""Integration tests for RAGManager."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from langchain.schema import Document

from deep_agent_system.rag.manager import RAGManager, CombinedContext, RetrievalResult
from deep_agent_system.rag.graph_store import GraphContext, Entity, Relationship
from deep_agent_system.config.models import RAGConfig, ChromaDBConfig, Neo4jConfig, RetrievalType


@pytest.fixture
def rag_config():
    """Create RAG configuration for testing."""
    return RAGConfig(
        embedding_model="text-embedding-ada-002",
        embedding_dimension=1536,
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=3,
        similarity_threshold=0.7,
        vector_store_enabled=True,
        graph_store_enabled=True
    )


@pytest.fixture
def chroma_config():
    """Create ChromaDB configuration for testing."""
    return ChromaDBConfig(
        host="localhost",
        port=8000,
        collection_name="test_collection",
        persist_directory="./test_data"
    )


@pytest.fixture
def neo4j_config():
    """Create Neo4j configuration for testing."""
    return Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test_password",
        database="test_db"
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Python is a programming language used for AI and machine learning.",
            metadata={"source": "test_doc_1.txt", "category": "Programming"}
        ),
        Document(
            page_content="ChromaDB is a vector database for storing embeddings efficiently.",
            metadata={"source": "test_doc_2.txt", "category": "Database"}
        ),
        Document(
            page_content="Neo4j is a graph database that stores data as nodes and relationships.",
            metadata={"source": "test_doc_3.txt", "category": "Database"}
        )
    ]


@pytest.fixture
def mock_vector_results():
    """Create mock vector search results."""
    return [
        (Document(page_content="Vector result 1", metadata={"source": "vec1.txt"}), 0.9),
        (Document(page_content="Vector result 2", metadata={"source": "vec2.txt"}), 0.8),
        (Document(page_content="Vector result 3", metadata={"source": "vec3.txt"}), 0.7)
    ]


@pytest.fixture
def mock_graph_context():
    """Create mock graph context."""
    entities = [
        Entity(name="Python", type="TECHNOLOGY", properties={"confidence": 0.9}, source_document="test.txt"),
        Entity(name="ChromaDB", type="DATABASE", properties={"confidence": 0.8}, source_document="test.txt")
    ]
    
    relationships = [
        Relationship(
            source_entity="Python",
            target_entity="ChromaDB",
            relationship_type="USES",
            properties={"confidence": 0.7},
            source_document="test.txt"
        )
    ]
    
    return GraphContext(
        entities=entities,
        relationships=relationships,
        subgraph_query="Python ChromaDB",
        relevance_score=0.8
    )


class TestRAGManager:
    """Test cases for RAGManager class."""
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_initialization_both_stores(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config):
        """Test RAGManager initialization with both stores."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        
        # Initialize manager
        manager = RAGManager(rag_config, chroma_config, neo4j_config)
        
        # Verify initialization
        assert manager.rag_config == rag_config
        assert manager.vector_store == mock_chroma_instance
        assert manager.graph_store == mock_neo4j_instance
        
        # Verify stores were initialized with correct configs
        mock_chroma.assert_called_once_with(chroma_config, rag_config)
        mock_neo4j.assert_called_once_with(neo4j_config, rag_config)
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    def test_initialization_vector_only(self, mock_chroma, rag_config, chroma_config):
        """Test RAGManager initialization with vector store only."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Initialize manager without graph config
        manager = RAGManager(rag_config, chroma_config, None)
        
        # Verify initialization
        assert manager.vector_store == mock_chroma_instance
        assert manager.graph_store is None
    
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_initialization_graph_only(self, mock_neo4j, rag_config, neo4j_config):
        """Test RAGManager initialization with graph store only."""
        # Setup mocks
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        
        # Initialize manager without vector config
        manager = RAGManager(rag_config, None, neo4j_config)
        
        # Verify initialization
        assert manager.vector_store is None
        assert manager.graph_store == mock_neo4j_instance
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_ingest_documents(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config, sample_documents):
        """Test document ingestion into both stores."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        mock_chroma_instance.ingest_documents.return_value = ["doc1", "doc2", "doc3"]
        
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        mock_neo4j_instance.ingest_documents.return_value = True
        
        # Initialize manager
        manager = RAGManager(rag_config, chroma_config, neo4j_config)
        
        # Test ingestion
        results = manager.ingest_documents(sample_documents)
        
        # Verify results
        assert results["total_documents"] == 3
        assert results["vector_store"]["success"] is True
        assert results["vector_store"]["document_ids"] == ["doc1", "doc2", "doc3"]
        assert results["graph_store"]["success"] is True
        
        # Verify both stores were called
        mock_chroma_instance.ingest_documents.assert_called_once_with(sample_documents)
        mock_neo4j_instance.ingest_documents.assert_called_once_with(sample_documents)
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_retrieve_vector_context(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config, mock_vector_results):
        """Test vector context retrieval."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        mock_chroma_instance.similarity_search.return_value = mock_vector_results
        
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        
        # Initialize manager
        manager = RAGManager(rag_config, chroma_config, neo4j_config)
        
        # Test retrieval
        results = manager.retrieve_vector_context("test query", k=3)
        
        # Verify results
        assert len(results) == 3
        assert results == mock_vector_results
        
        # Verify vector store was called correctly
        mock_chroma_instance.similarity_search.assert_called_once_with(
            query="test query",
            k=3,
            filter_metadata=None,
            similarity_threshold=None
        )
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_retrieve_graph_context(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config, mock_graph_context):
        """Test graph context retrieval."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        mock_neo4j_instance.query_graph.return_value = mock_graph_context
        
        # Initialize manager
        manager = RAGManager(rag_config, chroma_config, neo4j_config)
        
        # Test retrieval
        result = manager.retrieve_graph_context("test query", depth=2)
        
        # Verify results
        assert result == mock_graph_context
        
        # Verify graph store was called correctly
        mock_neo4j_instance.query_graph.assert_called_once_with("test query", 2, 50)
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_hybrid_retrieve(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config, mock_vector_results, mock_graph_context):
        """Test hybrid retrieval combining both approaches."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        mock_chroma_instance.similarity_search.return_value = mock_vector_results
        
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        mock_neo4j_instance.query_graph.return_value = mock_graph_context
        
        # Initialize manager
        manager = RAGManager(rag_config, chroma_config, neo4j_config)
        
        # Test hybrid retrieval
        result = manager.hybrid_retrieve("test query", k=3, depth=2)
        
        # Verify results
        assert isinstance(result, CombinedContext)
        assert result.vector_results == mock_vector_results
        assert result.graph_context == mock_graph_context
        assert len(result.combined_results) > 0
        assert result.retrieval_strategy == "hybrid"
        assert result.total_score > 0
        
        # Verify both stores were called
        mock_chroma_instance.similarity_search.assert_called_once()
        mock_neo4j_instance.query_graph.assert_called_once()
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_get_retrieval_stats(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config):
        """Test getting retrieval statistics."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        mock_chroma_instance.get_collection_stats.return_value = {"document_count": 100}
        
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        mock_neo4j_instance.get_graph_stats.return_value = {"node_count": 50}
        
        # Initialize manager
        manager = RAGManager(rag_config, chroma_config, neo4j_config)
        
        # Test stats
        stats = manager.get_retrieval_stats()
        
        # Verify stats
        assert stats["vector_store"]["document_count"] == 100
        assert stats["graph_store"]["node_count"] == 50
        assert stats["rag_config"]["vector_enabled"] is True
        assert stats["rag_config"]["graph_enabled"] is True
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_clear_all_data(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config):
        """Test clearing all data from both stores."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        mock_chroma_instance.clear_collection.return_value = True
        
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        mock_neo4j_instance.clear_graph.return_value = True
        
        # Initialize manager
        manager = RAGManager(rag_config, chroma_config, neo4j_config)
        
        # Test clearing
        results = manager.clear_all_data()
        
        # Verify results
        assert results["vector_store"] is True
        assert results["graph_store"] is True
        
        # Verify both stores were called
        mock_chroma_instance.clear_collection.assert_called_once()
        mock_neo4j_instance.clear_graph.assert_called_once()
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_search_by_retrieval_type_vector(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config, mock_vector_results):
        """Test searching by specific retrieval type - vector."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        mock_chroma_instance.similarity_search.return_value = mock_vector_results
        
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        
        # Initialize manager
        manager = RAGManager(rag_config, chroma_config, neo4j_config)
        
        # Test vector search
        results = manager.search_by_retrieval_type("test query", RetrievalType.VECTOR, k=3)
        
        # Verify results
        assert results == mock_vector_results
        mock_chroma_instance.similarity_search.assert_called_once()
        mock_neo4j_instance.query_graph.assert_not_called()
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_search_by_retrieval_type_graph(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config, mock_graph_context):
        """Test searching by specific retrieval type - graph."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        mock_neo4j_instance.query_graph.return_value = mock_graph_context
        
        # Initialize manager
        manager = RAGManager(rag_config, chroma_config, neo4j_config)
        
        # Test graph search
        result = manager.search_by_retrieval_type("test query", RetrievalType.GRAPH, depth=2)
        
        # Verify results
        assert result == mock_graph_context
        mock_neo4j_instance.query_graph.assert_called_once()
        mock_chroma_instance.similarity_search.assert_not_called()
    
    @patch('deep_agent_system.rag.manager.ChromaDBStore')
    @patch('deep_agent_system.rag.manager.Neo4jStore')
    def test_context_manager(self, mock_neo4j, mock_chroma, rag_config, chroma_config, neo4j_config):
        """Test using RAGManager as a context manager."""
        # Setup mocks
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance
        
        # Test context manager
        with RAGManager(rag_config, chroma_config, neo4j_config) as manager:
            assert manager.vector_store == mock_chroma_instance
            assert manager.graph_store == mock_neo4j_instance
        
        # Verify close was called
        mock_neo4j_instance.close.assert_called_once()


class TestRetrievalResult:
    """Test cases for RetrievalResult dataclass."""
    
    def test_retrieval_result_creation(self):
        """Test creating a RetrievalResult instance."""
        result = RetrievalResult(
            content="Test content",
            metadata={"source": "test.txt"},
            score=0.8,
            source_type=RetrievalType.VECTOR,
            source_id="vector_1"
        )
        
        assert result.content == "Test content"
        assert result.metadata["source"] == "test.txt"
        assert result.score == 0.8
        assert result.source_type == RetrievalType.VECTOR
        assert result.source_id == "vector_1"


class TestCombinedContext:
    """Test cases for CombinedContext model."""
    
    def test_combined_context_creation(self):
        """Test creating a CombinedContext instance."""
        vector_results = [
            (Document(page_content="Test", metadata={}), 0.8)
        ]
        
        combined_results = [
            RetrievalResult(
                content="Test",
                metadata={},
                score=0.8,
                source_type=RetrievalType.VECTOR,
                source_id="vector_1"
            )
        ]
        
        context = CombinedContext(
            vector_results=vector_results,
            graph_context=None,
            combined_results=combined_results,
            total_score=0.8,
            retrieval_strategy="vector_only"
        )
        
        assert len(context.vector_results) == 1
        assert context.graph_context is None
        assert len(context.combined_results) == 1
        assert context.total_score == 0.8
        assert context.retrieval_strategy == "vector_only"


if __name__ == "__main__":
    pytest.main([__file__])