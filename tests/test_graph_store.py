"""Unit tests for Neo4j graph store integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from langchain.schema import Document

from deep_agent_system.rag.graph_store import Neo4jStore, Entity, Relationship, GraphContext
from deep_agent_system.config.models import Neo4jConfig, RAGConfig


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
def rag_config():
    """Create RAG configuration for testing."""
    return RAGConfig(
        embedding_model="text-embedding-ada-002",
        embedding_dimension=1536,
        chunk_size=500,
        chunk_overlap=50,
        retrieval_k=3,
        similarity_threshold=0.7
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Python is a programming language. ChromaDB uses vector embeddings.",
            metadata={"source": "test_doc_1.txt", "category": "Programming"}
        ),
        Document(
            page_content="Neo4j is a graph database. LangChain implements AI workflows.",
            metadata={"source": "test_doc_2.txt", "category": "Database"}
        ),
        Document(
            page_content="class AgentSystem implements BaseAgent. The system uses ChromaDB for storage.",
            metadata={"source": "test_doc_3.py", "category": "Code"}
        )
    ]


class TestNeo4jStore:
    """Test cases for Neo4jStore class."""
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_initialization(self, mock_driver, neo4j_config, rag_config):
        """Test Neo4jStore initialization."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        mock_session.run.return_value = None
        
        # Initialize store
        store = Neo4jStore(neo4j_config, rag_config)
        
        # Verify initialization
        assert store.neo4j_config == neo4j_config
        assert store.rag_config == rag_config
        assert store.driver == mock_driver_instance
        
        # Verify driver was called with correct parameters
        mock_driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "test_password")
        )
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_extract_entities_and_relationships(self, mock_driver, neo4j_config, rag_config, sample_documents):
        """Test entity and relationship extraction."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        
        # Initialize store
        store = Neo4jStore(neo4j_config, rag_config)
        
        # Test extraction
        entities, relationships = store.extract_entities_and_relationships(sample_documents)
        
        # Verify entities were extracted
        assert len(entities) > 0
        entity_names = [e.name for e in entities]
        assert any("Python" in name for name in entity_names)
        assert any("ChromaDB" in name for name in entity_names)
        
        # Verify relationships were extracted
        assert len(relationships) > 0
        rel_types = [r.relationship_type for r in relationships]
        assert any(rel_type in ["USES", "IMPLEMENTS"] for rel_type in rel_types)
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_ingest_documents(self, mock_driver, neo4j_config, rag_config, sample_documents):
        """Test document ingestion."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        
        # Initialize store
        store = Neo4jStore(neo4j_config, rag_config)
        
        # Test ingestion
        result = store.ingest_documents(sample_documents)
        
        # Verify ingestion was successful
        assert result is True
        
        # Verify session.run was called multiple times (for documents, entities, relationships)
        assert mock_session.run.call_count > len(sample_documents)
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_ingest_empty_documents(self, mock_driver, neo4j_config, rag_config):
        """Test ingesting empty document list."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        
        # Initialize store
        store = Neo4jStore(neo4j_config, rag_config)
        
        # Test with empty list
        result = store.ingest_documents([])
        
        # Verify no ingestion occurred
        assert result is True
        # Only index creation queries should have been called during initialization
        initial_call_count = mock_session.run.call_count
        
        # Reset and test empty ingestion
        mock_session.run.reset_mock()
        result = store.ingest_documents([])
        assert result is True
        assert mock_session.run.call_count == 0
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_query_graph(self, mock_driver, neo4j_config, rag_config):
        """Test graph querying."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        
        # Mock query results
        mock_entity_results = [
            {
                'name': 'Python',
                'type': 'TECHNOLOGY',
                'properties': {'confidence': 0.8},
                'source_document': 'test.txt'
            }
        ]
        
        mock_rel_results = [
            {
                'source_entity': 'Python',
                'target_entity': 'ChromaDB',
                'relationship_type': 'USES',
                'properties': {'confidence': 0.7},
                'source_document': 'test.txt'
            }
        ]
        
        # Configure mock to return different results for different queries
        def side_effect(query, params=None):
            mock_result = Mock()
            if 'MATCH (e:Entity)' in query:
                mock_result.__iter__ = lambda self: iter([Mock(**entity) for entity in mock_entity_results])
                return mock_result
            elif 'MATCH (source:Entity)-[r]->(target:Entity)' in query:
                mock_result.__iter__ = lambda self: iter([Mock(**rel) for rel in mock_rel_results])
                return mock_result
            return Mock(__iter__=lambda self: iter([]))
        
        mock_session.run.side_effect = side_effect
        
        # Initialize store
        store = Neo4jStore(neo4j_config, rag_config)
        
        # Test query
        context = store.query_graph("Python ChromaDB")
        
        # Verify results
        assert isinstance(context, GraphContext)
        assert len(context.entities) > 0
        assert len(context.relationships) > 0
        assert context.subgraph_query == "Python ChromaDB"
        assert 0.0 <= context.relevance_score <= 1.0
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_get_connected_entities(self, mock_driver, neo4j_config, rag_config):
        """Test getting connected entities."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        
        # Mock connected entity results
        mock_results = [
            {'name': 'ChromaDB', 'type': 'TECHNOLOGY', 'properties': {'confidence': 0.8}, 'source_document': 'test.txt'},
            {'name': 'LangChain', 'type': 'TECHNOLOGY', 'properties': {'confidence': 0.9}, 'source_document': 'test.txt'}
        ]
        
        mock_session.run.return_value = [Mock(**result) for result in mock_results]
        
        # Initialize store
        store = Neo4jStore(neo4j_config, rag_config)
        
        # Test getting connected entities
        entities = store.get_connected_entities("Python", depth=2)
        
        # Verify results
        assert len(entities) == 2
        assert all(isinstance(entity, Entity) for entity in entities)
        
        # Verify session.run was called
        mock_session.run.assert_called()
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_get_graph_stats(self, mock_driver, neo4j_config, rag_config):
        """Test getting graph statistics."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        
        # Mock stats results
        node_results = [
            Mock(labels=['Entity'], count=10),
            Mock(labels=['Document'], count=5)
        ]
        
        rel_results = [
            Mock(type='USES', count=8),
            Mock(type='IMPLEMENTS', count=3)
        ]
        
        def side_effect(query, params=None):
            if 'MATCH (n)' in query:
                return node_results
            elif 'MATCH ()-[r]->' in query:
                return rel_results
            return []
        
        mock_session.run.side_effect = side_effect
        
        # Initialize store
        store = Neo4jStore(neo4j_config, rag_config)
        
        # Test stats
        stats = store.get_graph_stats()
        
        # Verify stats
        assert stats["database"] == "test_db"
        assert "node_counts" in stats
        assert "relationship_counts" in stats
        assert "total_nodes" in stats
        assert "total_relationships" in stats
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_clear_graph(self, mock_driver, neo4j_config, rag_config):
        """Test clearing the graph."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        
        # Initialize store
        store = Neo4jStore(neo4j_config, rag_config)
        
        # Test clearing
        result = store.clear_graph()
        
        # Verify clearing
        assert result is True
        
        # Verify delete query was called
        delete_calls = [call for call in mock_session.run.call_args_list 
                       if 'DETACH DELETE' in str(call)]
        assert len(delete_calls) > 0
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_execute_cypher_query(self, mock_driver, neo4j_config, rag_config):
        """Test executing custom Cypher queries."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        
        # Mock query results
        mock_record = Mock()
        mock_record.data.return_value = {'name': 'Python', 'count': 5}
        mock_session.run.return_value = [mock_record]
        
        # Initialize store
        store = Neo4jStore(neo4j_config, rag_config)
        
        # Test custom query
        query = "MATCH (n:Entity) RETURN n.name as name, count(n) as count"
        results = store.execute_cypher_query(query)
        
        # Verify results
        assert len(results) == 1
        assert results[0]['name'] == 'Python'
        assert results[0]['count'] == 5
        
        # Verify session.run was called with the query
        mock_session.run.assert_called_with(query, {})
    
    @patch('deep_agent_system.rag.graph_store.GraphDatabase.driver')
    def test_context_manager(self, mock_driver, neo4j_config, rag_config):
        """Test using Neo4jStore as a context manager."""
        # Setup mocks
        mock_driver_instance = Mock()
        mock_driver.return_value = mock_driver_instance
        mock_session = Mock()
        mock_session_context = Mock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver_instance.session.return_value = mock_session_context
        
        # Test context manager
        with Neo4jStore(neo4j_config, rag_config) as store:
            assert store.driver == mock_driver_instance
        
        # Verify close was called
        mock_driver_instance.close.assert_called_once()


class TestEntityAndRelationship:
    """Test cases for Entity and Relationship dataclasses."""
    
    def test_entity_creation(self):
        """Test creating an Entity instance."""
        entity = Entity(
            name="Python",
            type="TECHNOLOGY",
            properties={"confidence": 0.8},
            source_document="test.txt"
        )
        
        assert entity.name == "Python"
        assert entity.type == "TECHNOLOGY"
        assert entity.properties["confidence"] == 0.8
        assert entity.source_document == "test.txt"
    
    def test_relationship_creation(self):
        """Test creating a Relationship instance."""
        relationship = Relationship(
            source_entity="Python",
            target_entity="ChromaDB",
            relationship_type="USES",
            properties={"confidence": 0.7},
            source_document="test.txt"
        )
        
        assert relationship.source_entity == "Python"
        assert relationship.target_entity == "ChromaDB"
        assert relationship.relationship_type == "USES"
        assert relationship.properties["confidence"] == 0.7
        assert relationship.source_document == "test.txt"


class TestGraphContext:
    """Test cases for GraphContext model."""
    
    def test_graph_context_creation(self):
        """Test creating a GraphContext instance."""
        entity = Entity(
            name="Python",
            type="TECHNOLOGY",
            properties={},
            source_document="test.txt"
        )
        
        relationship = Relationship(
            source_entity="Python",
            target_entity="ChromaDB",
            relationship_type="USES",
            properties={},
            source_document="test.txt"
        )
        
        context = GraphContext(
            entities=[entity],
            relationships=[relationship],
            subgraph_query="test query",
            relevance_score=0.8
        )
        
        assert len(context.entities) == 1
        assert len(context.relationships) == 1
        assert context.subgraph_query == "test query"
        assert context.relevance_score == 0.8


if __name__ == "__main__":
    pytest.main([__file__])