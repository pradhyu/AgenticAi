"""Unit tests for ChromaDB vector store integration."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

from langchain.schema import Document

from deep_agent_system.rag.vector_store import ChromaDBStore, DocumentChunk
from deep_agent_system.config.models import ChromaDBConfig, RAGConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def chroma_config(temp_dir):
    """Create ChromaDB configuration for testing."""
    return ChromaDBConfig(
        host="localhost",
        port=8000,
        collection_name="test_collection",
        persist_directory=temp_dir
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
            page_content="This is a test document about machine learning and artificial intelligence.",
            metadata={"source": "test_doc_1.txt", "category": "AI"}
        ),
        Document(
            page_content="Python is a programming language used for data science and web development.",
            metadata={"source": "test_doc_2.txt", "category": "Programming"}
        ),
        Document(
            page_content="ChromaDB is a vector database for storing and retrieving embeddings efficiently.",
            metadata={"source": "test_doc_3.txt", "category": "Database"}
        )
    ]


class TestChromaDBStore:
    """Test cases for ChromaDBStore class."""
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_initialization(self, mock_embedding_func, mock_client, chroma_config, rag_config):
        """Test ChromaDBStore initialization."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Verify initialization
        assert store.chroma_config == chroma_config
        assert store.rag_config == rag_config
        assert store.client == mock_client_instance
        assert store.collection == mock_collection
        assert store.text_splitter is not None
        
        # Verify client was called with correct parameters
        mock_client.assert_called_once()
        mock_client_instance.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            embedding_function=mock_embedding_func.return_value,
            metadata={"description": "Deep Agent System knowledge base"}
        )
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_ingest_documents(self, mock_embedding_func, mock_client, chroma_config, rag_config, sample_documents):
        """Test document ingestion."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Test ingestion
        result_ids = store.ingest_documents(sample_documents)
        
        # Verify collection.add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        
        # Verify the call arguments
        assert 'ids' in call_args
        assert 'documents' in call_args
        assert 'metadatas' in call_args
        assert len(call_args['ids']) > 0
        assert len(call_args['documents']) > 0
        assert len(call_args['metadatas']) > 0
        assert len(result_ids) > 0
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_ingest_empty_documents(self, mock_embedding_func, mock_client, chroma_config, rag_config):
        """Test ingesting empty document list."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Test with empty list
        result_ids = store.ingest_documents([])
        
        # Verify no documents were added
        assert result_ids == []
        mock_collection.add.assert_not_called()
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_similarity_search(self, mock_embedding_func, mock_client, chroma_config, rag_config):
        """Test similarity search functionality."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Mock search results
        mock_collection.query.return_value = {
            'documents': [['Test document content', 'Another document']],
            'metadatas': [[{'source': 'test1.txt'}, {'source': 'test2.txt'}]],
            'distances': [[0.1, 0.3]]
        }
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Test search
        results = store.similarity_search("test query", k=2)
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc, score in results)
        assert all(isinstance(score, float) for doc, score in results)
        
        # Verify collection.query was called correctly
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=2,
            where=None
        )
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_similarity_search_with_filters(self, mock_embedding_func, mock_client, chroma_config, rag_config):
        """Test similarity search with metadata filters."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Mock search results
        mock_collection.query.return_value = {
            'documents': [['Filtered document']],
            'metadatas': [[{'source': 'filtered.txt', 'category': 'AI'}]],
            'distances': [[0.2]]
        }
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Test search with filters
        filter_metadata = {"category": "AI"}
        results = store.similarity_search("test query", filter_metadata=filter_metadata)
        
        # Verify collection.query was called with filters
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,  # default k from rag_config
            where=filter_metadata
        )
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_similarity_search_empty_query(self, mock_embedding_func, mock_client, chroma_config, rag_config):
        """Test similarity search with empty query."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Test with empty query
        results = store.similarity_search("")
        
        # Verify no search was performed
        assert results == []
        mock_collection.query.assert_not_called()
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_get_collection_stats(self, mock_embedding_func, mock_client, chroma_config, rag_config):
        """Test getting collection statistics."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Test stats
        stats = store.get_collection_stats()
        
        # Verify stats
        assert stats["collection_name"] == "test_collection"
        assert stats["document_count"] == 42
        assert stats["embedding_model"] == "text-embedding-ada-002"
        assert stats["chunk_size"] == 500
        assert stats["chunk_overlap"] == 50
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_delete_documents(self, mock_embedding_func, mock_client, chroma_config, rag_config):
        """Test document deletion."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Test deletion
        doc_ids = ["doc1", "doc2", "doc3"]
        result = store.delete_documents(doc_ids)
        
        # Verify deletion
        assert result is True
        mock_collection.delete.assert_called_once_with(ids=doc_ids)
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_clear_collection(self, mock_embedding_func, mock_client, chroma_config, rag_config):
        """Test clearing collection."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Test clearing
        result = store.clear_collection()
        
        # Verify clearing
        assert result is True
        mock_client_instance.delete_collection.assert_called_once_with(name="test_collection")
    
    @patch('deep_agent_system.rag.vector_store.chromadb.PersistentClient')
    @patch('deep_agent_system.rag.vector_store.embedding_functions.OpenAIEmbeddingFunction')
    def test_search_by_metadata(self, mock_embedding_func, mock_client, chroma_config, rag_config):
        """Test searching by metadata only."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Mock get results
        mock_collection.get.return_value = {
            'documents': ['Doc 1', 'Doc 2'],
            'metadatas': [{'category': 'AI'}, {'category': 'AI'}]
        }
        
        # Initialize store
        store = ChromaDBStore(chroma_config, rag_config)
        
        # Test metadata search
        metadata_filter = {"category": "AI"}
        results = store.search_by_metadata(metadata_filter, limit=10)
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        
        # Verify collection.get was called correctly
        mock_collection.get.assert_called_once_with(
            where=metadata_filter,
            limit=10
        )


class TestDocumentChunk:
    """Test cases for DocumentChunk model."""
    
    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk instance."""
        chunk = DocumentChunk(
            id="test_chunk_1",
            content="This is test content",
            metadata={"source": "test.txt", "chunk_index": 0}
        )
        
        assert chunk.id == "test_chunk_1"
        assert chunk.content == "This is test content"
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.embedding is None
    
    def test_document_chunk_with_embedding(self):
        """Test creating a DocumentChunk with embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk = DocumentChunk(
            id="test_chunk_2",
            content="Content with embedding",
            metadata={"source": "test2.txt"},
            embedding=embedding
        )
        
        assert chunk.embedding == embedding


if __name__ == "__main__":
    pytest.main([__file__])