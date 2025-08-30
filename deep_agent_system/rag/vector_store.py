"""ChromaDB vector store integration for RAG functionality."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pydantic import BaseModel

from ..config.models import ChromaDBConfig, RAGConfig


logger = logging.getLogger(__name__)


class DocumentChunk(BaseModel):
    """Represents a document chunk with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class ChromaDBStore:
    """ChromaDB vector store for document storage and retrieval."""
    
    def __init__(self, chroma_config: ChromaDBConfig, rag_config: RAGConfig):
        """Initialize ChromaDB store with configuration.
        
        Args:
            chroma_config: ChromaDB-specific configuration
            rag_config: General RAG configuration
        """
        self.chroma_config = chroma_config
        self.rag_config = rag_config
        self.client = None
        self.collection = None
        self.text_splitter = None
        self.embedding_function = None
        
        self._initialize_client()
        self._initialize_text_splitter()
        self._initialize_embedding_function()
        self._initialize_collection()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            # Create persistence directory if it doesn't exist
            persist_dir = Path(self.chroma_config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            settings = Settings(
                persist_directory=str(persist_dir),
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=settings
            )
            
            logger.info(f"ChromaDB client initialized with persistence at {persist_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _initialize_text_splitter(self) -> None:
        """Initialize text splitter for document chunking."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rag_config.chunk_size,
            chunk_overlap=self.rag_config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"Text splitter initialized with chunk_size={self.rag_config.chunk_size}")
    
    def _initialize_embedding_function(self) -> None:
        """Initialize embedding function."""
        try:
            # Use OpenAI embeddings (requires OPENAI_API_KEY environment variable)
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                model_name=self.rag_config.embedding_model
            )
            logger.info(f"Embedding function initialized with model {self.rag_config.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding function: {e}")
            raise
    
    def _initialize_collection(self) -> None:
        """Initialize or get ChromaDB collection."""
        try:
            # Try to get existing collection or create new one
            self.collection = self.client.get_or_create_collection(
                name=self.chroma_config.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Deep Agent System knowledge base"}
            )
            
            logger.info(f"Collection '{self.chroma_config.collection_name}' initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def ingest_documents(self, documents: List[Document]) -> List[str]:
        """Ingest documents into the vector store.
        
        Args:
            documents: List of LangChain Document objects to ingest
            
        Returns:
            List of document IDs that were added
            
        Raises:
            Exception: If document ingestion fails
        """
        try:
            if not documents:
                logger.warning("No documents provided for ingestion")
                return []
            
            # Split documents into chunks
            all_chunks = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc.metadata.get('source', 'unknown')}_{i}_{uuid.uuid4().hex[:8]}"
                    chunk_metadata = {
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_id": chunk_id
                    }
                    all_chunks.append(DocumentChunk(
                        id=chunk_id,
                        content=chunk,
                        metadata=chunk_metadata
                    ))
            
            if not all_chunks:
                logger.warning("No chunks created from documents")
                return []
            
            # Prepare data for ChromaDB
            ids = [chunk.id for chunk in all_chunks]
            documents_text = [chunk.content for chunk in all_chunks]
            metadatas = [chunk.metadata for chunk in all_chunks]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents_text,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully ingested {len(all_chunks)} chunks from {len(documents)} documents")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search in the vector store.
        
        Args:
            query: Search query text
            k: Number of results to return (defaults to rag_config.retrieval_k)
            filter_metadata: Metadata filters to apply
            similarity_threshold: Minimum similarity score (defaults to rag_config.similarity_threshold)
            
        Returns:
            List of tuples containing (Document, similarity_score)
            
        Raises:
            Exception: If search fails
        """
        try:
            if not query.strip():
                logger.warning("Empty query provided for similarity search")
                return []
            
            k = k or self.rag_config.retrieval_k
            similarity_threshold = similarity_threshold or self.rag_config.similarity_threshold
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_metadata
            )
            
            if not results['documents'] or not results['documents'][0]:
                logger.info(f"No results found for query: {query[:50]}...")
                return []
            
            # Convert results to Document objects with similarity scores
            documents_with_scores = []
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
            distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
            
            for doc_text, metadata, distance in zip(documents, metadatas, distances):
                # Convert distance to similarity score (ChromaDB uses cosine distance)
                similarity_score = 1.0 - distance
                
                # Apply similarity threshold
                if similarity_score >= similarity_threshold:
                    document = Document(
                        page_content=doc_text,
                        metadata=metadata or {}
                    )
                    documents_with_scores.append((document, similarity_score))
            
            logger.info(f"Found {len(documents_with_scores)} relevant documents for query")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.chroma_config.collection_name,
                "document_count": count,
                "embedding_model": self.rag_config.embedding_model,
                "chunk_size": self.rag_config.chunk_size,
                "chunk_overlap": self.rag_config.chunk_overlap
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the collection.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if not document_ids:
                logger.warning("No document IDs provided for deletion")
                return True
            
            self.collection.delete(ids=document_ids)
            logger.info(f"Successfully deleted {len(document_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection.
        
        Returns:
            True if clearing was successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.chroma_config.collection_name)
            self._initialize_collection()
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific document.
        
        Args:
            document_id: ID of the document to update
            metadata: New metadata to set
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            self.collection.update(
                ids=[document_id],
                metadatas=[metadata]
            )
            logger.info(f"Successfully updated metadata for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document metadata: {e}")
            return False
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 100) -> List[Document]:
        """Search documents by metadata filters only.
        
        Args:
            metadata_filter: Metadata filters to apply
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit
            )
            
            documents = []
            if results['documents']:
                for doc_text, metadata in zip(results['documents'], results['metadatas'] or []):
                    document = Document(
                        page_content=doc_text,
                        metadata=metadata or {}
                    )
                    documents.append(document)
            
            logger.info(f"Found {len(documents)} documents matching metadata filter")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search by metadata: {e}")
            return []