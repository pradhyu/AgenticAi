"""Unified RAG manager that coordinates vector and graph retrieval."""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass

from langchain.schema import Document
from pydantic import BaseModel

from .vector_store import ChromaDBStore
from .graph_store import Neo4jStore, GraphContext
from ..config.models import RAGConfig, ChromaDBConfig, Neo4jConfig, RetrievalType


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result with source information."""
    content: str
    metadata: Dict[str, Any]
    score: float
    source_type: RetrievalType
    source_id: str


class CombinedContext(BaseModel):
    """Represents combined context from multiple retrieval sources."""
    vector_results: List[Tuple[Document, float]]
    graph_context: Optional[GraphContext]
    combined_results: List[RetrievalResult]
    total_score: float
    retrieval_strategy: str


class RAGManager:
    """Unified RAG manager that coordinates vector and graph retrieval."""
    
    def __init__(
        self,
        rag_config: RAGConfig,
        chroma_config: Optional[ChromaDBConfig] = None,
        neo4j_config: Optional[Neo4jConfig] = None
    ):
        """Initialize RAG manager with configuration.
        
        Args:
            rag_config: General RAG configuration
            chroma_config: ChromaDB configuration (optional)
            neo4j_config: Neo4j configuration (optional)
        """
        self.rag_config = rag_config
        self.vector_store: Optional[ChromaDBStore] = None
        self.graph_store: Optional[Neo4jStore] = None
        
        # Initialize stores based on configuration and availability
        if rag_config.vector_store_enabled and chroma_config:
            try:
                self.vector_store = ChromaDBStore(chroma_config, rag_config)
                logger.info("Vector store initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                
        if rag_config.graph_store_enabled and neo4j_config:
            try:
                self.graph_store = Neo4jStore(neo4j_config, rag_config)
                logger.info("Graph store initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize graph store: {e}")
    
    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Ingest documents into both vector and graph stores.
        
        Args:
            documents: List of documents to ingest
            
        Returns:
            Dictionary containing ingestion results
        """
        results = {
            "vector_store": {"success": False, "document_ids": []},
            "graph_store": {"success": False},
            "total_documents": len(documents)
        }
        
        if not documents:
            logger.warning("No documents provided for ingestion")
            return results
        
        # Ingest into vector store
        if self.vector_store:
            try:
                doc_ids = self.vector_store.ingest_documents(documents)
                results["vector_store"]["success"] = True
                results["vector_store"]["document_ids"] = doc_ids
                logger.info(f"Successfully ingested {len(doc_ids)} chunks into vector store")
            except Exception as e:
                logger.error(f"Failed to ingest documents into vector store: {e}")
        
        # Ingest into graph store
        if self.graph_store:
            try:
                success = self.graph_store.ingest_documents(documents)
                results["graph_store"]["success"] = success
                if success:
                    logger.info("Successfully ingested documents into graph store")
            except Exception as e:
                logger.error(f"Failed to ingest documents into graph store: {e}")
        
        return results
    
    def retrieve_vector_context(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Retrieve context using vector similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Metadata filters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vector_store:
            logger.warning("Vector store not available for retrieval")
            return []
        
        try:
            return self.vector_store.similarity_search(
                query=query,
                k=k,
                filter_metadata=filter_metadata,
                similarity_threshold=similarity_threshold
            )
        except Exception as e:
            logger.error(f"Failed to retrieve vector context: {e}")
            return []
    
    def retrieve_graph_context(
        self,
        query: str,
        depth: int = 2,
        limit: int = 50
    ) -> Optional[GraphContext]:
        """Retrieve context using graph traversal.
        
        Args:
            query: Search query
            depth: Maximum traversal depth
            limit: Maximum number of results
            
        Returns:
            GraphContext or None if failed
        """
        if not self.graph_store:
            logger.warning("Graph store not available for retrieval")
            return None
        
        try:
            return self.graph_store.query_graph(query, depth, limit)
        except Exception as e:
            logger.error(f"Failed to retrieve graph context: {e}")
            return None
    
    def hybrid_retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        depth: int = 2,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> CombinedContext:
        """Perform hybrid retrieval combining vector and graph approaches.
        
        Args:
            query: Search query
            k: Number of vector results to return
            depth: Graph traversal depth
            vector_weight: Weight for vector results in scoring
            graph_weight: Weight for graph results in scoring
            filter_metadata: Metadata filters for vector search
            
        Returns:
            CombinedContext containing results from both approaches
        """
        k = k or self.rag_config.retrieval_k
        
        # Retrieve from vector store
        vector_results = self.retrieve_vector_context(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )
        
        # Retrieve from graph store
        graph_context = self.retrieve_graph_context(
            query=query,
            depth=depth
        )
        
        # Combine and rank results
        combined_results = self._combine_and_rank_results(
            vector_results=vector_results,
            graph_context=graph_context,
            vector_weight=vector_weight,
            graph_weight=graph_weight
        )
        
        # Calculate total score
        total_score = sum(result.score for result in combined_results)
        
        # Determine strategy used
        strategy = self._determine_strategy(vector_results, graph_context)
        
        return CombinedContext(
            vector_results=vector_results,
            graph_context=graph_context,
            combined_results=combined_results,
            total_score=total_score,
            retrieval_strategy=strategy
        )
    
    def _combine_and_rank_results(
        self,
        vector_results: List[Tuple[Document, float]],
        graph_context: Optional[GraphContext],
        vector_weight: float,
        graph_weight: float
    ) -> List[RetrievalResult]:
        """Combine and rank results from different retrieval methods.
        
        Args:
            vector_results: Results from vector search
            graph_context: Results from graph search
            vector_weight: Weight for vector results
            graph_weight: Weight for graph results
            
        Returns:
            List of ranked RetrievalResult objects
        """
        combined_results = []
        
        # Add vector results
        for i, (doc, score) in enumerate(vector_results):
            result = RetrievalResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score * vector_weight,
                source_type=RetrievalType.VECTOR,
                source_id=f"vector_{i}"
            )
            combined_results.append(result)
        
        # Add graph results
        if graph_context and graph_context.entities:
            # Create content from entities and relationships
            graph_content_parts = []
            
            # Add entity information
            for entity in graph_context.entities:
                entity_info = f"Entity: {entity.name} (Type: {entity.type})"
                if entity.properties:
                    entity_info += f" - Properties: {entity.properties}"
                graph_content_parts.append(entity_info)
            
            # Add relationship information
            for rel in graph_context.relationships:
                rel_info = f"Relationship: {rel.source_entity} -{rel.relationship_type}-> {rel.target_entity}"
                if rel.properties:
                    rel_info += f" - Properties: {rel.properties}"
                graph_content_parts.append(rel_info)
            
            if graph_content_parts:
                graph_content = "\n".join(graph_content_parts)
                result = RetrievalResult(
                    content=graph_content,
                    metadata={
                        "entity_count": len(graph_context.entities),
                        "relationship_count": len(graph_context.relationships),
                        "subgraph_query": graph_context.subgraph_query
                    },
                    score=graph_context.relevance_score * graph_weight,
                    source_type=RetrievalType.GRAPH,
                    source_id="graph_0"
                )
                combined_results.append(result)
        
        # Sort by score (descending)
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
    
    def _determine_strategy(
        self,
        vector_results: List[Tuple[Document, float]],
        graph_context: Optional[GraphContext]
    ) -> str:
        """Determine which retrieval strategy was most effective.
        
        Args:
            vector_results: Results from vector search
            graph_context: Results from graph search
            
        Returns:
            String describing the strategy used
        """
        has_vector = len(vector_results) > 0
        has_graph = graph_context and len(graph_context.entities) > 0
        
        if has_vector and has_graph:
            return "hybrid"
        elif has_vector:
            return "vector_only"
        elif has_graph:
            return "graph_only"
        else:
            return "no_results"
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval systems.
        
        Returns:
            Dictionary containing statistics from both stores
        """
        stats = {
            "vector_store": None,
            "graph_store": None,
            "rag_config": {
                "vector_enabled": self.rag_config.vector_store_enabled,
                "graph_enabled": self.rag_config.graph_store_enabled,
                "retrieval_k": self.rag_config.retrieval_k,
                "similarity_threshold": self.rag_config.similarity_threshold
            }
        }
        
        if self.vector_store:
            try:
                stats["vector_store"] = self.vector_store.get_collection_stats()
            except Exception as e:
                logger.error(f"Failed to get vector store stats: {e}")
                stats["vector_store"] = {"error": str(e)}
        
        if self.graph_store:
            try:
                stats["graph_store"] = self.graph_store.get_graph_stats()
            except Exception as e:
                logger.error(f"Failed to get graph store stats: {e}")
                stats["graph_store"] = {"error": str(e)}
        
        return stats
    
    def clear_all_data(self) -> Dict[str, bool]:
        """Clear all data from both stores.
        
        Returns:
            Dictionary indicating success for each store
        """
        results = {"vector_store": False, "graph_store": False}
        
        if self.vector_store:
            try:
                results["vector_store"] = self.vector_store.clear_collection()
            except Exception as e:
                logger.error(f"Failed to clear vector store: {e}")
        
        if self.graph_store:
            try:
                results["graph_store"] = self.graph_store.clear_graph()
            except Exception as e:
                logger.error(f"Failed to clear graph store: {e}")
        
        return results
    
    def search_by_retrieval_type(
        self,
        query: str,
        retrieval_type: RetrievalType,
        **kwargs
    ) -> Union[List[Tuple[Document, float]], GraphContext, None]:
        """Search using a specific retrieval type.
        
        Args:
            query: Search query
            retrieval_type: Type of retrieval to use
            **kwargs: Additional arguments for the specific retrieval method
            
        Returns:
            Results based on the retrieval type
        """
        if retrieval_type == RetrievalType.VECTOR:
            return self.retrieve_vector_context(query, **kwargs)
        elif retrieval_type == RetrievalType.GRAPH:
            return self.retrieve_graph_context(query, **kwargs)
        elif retrieval_type == RetrievalType.HYBRID:
            return self.hybrid_retrieve(query, **kwargs)
        else:
            logger.error(f"Unknown retrieval type: {retrieval_type}")
            return None
    
    def close(self) -> None:
        """Close connections to both stores."""
        if self.graph_store:
            self.graph_store.close()
            logger.info("Graph store connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()