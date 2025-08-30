"""Neo4j graph store integration for RAG functionality."""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import re
import uuid
from dataclasses import dataclass

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
from langchain.schema import Document
from pydantic import BaseModel

from ..config.models import Neo4jConfig, RAGConfig


logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str
    properties: Dict[str, Any]
    source_document: str


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_entity: str
    target_entity: str
    relationship_type: str
    properties: Dict[str, Any]
    source_document: str


class GraphContext(BaseModel):
    """Represents graph-based context for RAG."""
    entities: List[Entity]
    relationships: List[Relationship]
    subgraph_query: str
    relevance_score: float


class Neo4jStore:
    """Neo4j graph store for knowledge graph storage and retrieval."""
    
    def __init__(self, neo4j_config: Neo4jConfig, rag_config: RAGConfig):
        """Initialize Neo4j store with configuration.
        
        Args:
            neo4j_config: Neo4j-specific configuration
            rag_config: General RAG configuration
        """
        self.neo4j_config = neo4j_config
        self.rag_config = rag_config
        self.driver: Optional[Driver] = None
        
        self._initialize_driver()
        self._create_indexes()
    
    def _initialize_driver(self) -> None:
        """Initialize Neo4j driver."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_config.uri,
                auth=(self.neo4j_config.username, self.neo4j_config.password)
            )
            
            # Test connection
            with self.driver.session(database=self.neo4j_config.database) as session:
                session.run("RETURN 1")
            
            logger.info(f"Neo4j driver initialized for database '{self.neo4j_config.database}'")
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise
    
    def _create_indexes(self) -> None:
        """Create necessary indexes for performance."""
        try:
            with self.driver.session(database=self.neo4j_config.database) as session:
                # Create indexes for entities
                session.run("""
                    CREATE INDEX entity_name_index IF NOT EXISTS 
                    FOR (e:Entity) ON (e.name)
                """)
                
                session.run("""
                    CREATE INDEX entity_type_index IF NOT EXISTS 
                    FOR (e:Entity) ON (e.type)
                """)
                
                # Create index for documents
                session.run("""
                    CREATE INDEX document_source_index IF NOT EXISTS 
                    FOR (d:Document) ON (d.source)
                """)
                
                logger.info("Neo4j indexes created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            # Don't raise here as indexes are for performance, not functionality
    
    def extract_entities_and_relationships(self, documents: List[Document]) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from documents.
        
        This is a simple rule-based extraction. In production, you might want to use
        more sophisticated NLP models like spaCy, NLTK, or transformer-based NER models.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []
        
        # Simple patterns for entity extraction
        entity_patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Simple name pattern
            'TECHNOLOGY': r'\b(?:Python|Java|JavaScript|React|Django|Flask|ChromaDB|Neo4j|LangChain|OpenAI|GPT|AI|ML|API)\b',
            'CONCEPT': r'\b(?:machine learning|artificial intelligence|database|vector|embedding|retrieval|agent|workflow)\b',
            'FILE': r'\b\w+\.(py|js|txt|md|json|yaml|yml)\b',
            'CLASS': r'\bclass [A-Z][a-zA-Z]*\b',
            'FUNCTION': r'\bdef [a-z_][a-zA-Z0-9_]*\b'
        }
        
        # Simple patterns for relationships
        relationship_patterns = [
            (r'(\w+) uses (\w+)', 'USES'),
            (r'(\w+) extends (\w+)', 'EXTENDS'),
            (r'(\w+) implements (\w+)', 'IMPLEMENTS'),
            (r'(\w+) contains (\w+)', 'CONTAINS'),
            (r'(\w+) depends on (\w+)', 'DEPENDS_ON'),
            (r'(\w+) is a (\w+)', 'IS_A'),
            (r'(\w+) has (\w+)', 'HAS')
        ]
        
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            content = doc.page_content
            
            # Extract entities
            doc_entities = set()
            for entity_type, pattern in entity_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    entity_name = match.strip()
                    if entity_name and len(entity_name) > 1:
                        entity = Entity(
                            name=entity_name,
                            type=entity_type,
                            properties={'confidence': 0.8},
                            source_document=source
                        )
                        entities.append(entity)
                        doc_entities.add(entity_name.lower())
            
            # Extract relationships
            for pattern, rel_type in relationship_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        source_entity, target_entity = match
                        source_entity = source_entity.strip()
                        target_entity = target_entity.strip()
                        
                        if (source_entity and target_entity and 
                            len(source_entity) > 1 and len(target_entity) > 1):
                            relationship = Relationship(
                                source_entity=source_entity,
                                target_entity=target_entity,
                                relationship_type=rel_type,
                                properties={'confidence': 0.7},
                                source_document=source
                            )
                            relationships.append(relationship)
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
        return entities, relationships
    
    def ingest_documents(self, documents: List[Document]) -> bool:
        """Ingest documents into the graph store.
        
        Args:
            documents: List of documents to ingest
            
        Returns:
            True if ingestion was successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided for ingestion")
                return True
            
            # Extract entities and relationships
            entities, relationships = self.extract_entities_and_relationships(documents)
            
            with self.driver.session(database=self.neo4j_config.database) as session:
                # Create document nodes
                for doc in documents:
                    doc_id = str(uuid.uuid4())
                    session.run("""
                        CREATE (d:Document {
                            id: $doc_id,
                            source: $source,
                            content: $content,
                            metadata: $metadata
                        })
                    """, {
                        'doc_id': doc_id,
                        'source': doc.metadata.get('source', 'unknown'),
                        'content': doc.page_content,
                        'metadata': str(doc.metadata)
                    })
                
                # Create entity nodes
                for entity in entities:
                    session.run("""
                        MERGE (e:Entity {name: $name, type: $type})
                        SET e.properties = $properties,
                            e.source_document = $source_document
                    """, {
                        'name': entity.name,
                        'type': entity.type,
                        'properties': entity.properties,
                        'source_document': entity.source_document
                    })
                
                # Create relationships
                for rel in relationships:
                    session.run(f"""
                        MATCH (source:Entity {{name: $source_entity}})
                        MATCH (target:Entity {{name: $target_entity}})
                        MERGE (source)-[r:{rel.relationship_type}]->(target)
                        SET r.properties = $properties,
                            r.source_document = $source_document
                    """, {
                        'source_entity': rel.source_entity,
                        'target_entity': rel.target_entity,
                        'properties': rel.properties,
                        'source_document': rel.source_document
                    })
                
                # Create relationships between entities and documents
                for entity in entities:
                    session.run("""
                        MATCH (e:Entity {name: $entity_name})
                        MATCH (d:Document {source: $source})
                        MERGE (e)-[:MENTIONED_IN]->(d)
                    """, {
                        'entity_name': entity.name,
                        'source': entity.source_document
                    })
            
            logger.info(f"Successfully ingested {len(documents)} documents into graph store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest documents: {e}")
            return False
    
    def query_graph(self, query: str, depth: int = 2, limit: int = 50) -> GraphContext:
        """Query the graph for relevant entities and relationships.
        
        Args:
            query: Search query
            depth: Maximum traversal depth
            limit: Maximum number of results
            
        Returns:
            GraphContext containing relevant graph data
        """
        try:
            # Simple keyword-based matching for entities
            query_terms = query.lower().split()
            
            with self.driver.session(database=self.neo4j_config.database) as session:
                # Find relevant entities
                entity_query = """
                    MATCH (e:Entity)
                    WHERE ANY(term IN $query_terms WHERE toLower(e.name) CONTAINS term)
                    RETURN e.name as name, e.type as type, e.properties as properties, 
                           e.source_document as source_document
                    LIMIT $limit
                """
                
                entity_results = session.run(entity_query, {
                    'query_terms': query_terms,
                    'limit': limit
                })
                
                entities = []
                entity_names = set()
                for record in entity_results:
                    entity = Entity(
                        name=record['name'],
                        type=record['type'],
                        properties=record['properties'] or {},
                        source_document=record['source_document']
                    )
                    entities.append(entity)
                    entity_names.add(record['name'])
                
                # Find relationships involving these entities
                if entity_names:
                    rel_query = f"""
                        MATCH (source:Entity)-[r]->(target:Entity)
                        WHERE source.name IN $entity_names OR target.name IN $entity_names
                        RETURN source.name as source_entity, target.name as target_entity,
                               type(r) as relationship_type, r.properties as properties,
                               r.source_document as source_document
                        LIMIT $limit
                    """
                    
                    rel_results = session.run(rel_query, {
                        'entity_names': list(entity_names),
                        'limit': limit
                    })
                    
                    relationships = []
                    for record in rel_results:
                        relationship = Relationship(
                            source_entity=record['source_entity'],
                            target_entity=record['target_entity'],
                            relationship_type=record['relationship_type'],
                            properties=record['properties'] or {},
                            source_document=record['source_document']
                        )
                        relationships.append(relationship)
                else:
                    relationships = []
                
                # Calculate relevance score based on number of matches
                relevance_score = min(1.0, len(entities) / max(1, len(query_terms)))
                
                return GraphContext(
                    entities=entities,
                    relationships=relationships,
                    subgraph_query=query,
                    relevance_score=relevance_score
                )
                
        except Exception as e:
            logger.error(f"Failed to query graph: {e}")
            return GraphContext(
                entities=[],
                relationships=[],
                subgraph_query=query,
                relevance_score=0.0
            )
    
    def get_connected_entities(self, entity_name: str, depth: int = 2) -> List[Entity]:
        """Get entities connected to a specific entity.
        
        Args:
            entity_name: Name of the central entity
            depth: Maximum traversal depth
            
        Returns:
            List of connected entities
        """
        try:
            with self.driver.session(database=self.neo4j_config.database) as session:
                query = f"""
                    MATCH (start:Entity {{name: $entity_name}})
                    CALL apoc.path.subgraphNodes(start, {{
                        maxLevel: $depth,
                        relationshipFilter: null,
                        labelFilter: "Entity"
                    }}) YIELD node
                    RETURN node.name as name, node.type as type, 
                           node.properties as properties, node.source_document as source_document
                """
                
                # Fallback query if APOC is not available
                fallback_query = f"""
                    MATCH (start:Entity {{name: $entity_name}})
                    MATCH (start)-[*1..{depth}]-(connected:Entity)
                    RETURN DISTINCT connected.name as name, connected.type as type,
                           connected.properties as properties, connected.source_document as source_document
                """
                
                try:
                    results = session.run(query, {'entity_name': entity_name, 'depth': depth})
                except Exception:
                    # Use fallback if APOC is not available
                    results = session.run(fallback_query, {'entity_name': entity_name})
                
                entities = []
                for record in results:
                    entity = Entity(
                        name=record['name'],
                        type=record['type'],
                        properties=record['properties'] or {},
                        source_document=record['source_document']
                    )
                    entities.append(entity)
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to get connected entities: {e}")
            return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        try:
            with self.driver.session(database=self.neo4j_config.database) as session:
                # Count nodes by type
                node_counts = {}
                result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
                for record in result:
                    labels = record['labels']
                    count = record['count']
                    for label in labels:
                        node_counts[label] = node_counts.get(label, 0) + count
                
                # Count relationships by type
                rel_counts = {}
                result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
                for record in result:
                    rel_counts[record['type']] = record['count']
                
                return {
                    "database": self.neo4j_config.database,
                    "node_counts": node_counts,
                    "relationship_counts": rel_counts,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": sum(rel_counts.values())
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {"error": str(e)}
    
    def clear_graph(self) -> bool:
        """Clear all data from the graph.
        
        Returns:
            True if clearing was successful, False otherwise
        """
        try:
            with self.driver.session(database=self.neo4j_config.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            
            logger.info("Graph cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return False
    
    def execute_cypher_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        try:
            with self.driver.session(database=self.neo4j_config.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Failed to execute Cypher query: {e}")
            return []
    
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()