# RAG Setup Guide

This guide covers setting up and configuring Retrieval-Augmented Generation (RAG) capabilities in the Deep Agent System, including both vector-based (ChromaDB) and graph-based (Neo4j) knowledge retrieval.

## Table of Contents

- [Overview](#overview)
- [ChromaDB Setup (Vector RAG)](#chromadb-setup-vector-rag)
- [Neo4j Setup (Graph RAG)](#neo4j-setup-graph-rag)
- [Knowledge Base Ingestion](#knowledge-base-ingestion)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Overview

The Deep Agent System supports dual RAG capabilities:

- **Vector RAG (ChromaDB)**: Semantic similarity search using embeddings
- **Graph RAG (Neo4j)**: Relationship-based knowledge retrieval using graph structures

Both systems can work independently or together to provide comprehensive context for agent responses.

### Benefits of Dual RAG

1. **Vector RAG**: Excellent for finding semantically similar content
2. **Graph RAG**: Superior for understanding relationships and connections
3. **Hybrid Approach**: Combines both for comprehensive context retrieval

## ChromaDB Setup (Vector RAG)

### Installation Options

#### Option 1: Local ChromaDB (Embedded)
ChromaDB is included in the project dependencies and runs embedded by default.

```bash
# Already included in requirements
# No additional installation needed
```

#### Option 2: ChromaDB Server
For production or multi-user scenarios:

```bash
# Install ChromaDB server
pip install chromadb[server]

# Start ChromaDB server
chroma run --host localhost --port 8000
```

#### Option 3: Docker ChromaDB
```bash
# Run ChromaDB in Docker
docker run -p 8000:8000 chromadb/chroma:latest

# Or with persistent storage
docker run -p 8000:8000 \
  -v ./chroma-data:/chroma/chroma \
  chromadb/chroma:latest
```

### Configuration

Configure ChromaDB in your `.env` file:

```bash
# Enable vector store
VECTOR_STORE_ENABLED=true

# ChromaDB connection (for server mode)
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Collection settings
CHROMADB_COLLECTION_NAME=deep_agent_knowledge
CHROMADB_PERSIST_DIRECTORY=./data/chromadb

# Embedding settings
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# Performance settings
CHROMADB_BATCH_SIZE=1000
CHROMADB_QUERY_TIMEOUT=30
```

### Verification

Test ChromaDB connection:

```bash
# Check system status
deep-agent status

# Test vector store directly
curl http://localhost:8000/api/v1/heartbeat
```

## Neo4j Setup (Graph RAG)

### Installation Options

#### Option 1: Neo4j Desktop
1. Download Neo4j Desktop from [neo4j.com](https://neo4j.com/download/)
2. Install and create a new database
3. Start the database and note the connection details

#### Option 2: Neo4j Community Server

**macOS:**
```bash
# Install with Homebrew
brew install neo4j

# Start Neo4j
neo4j start

# Access at http://localhost:7474
```

**Linux (Ubuntu/Debian):**
```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt update
sudo apt install neo4j

# Start Neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j
```

**Windows:**
1. Download Neo4j Community Server
2. Extract and run `bin/neo4j.bat console`
3. Access at http://localhost:7474

#### Option 3: Docker Neo4j
```bash
# Run Neo4j in Docker
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Access Neo4j Browser at http://localhost:7474
```

### Initial Setup

1. **Access Neo4j Browser**: http://localhost:7474
2. **Login**: Username: `neo4j`, Password: `neo4j` (first time)
3. **Change Password**: Set a secure password
4. **Verify Connection**: Run `MATCH (n) RETURN count(n)`

### Configuration

Configure Neo4j in your `.env` file:

```bash
# Enable graph store
GRAPH_STORE_ENABLED=true

# Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Performance settings
NEO4J_MAX_CONNECTION_LIFETIME=3600
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=60
NEO4J_QUERY_TIMEOUT=30
```

### Verification

Test Neo4j connection:

```bash
# Check system status
deep-agent status

# Test with cypher-shell (if installed)
cypher-shell -u neo4j -p your_password "MATCH (n) RETURN count(n)"
```

## Knowledge Base Ingestion

### Document Types Supported

- **Text Files**: `.txt`, `.md`, `.rst`
- **Code Files**: `.py`, `.js`, `.java`, `.cpp`, etc.
- **Documentation**: README files, API docs, wikis
- **Configuration**: `.json`, `.yaml`, `.toml`
- **Data Files**: `.csv`, `.json` (structured data)

### Ingestion Methods

#### Method 1: Command Line Ingestion

```bash
# Ingest a directory
deep-agent ingest --path ./docs --recursive

# Ingest specific files
deep-agent ingest --files doc1.md doc2.txt code.py

# Ingest with metadata
deep-agent ingest --path ./api-docs --type api_documentation --tags "api,rest,v1"

# Ingest code repository
deep-agent ingest --path ./src --type source_code --language python
```

#### Method 2: Programmatic Ingestion

```python
from deep_agent_system.rag.manager import RAGManager
import asyncio

async def ingest_knowledge_base():
    rag_manager = RAGManager()
    
    # Ingest documents with metadata
    documents = [
        {
            "content": "API endpoint documentation...",
            "metadata": {
                "type": "api_doc",
                "version": "v1",
                "tags": ["authentication", "users"]
            }
        },
        {
            "content": "Code example for user authentication...",
            "metadata": {
                "type": "code_example",
                "language": "python",
                "framework": "fastapi"
            }
        }
    ]
    
    # Ingest to vector store
    await rag_manager.ingest_documents(documents)
    
    # Build knowledge graph
    await rag_manager.build_knowledge_graph(documents)

# Run ingestion
asyncio.run(ingest_knowledge_base())
```

#### Method 3: Batch Ingestion

```bash
# Create ingestion configuration
cat > ingestion_config.yaml << EOF
sources:
  - path: ./docs
    type: documentation
    recursive: true
    include_patterns: ["*.md", "*.rst"]
    exclude_patterns: ["**/node_modules/**"]
    
  - path: ./src
    type: source_code
    recursive: true
    include_patterns: ["*.py", "*.js"]
    metadata:
      project: "deep-agent-system"
      language: "python"
      
  - path: ./examples
    type: code_examples
    recursive: true
    metadata:
      category: "examples"
EOF

# Run batch ingestion
deep-agent ingest --config ingestion_config.yaml
```

### Document Processing Pipeline

The ingestion process includes several steps:

1. **Document Loading**: Read files and extract content
2. **Text Extraction**: Extract text from various file formats
3. **Chunking**: Split documents into manageable chunks
4. **Metadata Extraction**: Extract relevant metadata
5. **Embedding Generation**: Create vector embeddings
6. **Vector Storage**: Store in ChromaDB
7. **Graph Construction**: Build knowledge graph in Neo4j

### Chunking Strategies

Configure chunking in your `.env` file:

```bash
# Chunk size (characters)
RAG_CHUNK_SIZE=1000

# Overlap between chunks
RAG_CHUNK_OVERLAP=200

# Chunking strategy: fixed, semantic, or adaptive
RAG_CHUNKING_STRATEGY=semantic

# Minimum chunk size
RAG_MIN_CHUNK_SIZE=100
```

#### Chunking Examples

**Fixed Size Chunking:**
```python
# Fixed 1000-character chunks with 200-character overlap
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_CHUNKING_STRATEGY=fixed
```

**Semantic Chunking:**
```python
# Chunk by semantic boundaries (paragraphs, sections)
RAG_CHUNKING_STRATEGY=semantic
RAG_SEMANTIC_THRESHOLD=0.8
```

**Code-Aware Chunking:**
```python
# Special handling for code files
RAG_CODE_CHUNKING=true
RAG_CODE_CHUNK_BY_FUNCTION=true
RAG_CODE_PRESERVE_CONTEXT=true
```

## Configuration

### Basic RAG Configuration

```bash
# .env file
# Enable both RAG systems
VECTOR_STORE_ENABLED=true
GRAPH_STORE_ENABLED=true

# Embedding configuration
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# Retrieval settings
RAG_RETRIEVAL_K=5
RAG_SIMILARITY_THRESHOLD=0.7

# Hybrid retrieval weights
RAG_VECTOR_WEIGHT=0.7
RAG_GRAPH_WEIGHT=0.3
```

### Advanced RAG Configuration

```bash
# Performance optimization
RAG_PARALLEL_RETRIEVAL=true
RAG_MAX_PARALLEL_QUERIES=5
RAG_RETRIEVAL_TIMEOUT=10

# Caching
ENABLE_EMBEDDING_CACHE=true
EMBEDDING_CACHE_SIZE=10000
EMBEDDING_CACHE_TTL=3600

# Query expansion
ENABLE_QUERY_EXPANSION=true
QUERY_EXPANSION_TERMS=3

# Re-ranking
ENABLE_RESULT_RERANKING=true
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Agent-Specific RAG Settings

```bash
# Developer Agent - prioritize code examples
DEVELOPER_RAG_CODE_WEIGHT=0.8
DEVELOPER_RAG_DOC_WEIGHT=0.2

# Architect Agent - prioritize design documents
ARCHITECT_RAG_DESIGN_WEIGHT=0.7
ARCHITECT_RAG_PATTERN_WEIGHT=0.3

# Code Reviewer Agent - prioritize best practices
CODE_REVIEWER_RAG_GUIDELINES_WEIGHT=0.6
CODE_REVIEWER_RAG_EXAMPLES_WEIGHT=0.4
```

## Usage Examples

### Basic RAG Queries

```bash
# Questions that will use RAG context
deep-agent "How do we handle authentication in our system?"
deep-agent "What's our standard error handling pattern?"
deep-agent "Show me examples of our API endpoints"
```

### Targeted RAG Queries

```bash
# Query specific document types
deep-agent "Find API documentation for user management" --rag-filter "type:api_doc"

# Query by tags
deep-agent "Show authentication examples" --rag-filter "tags:authentication"

# Query by language
deep-agent "Python code examples for database operations" --rag-filter "language:python"
```

### RAG with Context Control

```bash
# Specify retrieval method
deep-agent "Design patterns in our codebase" --rag-method vector
deep-agent "Related components to user service" --rag-method graph
deep-agent "Comprehensive analysis of user system" --rag-method hybrid

# Control context amount
deep-agent "Quick overview of API structure" --rag-k 3
deep-agent "Detailed analysis of authentication flow" --rag-k 10
```

### Programmatic RAG Usage

```python
from deep_agent_system.rag.manager import RAGManager

async def query_knowledge_base():
    rag_manager = RAGManager()
    
    # Vector search
    vector_results = await rag_manager.retrieve_vector_context(
        query="user authentication patterns",
        k=5,
        filters={"type": "code_example"}
    )
    
    # Graph search
    graph_results = await rag_manager.retrieve_graph_context(
        query="user service dependencies",
        depth=2,
        relationship_types=["DEPENDS_ON", "IMPLEMENTS"]
    )
    
    # Hybrid search
    hybrid_results = await rag_manager.hybrid_retrieve(
        query="authentication implementation",
        vector_weight=0.7,
        graph_weight=0.3
    )
    
    return hybrid_results
```

## Performance Optimization

### Vector Store Optimization

```bash
# Optimize embedding generation
EMBEDDING_BATCH_SIZE=100
EMBEDDING_MAX_RETRIES=3
EMBEDDING_TIMEOUT=30

# Optimize vector search
VECTOR_SEARCH_ALGORITHM=hnsw
VECTOR_INDEX_PARAMS='{"M": 16, "efConstruction": 200}'
VECTOR_SEARCH_PARAMS='{"ef": 100}'

# Memory optimization
VECTOR_STORE_MEMORY_LIMIT=4GB
VECTOR_STORE_CACHE_SIZE=1000
```

### Graph Store Optimization

```bash
# Query optimization
NEO4J_QUERY_CACHE_SIZE=1000
NEO4J_ENABLE_QUERY_CACHE=true
NEO4J_QUERY_TIMEOUT=30

# Index optimization
NEO4J_AUTO_INDEX=true
NEO4J_INDEX_PROPERTIES=name,type,content_hash

# Memory settings
NEO4J_HEAP_SIZE=2G
NEO4J_PAGE_CACHE_SIZE=1G
```

### Retrieval Optimization

```bash
# Parallel processing
RAG_PARALLEL_RETRIEVAL=true
RAG_MAX_PARALLEL_QUERIES=5
RAG_WORKER_THREADS=4

# Caching strategies
ENABLE_QUERY_CACHE=true
QUERY_CACHE_SIZE=1000
QUERY_CACHE_TTL=1800

# Result optimization
ENABLE_RESULT_DEDUPLICATION=true
RESULT_SIMILARITY_THRESHOLD=0.95
```

## Troubleshooting

### Common Issues

#### ChromaDB Connection Issues

```bash
# Check if ChromaDB is running
curl http://localhost:8000/api/v1/heartbeat

# Check ChromaDB logs
docker logs chromadb  # if using Docker

# Test connection
python -c "import chromadb; client = chromadb.Client(); print('Connected!')"
```

#### Neo4j Connection Issues

```bash
# Check Neo4j status
sudo systemctl status neo4j  # Linux
brew services list | grep neo4j  # macOS

# Test connection
cypher-shell -u neo4j -p password "RETURN 1"

# Check Neo4j logs
tail -f /var/log/neo4j/neo4j.log  # Linux
tail -f /usr/local/var/log/neo4j/neo4j.log  # macOS
```

#### Embedding Issues

```bash
# Test OpenAI API connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/embeddings \
     -d '{"input": "test", "model": "text-embedding-ada-002"}'

# Check embedding cache
ls -la ./data/embeddings_cache/

# Clear embedding cache
rm -rf ./data/embeddings_cache/*
```

### Performance Issues

#### Slow Retrieval

```bash
# Reduce retrieval count
RAG_RETRIEVAL_K=3

# Increase similarity threshold
RAG_SIMILARITY_THRESHOLD=0.8

# Enable caching
ENABLE_QUERY_CACHE=true
ENABLE_EMBEDDING_CACHE=true
```

#### Memory Issues

```bash
# Reduce chunk size
RAG_CHUNK_SIZE=500

# Limit concurrent operations
RAG_MAX_PARALLEL_QUERIES=2

# Reduce cache sizes
VECTOR_STORE_CACHE_SIZE=500
EMBEDDING_CACHE_SIZE=5000
```

#### Database Performance

```bash
# ChromaDB optimization
CHROMADB_BATCH_SIZE=500
CHROMADB_QUERY_TIMEOUT=60

# Neo4j optimization
NEO4J_QUERY_TIMEOUT=60
NEO4J_MAX_CONNECTION_POOL_SIZE=25
```

### Debugging RAG Issues

#### Enable Debug Logging

```bash
# Enable RAG debug logging
LOG_LEVEL=DEBUG
LOG_RAG_RETRIEVALS=true
LOG_EMBEDDING_OPERATIONS=true
```

#### Test RAG Components

```bash
# Test vector retrieval
deep-agent test-rag --method vector --query "test query"

# Test graph retrieval
deep-agent test-rag --method graph --query "test query"

# Test hybrid retrieval
deep-agent test-rag --method hybrid --query "test query"
```

#### Inspect Knowledge Base

```bash
# List collections
deep-agent list-collections

# Show collection stats
deep-agent collection-stats --collection deep_agent_knowledge

# Query collection directly
deep-agent query-collection --collection deep_agent_knowledge --query "test"
```

### Data Quality Issues

#### Improve Ingestion Quality

```bash
# Use better chunking
RAG_CHUNKING_STRATEGY=semantic
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=100

# Add metadata
deep-agent ingest --path ./docs --metadata-extractor advanced

# Clean existing data
deep-agent clean-collection --collection deep_agent_knowledge
deep-agent reingest --path ./docs --force
```

#### Monitor Retrieval Quality

```bash
# Enable retrieval metrics
ENABLE_RETRIEVAL_METRICS=true
METRICS_LOG_FILE=./logs/retrieval_metrics.log

# Test retrieval quality
deep-agent test-retrieval --queries ./test_queries.txt --ground-truth ./expected_results.json
```

This comprehensive RAG setup guide should help you configure and optimize both vector and graph-based knowledge retrieval for the Deep Agent System.