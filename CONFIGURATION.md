# Configuration Guide

This guide covers all configuration options for the Deep Agent System, including environment variables, prompt customization, and advanced settings.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Prompt Configuration](#prompt-configuration)
- [Database Configuration](#database-configuration)
- [Agent Configuration](#agent-configuration)
- [Performance Tuning](#performance-tuning)
- [Security Settings](#security-settings)
- [Development Settings](#development-settings)

## Environment Variables

All configuration is managed through environment variables, typically stored in a `.env` file.

### Core Configuration

#### LLM Configuration
```bash
# Required: OpenAI API Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=4000

# Optional: Alternative LLM providers
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

**Options:**
- `OPENAI_MODEL`: `gpt-4-turbo-preview`, `gpt-4`, `gpt-3.5-turbo`
- `OPENAI_TEMPERATURE`: 0.0 (deterministic) to 1.0 (creative)
- `OPENAI_MAX_TOKENS`: Maximum tokens per response (1-4000)

#### Prompt Configuration
```bash
# Prompt directory path (relative to project root)
PROMPT_DIR=./prompts

# Prompt reloading
ENABLE_HOT_RELOAD=false
```

### Database Configuration

#### ChromaDB (Vector Store)
```bash
# Enable/disable vector store
VECTOR_STORE_ENABLED=true

# ChromaDB connection
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
CHROMADB_COLLECTION_NAME=deep_agent_knowledge
CHROMADB_PERSIST_DIRECTORY=./data/chromadb

# ChromaDB authentication (if required)
CHROMADB_AUTH_PROVIDER=basic
CHROMADB_AUTH_CREDENTIALS=username:password
```

#### Neo4j (Graph Database)
```bash
# Enable/disable graph store
GRAPH_STORE_ENABLED=true

# Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Neo4j connection pool settings
NEO4J_MAX_CONNECTION_LIFETIME=3600
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=60
```

### RAG Configuration

#### Embedding Settings
```bash
# Embedding model
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# Alternative embedding models
# EMBEDDING_MODEL=text-embedding-3-small  # 1536 dimensions
# EMBEDDING_MODEL=text-embedding-3-large  # 3072 dimensions
```

#### Retrieval Parameters
```bash
# Document chunking
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200

# Retrieval settings
RAG_RETRIEVAL_K=5
RAG_SIMILARITY_THRESHOLD=0.7

# Hybrid retrieval weights
RAG_VECTOR_WEIGHT=0.7
RAG_GRAPH_WEIGHT=0.3
```

### Agent Configuration

#### General Agent Settings
```bash
# Agent timeouts
AGENT_TIMEOUT=300
MAX_RETRIES=3

# Agent communication
ENABLE_AGENT_LOGGING=true
AGENT_COMMUNICATION_TIMEOUT=30
```

#### Specific Agent Settings
```bash
# Analyst Agent
ANALYST_CONFIDENCE_THRESHOLD=0.8
ANALYST_MAX_ROUTING_ATTEMPTS=3

# Developer Agent
DEVELOPER_MAX_CODE_LENGTH=5000
DEVELOPER_ENABLE_SYNTAX_CHECK=true

# Code Reviewer Agent
CODE_REVIEWER_SEVERITY_THRESHOLD=medium
CODE_REVIEWER_MAX_SUGGESTIONS=10

# Tester Agent
TESTER_DEFAULT_FRAMEWORK=pytest
TESTER_COVERAGE_THRESHOLD=80

# Architect Agent
ARCHITECT_DIAGRAM_FORMAT=mermaid
ARCHITECT_MAX_COMPONENTS=20
```

### CLI Configuration

#### Rich CLI Settings
```bash
# Appearance
CLI_THEME=dark
CLI_SYNTAX_HIGHLIGHTING=true
CLI_SHOW_PROGRESS=true
CLI_MAX_WIDTH=120

# Behavior
CLI_AUTO_PAGER=true
CLI_CONFIRM_DESTRUCTIVE=true
CLI_HISTORY_SIZE=1000
```

### Logging Configuration

#### Basic Logging
```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Log format: json, text
LOG_FORMAT=json

# Log file path
LOG_FILE=./logs/deep_agent_system.log

# Enable debug logging
ENABLE_DEBUG_LOGGING=false
```

#### Advanced Logging
```bash
# Structured logging
ENABLE_STRUCTURED_LOGGING=true
LOG_CORRELATION_ID=true

# Performance logging
ENABLE_PERFORMANCE_LOGGING=true
LOG_SLOW_QUERIES=true
SLOW_QUERY_THRESHOLD=1.0

# Agent interaction logging
LOG_AGENT_COMMUNICATIONS=true
LOG_RAG_RETRIEVALS=true
LOG_WORKFLOW_STEPS=true
```

### Workflow Configuration

#### LangGraph Settings
```bash
# Workflow execution
WORKFLOW_TIMEOUT=600
WORKFLOW_MAX_STEPS=50

# Checkpointing
ENABLE_WORKFLOW_CHECKPOINTS=true
WORKFLOW_CHECKPOINT_DIR=./data/checkpoints
CHECKPOINT_INTERVAL=10

# Parallel execution
WORKFLOW_MAX_PARALLEL_AGENTS=3
WORKFLOW_PARALLEL_TIMEOUT=120
```

### Security Configuration

#### API Security
```bash
# Rate limiting
API_RATE_LIMIT=100
API_RATE_WINDOW=60

# Input validation
ENABLE_INPUT_VALIDATION=true
MAX_INPUT_LENGTH=10000
SANITIZE_OUTPUTS=true

# Content filtering
ENABLE_CONTENT_FILTER=true
BLOCKED_PATTERNS=password,secret,token
```

#### Data Security
```bash
# Encryption
ENCRYPT_STORED_DATA=false
ENCRYPTION_KEY_PATH=./keys/encryption.key

# Data retention
DATA_RETENTION_DAYS=30
AUTO_CLEANUP_ENABLED=true
```

### Performance Configuration

#### Memory Management
```bash
# Memory limits
MAX_MEMORY_USAGE=8GB
ENABLE_MEMORY_MONITORING=true
MEMORY_WARNING_THRESHOLD=0.8

# Caching
ENABLE_RESPONSE_CACHING=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Connection pooling
DATABASE_POOL_SIZE=10
DATABASE_POOL_TIMEOUT=30
```

#### Optimization Settings
```bash
# Async settings
MAX_CONCURRENT_REQUESTS=10
ASYNC_TIMEOUT=300

# Batch processing
ENABLE_BATCH_PROCESSING=true
BATCH_SIZE=50
BATCH_TIMEOUT=60
```

## Configuration Files

### Environment File Structure

Create a `.env` file in your project root:

```bash
# .env file structure
# =============================================================================
# Core Configuration
# =============================================================================
OPENAI_API_KEY=your_key_here
PROMPT_DIR=./prompts

# =============================================================================
# Database Configuration  
# =============================================================================
VECTOR_STORE_ENABLED=true
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

GRAPH_STORE_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# =============================================================================
# RAG Configuration
# =============================================================================
EMBEDDING_MODEL=text-embedding-ada-002
RAG_CHUNK_SIZE=1000
RAG_RETRIEVAL_K=5

# =============================================================================
# Agent Configuration
# =============================================================================
AGENT_TIMEOUT=300
MAX_RETRIES=3

# =============================================================================
# CLI Configuration
# =============================================================================
CLI_THEME=dark
CLI_SYNTAX_HIGHLIGHTING=true

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL=INFO
LOG_FILE=./logs/deep_agent_system.log
```

### Configuration Validation

Validate your configuration:

```bash
# Check configuration
deep-agent status --verbose

# Validate specific components
deep-agent validate-config --component database
deep-agent validate-config --component rag
deep-agent validate-config --component agents
```

## Prompt Configuration

### Prompt Directory Structure

```
prompts/
├── analyst/
│   ├── classification.txt    # Question classification logic
│   ├── routing.txt          # Agent routing decisions
│   └── aggregation.txt      # Multi-agent response aggregation
├── architect/
│   ├── design.txt           # System design prompts
│   ├── explanation.txt      # Architecture explanations
│   └── diagram.txt          # Diagram generation
├── developer/
│   ├── implementation.txt   # Code implementation
│   ├── debugging.txt        # Debugging assistance
│   └── optimization.txt     # Code optimization
├── code_reviewer/
│   ├── review.txt           # Code review analysis
│   ├── security.txt         # Security assessment
│   └── performance.txt      # Performance review
└── tester/
    ├── test_creation.txt    # Test generation
    ├── strategy.txt         # Testing strategy
    └── validation.txt       # Test validation
```

### Prompt Template Variables

Available variables in prompts:

```bash
# Core variables
{question}              # User's original question
{context}               # Retrieved RAG context
{agent_type}            # Current agent type
{conversation_history}  # Previous conversation

# Agent-specific variables
{code_snippet}          # Code being reviewed (Code Reviewer)
{architecture_type}     # Architecture pattern (Architect)
{test_framework}        # Testing framework (Tester)
{language}              # Programming language (Developer)

# System variables
{timestamp}             # Current timestamp
{session_id}            # Session identifier
{confidence_threshold}  # Confidence threshold
```

### Custom Prompt Examples

#### Developer Agent Prompt
```text
# prompts/developer/implementation.txt

You are an expert software developer with deep knowledge of multiple programming languages and frameworks.

## Context
Question: {question}
Language: {language}
Framework: {framework}

## Retrieved Context
{context}

## Instructions
1. Analyze the question to understand the implementation requirements
2. Consider best practices for {language}
3. Provide clean, well-documented code
4. Include error handling where appropriate
5. Explain your implementation choices

## Response Format
- Brief explanation of the approach
- Complete code implementation
- Usage examples
- Additional considerations

Generate a comprehensive implementation that follows {language} best practices.
```

#### Code Reviewer Prompt
```text
# prompts/code_reviewer/review.txt

You are a senior code reviewer with expertise in code quality, security, and performance.

## Code to Review
{code_snippet}

## Context
{context}

## Review Criteria
1. Code quality and readability
2. Security vulnerabilities
3. Performance considerations
4. Best practices adherence
5. Potential bugs or issues

## Response Format
### Summary
Brief overview of the code quality

### Issues Found
- **Severity**: High/Medium/Low
- **Issue**: Description
- **Recommendation**: How to fix

### Positive Aspects
What the code does well

### Overall Rating
Score: X/10
Recommendation: Approve/Request Changes/Reject
```

## Database Configuration

### ChromaDB Configuration

#### Basic Setup
```bash
# Local ChromaDB
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
CHROMADB_PERSIST_DIRECTORY=./data/chromadb

# Collection settings
CHROMADB_COLLECTION_NAME=deep_agent_knowledge
CHROMADB_DISTANCE_FUNCTION=cosine
```

#### Advanced ChromaDB Settings
```bash
# Performance tuning
CHROMADB_BATCH_SIZE=1000
CHROMADB_MAX_BATCH_SIZE=5000
CHROMADB_QUERY_TIMEOUT=30

# Memory settings
CHROMADB_MEMORY_LIMIT=4GB
CHROMADB_CACHE_SIZE=1000

# Authentication (if using ChromaDB server)
CHROMADB_AUTH_PROVIDER=basic
CHROMADB_AUTH_CREDENTIALS=username:password
CHROMADB_SSL_ENABLED=false
```

### Neo4j Configuration

#### Basic Setup
```bash
# Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

#### Advanced Neo4j Settings
```bash
# Connection pool
NEO4J_MAX_CONNECTION_LIFETIME=3600
NEO4J_MAX_CONNECTION_POOL_SIZE=50
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=60

# Query settings
NEO4J_QUERY_TIMEOUT=30
NEO4J_MAX_QUERY_SIZE=10000

# SSL/TLS
NEO4J_ENCRYPTED=false
NEO4J_TRUST_STRATEGY=TRUST_ALL_CERTIFICATES
```

## Agent Configuration

### Agent-Specific Settings

#### Analyst Agent
```bash
# Classification settings
ANALYST_CONFIDENCE_THRESHOLD=0.8
ANALYST_MAX_ROUTING_ATTEMPTS=3
ANALYST_ENABLE_MULTI_ROUTING=true

# Aggregation settings
ANALYST_AGGREGATION_METHOD=weighted
ANALYST_MAX_RESPONSES=5
```

#### Developer Agent
```bash
# Code generation
DEVELOPER_MAX_CODE_LENGTH=5000
DEVELOPER_ENABLE_SYNTAX_CHECK=true
DEVELOPER_DEFAULT_LANGUAGE=python

# Code quality
DEVELOPER_ENFORCE_STYLE=true
DEVELOPER_STYLE_GUIDE=pep8
```

#### Code Reviewer Agent
```bash
# Review settings
CODE_REVIEWER_SEVERITY_THRESHOLD=medium
CODE_REVIEWER_MAX_SUGGESTIONS=10
CODE_REVIEWER_ENABLE_AUTO_FIX=false

# Security scanning
CODE_REVIEWER_SECURITY_SCAN=true
CODE_REVIEWER_VULNERABILITY_DB=./data/vulnerabilities.json
```

#### Tester Agent
```bash
# Test generation
TESTER_DEFAULT_FRAMEWORK=pytest
TESTER_COVERAGE_THRESHOLD=80
TESTER_GENERATE_MOCKS=true

# Test types
TESTER_ENABLE_UNIT_TESTS=true
TESTER_ENABLE_INTEGRATION_TESTS=true
TESTER_ENABLE_E2E_TESTS=false
```

#### Architect Agent
```bash
# Design settings
ARCHITECT_DIAGRAM_FORMAT=mermaid
ARCHITECT_MAX_COMPONENTS=20
ARCHITECT_INCLUDE_PATTERNS=true

# Documentation
ARCHITECT_GENERATE_DOCS=true
ARCHITECT_DOC_FORMAT=markdown
```

## Performance Tuning

### Memory Optimization
```bash
# Memory limits
MAX_MEMORY_USAGE=8GB
ENABLE_MEMORY_MONITORING=true
MEMORY_WARNING_THRESHOLD=0.8

# Garbage collection
ENABLE_AGGRESSIVE_GC=false
GC_THRESHOLD=0.9
```

### Database Performance
```bash
# Connection pooling
DATABASE_POOL_SIZE=10
DATABASE_POOL_TIMEOUT=30
DATABASE_MAX_OVERFLOW=20

# Query optimization
ENABLE_QUERY_CACHING=true
QUERY_CACHE_SIZE=1000
QUERY_TIMEOUT=30
```

### RAG Performance
```bash
# Embedding caching
ENABLE_EMBEDDING_CACHE=true
EMBEDDING_CACHE_SIZE=10000
EMBEDDING_CACHE_TTL=3600

# Retrieval optimization
RAG_PARALLEL_RETRIEVAL=true
RAG_MAX_PARALLEL_QUERIES=5
RAG_RETRIEVAL_TIMEOUT=10
```

## Security Settings

### Input Validation
```bash
# Content filtering
ENABLE_INPUT_VALIDATION=true
MAX_INPUT_LENGTH=10000
BLOCKED_PATTERNS=password,secret,api_key

# Sanitization
SANITIZE_OUTPUTS=true
ESCAPE_HTML=true
REMOVE_SENSITIVE_DATA=true
```

### API Security
```bash
# Rate limiting
API_RATE_LIMIT=100
API_RATE_WINDOW=60
ENABLE_IP_WHITELIST=false

# Authentication
REQUIRE_API_KEY=false
API_KEY_HEADER=X-API-Key
```

## Development Settings

### Debug Configuration
```bash
# Development mode
DEBUG_MODE=false
DEVELOPMENT_MODE=false
ENABLE_HOT_RELOAD=false

# Testing
TEST_MODE=false
MOCK_EXTERNAL_APIS=false
USE_TEST_DATABASE=false
```

### Monitoring
```bash
# Performance monitoring
ENABLE_PERFORMANCE_MONITORING=true
METRICS_COLLECTION_INTERVAL=60
METRICS_RETENTION_DAYS=7

# Health checks
ENABLE_HEALTH_CHECKS=true
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=5
```

## Configuration Best Practices

### Environment-Specific Configuration

#### Development
```bash
# .env.development
DEBUG_MODE=true
LOG_LEVEL=DEBUG
ENABLE_HOT_RELOAD=true
MOCK_EXTERNAL_APIS=true
```

#### Production
```bash
# .env.production
DEBUG_MODE=false
LOG_LEVEL=INFO
ENABLE_PERFORMANCE_MONITORING=true
ENCRYPT_STORED_DATA=true
```

#### Testing
```bash
# .env.testing
TEST_MODE=true
USE_TEST_DATABASE=true
LOG_LEVEL=WARNING
DISABLE_EXTERNAL_APIS=true
```

### Configuration Validation

Create a configuration validation script:

```python
# scripts/validate_config.py
import os
from typing import Dict, Any

def validate_config() -> Dict[str, Any]:
    """Validate configuration settings."""
    errors = []
    warnings = []
    
    # Required settings
    required_vars = [
        'OPENAI_API_KEY',
        'PROMPT_DIR'
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required variable: {var}")
    
    # Validate numeric settings
    numeric_vars = {
        'RAG_CHUNK_SIZE': (100, 5000),
        'RAG_RETRIEVAL_K': (1, 20),
        'AGENT_TIMEOUT': (30, 3600)
    }
    
    for var, (min_val, max_val) in numeric_vars.items():
        value = os.getenv(var)
        if value:
            try:
                num_value = int(value)
                if not min_val <= num_value <= max_val:
                    warnings.append(f"{var}={num_value} outside recommended range [{min_val}, {max_val}]")
            except ValueError:
                errors.append(f"Invalid numeric value for {var}: {value}")
    
    return {
        'errors': errors,
        'warnings': warnings,
        'valid': len(errors) == 0
    }

if __name__ == "__main__":
    result = validate_config()
    if result['errors']:
        print("Configuration errors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['warnings']:
        print("Configuration warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    if result['valid']:
        print("Configuration is valid!")
```

### Configuration Templates

Create configuration templates for different scenarios:

```bash
# templates/minimal.env
OPENAI_API_KEY=your_key_here
PROMPT_DIR=./prompts
LOG_LEVEL=INFO

# templates/full-rag.env
OPENAI_API_KEY=your_key_here
PROMPT_DIR=./prompts
VECTOR_STORE_ENABLED=true
GRAPH_STORE_ENABLED=true
CHROMADB_HOST=localhost
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# templates/production.env
OPENAI_API_KEY=your_key_here
PROMPT_DIR=./prompts
LOG_LEVEL=INFO
ENABLE_PERFORMANCE_MONITORING=true
API_RATE_LIMIT=1000
ENCRYPT_STORED_DATA=true
```

This comprehensive configuration guide should help users understand and customize all aspects of the Deep Agent System according to their specific needs and deployment scenarios.