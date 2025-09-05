# Usage Examples

This guide provides comprehensive examples of how to use the Deep Agent System for various development tasks and scenarios.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Agent-Specific Examples](#agent-specific-examples)
- [Workflow Examples](#workflow-examples)
- [RAG Integration Examples](#rag-integration-examples)
- [CLI Command Examples](#cli-command-examples)
- [Interactive Session Examples](#interactive-session-examples)
- [Advanced Use Cases](#advanced-use-cases)

## Basic Usage

### Simple Questions

```bash
# Ask a general programming question
deep-agent "How do I create a REST API in Python?"

# Get architecture advice
deep-agent "What's the best database for a chat application?"

# Request code review
deep-agent "Review this Python function for improvements"

# Generate tests
deep-agent "Create unit tests for a user authentication system"
```

### With Output Formatting

```bash
# JSON output for programmatic use
deep-agent --format json "Explain dependency injection"

# Plain text output
deep-agent --format plain "List Python web frameworks"

# Rich formatted output (default)
deep-agent --format rich "Design a microservices architecture"
```

### Verbose and Debug Modes

```bash
# Verbose output shows routing and processing details
deep-agent --verbose "Implement a caching layer"

# Debug mode shows detailed system information
deep-agent --debug "Troubleshoot this error message"
```

## Agent-Specific Examples

### Analyst Agent (Automatic Routing)

The Analyst agent automatically routes questions to appropriate specialists:

```bash
# Architecture questions → Architect Agent
deep-agent "Design a scalable e-commerce platform"

# Implementation questions → Developer Agent  
deep-agent "Implement user authentication with JWT"

# Code quality questions → Code Reviewer Agent
deep-agent "Review this code for security vulnerabilities"

# Testing questions → Tester Agent
deep-agent "Create integration tests for this API"
```

### Architect Agent Examples

```bash
# System design questions
deep-agent "Design a real-time chat application architecture"

# Technology stack recommendations
deep-agent "Choose the best tech stack for a data analytics platform"

# Scalability planning
deep-agent "How to scale a web application to handle 1M users?"

# Pattern recommendations
deep-agent "When should I use the Observer pattern?"

# Database design
deep-agent "Design a database schema for a social media platform"
```

**Example Output:**
```
┌─ Response from Architect Agent ─────────────────────────────────────┐
│ # Real-time Chat Application Architecture                           │
│                                                                     │
│ ## High-Level Architecture                                          │
│                                                                     │
│ ```mermaid                                                          │
│ graph TB                                                            │
│     Client[Web/Mobile Clients]                                      │
│     LB[Load Balancer]                                              │
│     API[API Gateway]                                               │
│     Auth[Auth Service]                                             │
│     Chat[Chat Service]                                             │
│     WS[WebSocket Manager]                                          │
│     Queue[Message Queue]                                           │
│     DB[(Database)]                                                 │
│     Cache[(Redis Cache)]                                           │
│                                                                     │
│     Client --> LB                                                  │
│     LB --> API                                                     │
│     API --> Auth                                                   │
│     API --> Chat                                                   │
│     Chat --> WS                                                    │
│     Chat --> Queue                                                 │
│     Chat --> DB                                                    │
│     Chat --> Cache                                                 │
│ ```                                                                 │
│                                                                     │
│ ## Key Components                                                   │
│                                                                     │
│ 1. **WebSocket Manager**: Handles real-time connections            │
│ 2. **Message Queue**: Ensures reliable message delivery            │
│ 3. **Redis Cache**: Stores active sessions and recent messages     │
│ 4. **Database**: Persistent storage for chat history               │
│                                                                     │
│ ## Technology Recommendations                                       │
│                                                                     │
│ - **Backend**: Node.js with Socket.io or Python with FastAPI       │
│ - **Database**: PostgreSQL for persistence, Redis for caching      │
│ - **Message Queue**: Redis Pub/Sub or Apache Kafka                 │
│ - **Frontend**: React with Socket.io-client                        │
└─────────────────────────────────────────────────────── Confidence: 0.92 ┘
```

### Developer Agent Examples

```bash
# Code implementation
deep-agent "Implement a REST API endpoint for user registration"

# Framework-specific questions
deep-agent "Create a React component for file upload with progress"

# Algorithm implementation
deep-agent "Implement a binary search algorithm in Python"

# Database operations
deep-agent "Write SQL queries for a blog application"

# Error handling
deep-agent "Add proper error handling to this async function"
```

**Example Output:**
```
┌─ Response from Developer Agent ─────────────────────────────────────┐
│ # User Registration API Endpoint                                    │
│                                                                     │
│ Here's a complete implementation using FastAPI:                     │
│                                                                     │
│ ```python                                                           │
│ from fastapi import FastAPI, HTTPException, Depends                 │
│ from pydantic import BaseModel, EmailStr                           │
│ from passlib.context import CryptContext                           │
│ from sqlalchemy.orm import Session                                  │
│ import re                                                           │
│                                                                     │
│ app = FastAPI()                                                     │
│ pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")   │
│                                                                     │
│ class UserRegistration(BaseModel):                                  │
│     username: str                                                   │
│     email: EmailStr                                                 │
│     password: str                                                   │
│     full_name: str                                                  │
│                                                                     │
│ class UserResponse(BaseModel):                                      │
│     id: int                                                         │
│     username: str                                                   │
│     email: str                                                      │
│     full_name: str                                                  │
│     is_active: bool                                                 │
│                                                                     │
│ def validate_password(password: str) -> bool:                       │
│     """Validate password strength."""                               │
│     if len(password) < 8:                                          │
│         return False                                                │
│     if not re.search(r"[A-Z]", password):                         │
│         return False                                                │
│     if not re.search(r"[a-z]", password):                         │
│         return False                                                │
│     if not re.search(r"\d", password):                            │
│         return False                                                │
│     return True                                                     │
│                                                                     │
│ @app.post("/register", response_model=UserResponse)                 │
│ async def register_user(                                            │
│     user_data: UserRegistration,                                    │
│     db: Session = Depends(get_db)                                   │
│ ):                                                                  │
│     """Register a new user."""                                      │
│     # Validate password                                             │
│     if not validate_password(user_data.password):                   │
│         raise HTTPException(                                        │
│             status_code=400,                                        │
│             detail="Password must be at least 8 characters with uppercase, lowercase, and number"                                                                     │
│         )                                                           │
│                                                                     │
│     # Check if user exists                                          │
│     existing_user = db.query(User).filter(                         │
│         (User.username == user_data.username) |                     │
│         (User.email == user_data.email)                             │
│     ).first()                                                       │
│                                                                     │
│     if existing_user:                                               │
│         raise HTTPException(                                        │
│             status_code=400,                                        │
│             detail="Username or email already registered"           │
│         )                                                           │
│                                                                     │
│     # Hash password                                                 │
│     hashed_password = pwd_context.hash(user_data.password)          │
│                                                                     │
│     # Create user                                                   │
│     db_user = User(                                                 │
│         username=user_data.username,                                │
│         email=user_data.email,                                      │
│         hashed_password=hashed_password,                            │
│         full_name=user_data.full_name,                              │
│         is_active=True                                              │
│     )                                                               │
│                                                                     │
│     db.add(db_user)                                                 │
│     db.commit()                                                     │
│     db.refresh(db_user)                                             │
│                                                                     │
│     return UserResponse(                                            │
│         id=db_user.id,                                              │
│         username=db_user.username,                                  │
│         email=db_user.email,                                        │
│         full_name=db_user.full_name,                                │
│         is_active=db_user.is_active                                 │
│     )                                                               │
│ ```                                                                 │
│                                                                     │
│ ## Usage Example                                                    │
│                                                                     │
│ ```bash                                                             │
│ curl -X POST "http://localhost:8000/register" \                     │
│      -H "Content-Type: application/json" \                         │
│      -d '{                                                          │
│        "username": "johndoe",                                       │
│        "email": "john@example.com",                                 │
│        "password": "SecurePass123",                                 │
│        "full_name": "John Doe"                                      │
│      }'                                                             │
│ ```                                                                 │
│                                                                     │
│ ## Key Features                                                     │
│                                                                     │
│ - Password strength validation                                      │
│ - Duplicate user checking                                           │
│ - Secure password hashing                                           │
│ - Proper error handling                                             │
│ - Type validation with Pydantic                                     │
└─────────────────────────────────────────────────────── Confidence: 0.95 ┘
```

### Code Reviewer Agent Examples

```bash
# Security review
deep-agent "Review this authentication function for security issues"

# Performance review
deep-agent "Analyze this database query for performance problems"

# Code quality review
deep-agent "Review this class design for maintainability"

# Best practices review
deep-agent "Check if this React component follows best practices"
```

### Tester Agent Examples

```bash
# Unit test generation
deep-agent "Create unit tests for this user service class"

# Integration test creation
deep-agent "Write integration tests for this REST API"

# Test strategy planning
deep-agent "Design a testing strategy for a microservices application"

# Mock creation
deep-agent "Create mocks for external API dependencies"
```

## Workflow Examples

### Full Development Workflow

```bash
# 1. Architecture Design
deep-agent "Design a task management application with real-time updates"

# 2. Implementation Planning
deep-agent "Break down the task management app into implementable components"

# 3. Core Implementation
deep-agent "Implement the Task model with CRUD operations in Python"

# 4. Code Review
deep-agent "Review the Task model implementation for improvements"

# 5. Test Creation
deep-agent "Create comprehensive tests for the Task model"

# 6. API Development
deep-agent "Implement REST API endpoints for task management"

# 7. Frontend Integration
deep-agent "Create React components for task management UI"
```

### Code Quality Improvement Workflow

```bash
# 1. Initial Review
deep-agent "Review this legacy code for modernization opportunities"

# 2. Refactoring Suggestions
deep-agent "Suggest refactoring strategies for this monolithic function"

# 3. Implementation
deep-agent "Refactor this code to use dependency injection"

# 4. Testing
deep-agent "Create tests for the refactored code"

# 5. Performance Analysis
deep-agent "Analyze performance improvements after refactoring"
```

## RAG Integration Examples

### Using Knowledge Base Context

When RAG is enabled, agents automatically retrieve relevant context:

```bash
# The system will search your knowledge base for relevant information
deep-agent "How do we handle authentication in our application?"

# Context from your codebase and documentation will be included
deep-agent "What's our standard error handling pattern?"

# API documentation will be retrieved for implementation questions
deep-agent "Implement a new endpoint following our API standards"
```

### Knowledge Base Ingestion

```bash
# Ingest documentation
deep-agent ingest --path ./docs --type documentation

# Ingest code examples
deep-agent ingest --path ./examples --type code_examples

# Ingest API specifications
deep-agent ingest --file api-spec.yaml --type api_spec
```

## CLI Command Examples

### Status and Information Commands

```bash
# Check system status
deep-agent status

# List available agents
deep-agent agents

# Show version information
deep-agent version

# Validate configuration
deep-agent validate-config
```

### Configuration Commands

```bash
# Use custom configuration file
deep-agent --config ./custom.env "Your question"

# Test configuration
deep-agent test-config --config ./production.env

# Show current configuration
deep-agent show-config
```

### Maintenance Commands

```bash
# Clear cache
deep-agent clear-cache

# Reload prompts
deep-agent reload-prompts

# Update knowledge base
deep-agent update-kb --path ./new-docs
```

## Interactive Session Examples

### Starting Interactive Mode

```bash
deep-agent interactive
```

### Interactive Commands

```
Welcome to Deep Agent System Interactive Mode

Question: /help
┌─ Help ──────────────────────────────────────────────────────────────┐
│ Available Commands:                                                 │
│                                                                     │
│ /help    - Show this help message                                   │
│ /status  - Show system status                                       │
│ /agents  - List available agents                                    │
│ /reload  - Reload prompts and configuration                         │
│ /quit    - Exit the session                                         │
└─────────────────────────────────────────────────────────────────────┘

Question: /status
┌─ System Status ─────────────────────────────────────────────────────┐
│ Component                Status                                      │
│ ─────────────────────────────────────────────────────────────────── │
│ Initialized              ✓                                          │
│ Running                  ✓                                          │
│ Agents                   5                                          │
│ Vector Store             ✓                                          │
│ Graph Store              ✓                                          │
│ Prompt Manager           ✓                                          │
└─────────────────────────────────────────────────────────────────────┘

Question: Design a user authentication system

┌─ Response from Architect Agent ─────────────────────────────────────┐
│ # User Authentication System Design                                 │
│                                                                     │
│ ## Architecture Overview                                            │
│ [Detailed architecture response...]                                 │
└─────────────────────────────────────────────────────── Confidence: 0.91 ┘

Question: Now implement the login endpoint

┌─ Response from Developer Agent ─────────────────────────────────────┐
│ # Login Endpoint Implementation                                     │
│                                                                     │
│ Based on the authentication system design, here's the login        │
│ endpoint implementation:                                            │
│ [Code implementation...]                                            │
└─────────────────────────────────────────────────────── Confidence: 0.94 ┘

Question: /quit
Goodbye!
```

### Conversation Context

The system maintains conversation context across questions:

```
Question: Design a blog application

Question: Now implement the blog post model
# This will reference the previous architecture design

Question: Add comments functionality to the blog
# This will build on both previous responses

Question: Create tests for the comment system
# This will test the comment functionality in context
```

## Advanced Use Cases

### Multi-Agent Collaboration

```bash
# Complex questions that require multiple agents
deep-agent "Design and implement a secure file upload system with virus scanning"

# This might involve:
# 1. Architect Agent: System design
# 2. Developer Agent: Implementation
# 3. Code Reviewer Agent: Security review
# 4. Tester Agent: Security testing strategy
```

### Domain-Specific Workflows

#### Web Development Workflow
```bash
# Frontend development
deep-agent "Create a responsive navigation component in React"
deep-agent "Add accessibility features to the navigation"
deep-agent "Optimize the component for performance"
deep-agent "Create tests for the navigation component"

# Backend development
deep-agent "Design a RESTful API for user management"
deep-agent "Implement rate limiting for the API"
deep-agent "Add comprehensive error handling"
deep-agent "Create API documentation"
```

#### Data Science Workflow
```bash
# Data analysis
deep-agent "Design a data pipeline for real-time analytics"
deep-agent "Implement data validation and cleaning"
deep-agent "Create machine learning model training pipeline"
deep-agent "Add model monitoring and evaluation"
```

#### DevOps Workflow
```bash
# Infrastructure
deep-agent "Design a CI/CD pipeline for microservices"
deep-agent "Implement Docker containerization strategy"
deep-agent "Create Kubernetes deployment configurations"
deep-agent "Add monitoring and alerting setup"
```

### Custom Prompt Integration

```bash
# Using custom prompts for specific domains
export PROMPT_DIR=./custom-prompts/fintech
deep-agent "Implement payment processing with PCI compliance"

# Domain-specific knowledge base
export CHROMADB_COLLECTION_NAME=fintech_knowledge
deep-agent "What are the regulatory requirements for this feature?"
```

### Batch Processing

```bash
# Process multiple questions from a file
deep-agent batch --input questions.txt --output responses.json

# questions.txt content:
# Design a user authentication system
# Implement JWT token validation
# Create user registration tests
# Review authentication security
```

### Integration with External Tools

```bash
# Generate code and save to file
deep-agent "Implement user service class" --output user_service.py

# Review existing code file
deep-agent "Review this file for improvements" --input existing_code.py

# Generate documentation
deep-agent "Create API documentation" --format markdown --output api_docs.md
```

### Performance Optimization Examples

```bash
# Database optimization
deep-agent "Optimize this slow database query"
deep-agent "Design database indexes for better performance"

# Code optimization
deep-agent "Optimize this algorithm for better time complexity"
deep-agent "Reduce memory usage in this data processing function"

# System optimization
deep-agent "Optimize this microservice for high throughput"
deep-agent "Design caching strategy for this API"
```

### Error Handling and Debugging

```bash
# Debug assistance
deep-agent "Debug this error: AttributeError in user authentication"
deep-agent "Explain this stack trace and suggest fixes"

# Error handling implementation
deep-agent "Add comprehensive error handling to this API"
deep-agent "Implement retry logic for external API calls"
```

These examples demonstrate the versatility and power of the Deep Agent System across various development scenarios and use cases. The system's ability to maintain context, route questions appropriately, and provide specialized expertise makes it a valuable tool for developers at all levels.