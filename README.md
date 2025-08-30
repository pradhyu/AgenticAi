# Deep Agent System

A sophisticated multi-agent system using LangChain and LangGraph that intelligently routes questions to specialized agents based on content analysis.

## Features

- **Multi-Agent Architecture**: Analyst, Architect, Developer, Code Reviewer, and Tester agents
- **Intelligent Routing**: Automatic question classification and routing to appropriate specialists
- **RAG Integration**: Both vector-based (ChromaDB) and graph-based (Neo4j) retrieval
- **Dynamic Workflows**: LangGraph-powered multi-agent collaboration
- **Rich CLI Interface**: Beautiful command-line interface with syntax highlighting
- **Configurable Prompts**: Externalized prompts for easy customization

## Quick Start

### Prerequisites

- Python 3.11 or higher
- UV package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deep-agent-system
```

2. Create and activate virtual environment with UV:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. Set up environment variables:
```bash
cp .env.template .env
# Edit .env with your API keys and configuration
```

### Configuration

Copy `.env.template` to `.env` and configure the following:

- **OpenAI API Key**: Required for LLM functionality
- **ChromaDB Settings**: Vector database configuration
- **Neo4j Credentials**: Graph database configuration
- **Other Settings**: Adjust as needed for your environment

### Usage

```bash
# Ask a question to the agent system
deep-agent "How do I implement a REST API in Python?"

# Enable verbose output
deep-agent --verbose "Design a microservices architecture"
```

## Architecture

The system consists of:

- **Analyst Agent**: Routes questions to appropriate specialists
- **Architect Agent**: Handles solution design and architecture questions
- **Developer Agent**: Manages code implementation and programming tasks
- **Code Reviewer Agent**: Provides code analysis and improvement suggestions
- **Tester Agent**: Creates tests and validation strategies

## Development Status

🚧 **Under Development** - This project is currently being implemented following a spec-driven development approach.

### Implementation Progress

- [x] Project structure and environment setup
- [ ] Core configuration and prompt management
- [ ] RAG integration layer
- [ ] Base agent framework
- [ ] Specialized agent implementations
- [ ] Inter-agent communication system
- [ ] LangGraph workflow engine
- [ ] Rich CLI interface
- [ ] Main application orchestration
- [ ] Error handling and logging
- [ ] Comprehensive test suite
- [ ] Documentation and examples

## Contributing

This project follows a structured development approach using specifications and task-driven implementation. See the `.kiro/specs/` directory for detailed requirements, design, and implementation plans.

## License

MIT License - see LICENSE file for details.