# Installation Guide

This guide provides detailed installation instructions for the Deep Agent System across different platforms and deployment scenarios.

## Table of Contents

- [System Requirements](#system-requirements)
- [Platform-Specific Installation](#platform-specific-installation)
- [Database Setup](#database-setup)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for installation and data
- **Network**: Internet connection for API calls and package downloads

### Recommended Requirements
- **Python**: 3.12 (latest stable)
- **Memory**: 16GB RAM for optimal performance
- **Storage**: 10GB free space for knowledge bases
- **CPU**: Multi-core processor for concurrent agent operations

## Platform-Specific Installation

### macOS

#### Prerequisites
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11+
brew install python@3.11

# Install Git
brew install git
```

#### Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
```

#### Install Deep Agent System
```bash
git clone https://github.com/your-org/deep-agent-system.git
cd deep-agent-system
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update package list
sudo apt update

# Install Python 3.11+
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install Git and curl
sudo apt install git curl build-essential

# Install system dependencies
sudo apt install libssl-dev libffi-dev
```

#### Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### Install Deep Agent System
```bash
git clone https://github.com/your-org/deep-agent-system.git
cd deep-agent-system
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Linux (CentOS/RHEL/Fedora)

#### Prerequisites
```bash
# For CentOS/RHEL
sudo yum update
sudo yum install python3.11 python3.11-devel git curl gcc

# For Fedora
sudo dnf update
sudo dnf install python3.11 python3.11-devel git curl gcc
```

#### Install UV and Deep Agent System
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

git clone https://github.com/your-org/deep-agent-system.git
cd deep-agent-system
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Windows

#### Prerequisites
1. **Install Python 3.11+**:
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Install Git**:
   - Download from [git-scm.com](https://git-scm.com/download/win)

3. **Install Visual Studio Build Tools** (for some dependencies):
   - Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### Install UV
```powershell
# Using PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

#### Install Deep Agent System
```powershell
git clone https://github.com/your-org/deep-agent-system.git
cd deep-agent-system
uv venv
.venv\Scripts\activate
uv pip install -e .
```

### Docker Installation

#### Using Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  deep-agent-system:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      - chromadb
      - neo4j

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chromadb_data:/chroma/chroma

  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data

volumes:
  chromadb_data:
  neo4j_data:
```

```bash
# Start services
docker-compose up -d

# Run commands
docker-compose exec deep-agent-system deep-agent "Your question here"
```

## Database Setup

### ChromaDB Setup

#### Local Installation
```bash
# ChromaDB is included in dependencies
# For standalone server:
pip install chromadb[server]
chroma run --host localhost --port 8000
```

#### Docker Installation
```bash
docker run -p 8000:8000 chromadb/chroma:latest
```

#### Configuration
```bash
# In .env
VECTOR_STORE_ENABLED=true
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
CHROMADB_PERSIST_DIRECTORY=./data/chromadb
```

### Neo4j Setup

#### Local Installation

**macOS:**
```bash
brew install neo4j
neo4j start
```

**Linux:**
```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j

# Start Neo4j
sudo systemctl start neo4j
```

**Windows:**
1. Download Neo4j Desktop from [neo4j.com](https://neo4j.com/download/)
2. Install and create a new database
3. Start the database

#### Docker Installation
```bash
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

#### Configuration
```bash
# In .env
GRAPH_STORE_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## Configuration

### Environment Variables

Create your `.env` file:
```bash
cp .env.template .env
```

### Required Configuration
```bash
# OpenAI API Key (required)
OPENAI_API_KEY=sk-your-api-key-here

# Prompt directory
PROMPT_DIR=./prompts
```

### Optional Configuration
```bash
# Database settings
VECTOR_STORE_ENABLED=true
GRAPH_STORE_ENABLED=true

# Performance settings
RAG_CHUNK_SIZE=1000
RAG_RETRIEVAL_K=5
AGENT_TIMEOUT=300

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/deep_agent_system.log
```

### Prompt Setup

The system comes with default prompts, but you can customize them:

```bash
# Prompts are in the prompts/ directory
ls prompts/
# analyst/  architect/  code_reviewer/  developer/  tester/

# Edit prompts as needed
nano prompts/developer/implementation.txt
```

## Verification

### Basic Verification
```bash
# Check installation
deep-agent version

# Check system status
deep-agent status

# Test with a simple question
deep-agent "Hello, can you help me?"
```

### Advanced Verification
```bash
# Test with verbose output
deep-agent --verbose "Test the system functionality"

# Test interactive mode
deep-agent interactive
# Type: /help
# Type: /status
# Type: /quit
```

### Database Verification
```bash
# Check ChromaDB connection
curl http://localhost:8000/api/v1/heartbeat

# Check Neo4j connection (if installed)
# Open browser: http://localhost:7474
# Or use cypher-shell: cypher-shell -u neo4j -p password
```

## Troubleshooting

### Common Installation Issues

#### Python Version Issues
```bash
# Check Python version
python --version
python3 --version

# If wrong version, install correct Python
# macOS: brew install python@3.11
# Ubuntu: sudo apt install python3.11
```

#### UV Installation Issues
```bash
# If UV not found after installation
source ~/.bashrc  # or ~/.zshrc

# Manual installation
pip install uv

# Verify installation
uv --version
```

#### Permission Issues (Linux/macOS)
```bash
# If permission denied during installation
sudo chown -R $USER:$USER ~/.local
sudo chown -R $USER:$USER ~/.cache

# Or install in user directory
uv pip install --user -e .
```

#### Windows-Specific Issues
```powershell
# If execution policy prevents script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# If build tools missing
# Install Visual Studio Build Tools from Microsoft
```

### Database Connection Issues

#### ChromaDB Issues
```bash
# Check if ChromaDB is running
curl http://localhost:8000/api/v1/heartbeat

# Start ChromaDB server
chroma run --host localhost --port 8000

# Check firewall settings
sudo ufw allow 8000  # Linux
```

#### Neo4j Issues
```bash
# Check Neo4j status
sudo systemctl status neo4j  # Linux
brew services list | grep neo4j  # macOS

# Check Neo4j logs
sudo journalctl -u neo4j  # Linux
tail -f /usr/local/var/log/neo4j/neo4j.log  # macOS

# Reset Neo4j password
neo4j-admin set-initial-password newpassword
```

### Configuration Issues

#### API Key Issues
```bash
# Verify API key is set
grep OPENAI_API_KEY .env

# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### Prompt Loading Issues
```bash
# Check prompt directory exists
ls -la prompts/

# Check prompt files
find prompts/ -name "*.txt" -exec wc -l {} +

# Verify prompt directory in config
grep PROMPT_DIR .env
```

### Performance Issues

#### Memory Issues
```bash
# Monitor memory usage
top -p $(pgrep -f deep-agent)

# Reduce memory usage in .env
RAG_CHUNK_SIZE=500
RAG_RETRIEVAL_K=3
WORKFLOW_MAX_STEPS=25
```

#### Slow Response Issues
```bash
# Enable performance logging
LOG_LEVEL=DEBUG
ENABLE_DEBUG_LOGGING=true

# Optimize database queries
RAG_SIMILARITY_THRESHOLD=0.8
AGENT_TIMEOUT=60
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**:
   ```bash
   tail -f logs/deep_agent_system.log
   ```

2. **Run with debug mode**:
   ```bash
   deep-agent --debug "test question"
   ```

3. **Check system status**:
   ```bash
   deep-agent status --verbose
   ```

4. **Report issues**:
   - Create an issue on GitHub with:
     - Your operating system and Python version
     - Complete error messages
     - Steps to reproduce the problem
     - Your configuration (without sensitive data)

## Next Steps

After successful installation:

1. **Configure your knowledge base**: See [RAG Setup Guide](RAG_SETUP.md)
2. **Customize prompts**: See [Prompt Customization Guide](PROMPTS.md)
3. **Explore examples**: See [Usage Examples](EXAMPLES.md)
4. **Set up monitoring**: See [Monitoring Guide](MONITORING.md)