#!/usr/bin/env python3
"""
Demonstration of the Rich CLI interface capabilities.

This script shows the various formatting and display features
of the CLIManager without requiring a full system setup.
"""

from io import StringIO
from rich.console import Console

from deep_agent_system.cli.manager import CLIManager
from deep_agent_system.models.messages import Response
from datetime import datetime


def main():
    """Demonstrate CLI manager capabilities."""
    
    # Create CLI manager with real console for demonstration
    cli_manager = CLIManager()
    
    print("=" * 60)
    print("Deep Agent System CLI Manager Demonstration")
    print("=" * 60)
    
    # 1. Welcome banner
    print("\n1. Welcome Banner:")
    cli_manager.display_welcome_banner()
    
    # 2. Question display
    print("\n2. Question Display:")
    cli_manager.display_question(
        "How do I implement a REST API in Python using Flask?",
        "developer_user"
    )
    
    # 3. Response with code formatting
    print("\n3. Response with Code Formatting:")
    response_with_code = Response(
        message_id="demo-123",
        agent_id="developer_agent",
        content="""Here's how to create a simple REST API with Flask:

First, install Flask:
```bash
pip install flask
```

Then create your API:
```python
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify([
        {'id': 1, 'name': 'Alice'},
        {'id': 2, 'name': 'Bob'}
    ])

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    # Process the data here
    return jsonify({'status': 'created'}), 201

if __name__ == '__main__':
    app.run(debug=True)
```

You can test it with `curl` or any HTTP client:
```bash
curl http://localhost:5000/api/users
```

This creates a basic REST API with GET and POST endpoints.""",
        confidence_score=0.95,
        context_used=["flask_documentation", "rest_api_patterns"],
        metadata={"language": "python", "framework": "flask"},
        timestamp=datetime.now()
    )
    
    cli_manager.display_response(response_with_code, show_metadata=True)
    
    # 4. System status
    print("\n4. System Status Display:")
    sample_status = {
        "is_initialized": True,
        "is_running": True,
        "agents_count": 5,
        "components": {
            "config_manager": True,
            "prompt_manager": True,
            "rag_manager": True,
            "communication_manager": True,
            "workflow_manager": False,
        },
        "agents": [
            {"agent_id": "analyst", "agent_type": "analyst", "is_active": True},
            {"agent_id": "architect", "agent_type": "architect", "is_active": True},
            {"agent_id": "developer", "agent_type": "developer", "is_active": True},
            {"agent_id": "code_reviewer", "agent_type": "code_reviewer", "is_active": False},
            {"agent_id": "tester", "agent_type": "tester", "is_active": True},
        ]
    }
    
    cli_manager.display_system_status(sample_status)
    
    # 5. Help display
    print("\n5. Help System:")
    sample_commands = {
        "/help": "Show available commands and their descriptions",
        "/status": "Show current system status and agent information",
        "/agents": "List all registered agents and their status",
        "/history": "Show conversation history",
        "/clear": "Clear the screen",
        "/reload": "Reload system prompts and configuration",
        "/debug": "Toggle debug mode on/off",
        "/quit": "Exit the interactive session"
    }
    
    cli_manager.display_help(sample_commands)
    
    # 6. Different message types
    print("\n6. Different Message Types:")
    
    cli_manager.display_success("System initialized successfully!")
    cli_manager.display_info("Loading configuration from .env file...")
    cli_manager.display_warning("Some optional components are not available.")
    cli_manager.display_error(
        "Failed to connect to database", 
        "Connection timeout after 30 seconds. Please check your network connection."
    )
    
    # 7. Code formatting examples
    print("\n7. Code Formatting Examples:")
    
    python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Generate first 10 Fibonacci numbers
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
"""
    
    syntax_obj = cli_manager.format_code_block(
        python_code.strip(), 
        language="python", 
        line_numbers=True
    )
    
    cli_manager.print("\n[bold cyan]Python Code Example:[/bold cyan]")
    cli_manager.print(syntax_obj)
    
    # 8. Tree view example
    print("\n8. Tree View Example:")
    
    sample_data = {
        "system": {
            "version": "1.0.0",
            "status": "running",
            "uptime": "2 hours 15 minutes"
        },
        "agents": {
            "active": ["analyst", "developer", "tester"],
            "inactive": ["code_reviewer"],
            "total_count": 4
        },
        "configuration": {
            "debug_mode": False,
            "log_level": "INFO",
            "max_workers": 4,
            "features": ["rag", "workflows", "monitoring"]
        }
    }
    
    tree = cli_manager.create_tree_view("System Information", sample_data)
    cli_manager.print(tree)
    
    print("\n" + "=" * 60)
    print("CLI Manager Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()