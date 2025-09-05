"""End-to-end tests for CLI functionality."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from deep_agent_system.cli.main import app
from deep_agent_system.models.messages import Response


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_env_setup(self, temp_dir):
        """Setup mock environment for CLI tests."""
        # Create test config file
        config_file = temp_dir / ".env"
        config_file.write_text("""
OPENAI_API_KEY=test-api-key
NEO4J_PASSWORD=test-password
PROMPT_DIR=./prompts
VECTOR_STORE_ENABLED=false
GRAPH_STORE_ENABLED=false
""")
        
        # Create prompt directory structure
        prompt_dir = temp_dir / "prompts"
        prompt_dir.mkdir()
        for agent_type in ["analyst", "architect", "developer", "code_reviewer", "tester"]:
            agent_dir = prompt_dir / agent_type
            agent_dir.mkdir()
            (agent_dir / "default.txt").write_text(f"Mock prompt for {agent_type}")
        
        return config_file
    
    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Deep Agent System" in result.stdout
        assert "Version:" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    def test_ask_command_success(self, mock_init_system, runner, mock_env_setup):
        """Test successful ask command."""
        # Mock system and response
        mock_system = MagicMock()
        mock_response = Response(
            message_id="test",
            agent_id="analyst_agent",
            content="This is a test response from the analyst agent.",
            confidence_score=0.95
        )
        mock_system.process_message = AsyncMock(return_value=mock_response)
        mock_init_system.return_value = mock_system
        
        # Run command
        result = runner.invoke(app, [
            "ask", 
            "What is machine learning?",
            "--config", str(mock_env_setup)
        ])
        
        assert result.exit_code == 0
        assert "Question" in result.stdout
        assert "What is machine learning?" in result.stdout
        assert "Response from analyst_agent" in result.stdout
        assert "test response" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    def test_ask_command_json_output(self, mock_init_system, runner, mock_env_setup):
        """Test ask command with JSON output format."""
        # Mock system and response
        mock_system = MagicMock()
        mock_response = Response(
            message_id="test",
            agent_id="analyst_agent",
            content="JSON response test",
            confidence_score=0.85
        )
        mock_system.process_message = AsyncMock(return_value=mock_response)
        mock_init_system.return_value = mock_system
        
        # Run command with JSON format
        result = runner.invoke(app, [
            "ask", 
            "Test question",
            "--config", str(mock_env_setup),
            "--format", "json"
        ])
        
        assert result.exit_code == 0
        assert '"agent_id": "analyst_agent"' in result.stdout
        assert '"content": "JSON response test"' in result.stdout
        assert '"confidence_score": 0.85' in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    def test_ask_command_plain_output(self, mock_init_system, runner, mock_env_setup):
        """Test ask command with plain output format."""
        # Mock system and response
        mock_system = MagicMock()
        mock_response = Response(
            message_id="test",
            agent_id="analyst_agent",
            content="Plain text response",
            confidence_score=0.90
        )
        mock_system.process_message = AsyncMock(return_value=mock_response)
        mock_init_system.return_value = mock_system
        
        # Run command with plain format
        result = runner.invoke(app, [
            "ask", 
            "Test question",
            "--config", str(mock_env_setup),
            "--format", "plain"
        ])
        
        assert result.exit_code == 0
        assert "Plain text response" in result.stdout
        # Should not contain rich formatting
        assert "Response from" not in result.stdout
        assert "Confidence:" not in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    def test_ask_command_system_error(self, mock_init_system, runner, mock_env_setup):
        """Test ask command with system error."""
        # Mock system initialization failure
        mock_init_system.side_effect = Exception("System initialization failed")
        
        # Run command
        result = runner.invoke(app, [
            "ask", 
            "Test question",
            "--config", str(mock_env_setup)
        ])
        
        assert result.exit_code == 1
        assert "System Error" in result.stdout or "Unexpected error" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    def test_status_command(self, mock_init_system, runner, mock_env_setup):
        """Test status command."""
        # Mock system status
        mock_system = MagicMock()
        mock_system.get_system_status.return_value = {
            "is_initialized": True,
            "is_running": True,
            "agents_count": 5,
            "components": {
                "config_manager": True,
                "prompt_manager": True,
                "rag_manager": False,
                "communication_manager": True,
                "workflow_executor": True,
            }
        }
        mock_init_system.return_value = mock_system
        
        # Run command
        result = runner.invoke(app, [
            "status",
            "--config", str(mock_env_setup)
        ])
        
        assert result.exit_code == 0
        assert "System Status" in result.stdout
        assert "Initialized: ✓" in result.stdout
        assert "Running: ✓" in result.stdout
        assert "Agents: 5" in result.stdout
    
    def test_ask_command_missing_question(self, runner):
        """Test ask command without question argument."""
        result = runner.invoke(app, ["ask"])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "Error" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    def test_verbose_logging(self, mock_init_system, runner, mock_env_setup):
        """Test verbose logging option."""
        # Mock system
        mock_system = MagicMock()
        mock_response = Response(
            message_id="test",
            agent_id="analyst_agent",
            content="Verbose test response",
            confidence_score=0.88
        )
        mock_system.process_message = AsyncMock(return_value=mock_response)
        mock_init_system.return_value = mock_system
        
        # Run command with verbose flag
        result = runner.invoke(app, [
            "ask", 
            "Test question",
            "--config", str(mock_env_setup),
            "--verbose"
        ])
        
        assert result.exit_code == 0
        # Verbose output should still work
        assert "Verbose test response" in result.stdout


class TestCLIInteractiveMode:
    """Tests for interactive CLI mode."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_env_setup(self, temp_dir):
        """Setup mock environment for CLI tests."""
        config_file = temp_dir / ".env"
        config_file.write_text("""
OPENAI_API_KEY=test-api-key
NEO4J_PASSWORD=test-password
PROMPT_DIR=./prompts
VECTOR_STORE_ENABLED=false
GRAPH_STORE_ENABLED=false
""")
        
        prompt_dir = temp_dir / "prompts"
        prompt_dir.mkdir()
        for agent_type in ["analyst"]:
            agent_dir = prompt_dir / agent_type
            agent_dir.mkdir()
            (agent_dir / "default.txt").write_text(f"Mock prompt for {agent_type}")
        
        return config_file
    
    @patch('deep_agent_system.cli.main.initialize_system')
    @patch('deep_agent_system.cli.main.Prompt.ask')
    def test_interactive_help_command(self, mock_prompt, mock_init_system, runner, mock_env_setup):
        """Test interactive help command."""
        # Mock system
        mock_system = MagicMock()
        mock_init_system.return_value = mock_system
        
        # Mock user input sequence: /help, then /quit
        mock_prompt.side_effect = ["/help", "/quit"]
        
        # Run interactive mode
        result = runner.invoke(app, [
            "interactive",
            "--config", str(mock_env_setup)
        ])
        
        assert result.exit_code == 0
        assert "Interactive Mode" in result.stdout
        assert "Available Commands" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    @patch('deep_agent_system.cli.main.Prompt.ask')
    def test_interactive_status_command(self, mock_prompt, mock_init_system, runner, mock_env_setup):
        """Test interactive status command."""
        # Mock system
        mock_system = MagicMock()
        mock_system.get_system_status.return_value = {
            "is_initialized": True,
            "is_running": True,
            "agents_count": 3,
            "components": {
                "config_manager": True,
                "prompt_manager": True,
            }
        }
        mock_init_system.return_value = mock_system
        
        # Mock user input: /status, then /quit
        mock_prompt.side_effect = ["/status", "/quit"]
        
        # Run interactive mode
        result = runner.invoke(app, [
            "interactive",
            "--config", str(mock_env_setup)
        ])
        
        assert result.exit_code == 0
        assert "System Status" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    @patch('deep_agent_system.cli.main.Prompt.ask')
    def test_interactive_agents_command(self, mock_prompt, mock_init_system, runner, mock_env_setup):
        """Test interactive agents command."""
        # Mock system
        mock_system = MagicMock()
        mock_system.list_agents.return_value = [
            {"agent_id": "analyst_agent", "agent_type": "analyst", "is_active": True},
            {"agent_id": "architect_agent", "agent_type": "architect", "is_active": False},
        ]
        mock_init_system.return_value = mock_system
        
        # Mock user input: /agents, then /quit
        mock_prompt.side_effect = ["/agents", "/quit"]
        
        # Run interactive mode
        result = runner.invoke(app, [
            "interactive",
            "--config", str(mock_env_setup)
        ])
        
        assert result.exit_code == 0
        assert "Registered Agents" in result.stdout
        assert "analyst_agent" in result.stdout
        assert "architect_agent" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    @patch('deep_agent_system.cli.main.Prompt.ask')
    def test_interactive_question_processing(self, mock_prompt, mock_init_system, runner, mock_env_setup):
        """Test processing questions in interactive mode."""
        # Mock system
        mock_system = MagicMock()
        mock_response = Response(
            message_id="test",
            agent_id="analyst_agent",
            content="Interactive response",
            confidence_score=0.92
        )
        mock_system.process_message = AsyncMock(return_value=mock_response)
        mock_init_system.return_value = mock_system
        
        # Mock user input: question, then /quit
        mock_prompt.side_effect = ["What is AI?", "/quit"]
        
        # Run interactive mode
        result = runner.invoke(app, [
            "interactive",
            "--config", str(mock_env_setup)
        ])
        
        assert result.exit_code == 0
        assert "Interactive response" in result.stdout
        assert "analyst_agent" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    @patch('deep_agent_system.cli.main.Prompt.ask')
    def test_interactive_reload_command(self, mock_prompt, mock_init_system, runner, mock_env_setup):
        """Test interactive reload command."""
        # Mock system
        mock_system = MagicMock()
        mock_system.reload_prompts = AsyncMock(return_value=True)
        mock_system.reload_configuration = AsyncMock(return_value=True)
        mock_init_system.return_value = mock_system
        
        # Mock user input: /reload, then /quit
        mock_prompt.side_effect = ["/reload", "/quit"]
        
        # Run interactive mode
        result = runner.invoke(app, [
            "interactive",
            "--config", str(mock_env_setup)
        ])
        
        assert result.exit_code == 0
        assert "reloaded successfully" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    @patch('deep_agent_system.cli.main.Prompt.ask')
    def test_interactive_unknown_command(self, mock_prompt, mock_init_system, runner, mock_env_setup):
        """Test interactive mode with unknown command."""
        # Mock system
        mock_system = MagicMock()
        mock_init_system.return_value = mock_system
        
        # Mock user input: unknown command, then /quit
        mock_prompt.side_effect = ["/unknown", "/quit"]
        
        # Run interactive mode
        result = runner.invoke(app, [
            "interactive",
            "--config", str(mock_env_setup)
        ])
        
        assert result.exit_code == 0
        assert "Unknown command" in result.stdout


class TestCLIErrorHandling:
    """Tests for CLI error handling."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_invalid_config_file(self, runner):
        """Test handling of invalid config file."""
        result = runner.invoke(app, [
            "ask",
            "Test question",
            "--config", "/nonexistent/config.env"
        ])
        
        # Should handle the error gracefully
        assert result.exit_code != 0
    
    @patch('deep_agent_system.cli.main.initialize_system')
    def test_system_initialization_failure(self, mock_init_system, runner):
        """Test handling of system initialization failure."""
        from deep_agent_system.system import AgentSystemError
        
        # Mock initialization failure
        mock_init_system.side_effect = AgentSystemError("Initialization failed")
        
        result = runner.invoke(app, [
            "ask",
            "Test question"
        ])
        
        assert result.exit_code == 1
        assert "System Error" in result.stdout
    
    @patch('deep_agent_system.cli.main.initialize_system')
    def test_message_processing_failure(self, mock_init_system, runner):
        """Test handling of message processing failure."""
        # Mock system that fails during message processing
        mock_system = MagicMock()
        mock_system.process_message = AsyncMock(side_effect=Exception("Processing failed"))
        mock_init_system.return_value = mock_system
        
        result = runner.invoke(app, [
            "ask",
            "Test question"
        ])
        
        assert result.exit_code == 1
        assert "error" in result.stdout.lower()


class TestCLICommandLineArguments:
    """Tests for command line argument parsing."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_help_option(self, runner):
        """Test --help option."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Deep Agent System" in result.stdout
        assert "sophisticated multi-agent framework" in result.stdout
    
    def test_ask_help(self, runner):
        """Test help for ask command."""
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "Ask a question" in result.stdout
        assert "--config" in result.stdout
        assert "--verbose" in result.stdout
        assert "--debug" in result.stdout
        assert "--format" in result.stdout
    
    def test_interactive_help(self, runner):
        """Test help for interactive command."""
        result = runner.invoke(app, ["interactive", "--help"])
        assert result.exit_code == 0
        assert "interactive session" in result.stdout
        assert "--config" in result.stdout
        assert "--verbose" in result.stdout
    
    def test_status_help(self, runner):
        """Test help for status command."""
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "system status" in result.stdout
        assert "--config" in result.stdout
    
    def test_version_help(self, runner):
        """Test help for version command."""
        result = runner.invoke(app, ["version", "--help"])
        assert result.exit_code == 0
        assert "version information" in result.stdout