"""Unit tests for CLIManager with Rich text support."""

import pytest
from datetime import datetime
from io import StringIO
from unittest.mock import Mock, patch

from rich.console import Console
from rich.syntax import Syntax
from rich.tree import Tree

from deep_agent_system.cli.manager import CLIManager
from deep_agent_system.config.models import AgentType
from deep_agent_system.models.messages import Response


class TestCLIManager:
    """Test cases for CLIManager class."""
    
    @pytest.fixture
    def mock_console(self):
        """Create a mock console for testing."""
        return Mock(spec=Console)
    
    @pytest.fixture
    def cli_manager(self, mock_console):
        """Create CLIManager instance with mock console."""
        return CLIManager(console=mock_console)
    
    @pytest.fixture
    def sample_response(self):
        """Create a sample response for testing."""
        return Response(
            message_id="test-msg-123",
            agent_id="analyst_agent",
            content="This is a test response with `inline code` and:\n\n```python\nprint('hello world')\n```",
            confidence_score=0.95,
            context_used=["context1", "context2"],
            workflow_id="workflow-123",
            metadata={"test_key": "test_value"},
            timestamp=datetime(2024, 1, 1, 12, 0, 0)
        )
    
    def test_init_with_console(self, mock_console):
        """Test CLIManager initialization with provided console."""
        manager = CLIManager(console=mock_console)
        assert manager.console == mock_console
        assert manager._progress_tasks == {}
    
    def test_init_without_console(self):
        """Test CLIManager initialization without console creates new one."""
        manager = CLIManager()
        assert manager.console is not None
        assert isinstance(manager.console, Console)
    
    def test_display_welcome_banner(self, cli_manager, mock_console):
        """Test welcome banner display."""
        cli_manager.display_welcome_banner()
        
        # Verify console.print was called
        mock_console.print.assert_called_once()
        
        # Check that a Panel was passed
        call_args = mock_console.print.call_args[0]
        assert len(call_args) == 1
        # The panel should contain the banner content
    
    def test_display_question(self, cli_manager, mock_console):
        """Test question display formatting."""
        question = "What is the meaning of life?"
        sender_id = "user123"
        
        cli_manager.display_question(question, sender_id)
        
        # Verify console.print was called
        mock_console.print.assert_called_once()
    
    def test_display_response_basic(self, cli_manager, mock_console, sample_response):
        """Test basic response display."""
        cli_manager.display_response(sample_response)
        
        # Should call print at least once for the main response
        assert mock_console.print.call_count >= 1
    
    def test_display_response_with_metadata(self, cli_manager, mock_console, sample_response):
        """Test response display with metadata."""
        cli_manager.display_response(sample_response, show_metadata=True)
        
        # Should call print multiple times (response + metadata)
        assert mock_console.print.call_count >= 2
    
    def test_display_response_without_code_formatting(self, cli_manager, mock_console, sample_response):
        """Test response display without code formatting."""
        cli_manager.display_response(sample_response, format_code=False)
        
        # Verify console.print was called
        mock_console.print.assert_called()
    
    def test_format_response_content_no_code(self, cli_manager):
        """Test content formatting without code blocks."""
        content = "This is plain text with no code."
        result = cli_manager._format_response_content(content, format_code=True)
        
        assert isinstance(result, str)
        assert "plain text" in result
    
    def test_format_response_content_with_inline_code(self, cli_manager):
        """Test content formatting with inline code."""
        content = "Use the `print()` function to output text."
        result = cli_manager._format_response_content(content, format_code=True)
        
        assert isinstance(result, str)
        assert "[bold cyan]print()[/bold cyan]" in result
    
    def test_format_response_content_with_code_blocks(self, cli_manager):
        """Test content formatting with code blocks."""
        content = "Here's some Python code:\n\n```python\nprint('hello')\n```\n\nThat's it!"
        result = cli_manager._format_response_content(content, format_code=True)
        
        # Should return a Group with multiple parts
        from rich.console import Group
        assert isinstance(result, Group)
    
    def test_format_response_content_no_formatting(self, cli_manager):
        """Test content formatting disabled."""
        content = "Code: `print()` and ```python\nprint('test')\n```"
        result = cli_manager._format_response_content(content, format_code=False)
        
        assert isinstance(result, str)
        assert result == content  # Should be unchanged
    
    def test_display_system_status(self, cli_manager, mock_console):
        """Test system status display."""
        status = {
            "is_initialized": True,
            "is_running": True,
            "agents_count": 3,
            "components": {
                "config_manager": True,
                "prompt_manager": True,
                "rag_manager": False,
            },
            "agents": [
                {"agent_id": "analyst", "agent_type": "analyst", "is_active": True},
                {"agent_id": "developer", "agent_type": "developer", "is_active": False},
            ]
        }
        
        cli_manager.display_system_status(status)
        
        # Should call print multiple times (status table + agents table)
        assert mock_console.print.call_count >= 2
    
    def test_display_agents_table_with_agents(self, cli_manager, mock_console):
        """Test agents table display with agents."""
        agents = [
            {"agent_id": "analyst", "agent_type": "analyst", "is_active": True},
            {"agent_id": "developer", "agent_type": "developer", "is_active": False},
        ]
        
        cli_manager.display_agents_table(agents)
        
        mock_console.print.assert_called_once()
    
    def test_display_agents_table_empty(self, cli_manager, mock_console):
        """Test agents table display with no agents."""
        cli_manager.display_agents_table([])
        
        mock_console.print.assert_called_once()
        # Should display "No agents registered" message
    
    def test_display_help(self, cli_manager, mock_console):
        """Test help display."""
        commands = {
            "/help": "Show this help",
            "/status": "Show system status",
            "/quit": "Exit the session",
        }
        
        cli_manager.display_help(commands)
        
        mock_console.print.assert_called_once()
    
    def test_display_error(self, cli_manager, mock_console):
        """Test error message display."""
        error = "Something went wrong"
        details = "Detailed error information"
        
        cli_manager.display_error(error, details)
        
        mock_console.print.assert_called_once()
    
    def test_display_error_no_details(self, cli_manager, mock_console):
        """Test error message display without details."""
        error = "Something went wrong"
        
        cli_manager.display_error(error)
        
        mock_console.print.assert_called_once()
    
    def test_display_warning(self, cli_manager, mock_console):
        """Test warning message display."""
        warning = "This is a warning"
        
        cli_manager.display_warning(warning)
        
        mock_console.print.assert_called_once()
    
    def test_display_info(self, cli_manager, mock_console):
        """Test info message display."""
        info = "This is information"
        
        cli_manager.display_info(info)
        
        mock_console.print.assert_called_once()
    
    def test_display_success(self, cli_manager, mock_console):
        """Test success message display."""
        message = "Operation completed successfully"
        
        cli_manager.display_success(message)
        
        mock_console.print.assert_called_once()
    
    def test_create_progress_context(self, cli_manager):
        """Test progress context creation."""
        # Use real console for this test since Progress needs console methods
        from rich.console import Console
        real_console = Console()
        cli_manager.console = real_console
        
        progress = cli_manager.create_progress_context("Testing...")
        
        from rich.progress import Progress
        assert isinstance(progress, Progress)
    
    def test_format_code_block(self, cli_manager):
        """Test code block formatting."""
        code = "print('hello world')"
        syntax = cli_manager.format_code_block(code, language="python")
        
        assert isinstance(syntax, Syntax)
    
    def test_format_code_block_with_options(self, cli_manager):
        """Test code block formatting with options."""
        code = "console.log('hello');"
        syntax = cli_manager.format_code_block(
            code, 
            language="javascript",
            line_numbers=True,
            theme="github-dark"
        )
        
        assert isinstance(syntax, Syntax)
    
    def test_create_tree_view(self, cli_manager):
        """Test tree view creation."""
        data = {
            "system": {
                "status": "running",
                "agents": ["analyst", "developer"],
            },
            "config": {
                "debug": True,
                "timeout": 30,
            }
        }
        
        tree = cli_manager.create_tree_view("System Info", data)
        
        assert isinstance(tree, Tree)
    
    def test_create_tree_view_with_lists(self, cli_manager):
        """Test tree view creation with list data."""
        data = {
            "agents": ["agent1", "agent2", "agent3", "agent4", "agent5", "agent6"],
            "status": "active"
        }
        
        tree = cli_manager.create_tree_view("Agents", data)
        
        assert isinstance(tree, Tree)
    
    def test_get_agent_color_analyst(self, cli_manager):
        """Test agent color detection for analyst."""
        color = cli_manager._get_agent_color("analyst_agent")
        assert color == "blue"
    
    def test_get_agent_color_architect(self, cli_manager):
        """Test agent color detection for architect."""
        color = cli_manager._get_agent_color("architect_agent")
        assert color == "green"
    
    def test_get_agent_color_developer(self, cli_manager):
        """Test agent color detection for developer."""
        color = cli_manager._get_agent_color("developer_agent")
        assert color == "yellow"
    
    def test_get_agent_color_reviewer(self, cli_manager):
        """Test agent color detection for code reviewer."""
        color = cli_manager._get_agent_color("code_reviewer_agent")
        assert color == "red"
    
    def test_get_agent_color_tester(self, cli_manager):
        """Test agent color detection for tester."""
        color = cli_manager._get_agent_color("tester_agent")
        assert color == "magenta"
    
    def test_get_agent_color_unknown(self, cli_manager):
        """Test agent color detection for unknown agent."""
        color = cli_manager._get_agent_color("unknown_agent")
        assert color == "blue"  # Default primary color
    
    def test_clear_screen(self, cli_manager, mock_console):
        """Test screen clearing."""
        cli_manager.clear_screen()
        
        mock_console.clear.assert_called_once()
    
    def test_print(self, cli_manager, mock_console):
        """Test print method."""
        cli_manager.print("Hello", "World", style="bold")
        
        mock_console.print.assert_called_once_with("Hello", "World", style="bold")
    
    @patch('rich.prompt.Prompt.ask')
    def test_input(self, mock_ask, cli_manager):
        """Test input method."""
        mock_ask.return_value = "user input"
        
        result = cli_manager.input("Enter something: ")
        
        assert result == "user input"
        mock_ask.assert_called_once()
    
    @patch('rich.prompt.Confirm.ask')
    def test_confirm(self, mock_confirm, cli_manager):
        """Test confirm method."""
        mock_confirm.return_value = True
        
        result = cli_manager.confirm("Are you sure?", default=False)
        
        assert result is True
        mock_confirm.assert_called_once()
    
    def test_display_response_metadata_empty(self, cli_manager, mock_console):
        """Test metadata display with empty metadata."""
        response = Response(
            message_id="test",
            agent_id="test_agent",
            content="Test content",
            metadata={},
            context_used=[]
        )
        
        # Should not call print if no metadata
        cli_manager._display_response_metadata(response)
        
        mock_console.print.assert_not_called()
    
    def test_display_response_metadata_with_data(self, cli_manager, mock_console, sample_response):
        """Test metadata display with actual data."""
        cli_manager._display_response_metadata(sample_response)
        
        # Should call print once for the metadata table
        mock_console.print.assert_called_once()


class TestCLIManagerIntegration:
    """Integration tests for CLIManager with real Rich components."""
    
    def test_real_console_integration(self):
        """Test CLIManager with real Rich console."""
        # Use StringIO to capture output
        string_io = StringIO()
        console = Console(file=string_io, width=80)
        manager = CLIManager(console=console)
        
        # Test basic functionality
        manager.display_info("Test message")
        
        output = string_io.getvalue()
        assert "Test message" in output
        assert "Info" in output
    
    def test_code_formatting_integration(self):
        """Test code formatting with real Rich components."""
        string_io = StringIO()
        console = Console(file=string_io, width=80)
        manager = CLIManager(console=console)
        
        # Test code block formatting
        code = "def hello():\n    print('world')"
        syntax = manager.format_code_block(code, language="python")
        
        # Should create valid Syntax object
        assert isinstance(syntax, Syntax)
        assert syntax.code == code
    
    def test_response_formatting_integration(self):
        """Test response formatting with real Rich components."""
        string_io = StringIO()
        console = Console(file=string_io, width=80)
        manager = CLIManager(console=console)
        
        response = Response(
            message_id="test",
            agent_id="developer_agent",
            content="Here's some code:\n\n```python\nprint('hello')\n```",
            confidence_score=0.9
        )
        
        manager.display_response(response)
        
        output = string_io.getvalue()
        assert "developer_agent" in output
        assert "Confidence: 0.90" in output