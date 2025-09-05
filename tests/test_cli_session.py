"""Integration tests for CLI session management."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from deep_agent_system.cli.manager import CLIManager
from deep_agent_system.cli.session import CommandProcessor, InputValidator, InteractiveSession
from deep_agent_system.models.messages import Response
from deep_agent_system.system import AgentSystem


class TestInputValidator:
    """Test cases for InputValidator class."""
    
    def test_validate_question_valid(self):
        """Test validation of valid questions."""
        valid_questions = [
            "What is Python?",
            "How do I implement a REST API?",
            "Can you help me debug this code?",
        ]
        
        for question in valid_questions:
            is_valid, error = InputValidator.validate_question(question)
            assert is_valid is True
            assert error is None
    
    def test_validate_question_empty(self):
        """Test validation of empty questions."""
        empty_questions = ["", "   ", "\n\t  "]
        
        for question in empty_questions:
            is_valid, error = InputValidator.validate_question(question)
            assert is_valid is False
            assert "empty" in error.lower()
    
    def test_validate_question_too_short(self):
        """Test validation of too short questions."""
        short_questions = ["Hi", "?", "OK"]
        
        for question in short_questions:
            is_valid, error = InputValidator.validate_question(question)
            assert is_valid is False
            assert "short" in error.lower()
    
    def test_validate_question_too_long(self):
        """Test validation of too long questions."""
        long_question = "x" * 10001  # Over the limit
        
        is_valid, error = InputValidator.validate_question(long_question)
        assert is_valid is False
        assert "long" in error.lower()
    
    def test_validate_command_valid(self):
        """Test validation of valid commands."""
        valid_commands = ["/help", "/status", "/quit", "/reload prompts"]
        
        for command in valid_commands:
            is_valid, error = InputValidator.validate_command(command)
            assert is_valid is True
            assert error is None
    
    def test_validate_command_invalid(self):
        """Test validation of invalid commands."""
        invalid_commands = ["", "   ", "help", "status", "not a command"]
        
        for command in invalid_commands:
            is_valid, error = InputValidator.validate_command(command)
            assert is_valid is False
            assert error is not None


class TestCommandProcessor:
    """Test cases for CommandProcessor class."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock interactive session."""
        session = Mock()
        session.cli_manager = Mock(spec=CLIManager)
        session.agent_system = Mock(spec=AgentSystem)
        session.conversation_history = Mock()
        session.debug_mode = False
        return session
    
    @pytest.fixture
    def command_processor(self, mock_session):
        """Create CommandProcessor instance."""
        return CommandProcessor(mock_session)
    
    def test_register_command(self, command_processor):
        """Test command registration."""
        def test_handler(args):
            return True
        
        command_processor.register_command(
            "test", 
            test_handler, 
            "Test command",
            aliases=["t"]
        )
        
        assert "test" in command_processor.commands
        assert "t" in command_processor.commands
        assert command_processor.commands["test"] == test_handler
        assert command_processor.commands["t"] == test_handler
        assert command_processor.command_help["test"] == "Test command"
    
    def test_default_commands_registered(self, command_processor):
        """Test that default commands are registered."""
        expected_commands = ["help", "status", "agents", "history", "clear", "reload", "debug", "quit"]
        
        for cmd in expected_commands:
            assert cmd in command_processor.commands
            assert cmd in command_processor.command_help
    
    @pytest.mark.asyncio
    async def test_process_command_valid(self, command_processor, mock_session):
        """Test processing valid commands."""
        # Test help command
        result = await command_processor.process_command("/help")
        assert result is True
        mock_session.cli_manager.display_help.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_command_invalid(self, command_processor, mock_session):
        """Test processing invalid commands."""
        result = await command_processor.process_command("/nonexistent")
        assert result is True
        mock_session.cli_manager.display_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_command_with_args(self, command_processor, mock_session):
        """Test processing commands with arguments."""
        result = await command_processor.process_command("/help status")
        assert result is True
        # Should call display_info for specific command help
        mock_session.cli_manager.display_info.assert_called_once()
    
    def test_get_command_completions(self, command_processor):
        """Test command completion functionality."""
        # Test partial matches
        completions = command_processor.get_command_completions("/he")
        assert "/help" in completions
        
        # Test no matches
        completions = command_processor.get_command_completions("/xyz")
        assert len(completions) == 0
        
        # Test empty input
        completions = command_processor.get_command_completions("/")
        assert len(completions) > 0  # Should return all commands
    
    @pytest.mark.asyncio
    async def test_handle_status_command(self, command_processor, mock_session):
        """Test status command handler."""
        mock_session.agent_system.get_system_status.return_value = {
            "is_initialized": True,
            "is_running": True,
            "agents_count": 3
        }
        
        result = await command_processor._handle_status([])
        assert result is True
        mock_session.agent_system.get_system_status.assert_called_once()
        mock_session.cli_manager.display_system_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_agents_command(self, command_processor, mock_session):
        """Test agents command handler."""
        mock_agents = [
            {"agent_id": "analyst", "agent_type": "analyst", "is_active": True}
        ]
        mock_session.agent_system.list_agents.return_value = mock_agents
        
        result = await command_processor._handle_agents([])
        assert result is True
        mock_session.agent_system.list_agents.assert_called_once()
        mock_session.cli_manager.display_agents_table.assert_called_once_with(mock_agents)
    
    def test_handle_quit_command(self, command_processor):
        """Test quit command handler."""
        result = command_processor._handle_quit([])
        assert result is False  # Should signal to exit
    
    def test_handle_clear_command(self, command_processor, mock_session):
        """Test clear command handler."""
        result = command_processor._handle_clear([])
        assert result is True
        mock_session.cli_manager.clear_screen.assert_called_once()
    
    def test_handle_debug_command(self, command_processor, mock_session):
        """Test debug command handler."""
        # Initially debug is False
        assert mock_session.debug_mode is False
        
        result = command_processor._handle_debug([])
        assert result is True
        assert mock_session.debug_mode is True
        
        # Toggle again
        result = command_processor._handle_debug([])
        assert result is True
        assert mock_session.debug_mode is False


class TestInteractiveSession:
    """Test cases for InteractiveSession class."""
    
    @pytest.fixture
    def mock_agent_system(self):
        """Create a mock agent system."""
        system = Mock(spec=AgentSystem)
        system.get_system_status.return_value = {
            "is_initialized": True,
            "is_running": True,
            "agents_count": 3
        }
        system.list_agents.return_value = []
        system.reload_prompts = AsyncMock(return_value=True)
        system.reload_configuration = AsyncMock(return_value=True)
        return system
    
    @pytest.fixture
    def mock_cli_manager(self):
        """Create a mock CLI manager."""
        return Mock(spec=CLIManager)
    
    @pytest.fixture
    def interactive_session(self, mock_agent_system, mock_cli_manager):
        """Create InteractiveSession instance."""
        return InteractiveSession(mock_agent_system, mock_cli_manager)
    
    def test_init(self, interactive_session, mock_agent_system, mock_cli_manager):
        """Test session initialization."""
        assert interactive_session.agent_system == mock_agent_system
        assert interactive_session.cli_manager == mock_cli_manager
        assert interactive_session.is_running is False
        assert interactive_session.debug_mode is False
        assert interactive_session.questions_asked == 0
        assert interactive_session.commands_executed == 0
    
    def test_add_custom_command(self, interactive_session):
        """Test adding custom commands."""
        def custom_handler(args):
            return True
        
        interactive_session.add_custom_command(
            "custom", 
            custom_handler, 
            "Custom command"
        )
        
        assert "custom" in interactive_session.command_processor.commands
    
    def test_get_session_stats(self, interactive_session):
        """Test getting session statistics."""
        stats = interactive_session.get_session_stats()
        
        expected_keys = [
            "is_running", "debug_mode", "questions_asked", 
            "commands_executed", "messages_in_history", 
            "responses_in_history", "session_start_time"
        ]
        
        for key in expected_keys:
            assert key in stats
    
    @pytest.mark.asyncio
    async def test_process_question(self, interactive_session, mock_agent_system, mock_cli_manager):
        """Test question processing."""
        # Mock response
        mock_response = Response(
            message_id="test",
            agent_id="test_agent",
            content="Test response"
        )
        mock_agent_system.process_message = AsyncMock(return_value=mock_response)
        
        # Mock progress context
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress.add_task = Mock(return_value="task_id")
        mock_progress.update = Mock()
        mock_cli_manager.create_progress_context.return_value = mock_progress
        
        question = "What is Python?"
        await interactive_session._process_question(question)
        
        # Verify interactions
        mock_cli_manager.display_question.assert_called_once_with(question)
        mock_agent_system.process_message.assert_called_once_with(question)
        mock_cli_manager.display_response.assert_called_once()
        
        # Check that question was added to history
        assert len(interactive_session.conversation_history.messages) == 1
        assert len(interactive_session.conversation_history.responses) == 1
    
    @pytest.mark.asyncio
    async def test_stop_session(self, interactive_session, mock_cli_manager):
        """Test session stopping."""
        interactive_session.is_running = True
        interactive_session.questions_asked = 5
        interactive_session.commands_executed = 3
        
        await interactive_session.stop()
        
        assert interactive_session.is_running is False
        mock_cli_manager.display_success.assert_called_once()


class TestInteractiveSessionIntegration:
    """Integration tests for InteractiveSession with real components."""
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test complete session lifecycle."""
        # Create mocks
        mock_system = Mock(spec=AgentSystem)
        mock_system.get_system_status.return_value = {"is_initialized": True}
        
        # Create session with real CLI manager
        from io import StringIO
        from rich.console import Console
        
        string_io = StringIO()
        console = Console(file=string_io, width=80)
        cli_manager = CLIManager(console=console)
        
        session = InteractiveSession(mock_system, cli_manager)
        
        # Test session stats
        stats = session.get_session_stats()
        assert stats["is_running"] is False
        assert stats["questions_asked"] == 0
        
        # Test custom command addition
        def test_command(args):
            return True
        
        session.add_custom_command("test", test_command, "Test command")
        assert "test" in session.command_processor.commands
    
    @pytest.mark.asyncio
    async def test_command_processing_integration(self):
        """Test command processing with real components."""
        mock_system = Mock(spec=AgentSystem)
        mock_system.get_system_status.return_value = {
            "is_initialized": True,
            "is_running": True,
            "agents_count": 2,
            "components": {"config_manager": True}
        }
        
        from io import StringIO
        from rich.console import Console
        
        string_io = StringIO()
        console = Console(file=string_io, width=80)
        cli_manager = CLIManager(console=console)
        
        session = InteractiveSession(mock_system, cli_manager)
        
        # Test status command
        result = await session.command_processor.process_command("/status")
        assert result is True
        
        output = string_io.getvalue()
        assert "System Status" in output
    
    def test_input_validation_integration(self):
        """Test input validation with various inputs."""
        validator = InputValidator()
        
        # Test realistic questions
        test_cases = [
            ("How do I implement authentication in Flask?", True),
            ("What's the difference between REST and GraphQL?", True),
            ("", False),
            ("Hi", False),
            ("x" * 10001, False),
        ]
        
        for question, expected_valid in test_cases:
            is_valid, error = validator.validate_question(question)
            assert is_valid == expected_valid
            if not expected_valid:
                assert error is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling in session components."""
        # Create system that raises errors
        mock_system = Mock(spec=AgentSystem)
        mock_system.get_system_status.side_effect = Exception("Test error")
        
        from io import StringIO
        from rich.console import Console
        
        string_io = StringIO()
        console = Console(file=string_io, width=80)
        cli_manager = CLIManager(console=console)
        
        session = InteractiveSession(mock_system, cli_manager)
        
        # Test that error is handled gracefully
        result = await session.command_processor.process_command("/status")
        assert result is True  # Should continue despite error
        
        output = string_io.getvalue()
        assert "Error" in output or "error" in output