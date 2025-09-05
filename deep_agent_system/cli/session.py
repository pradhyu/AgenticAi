"""Interactive CLI session management for the Deep Agent System."""

import asyncio
import logging
import shlex
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.prompt import Confirm, Prompt

from deep_agent_system.cli.manager import CLIManager
from deep_agent_system.models.messages import ConversationHistory, Message, MessageType
from deep_agent_system.system import AgentSystem, AgentSystemError


logger = logging.getLogger(__name__)


class CommandProcessor:
    """Processor for CLI commands with validation and help system."""
    
    def __init__(self, session: 'InteractiveSession'):
        """Initialize command processor.
        
        Args:
            session: Reference to the interactive session
        """
        self.session = session
        self.commands: Dict[str, Callable] = {}
        self.command_help: Dict[str, str] = {}
        self._register_default_commands()
    
    def register_command(
        self, 
        name: str, 
        handler: Callable, 
        help_text: str,
        aliases: Optional[List[str]] = None
    ) -> None:
        """Register a command with the processor.
        
        Args:
            name: Command name (without leading slash)
            handler: Command handler function
            help_text: Help text for the command
            aliases: Optional list of command aliases
        """
        self.commands[name] = handler
        self.command_help[name] = help_text
        
        # Register aliases
        if aliases:
            for alias in aliases:
                self.commands[alias] = handler
    
    def _register_default_commands(self) -> None:
        """Register default system commands."""
        self.register_command(
            "help", 
            self._handle_help, 
            "Show available commands and their descriptions",
            aliases=["h", "?"]
        )
        
        self.register_command(
            "status", 
            self._handle_status, 
            "Show current system status and agent information"
        )
        
        self.register_command(
            "agents", 
            self._handle_agents, 
            "List all registered agents and their status"
        )
        
        self.register_command(
            "history", 
            self._handle_history, 
            "Show conversation history",
            aliases=["hist"]
        )
        
        self.register_command(
            "clear", 
            self._handle_clear, 
            "Clear the screen",
            aliases=["cls"]
        )
        
        self.register_command(
            "reload", 
            self._handle_reload, 
            "Reload system prompts and configuration"
        )
        
        self.register_command(
            "debug", 
            self._handle_debug, 
            "Toggle debug mode on/off"
        )
        
        self.register_command(
            "quit", 
            self._handle_quit, 
            "Exit the interactive session",
            aliases=["exit", "q"]
        )
    
    async def process_command(self, command_line: str) -> bool:
        """Process a command line input.
        
        Args:
            command_line: Full command line including command and arguments
            
        Returns:
            True if session should continue, False if should exit
        """
        try:
            # Parse command and arguments
            parts = shlex.split(command_line.strip())
            if not parts:
                return True
            
            command = parts[0].lstrip('/')  # Remove leading slash if present
            args = parts[1:] if len(parts) > 1 else []
            
            # Find and execute command
            if command in self.commands:
                handler = self.commands[command]
                return await self._execute_command(handler, args)
            else:
                self.session.cli_manager.display_error(
                    f"Unknown command: /{command}",
                    "Type /help to see available commands."
                )
                return True
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            self.session.cli_manager.display_error(
                "Command processing error",
                str(e)
            )
            return True
    
    async def _execute_command(self, handler: Callable, args: List[str]) -> bool:
        """Execute a command handler with error handling.
        
        Args:
            handler: Command handler function
            args: Command arguments
            
        Returns:
            True if session should continue, False if should exit
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(args)
            else:
                return handler(args)
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            self.session.cli_manager.display_error(
                "Command execution error",
                str(e)
            )
            return True
    
    def get_command_completions(self, partial_command: str) -> List[str]:
        """Get command completions for partial input.
        
        Args:
            partial_command: Partial command string
            
        Returns:
            List of matching command names
        """
        partial = partial_command.lstrip('/').lower()
        matches = []
        
        for command in self.commands.keys():
            if command.startswith(partial):
                matches.append(f"/{command}")
        
        return sorted(matches)
    
    # Default command handlers
    
    def _handle_help(self, args: List[str]) -> bool:
        """Handle help command."""
        if args and args[0] in self.command_help:
            # Show help for specific command
            command = args[0]
            help_text = self.command_help[command]
            self.session.cli_manager.display_info(f"/{command}: {help_text}")
        else:
            # Show all commands
            commands = {f"/{cmd}": help_text for cmd, help_text in self.command_help.items()}
            self.session.cli_manager.display_help(commands)
        return True
    
    async def _handle_status(self, args: List[str]) -> bool:
        """Handle status command."""
        try:
            status = self.session.agent_system.get_system_status()
            self.session.cli_manager.display_system_status(status)
        except Exception as e:
            self.session.cli_manager.display_error("Failed to get system status", str(e))
        return True
    
    async def _handle_agents(self, args: List[str]) -> bool:
        """Handle agents command."""
        try:
            agents = self.session.agent_system.list_agents()
            self.session.cli_manager.display_agents_table(agents)
        except Exception as e:
            self.session.cli_manager.display_error("Failed to list agents", str(e))
        return True
    
    def _handle_history(self, args: List[str]) -> bool:
        """Handle history command."""
        try:
            count = 10  # Default
            if args and args[0].isdigit():
                count = int(args[0])
            
            history = self.session.conversation_history
            recent_messages = history.get_recent_messages(count)
            recent_responses = history.get_recent_responses(count)
            
            if not recent_messages and not recent_responses:
                self.session.cli_manager.display_info("No conversation history available.")
                return True
            
            # Display conversation history
            self.session.cli_manager.print("\n[bold cyan]Conversation History[/bold cyan]\n")
            
            # Interleave messages and responses by timestamp
            all_items = []
            for msg in recent_messages:
                all_items.append(('message', msg))
            for resp in recent_responses:
                all_items.append(('response', resp))
            
            # Sort by timestamp
            all_items.sort(key=lambda x: x[1].timestamp)
            
            for item_type, item in all_items[-count:]:
                if item_type == 'message':
                    self.session.cli_manager.display_question(item.content, item.sender_id)
                else:
                    self.session.cli_manager.display_response(item, show_metadata=False)
                    
        except Exception as e:
            self.session.cli_manager.display_error("Failed to show history", str(e))
        return True
    
    def _handle_clear(self, args: List[str]) -> bool:
        """Handle clear command."""
        self.session.cli_manager.clear_screen()
        return True
    
    async def _handle_reload(self, args: List[str]) -> bool:
        """Handle reload command."""
        try:
            with self.session.cli_manager.create_progress_context("Reloading system...") as progress:
                task = progress.add_task("Reloading prompts and configuration...", total=None)
                
                # Reload prompts
                prompts_ok = await self.session.agent_system.reload_prompts()
                
                # Reload configuration
                config_ok = await self.session.agent_system.reload_configuration()
                
                progress.update(task, description="Reload complete!", completed=True)
            
            if prompts_ok and config_ok:
                self.session.cli_manager.display_success("System reloaded successfully.")
            else:
                self.session.cli_manager.display_warning("Some components failed to reload.")
                
        except Exception as e:
            self.session.cli_manager.display_error("Failed to reload system", str(e))
        return True
    
    def _handle_debug(self, args: List[str]) -> bool:
        """Handle debug command."""
        self.session.debug_mode = not self.session.debug_mode
        status = "enabled" if self.session.debug_mode else "disabled"
        self.session.cli_manager.display_info(f"Debug mode {status}.")
        return True
    
    def _handle_quit(self, args: List[str]) -> bool:
        """Handle quit command."""
        return False  # Signal to exit session


class InputValidator:
    """Validator for user input with various validation rules."""
    
    @staticmethod
    def validate_question(question: str) -> Tuple[bool, Optional[str]]:
        """Validate a user question.
        
        Args:
            question: User question string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not question or not question.strip():
            return False, "Question cannot be empty."
        
        if len(question.strip()) < 3:
            return False, "Question is too short. Please provide more details."
        
        if len(question) > 10000:
            return False, "Question is too long. Please keep it under 10,000 characters."
        
        return True, None
    
    @staticmethod
    def validate_command(command: str) -> Tuple[bool, Optional[str]]:
        """Validate a command input.
        
        Args:
            command: Command string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command or not command.strip():
            return False, "Command cannot be empty."
        
        if not command.startswith('/'):
            return False, "Commands must start with '/'."
        
        return True, None


class InteractiveSession:
    """Interactive CLI session manager with command processing and conversation handling."""
    
    def __init__(
        self, 
        agent_system: AgentSystem,
        cli_manager: Optional[CLIManager] = None
    ):
        """Initialize interactive session.
        
        Args:
            agent_system: The agent system instance
            cli_manager: Optional CLI manager instance
        """
        self.agent_system = agent_system
        self.cli_manager = cli_manager or CLIManager()
        self.command_processor = CommandProcessor(self)
        self.input_validator = InputValidator()
        
        # Session state
        self.is_running = False
        self.debug_mode = False
        self.conversation_history = ConversationHistory()
        
        # Session statistics
        self.questions_asked = 0
        self.commands_executed = 0
        self.session_start_time = None
    
    async def start(self) -> None:
        """Start the interactive session."""
        if self.is_running:
            logger.warning("Session is already running")
            return
        
        try:
            self.is_running = True
            self.session_start_time = self.conversation_history.created_at
            
            # Display welcome banner
            self.cli_manager.display_welcome_banner()
            
            # Show initial help
            self._display_session_info()
            
            # Main interaction loop
            await self._interaction_loop()
            
        except KeyboardInterrupt:
            self.cli_manager.display_warning("Session interrupted by user.")
        except Exception as e:
            logger.error(f"Session error: {e}")
            self.cli_manager.display_error("Session error", str(e))
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the interactive session."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Display session summary
        self._display_session_summary()
        
        self.cli_manager.display_success("Session ended. Goodbye!")
    
    def _display_session_info(self) -> None:
        """Display session information and help."""
        info_text = (
            "[bold blue]Interactive Mode Started[/bold blue]\n\n"
            "Type your questions and press Enter to get responses from the agent system.\n\n"
            "[bold]Available Commands:[/bold]\n"
            "  /help     - Show available commands\n"
            "  /status   - Show system status\n"
            "  /agents   - List available agents\n"
            "  /history  - Show conversation history\n"
            "  /clear    - Clear the screen\n"
            "  /reload   - Reload prompts and configuration\n"
            "  /debug    - Toggle debug mode\n"
            "  /quit     - Exit the session\n\n"
            "[dim]Tip: Use Tab for command completion (if supported by your terminal)[/dim]"
        )
        
        self.cli_manager.print(info_text)
    
    def _display_session_summary(self) -> None:
        """Display session summary statistics."""
        if self.session_start_time:
            from datetime import datetime
            duration = datetime.utcnow() - self.session_start_time
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            duration_str = "Unknown"
        
        summary_text = (
            f"[bold cyan]Session Summary[/bold cyan]\n\n"
            f"Duration: {duration_str}\n"
            f"Questions asked: {self.questions_asked}\n"
            f"Commands executed: {self.commands_executed}\n"
            f"Messages in history: {len(self.conversation_history.messages)}"
        )
        
        self.cli_manager.print(summary_text)
    
    async def _interaction_loop(self) -> None:
        """Main interaction loop for processing user input."""
        while self.is_running:
            try:
                # Get user input
                user_input = self._get_user_input()
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if user_input.startswith('/'):
                    # Validate command
                    is_valid, error = self.input_validator.validate_command(user_input)
                    if not is_valid:
                        self.cli_manager.display_error("Invalid command", error)
                        continue
                    
                    # Process command
                    self.commands_executed += 1
                    should_continue = await self.command_processor.process_command(user_input)
                    if not should_continue:
                        break
                else:
                    # Validate question
                    is_valid, error = self.input_validator.validate_question(user_input)
                    if not is_valid:
                        self.cli_manager.display_error("Invalid question", error)
                        continue
                    
                    # Process as question
                    await self._process_question(user_input)
                    self.questions_asked += 1
                    
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                if self.cli_manager.confirm("\nDo you want to exit the session?", default=False):
                    break
                else:
                    self.cli_manager.print("\nContinuing session...")
                    continue
            except Exception as e:
                logger.error(f"Error in interaction loop: {e}")
                self.cli_manager.display_error("Interaction error", str(e))
                if self.debug_mode:
                    import traceback
                    self.cli_manager.print(f"[dim red]{traceback.format_exc()}[/dim red]")
    
    def _get_user_input(self) -> str:
        """Get user input with proper prompting.
        
        Returns:
            User input string
        """
        try:
            prompt_text = "[bold cyan]❯[/bold cyan]"
            return self.cli_manager.input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt()
    
    async def _process_question(self, question: str) -> None:
        """Process a user question through the agent system.
        
        Args:
            question: User question string
        """
        try:
            # Display the question
            self.cli_manager.display_question(question)
            
            # Create message for history
            message = Message(
                sender_id="user",
                recipient_id="system",
                content=question,
                message_type=MessageType.USER_QUESTION
            )
            self.conversation_history.add_message(message)
            
            # Process through agent system with progress indicator
            with self.cli_manager.create_progress_context("Processing question...") as progress:
                task = progress.add_task("Analyzing and routing question...", total=None)
                
                response = await self.agent_system.process_message(question)
                
                progress.update(task, description="Response received!", completed=True)
            
            # Add response to history
            self.conversation_history.add_response(response)
            
            # Display the response
            self.cli_manager.display_response(response, show_metadata=self.debug_mode)
            
        except AgentSystemError as e:
            self.cli_manager.display_error("Agent system error", str(e))
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            self.cli_manager.display_error("Processing error", str(e))
            if self.debug_mode:
                import traceback
                self.cli_manager.print(f"[dim red]{traceback.format_exc()}[/dim red]")
    
    def add_custom_command(
        self, 
        name: str, 
        handler: Callable, 
        help_text: str,
        aliases: Optional[List[str]] = None
    ) -> None:
        """Add a custom command to the session.
        
        Args:
            name: Command name
            handler: Command handler function
            help_text: Help text for the command
            aliases: Optional command aliases
        """
        self.command_processor.register_command(name, handler, help_text, aliases)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics.
        
        Returns:
            Dictionary containing session statistics
        """
        return {
            "is_running": self.is_running,
            "debug_mode": self.debug_mode,
            "questions_asked": self.questions_asked,
            "commands_executed": self.commands_executed,
            "messages_in_history": len(self.conversation_history.messages),
            "responses_in_history": len(self.conversation_history.responses),
            "session_start_time": self.session_start_time.isoformat() if self.session_start_time else None,
        }