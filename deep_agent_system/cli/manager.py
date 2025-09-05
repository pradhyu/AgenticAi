"""CLI Manager with Rich text support for the Deep Agent System."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from deep_agent_system.config.models import AgentType
from deep_agent_system.models.messages import Message, Response
from deep_agent_system.system import AgentSystem


class CLIManager:
    """Manager for CLI interface with Rich text formatting and display capabilities."""
    
    # Color schemes for different agent types
    AGENT_COLORS = {
        AgentType.ANALYST: "blue",
        AgentType.ARCHITECT: "green", 
        AgentType.DEVELOPER: "yellow",
        AgentType.CODE_REVIEWER: "red",
        AgentType.TESTER: "magenta",
    }
    
    # Default colors for fallback
    DEFAULT_COLORS = {
        "primary": "blue",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "info": "cyan",
        "muted": "dim white",
    }
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the CLI Manager.
        
        Args:
            console: Optional Rich Console instance. If None, creates a new one.
        """
        self.console = console or Console()
        self._progress_tasks: Dict[str, Any] = {}
        
    def display_welcome_banner(self) -> None:
        """Display the welcome banner for the CLI."""
        banner_text = Text()
        banner_text.append("Deep Agent System", style="bold blue")
        banner_text.append("\n")
        banner_text.append("A sophisticated multi-agent framework using LangChain and LangGraph", style="dim")
        
        self.console.print(Panel(
            banner_text,
            title="Welcome",
            border_style="blue",
            padding=(1, 2)
        ))
    
    def display_question(self, question: str, sender_id: str = "user") -> None:
        """Display a user question with proper formatting.
        
        Args:
            question: The question text
            sender_id: ID of the sender
        """
        # Format question with proper styling
        question_text = Text(question, style="white")
        
        self.console.print(Panel(
            question_text,
            title=f"[bold blue]Question from {sender_id}[/bold blue]",
            border_style="blue",
            padding=(0, 1)
        ))
    
    def display_response(
        self, 
        response: Response, 
        format_code: bool = True,
        show_metadata: bool = False
    ) -> None:
        """Display an agent response with rich formatting.
        
        Args:
            response: The response object to display
            format_code: Whether to apply syntax highlighting to code blocks
            show_metadata: Whether to show response metadata
        """
        # Get agent color
        agent_color = self._get_agent_color(response.agent_id)
        
        # Format the response content
        formatted_content = self._format_response_content(
            response.content, 
            format_code=format_code
        )
        
        # Create subtitle with confidence and timestamp
        subtitle_parts = [f"Confidence: {response.confidence_score:.2f}"]
        if response.timestamp:
            subtitle_parts.append(f"Time: {response.timestamp.strftime('%H:%M:%S')}")
        subtitle = " | ".join(subtitle_parts)
        
        # Display main response
        self.console.print(Panel(
            formatted_content,
            title=f"[bold {agent_color}]Response from {response.agent_id}[/bold {agent_color}]",
            subtitle=subtitle,
            border_style=agent_color,
            padding=(0, 1)
        ))
        
        # Display metadata if requested
        if show_metadata and (response.metadata or response.context_used):
            self._display_response_metadata(response)
    
    def _format_response_content(self, content: str, format_code: bool = True) -> Union[str, Group]:
        """Format response content with syntax highlighting for code blocks.
        
        Args:
            content: Raw response content
            format_code: Whether to apply syntax highlighting
            
        Returns:
            Formatted content for Rich display
        """
        if not format_code:
            return content
        
        # Look for code blocks in the content
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        inline_code_pattern = r'`([^`]+)`'
        
        # Find all code blocks
        code_blocks = list(re.finditer(code_block_pattern, content, re.DOTALL))
        
        if not code_blocks:
            # No code blocks, just format inline code
            formatted_content = re.sub(
                inline_code_pattern,
                lambda m: f"[bold cyan]{m.group(1)}[/bold cyan]",
                content
            )
            return formatted_content
        
        # Process content with code blocks
        parts = []
        last_end = 0
        
        for match in code_blocks:
            # Add text before code block
            if match.start() > last_end:
                text_part = content[last_end:match.start()]
                # Format inline code in text part
                text_part = re.sub(
                    inline_code_pattern,
                    lambda m: f"[bold cyan]{m.group(1)}[/bold cyan]",
                    text_part
                )
                if text_part.strip():
                    parts.append(text_part)
            
            # Add formatted code block
            language = match.group(1) or "text"
            code_content = match.group(2)
            
            syntax = Syntax(
                code_content,
                language,
                theme="monokai",
                line_numbers=len(code_content.split('\n')) > 5,
                word_wrap=True
            )
            parts.append(syntax)
            
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(content):
            remaining_text = content[last_end:]
            remaining_text = re.sub(
                inline_code_pattern,
                lambda m: f"[bold cyan]{m.group(1)}[/bold cyan]",
                remaining_text
            )
            if remaining_text.strip():
                parts.append(remaining_text)
        
        return Group(*parts) if len(parts) > 1 else (parts[0] if parts else content)
    
    def _display_response_metadata(self, response: Response) -> None:
        """Display response metadata in a formatted table.
        
        Args:
            response: Response object with metadata
        """
        if not response.metadata and not response.context_used:
            return
        
        table = Table(title="Response Metadata", show_header=True, header_style="bold cyan")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        # Add context information
        if response.context_used:
            table.add_row("Context Sources", str(len(response.context_used)))
            for i, context in enumerate(response.context_used[:3], 1):  # Show first 3
                table.add_row(f"  Source {i}", context[:50] + "..." if len(context) > 50 else context)
        
        # Add workflow information
        if response.workflow_id:
            table.add_row("Workflow ID", response.workflow_id)
        
        # Add custom metadata
        for key, value in response.metadata.items():
            if key not in ['context_used', 'workflow_id']:  # Avoid duplicates
                table.add_row(key.replace('_', ' ').title(), str(value))
        
        self.console.print(table)
    
    def display_system_status(self, status: Dict[str, Any]) -> None:
        """Display system status information.
        
        Args:
            status: System status dictionary
        """
        # Create main status table
        status_table = Table(title="System Status", show_header=True, header_style="bold blue")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="white")
        
        # Add basic status
        status_table.add_row("Initialized", "✓" if status.get("is_initialized") else "✗")
        status_table.add_row("Running", "✓" if status.get("is_running") else "✗")
        status_table.add_row("Agents Count", str(status.get("agents_count", 0)))
        
        # Add component status
        components = status.get("components", {})
        for component, enabled in components.items():
            component_name = component.replace("_", " ").title()
            status_icon = "✓" if enabled else "✗"
            status_table.add_row(component_name, status_icon)
        
        self.console.print(status_table)
        
        # Display agent details if available
        if "agents" in status and status["agents"]:
            self.display_agents_table(status["agents"])
    
    def display_agents_table(self, agents: List[Dict[str, Any]]) -> None:
        """Display a table of registered agents.
        
        Args:
            agents: List of agent information dictionaries
        """
        if not agents:
            self.console.print("[yellow]No agents registered.[/yellow]")
            return
        
        agents_table = Table(title="Registered Agents", show_header=True, header_style="bold green")
        agents_table.add_column("Agent ID", style="cyan")
        agents_table.add_column("Type", style="blue")
        agents_table.add_column("Status", style="white")
        
        for agent in agents:
            agent_id = agent.get("agent_id", "unknown")
            agent_type = agent.get("agent_type", "unknown")
            is_active = agent.get("is_active", False)
            
            status_icon = "✓" if is_active else "✗"
            status_color = "green" if is_active else "red"
            
            agents_table.add_row(
                agent_id,
                agent_type,
                f"[{status_color}]{status_icon}[/{status_color}]"
            )
        
        self.console.print(agents_table)
    
    def display_help(self, commands: Dict[str, str]) -> None:
        """Display help information for available commands.
        
        Args:
            commands: Dictionary of command names and descriptions
        """
        help_table = Table(title="Available Commands", show_header=True, header_style="bold yellow")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        for command, description in commands.items():
            help_table.add_row(command, description)
        
        self.console.print(Panel(
            help_table,
            title="Help",
            border_style="yellow",
            padding=(1, 2)
        ))
    
    def display_error(self, error: str, details: Optional[str] = None) -> None:
        """Display an error message with optional details.
        
        Args:
            error: Main error message
            details: Optional detailed error information
        """
        error_text = Text(error, style="bold red")
        
        if details:
            error_text.append("\n\n")
            error_text.append("Details:", style="bold")
            error_text.append(f"\n{details}", style="dim red")
        
        self.console.print(Panel(
            error_text,
            title="[bold red]Error[/bold red]",
            border_style="red",
            padding=(0, 1)
        ))
    
    def display_warning(self, warning: str) -> None:
        """Display a warning message.
        
        Args:
            warning: Warning message
        """
        self.console.print(Panel(
            warning,
            title="[bold yellow]Warning[/bold yellow]",
            border_style="yellow",
            padding=(0, 1)
        ))
    
    def display_info(self, info: str) -> None:
        """Display an informational message.
        
        Args:
            info: Information message
        """
        self.console.print(Panel(
            info,
            title="[bold cyan]Info[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        ))
    
    def display_success(self, message: str) -> None:
        """Display a success message.
        
        Args:
            message: Success message
        """
        self.console.print(Panel(
            message,
            title="[bold green]Success[/bold green]",
            border_style="green",
            padding=(0, 1)
        ))
    
    def create_progress_context(self, description: str) -> Progress:
        """Create a progress context for long-running operations.
        
        Args:
            description: Description of the operation
            
        Returns:
            Rich Progress context manager
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        )
    
    def format_code_block(
        self, 
        code: str, 
        language: str = "python",
        line_numbers: bool = False,
        theme: str = "monokai"
    ) -> Syntax:
        """Format a code block with syntax highlighting.
        
        Args:
            code: Code content
            language: Programming language for syntax highlighting
            line_numbers: Whether to show line numbers
            theme: Color theme for syntax highlighting
            
        Returns:
            Rich Syntax object for display
        """
        return Syntax(
            code,
            language,
            theme=theme,
            line_numbers=line_numbers,
            word_wrap=True
        )
    
    def create_tree_view(self, title: str, data: Dict[str, Any]) -> Tree:
        """Create a tree view for hierarchical data.
        
        Args:
            title: Tree title
            data: Hierarchical data dictionary
            
        Returns:
            Rich Tree object
        """
        tree = Tree(title)
        
        def add_items(parent, items):
            for key, value in items.items():
                if isinstance(value, dict):
                    branch = parent.add(f"[bold]{key}[/bold]")
                    add_items(branch, value)
                elif isinstance(value, list):
                    branch = parent.add(f"[bold]{key}[/bold] ({len(value)} items)")
                    for i, item in enumerate(value[:5]):  # Show first 5 items
                        branch.add(f"[dim]{i}: {str(item)[:50]}{'...' if len(str(item)) > 50 else ''}[/dim]")
                    if len(value) > 5:
                        branch.add(f"[dim]... and {len(value) - 5} more[/dim]")
                else:
                    parent.add(f"{key}: [green]{value}[/green]")
        
        add_items(tree, data)
        return tree
    
    def _get_agent_color(self, agent_id: str) -> str:
        """Get the color scheme for an agent based on its type.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Color name for the agent
        """
        # Try to determine agent type from ID
        agent_id_lower = agent_id.lower()
        
        if "analyst" in agent_id_lower:
            return self.AGENT_COLORS.get(AgentType.ANALYST, self.DEFAULT_COLORS["primary"])
        elif "architect" in agent_id_lower:
            return self.AGENT_COLORS.get(AgentType.ARCHITECT, self.DEFAULT_COLORS["success"])
        elif "developer" in agent_id_lower:
            return self.AGENT_COLORS.get(AgentType.DEVELOPER, self.DEFAULT_COLORS["warning"])
        elif "reviewer" in agent_id_lower or "review" in agent_id_lower:
            return self.AGENT_COLORS.get(AgentType.CODE_REVIEWER, self.DEFAULT_COLORS["error"])
        elif "tester" in agent_id_lower or "test" in agent_id_lower:
            return self.AGENT_COLORS.get(AgentType.TESTER, self.DEFAULT_COLORS["info"])
        else:
            return self.DEFAULT_COLORS["primary"]
    
    def clear_screen(self) -> None:
        """Clear the console screen."""
        self.console.clear()
    
    def print(self, *args, **kwargs) -> None:
        """Print to console with Rich formatting.
        
        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments for Rich console.print()
        """
        self.console.print(*args, **kwargs)
    
    def input(self, prompt: str = "", **kwargs) -> str:
        """Get user input with Rich formatting.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for Rich Prompt
            
        Returns:
            User input string
        """
        from rich.prompt import Prompt
        return Prompt.ask(prompt, console=self.console, **kwargs)
    
    def confirm(self, prompt: str = "", default: bool = False, **kwargs) -> bool:
        """Get user confirmation with Rich formatting.
        
        Args:
            prompt: Confirmation prompt
            default: Default value if user just presses Enter
            **kwargs: Additional arguments for Rich Confirm
            
        Returns:
            User confirmation boolean
        """
        from rich.prompt import Confirm
        return Confirm.ask(prompt, default=default, console=self.console, **kwargs)