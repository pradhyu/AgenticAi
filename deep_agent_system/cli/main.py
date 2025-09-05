"""Main CLI entry point for the Deep Agent System."""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from deep_agent_system.system import AgentSystem, AgentSystemError


# CLI application setup
app = typer.Typer(
    name="deep-agent",
    help="Deep Agent System - A sophisticated multi-agent framework",
    add_completion=False,
)
console = Console()

# Global system instance
_system: Optional[AgentSystem] = None
_shutdown_requested = False


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
        debug: Enable debug logging
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    console.print("\n[yellow]Shutdown requested. Cleaning up...[/yellow]")
    
    if _system:
        asyncio.create_task(shutdown_system())


async def shutdown_system() -> None:
    """Shutdown the system gracefully."""
    global _system
    if _system:
        try:
            await _system.stop()
            console.print("[green]System shutdown complete.[/green]")
        except Exception as e:
            console.print(f"[red]Error during shutdown: {e}[/red]")
        finally:
            _system = None


async def initialize_system(config_file: Optional[str] = None) -> AgentSystem:
    """Initialize the AgentSystem.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        Initialized AgentSystem instance
        
    Raises:
        AgentSystemError: If initialization fails
    """
    global _system
    
    if _system and _system.is_initialized:
        return _system
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Initializing Deep Agent System...", total=None)
        
        try:
            _system = AgentSystem(config_file)
            await _system.initialize()
            
            progress.update(task, description="Starting system components...")
            await _system.start()
            
            progress.update(task, description="System ready!", completed=True)
            
            return _system
            
        except Exception as e:
            progress.update(task, description=f"Initialization failed: {e}")
            raise AgentSystemError(f"Failed to initialize system: {e}") from e


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the agent system"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
    output_format: str = typer.Option("rich", "--format", "-f", help="Output format (rich, json, plain)"),
) -> None:
    """Ask a question to the Deep Agent System."""
    setup_logging(verbose, debug)
    
    async def process_question():
        try:
            # Initialize system
            system = await initialize_system(config)
            
            # Display question
            console.print(Panel(
                question,
                title="[bold blue]Question[/bold blue]",
                border_style="blue"
            ))
            
            # Process question
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing question...", total=None)
                
                response = await system.process_message(question)
                
                progress.update(task, description="Response received!", completed=True)
            
            # Display response
            if output_format == "json":
                import json
                response_data = {
                    "agent_id": response.agent_id,
                    "content": response.content,
                    "confidence_score": response.confidence_score,
                    "timestamp": response.timestamp.isoformat(),
                }
                console.print(json.dumps(response_data, indent=2))
            elif output_format == "plain":
                console.print(response.content)
            else:  # rich format
                console.print(Panel(
                    response.content,
                    title=f"[bold green]Response from {response.agent_id}[/bold green]",
                    subtitle=f"Confidence: {response.confidence_score:.2f}",
                    border_style="green"
                ))
            
        except AgentSystemError as e:
            console.print(f"[red]System Error: {e}[/red]")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            if debug:
                console.print_exception()
            raise typer.Exit(1)
        finally:
            if _system:
                await shutdown_system()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run async function
    asyncio.run(process_question())


@app.command()
def interactive(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
) -> None:
    """Start an interactive session with the Deep Agent System."""
    setup_logging(verbose, debug)
    
    async def run_interactive():
        try:
            # Initialize system
            system = await initialize_system(config)
            
            # Import session management components
            from deep_agent_system.cli.manager import CLIManager
            from deep_agent_system.cli.session import InteractiveSession
            
            # Create CLI manager and session
            cli_manager = CLIManager(console=console)
            session = InteractiveSession(system, cli_manager)
            
            # Set debug mode if requested
            session.debug_mode = debug
            
            # Start interactive session
            await session.start()
            
        except AgentSystemError as e:
            console.print(f"[red]System Error: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            if debug:
                console.print_exception()
            raise typer.Exit(1)
        finally:
            if _system:
                await shutdown_system()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run interactive session
    asyncio.run(run_interactive())





@app.command()
def status(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Show system status without starting interactive mode."""
    setup_logging(verbose)
    
    async def show_status():
        try:
            system = await initialize_system(config)
            status = system.get_system_status()
            
            # Display status
            console.print(Panel(
                f"[bold]System Status[/bold]\n\n"
                f"Initialized: {'✓' if status['is_initialized'] else '✗'}\n"
                f"Running: {'✓' if status['is_running'] else '✗'}\n"
                f"Agents: {status['agents_count']}\n\n"
                f"[bold]Components:[/bold]\n" +
                "\n".join([
                    f"  {comp.replace('_', ' ').title()}: {'✓' if enabled else '✗'}"
                    for comp, enabled in status['components'].items()
                ]),
                title="Deep Agent System",
                border_style="blue"
            ))
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        finally:
            if _system:
                await shutdown_system()
    
    asyncio.run(show_status())


@app.command()
def version() -> None:
    """Show version information."""
    from deep_agent_system import __version__
    
    console.print(Panel(
        f"[bold blue]Deep Agent System[/bold blue]\n"
        f"Version: {__version__}\n"
        f"A sophisticated multi-agent framework using LangChain and LangGraph",
        title="Version Information",
        border_style="blue"
    ))


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()