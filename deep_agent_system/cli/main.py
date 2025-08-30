"""Main CLI entry point for the Deep Agent System."""

import typer
from rich.console import Console

app = typer.Typer(
    name="deep-agent",
    help="Deep Agent System - A sophisticated multi-agent framework",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    question: str = typer.Argument(..., help="Question to ask the agent system"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Ask a question to the Deep Agent System."""
    if verbose:
        console.print("[bold blue]Deep Agent System[/bold blue] - Processing question...")
    
    console.print(f"[yellow]Question:[/yellow] {question}")
    console.print("[red]System not yet implemented. Please run the setup tasks first.[/red]")


if __name__ == "__main__":
    app()