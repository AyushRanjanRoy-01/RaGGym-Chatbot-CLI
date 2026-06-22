"""RAGGym command-line entrypoint.

    raggym version          # show version
    raggym config           # print resolved settings (secrets redacted)
    raggym ingest <path>    # Phase 1 — build the vector store from books
    raggym chat             # Phase 2 — launch the RAG chat UI
    raggym practice         # Phase 3 — start a coding/interview practice session
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from raggym import __version__
from raggym.config import get_settings
from raggym.core import configure_logging

app = typer.Typer(
    name="raggym",
    help="An open-source gym for RAG interview prep.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_SECRET_FIELDS = {"openai_api_key", "anthropic_api_key"}


@app.callback()
def _main() -> None:
    """Initialise logging for every command."""
    configure_logging()


@app.command()
def version() -> None:
    """Show the installed RAGGym version."""
    console.print(f"[bold cyan]RAGGym[/] v{__version__}")


@app.command()
def config() -> None:
    """Print the resolved configuration (API keys redacted)."""
    settings = get_settings()
    table = Table(title="RAGGym configuration", show_lines=False)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for name, value in settings.model_dump().items():
        if name in _SECRET_FIELDS:
            value = "***set***" if value else "(unset)"
        table.add_row(name, str(value))
    console.print(table)


@app.command()
def ingest(
    path: str = typer.Argument(None, help="PDF/dir to ingest (defaults to BOOKS_DIR)."),
) -> None:
    """Build the vector store from your book corpus. [Phase 1]"""
    console.print("[yellow]Ingestion arrives in Phase 1.[/] Drop PDFs into ./data/books/ first.")
    raise typer.Exit(code=0)


@app.command()
def chat() -> None:
    """Launch the RAG chat UI. [Phase 2]"""
    console.print("[yellow]Chat mode arrives in Phase 2.[/]")
    raise typer.Exit(code=0)


@app.command()
def practice() -> None:
    """Start a coding / interview practice session. [Phase 3]"""
    console.print("[yellow]Practice mode arrives in Phase 3.[/]")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
