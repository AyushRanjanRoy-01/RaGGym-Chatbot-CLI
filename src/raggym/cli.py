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
from raggym.practice.cli import practice_app

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
    path: str = typer.Argument(None, help="PDF or directory to ingest (defaults to BOOKS_DIR)."),
    limit_pages: int = typer.Option(
        None, "--limit-pages", "-n", help="Only ingest the first N pages per book (quick test)."
    ),
    recreate: bool = typer.Option(
        False, "--recreate", help="Drop and rebuild the collection before ingesting."
    ),
) -> None:
    """Build the vector store from your book corpus."""
    from raggym.ingestion import ingest_path

    console.print("[bold]Ingesting…[/] (first run may download the embedding model)")
    result = ingest_path(path, limit_pages=limit_pages, recreate=recreate)

    if result["books"] == 0:
        console.print("[red]No PDFs found.[/] Drop files into ./data/books/ or pass a path.")
        raise typer.Exit(code=1)

    table = Table(title="Ingestion complete")
    table.add_column("Book", style="cyan")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Secs", justify="right")
    for f in result["files"]:
        table.add_row(f["book"], str(f["pages"]), str(f["chunks"]), str(f["seconds"]))
    console.print(table)
    console.print(
        f"[green]Stored {result['chunks']} chunks from {result['books']} book(s).[/]"
    )


@app.command()
def chat() -> None:
    """Launch the RAG chat UI (Streamlit)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "apps" / "chat_app.py"
    src_path = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    console.print(f"[bold]Launching Streamlit chat…[/] ({app_path})")
    raise typer.Exit(
        code=subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)], env=env)
    )


@app.command("eval")
def evaluate_cmd(
    questions: str = typer.Option(None, "--questions", help="Path to a questions JSON file."),
) -> None:
    """Evaluate retrieval + answer quality with RAGAS (needs a provider + eval extra)."""
    from raggym.eval import evaluate_pipeline

    console.print("[bold]Running RAGAS evaluation…[/] (needs an LLM provider)")
    try:
        result = evaluate_pipeline(questions_path=questions)
    except Exception as exc:  # noqa: BLE001 — surface a friendly message, not a traceback
        console.print(f"[red]Eval unavailable:[/] {exc}")
        raise typer.Exit(code=1) from None
    console.print(result)


app.add_typer(practice_app, name="practice")


if __name__ == "__main__":
    app()
