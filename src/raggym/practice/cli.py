"""`raggym practice` sub-commands: new / grade / list."""

from __future__ import annotations

import typer
from rich.console import Console

practice_app = typer.Typer(
    help="Coding practice grounded in the corpus.", no_args_is_help=True, add_completion=False
)
console = Console()


@practice_app.command("new")
def new(topic: str = typer.Argument(..., help="Concept to practice, e.g. 'prompt chaining'.")):
    """Generate a new coding exercise from the corpus."""
    from raggym.practice.service import create_exercise

    console.print(f"[bold]Generating an exercise on:[/] {topic}")
    path = create_exercise(topic)
    console.print(f"[green]Created:[/] {path}")
    console.print(
        f"Edit [cyan]{path / 'solution.py'}[/], then run: "
        f"[bold]raggym practice grade \"{path}\"[/]"
    )


@practice_app.command("grade")
def grade(exercise_dir: str = typer.Argument(..., help="Path to an exercise directory.")):
    """Run the exercise tests and get an AI review of your solution."""
    from raggym.practice.service import grade_exercise

    result = grade_exercise(exercise_dir)
    tests = result["tests"]
    status = "[bold green]PASSED[/]" if tests["passed"] else "[bold red]FAILED[/]"
    console.print(f"Tests: {status}")
    if tests["output"].strip():
        console.print(tests["output"].strip())
    console.rule("AI Review")
    console.print(result["review"])


@practice_app.command("list")
def list_cmd():
    """List exercises in the workspace."""
    from raggym.practice.service import list_exercises

    items = list_exercises()
    if not items:
        console.print('No exercises yet. Create one: raggym practice new "prompt chaining"')
        return
    for it in items:
        console.print(
            f"- [cyan]{it['dir']}[/] — {it.get('concept', '?')} ({it.get('difficulty', '?')})"
        )
