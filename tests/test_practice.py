"""Round-trip test for the practice workspace + grader (no LLM)."""

from raggym.config import Settings
from raggym.practice.grader import run_tests
from raggym.practice.models import Exercise
from raggym.practice.workspace import list_exercises, write_exercise


def _exercise() -> Exercise:
    return Exercise(
        title="Reverse String",
        concept="string basics",
        difficulty="easy",
        statement="Implement a function that returns the reverse of a string.",
        function_name="reverse_string",
        starter_code="def reverse_string(s):\n    raise NotImplementedError\n",
        reference_solution="def reverse_string(s):\n    return s[::-1]\n",
        test_code=(
            "from solution import reverse_string\n\n"
            "def test_basic():\n    assert reverse_string('abc') == 'cba'\n\n"
            "def test_empty():\n    assert reverse_string('') == ''\n"
        ),
    )


def test_workspace_and_grader_roundtrip(tmp_path):
    settings = Settings(_env_file=None, workspace_dir=tmp_path)
    path = write_exercise(_exercise(), [{"n": 1, "tag": "AI p.1"}], settings=settings)

    assert (path / "solution.py").exists()
    assert (path / "test_exercise.py").exists()
    assert (path / "README.md").exists()

    # The stub raises NotImplementedError → tests fail.
    assert run_tests(path)["passed"] is False

    # A correct solution → tests pass.
    (path / "solution.py").write_text(
        "def reverse_string(s):\n    return s[::-1]\n", encoding="utf-8"
    )
    assert run_tests(path)["passed"] is True

    items = list_exercises(settings)
    assert any(it["function_name"] == "reverse_string" for it in items)
