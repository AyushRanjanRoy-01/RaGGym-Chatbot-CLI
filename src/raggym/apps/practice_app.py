"""Streamlit practice app — code in the browser, graded by Pyodide.

Launch with ``raggym practice web``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from raggym.apps.browser_ide import render_browser_ide
from raggym.practice import list_exercises

st.set_page_config(page_title="RAGGym · Practice", page_icon="🏋️", layout="wide")

_DEMO_STARTER = (
    "def reverse_string(s):\n"
    '    """Return the reverse of s."""\n'
    "    raise NotImplementedError\n"
)
_DEMO_TESTS = (
    "from solution import reverse_string\n\n"
    "def test_basic():\n    assert reverse_string('abc') == 'cba'\n\n"
    "def test_empty():\n    assert reverse_string('') == ''\n"
)

st.title("🏋️ Practice — code in the browser")
st.caption(
    "Edit the solution and click **Run tests**. Python runs entirely in your browser "
    "(Pyodide/WebAssembly) — nothing is sent to a server."
)

exercises = [e for e in list_exercises() if (Path(e["dir"]) / "test_exercise.py").exists()]

if exercises:
    labels = {f"{e.get('title') or Path(e['dir']).name}": e["dir"] for e in exercises}
    choice = st.selectbox("Exercise", list(labels))
    ex_dir = Path(labels[choice])
    starter = (ex_dir / "solution.py").read_text(encoding="utf-8")
    tests = (ex_dir / "test_exercise.py").read_text(encoding="utf-8")
    render_browser_ide(starter, tests, title=choice)
else:
    st.info('No saved exercises yet — run `raggym practice new "<topic>"`. Here is a demo:')
    render_browser_ide(_DEMO_STARTER, _DEMO_TESTS, title="Reverse String (demo)")
