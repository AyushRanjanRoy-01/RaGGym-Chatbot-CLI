"""Deployment entrypoint for Streamlit Community Cloud.

The actual app lives in ``src/raggym/apps/chat_app.py``. This thin wrapper keeps
the deployment entrypoint at the repository root and forwards root-level
Streamlit secrets into environment variables so pydantic-settings can read them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _load_streamlit_secrets_into_env() -> None:
    """Make Streamlit Cloud secrets visible to Settings without committing .env."""

    try:
        secrets = st.secrets
        for key in secrets:
            value = secrets[key]
            if isinstance(value, str | int | float | bool):
                os.environ.setdefault(key, str(value))
    except Exception:
        # Local runs can use .env; deployed runs should configure Streamlit secrets.
        return


_load_streamlit_secrets_into_env()

from raggym.apps import chat_app  # noqa: E402,F401
