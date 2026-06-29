"""Tests for evaluation compatibility helpers."""

import importlib
import sys

from raggym.eval.runner import _install_ragas_compat_shims


def test_ragas_vertexai_compat_shim_registers_missing_module(monkeypatch):
    module_name = "langchain_community.chat_models.vertexai"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    _install_ragas_compat_shims()

    module = importlib.import_module(module_name)
    assert hasattr(module, "ChatVertexAI")
