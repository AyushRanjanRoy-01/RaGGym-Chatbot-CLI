"""Tests for evaluation compatibility helpers."""

import builtins
import importlib
import sys

from raggym.eval.runner import _install_ragas_compat_shims


def test_ragas_vertexai_compat_shim_handles_missing_parent_package(monkeypatch):
    module_name = "langchain_community.chat_models.vertexai"
    for name in (
        "langchain_community.chat_models.vertexai",
        "langchain_community.chat_models",
        "langchain_community",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == module_name:
            raise ModuleNotFoundError(
                "No module named 'langchain_community'",
                name="langchain_community",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    _install_ragas_compat_shims()

    module = importlib.import_module(module_name)
    assert hasattr(module, "ChatVertexAI")
