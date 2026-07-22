# Contributing to RAGGym

Thanks for your interest! RAGGym is an open-source platform for learning and
practicing Retrieval-Augmented Generation.

## Dev setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |  macOS/Linux: source .venv/bin/activate
pip install -e ".[all]"
pre-commit install
```

## Before you push
```bash
ruff check .      # lint (and `ruff format .` to auto-format)
pytest            # tests must pass
```
Tests use fakes for LLM/vector calls, so they run with **no API keys and no
network**. Keep it that way — gate anything heavier behind `pytest.importorskip`.

## Pull requests
- **One concern per PR** — small, atomic, reviewable.
- Use **Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:`, `ci:`,
  `refactor:`, `test:`). The PR title should read like a changelog line.
- Fill in the PR template; make sure CI is green.
- No secrets in code or tests (`gitleaks` runs in CI).

## Project layout
See the [README architecture section](README.md) and `src/raggym/` — each layer
(`llm`, `embeddings`, `vectorstore`, `retrieval`, `agents`, `ingestion`,
`practice`, `eval`) is a small, swappable module.

## Good first contributions
- New `.env`-swappable provider (LLM / embeddings / vector store).
- A new retrieval technique behind a settings flag.
- More eval questions in `src/raggym/eval/questions.json`.
- Docs and examples.
