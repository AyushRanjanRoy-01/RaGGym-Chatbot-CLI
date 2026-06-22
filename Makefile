# RAGGym developer commands.
# On Windows without `make`, run the underlying commands shown here directly,
# or use Git Bash / WSL. Equivalent commands are in the README quickstart.

.PHONY: install install-all lint fmt test ingest chat practice clean

install:        ## Install base package + dev tools (editable)
	pip install -e ".[dev]"

install-all:    ## Install everything (rag, ingest, chat, eval, dev)
	pip install -e ".[all]"

lint:           ## Lint with ruff
	ruff check .

fmt:            ## Auto-format + fix with ruff
	ruff format .
	ruff check . --fix

test:           ## Run the test suite
	pytest

ingest:         ## Build the vector store from data/books/  [Phase 1]
	raggym ingest

chat:           ## Launch the RAG chat UI  [Phase 2]
	raggym chat

practice:       ## Start a practice session  [Phase 3]
	raggym practice

clean:          ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build src/*.egg-info
