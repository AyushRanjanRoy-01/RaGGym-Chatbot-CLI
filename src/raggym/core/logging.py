"""Structured logging via structlog.

Pretty, colorized console logs in dev; single-line JSON in prod (set
``LOG_JSON=true``). JSON logs are grep/jq-able, which matters for debugging a
RAG pipeline — e.g. find every query whose top retrieval score was poor::

    cat app.log | jq 'select(.event=="retrieval_complete" and .top_score < 0.5)'
"""

from __future__ import annotations

import logging
import sys

import structlog

from raggym.config import get_settings

_configured = False


def configure_logging() -> None:
    """Configure structlog + stdlib logging from settings. Idempotent."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if settings.log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
