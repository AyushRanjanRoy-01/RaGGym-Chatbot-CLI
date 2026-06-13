import logging
import sys
from pathlib import Path

import structlog
from config import settings

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "rag.log"


def setup_logging(level: str | None = None) -> None:
    log_level_str = level or settings.log_level
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    _LOG_DIR.mkdir(exist_ok=True)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.dev.ConsoleRenderer(colors=True)],
        foreign_pre_chain=shared_processors,
    )
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.processors.JSONRenderer()],
        foreign_pre_chain=shared_processors,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(file_formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.setLevel(log_level)

    for name in ("uvicorn", "uvicorn.error", "fastapi"):
        logging.getLogger(name).propagate = True

    log = structlog.get_logger()
    log.info("logging_initialized", level=log_level_str.upper(), log_file=str(_LOG_FILE))
