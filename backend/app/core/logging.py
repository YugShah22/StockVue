"""
AI Multi-Factor Stock Intelligence & Portfolio Decision-Support System
backend/app/core/logging.py

Structured logging setup.

Usage (anywhere in the codebase):
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Fetched prices", extra={"symbol": "RELIANCE", "bars": 252})

In development (LOG_JSON=False): human-readable coloured output.
In production  (LOG_JSON=True):  JSON lines, one object per log record.
"""

import logging
import sys
from typing import Any


def _build_formatter(*, json_mode: bool) -> logging.Formatter:
    """Return a plain or JSON-style formatter."""
    if json_mode:
        # Simple JSON formatter — replace with python-json-logger in later phases
        # if richer structured logging is needed.
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"name":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    return logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")


def configure_logging(*, level: str = "INFO", json_mode: bool = False) -> None:
    """
    Configure the root logger once at application startup.
    Call this from main.py lifespan before anything else logs.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_build_formatter(json_mode=json_mode))

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Always use __name__ as the name so log records carry the module path.
    Example:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Type alias for annotating logger parameters in service/domain code
# ---------------------------------------------------------------------------
Logger = logging.Logger


def log_context(**kwargs: Any) -> dict[str, Any]:
    """
    Helper to build a structured extra dict for logger calls.

    Usage:
        logger.info("Price ingested", extra=log_context(symbol="RELIANCE", bars=252))
    """
    return kwargs
