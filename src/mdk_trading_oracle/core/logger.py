"""Structured logging module using rich formatting."""

import logging
import sys
from rich.logging import RichHandler
from mdk_trading_oracle.core.config import get_settings


def get_logger(name: str = "mdk_oracle") -> logging.Logger:
    """Configure and return a structured rich logger."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
