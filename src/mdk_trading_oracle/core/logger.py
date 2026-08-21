"""Structured logging module using rich formatting with Turkish Time (Europe/Istanbul)."""

import logging
import os
from datetime import datetime
from typing import Optional

from rich.console import ConsoleRenderable
from rich.logging import RichHandler
from rich.traceback import Traceback

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.time import TURKEY_TZ


class TRTRichHandler(RichHandler):
    """RichHandler subclass enforcing Europe/Istanbul (TRT / UTC+3) timestamps."""

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Optional[Traceback],
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        path = os.path.basename(record.pathname)
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created, tz=TURKEY_TZ)

        return self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )


def get_logger(name: str = "mdk_oracle") -> logging.Logger:
    """Configure and return a structured rich logger operating in Turkish Time."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        handler = TRTRichHandler(
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
