from __future__ import annotations

from ._loguru_backend import configure, current_level
from ._loguru_backend import logger as loguru_logger
from .log_constants import Log_Type
from .log_file import (
    Logger,
    Logger_Interface,
    Reflection_Logger,
    String_Logger,
    configure_logging,
    get_default_logger,
    log,
    logger,
)
from .log_file import No_Logger as Print_Logger

__all__ = [
    "Log_Type",
    "Logger",
    "Logger_Interface",
    "Print_Logger",
    "Reflection_Logger",
    "String_Logger",
    "configure",
    "configure_logging",
    "current_level",
    "get_default_logger",
    "log",
    "logger",
    "loguru_logger",
]
