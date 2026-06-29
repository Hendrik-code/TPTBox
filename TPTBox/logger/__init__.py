from __future__ import annotations

from ._loguru_backend import configure, install_excepthook
from ._loguru_backend import logger as loguru_logger
from .log_constants import Log_Type
from .log_file import Logger, Logger_Interface, Reflection_Logger, String_Logger
from .log_file import No_Logger as Print_Logger

__all__ = [
    "Log_Type",
    "Logger",
    "Logger_Interface",
    "Print_Logger",
    "Reflection_Logger",
    "String_Logger",
    "configure",  # opt out of the global-Loguru take-over / customize sinks
    "install_excepthook",  # route uncaught exceptions through Loguru (opt-in)
    "loguru_logger",  # the configured Loguru logger; add your own sinks (JSON, files, ...) to it
]
