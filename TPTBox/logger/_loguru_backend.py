"""Loguru-backed emission layer for the TPTBox logger.

This is the only module that imports :mod:`loguru`.  The public logger classes in
:mod:`TPTBox.logger.log_file` keep owning everything that decides the output *bytes*
(prefix building, ``ltype``-in-``*text`` detection, ``verbose`` gating, the
``datatype_to_string`` conversion).  This module only takes the final, fully-built
message string and routes it to Loguru sinks:

* a colorized terminal sink writing to the live ``sys.stdout`` (so output stays
  capturable/redirectable just like the old ``print``), and
* per-instance file sinks for :class:`~TPTBox.logger.log_file.Logger`.

Each :class:`~TPTBox.logger.log_constants.Log_Type` maps to a custom Loguru level
whose ``color`` markup reproduces the exact ANSI code of the old ``type2bcolors``
table, so the terminal coloring renders identically while being level-driven.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

from TPTBox.logger.log_constants import Log_Type

__all__ = [
    "add_file_sink",
    "configure",
    "emit_file",
    "emit_terminal",
    "level_name",
    "logger",
    "remove_sink",
]

# Log_Type -> (loguru level name, severity, color markup).
# The markup is chosen so Loguru emits the SAME ANSI escape as the old `type2bcolors`
# (validated: <light-cyan> -> \033[96m, <bg blue> -> \033[44m, etc.). Empty color == default.
_LEVELS: dict[Log_Type, tuple[str, int, str]] = {
    Log_Type.TEXT: ("TPTBOX_TEXT", 20, ""),
    Log_Type.NEUTRAL: ("TPTBOX_NEUTRAL", 20, ""),
    Log_Type.SAVE: ("TPTBOX_SAVE", 22, "<light-cyan>"),
    Log_Type.WARNING: ("TPTBOX_WARNING", 30, "<light-yellow>"),
    Log_Type.WARNING_THROW: ("TPTBOX_WARNING_THROW", 30, "<light-yellow>"),
    Log_Type.LOG: ("TPTBOX_LOG", 20, "<light-blue>"),
    Log_Type.OK: ("TPTBOX_OK", 25, "<light-green>"),
    Log_Type.FAIL: ("TPTBOX_FAIL", 40, "<light-red>"),
    Log_Type.Yellow: ("TPTBOX_YELLOW", 20, "<yellow>"),
    Log_Type.STRANGE: ("TPTBOX_STRANGE", 10, "<light-magenta>"),
    Log_Type.UNDERLINE: ("TPTBOX_UNDERLINE", 20, "<underline>"),
    Log_Type.ITALICS: ("TPTBOX_ITALICS", 20, "<italic>"),
    Log_Type.BOLD: ("TPTBOX_BOLD", 20, "<bold>"),
    Log_Type.DOCKER: ("TPTBOX_DOCKER", 20, "<italic>"),
    Log_Type.TOTALSEG: ("TPTBOX_TOTALSEG", 20, "<italic>"),
    Log_Type.STAGE: ("TPTBOX_STAGE", 20, "<bg blue>"),
}

_TERMINAL_FORMAT = "<level>{message}</level>"
_FILE_FORMAT = "{message}"

_configured = False


def level_name(ltype: Log_Type) -> str:
    """Return the Loguru level name registered for a given ``Log_Type``."""
    return _LEVELS.get(ltype, _LEVELS[Log_Type.TEXT])[0]


def _ensure_levels() -> None:
    """Register the custom ``TPTBOX_*`` levels (idempotent)."""
    for name, no, color in _LEVELS.values():
        try:
            logger.level(name)
        except ValueError:
            logger.level(name, no=no, color=color)


def _terminal_sink(message) -> None:
    r"""Write a colorized record to the live ``sys.stdout``, honoring the call's ``end``.

    Loguru always appends a ``\n``; we strip it and append the original ``end`` so
    ``end="\r"`` progress lines survive unchanged.
    """
    end = message.record["extra"].get("tptbox_end", "\n")
    text = str(message)
    text = text.removesuffix("\n")
    # Look up sys.stdout at write-time (not at add-time) so redirect_stdout / capsys work.
    sys.stdout.write(text + end)


def configure(take_over: bool | None = None) -> None:
    """Configure the global Loguru logger for TPTBox (idempotent).

    Args:
        take_over: If True, remove Loguru's pre-existing handlers so only the
            TPTBox terminal sink is active (the default — matches the old logger
            which was the sole stdout writer). If False, leave any handlers a host
            application registered and only add TPTBox's filtered stdout sink.
            If None, read the ``TPTBOX_LOGGER_TAKEOVER`` env var (default True).
    """
    global _configured  # noqa: PLW0603
    _ensure_levels()
    if _configured:
        return
    if take_over is None:
        take_over = os.environ.get("TPTBOX_LOGGER_TAKEOVER", "1") not in ("0", "false", "False")
    if take_over:
        logger.remove()
    logger.add(
        _terminal_sink,
        format=_TERMINAL_FORMAT,
        colorize=True,
        level=0,
        filter=lambda r: r["extra"].get("tptbox_channel") == "terminal",
        enqueue=False,
        catch=False,
    )
    _configured = True


def emit_terminal(text: str, ltype: Log_Type = Log_Type.TEXT, end: str = "\n") -> None:
    """Emit one already-built, already-prefixed message to the terminal sink.

    ``text`` is passed as the single ``{message}`` value with NO format args, so literal
    ``{}``/``<>`` in the message are never interpreted.
    """
    if not _configured:
        configure()
    logger.bind(tptbox_channel="terminal", tptbox_end=end).log(level_name(ltype), text)


def add_file_sink(filepath, key, *, rotation=None, retention=None, enqueue: bool = False, mode: str = "w") -> int:
    """Register a Loguru file sink dedicated to one ``Logger`` instance.

    Args:
        filepath: Destination log file (Loguru owns/creates it).
        key: Unique id bound on each record so only this instance's lines land here.
        rotation/retention: Optional Loguru file rotation/retention policies.
        enqueue: If True, writes go through a background thread (thread/process-safe).
        mode: File open mode (``"w"`` truncates, matching the old behavior).

    Returns:
        The Loguru sink id (pass to :func:`remove_sink`).
    """
    if not _configured:
        configure()
    return logger.add(
        str(filepath),
        format=_FILE_FORMAT,
        colorize=False,
        level=0,
        filter=lambda r, _k=key: r["extra"].get("tptbox_file_id") == _k,
        rotation=rotation,
        retention=retention,
        enqueue=enqueue,
        mode=mode,
        catch=False,
    )


def add_file_stream_sink(stream, key, *, enqueue: bool = False) -> int:
    """Register a Loguru function sink writing ANSI-free lines to ``stream`` (a file handle).

    Unlike :func:`add_file_sink` (Loguru owns the file) this keeps the caller's handle, so
    ``flush()`` works and the call's ``end`` is honored. No rotation/retention.
    """
    if not _configured:
        configure()

    def _sink(message, _s=stream) -> None:
        end = message.record["extra"].get("tptbox_end", "\n")
        text = str(message)
        text = text.removesuffix("\n")
        _s.write(text + end)

    return logger.add(
        _sink,
        format=_FILE_FORMAT,
        colorize=False,
        level=0,
        filter=lambda r, _k=key: r["extra"].get("tptbox_file_id") == _k,
        enqueue=enqueue,
        catch=False,
    )


def emit_file(text: str, key, ltype: Log_Type = Log_Type.TEXT, end: str = "\n") -> None:
    """Emit one ANSI-free line to the file sink identified by ``key``."""
    if not _configured:
        configure()
    logger.bind(tptbox_channel="file", tptbox_file_id=key, tptbox_end=end).log(level_name(ltype), text)


def remove_sink(sink_id: int) -> None:
    """Remove a Loguru sink, tolerating an already-removed id (atexit double-remove)."""
    try:
        logger.remove(sink_id)
    except (ValueError, KeyError):
        pass
