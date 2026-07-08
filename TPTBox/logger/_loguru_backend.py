"""Loguru-backed emission layer.

Terminal output is written directly to ``sys.stdout`` with our own ANSI coloring;
file sinks and structured exception records go through Loguru so users can attach
their own sinks to :data:`loguru_logger`.
"""

from __future__ import annotations

import datetime
import os
import sys
import threading

from loguru import logger

from TPTBox.logger.log_constants import Log_Type, bcolors, type2bcolors

__all__ = [
    "SILENT_LEVEL",
    "_install_excepthook",
    "_install_warnings_hook",
    "_restore_warnings_hook",
    "_set_verbosity_level",
    "add_file_sink",
    "add_file_stream_sink",
    "configure",
    "current_level",
    "emit_exception",
    "emit_file",
    "emit_terminal",
    "level_name",
    "logger",
    "remove_sink",
]

# Threshold higher than any real level; used by silence().
SILENT_LEVEL = 1000

# Log_Type -> (loguru level name, severity, color markup).
# The markup is chosen so Loguru emits the SAME ANSI escape as the old `type2bcolors`
# (validated: <light-cyan> -> \033[96m, <bg blue> -> \033[44m, etc.). Empty color == default.
_LEVELS: dict[Log_Type, tuple[str, int, str]] = {
    Log_Type.STRANGE: ("TPTBOX_STRANGE", 10, "<light-magenta>"),
    Log_Type.TEXT: ("TPTBOX_TEXT", 20, ""),
    Log_Type.NEUTRAL: ("TPTBOX_NEUTRAL", 20, ""),
    Log_Type.LOG: ("TPTBOX_LOG", 20, "<light-blue>"),
    Log_Type.Yellow: ("TPTBOX_YELLOW", 20, "<yellow>"),
    Log_Type.UNDERLINE: ("TPTBOX_UNDERLINE", 20, "<underline>"),
    Log_Type.ITALICS: ("TPTBOX_ITALICS", 20, "<italic>"),
    Log_Type.BOLD: ("TPTBOX_BOLD", 20, "<bold>"),
    Log_Type.DOCKER: ("TPTBOX_DOCKER", 20, "<italic>"),
    Log_Type.TOTALSEG: ("TPTBOX_TOTALSEG", 20, "<italic>"),
    Log_Type.STAGE: ("TPTBOX_STAGE", 20, "<bg blue>"),
    Log_Type.SAVE: ("TPTBOX_SAVE", 22, "<light-cyan>"),
    Log_Type.OK: ("TPTBOX_OK", 25, "<light-green>"),
    Log_Type.WARNING: ("TPTBOX_WARNING", 30, "<light-yellow>"),
    Log_Type.WARNING_THROW: ("TPTBOX_WARNING_THROW", 30, "<light-yellow>"),
    Log_Type.FAIL: ("TPTBOX_FAIL", 40, "<light-red>"),
}

_LEVEL_ALIASES: dict[str, int] = {
    "TRACE": 5,
    "DEBUG": 10,
    "INFO": 20,
    "SUCCESS": 25,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_FILE_FORMAT = "{message}"

_configured = False
_current_level: int = 0
_stdout_lock = threading.Lock()


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


def _resolve_level(level: int | str) -> int:
    """Translate an integer/string/Log_Type level to a numeric severity threshold."""
    if isinstance(level, Log_Type):
        return _LEVELS[level][1]
    if isinstance(level, int):
        return level
    key = str(level).strip().upper()
    if key in _LEVEL_ALIASES:
        return _LEVEL_ALIASES[key]
    for name, no, _ in _LEVELS.values():
        if name == key or name.removeprefix("TPTBOX_") == key:
            return no
    raise ValueError(f"Unknown log level: {level!r}")


def _set_verbosity_level(level: int | str) -> None:
    """Set the global severity threshold for terminal output.

    Records with severity below this threshold are dropped by :func:`emit_terminal`.
    File sinks are not affected — use Loguru's per-sink ``level=`` when adding files.

    Args:
        level: Either a numeric severity (e.g. ``20``), a standard Loguru level name
            (``"WARNING"``, ``"ERROR"``, ...), or a TPTBox level name (``"TPTBOX_OK"``,
            ``"OK"``).
    """
    global _current_level  # noqa: PLW0603
    _current_level = _resolve_level(level)


def current_level() -> int:
    """Return the current global terminal severity threshold."""
    return _current_level


def configure(take_over: bool | None = None) -> None:
    """Configure the Loguru side of the logger (idempotent).

    Terminal output does NOT go through Loguru; this function only sets up levels and
    optionally strips Loguru's default stderr handler so :func:`emit_file` and
    :func:`emit_exception` records (which DO go through Loguru) don't get echoed with
    the ``time | LEVEL | module:func:line - msg`` default format.

    Args:
        take_over: If True (default), remove Loguru's built-in stderr handler (id 0)
            so the default file:function:line format never appears alongside TPTBox
            output. Any sinks a host application registered *before* importing TPTBox
            are preserved. If False, leave every existing Loguru handler alone. If
            None, read ``TPTBOX_LOGGER_TAKEOVER`` (default ``1``).
    """
    global _configured  # noqa: PLW0603
    _ensure_levels()
    env_level = os.environ.get("TPTBOX_LOG_LEVEL")
    if env_level:
        try:
            _set_verbosity_level(env_level)
        except ValueError:
            pass
    if _configured:
        return
    if take_over is None:
        take_over = os.environ.get("TPTBOX_LOGGER_TAKEOVER", "1") not in ("0", "false", "False")
    if take_over:
        # Remove only Loguru's built-in default stderr handler; leave anything a host
        # application already attached in place. `remove(0)` raises ValueError if id 0
        # was already removed by the user — treat that as a no-op.
        try:
            logger.remove(0)
        except ValueError:
            pass
    _configured = True

    # Env-var opt-ins, applied on first configure() only.
    from TPTBox.logger.log_file import _set_logger_silence  # lazy: avoids circular import

    for env_name, hook in (
        ("TPTBOX_SILENT", _set_logger_silence),
        ("TPTBOX_CAPTURE_WARNINGS", _install_warnings_hook),
        ("TPTBOX_CAPTURE_EXCEPTIONS", _install_excepthook),
    ):
        if _env_truthy(env_name):
            hook()


def _env_truthy(name: str) -> bool:
    """Return True if env var ``name`` is set to a truthy value."""
    return os.environ.get(name, "0") not in ("0", "", "false", "False", "no", "No")


def _timestamp() -> str:
    """Return the current wall-clock time as ``YYYY-MM-DD HH:MM:SS``."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_terminal_line(text: str, ltype: Log_Type = Log_Type.TEXT) -> str:
    """Return ``text`` wrapped with the ANSI color of ``ltype`` (no trailing newline).

    Empty ``text`` is returned unchanged so ``logger.print()`` prints a clean blank line.
    """
    if text == "":
        return ""
    color = type2bcolors.get(ltype, (bcolors.ENDC, ""))[0]
    if not color or color == bcolors.ENDC:
        return text
    return f"{color}{text}{bcolors.ENDC}"


def emit_terminal(text: str, ltype: Log_Type = Log_Type.TEXT, end: str = "\n") -> None:
    r"""Write one already-prefixed line directly to :data:`sys.stdout`.

    Bypasses Loguru's global routing entirely so the terminal output is not affected by
    whatever handlers a host application has active. Uses our own ANSI color for
    ``ltype`` (matching the legacy ``type2bcolors`` mapping) and honors ``end`` so
    ``end="\r"`` progress lines stay on one line.
    """
    if _LEVELS.get(ltype, _LEVELS[Log_Type.TEXT])[1] < _current_level:
        return
    line = format_terminal_line(text, ltype)
    with _stdout_lock:
        # Look up sys.stdout at write-time (not at add-time) so redirect_stdout / capsys work.
        sys.stdout.write(line + end)


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


def emit_exception(message: str, ltype: Log_Type = Log_Type.FAIL) -> None:
    """Emit a structured record carrying the *active* exception, for user sinks.

    The human-readable traceback text is still emitted separately by the facade
    (``print_error``). This extra record is bound to the ``"exception"`` channel so the
    default terminal/file sinks ignore it (no double traceback there); a user sink added
    with ``serialize=True`` (or any permissive filter) receives the structured exception.
    """
    if not _configured:
        configure()
    logger.opt(exception=True).bind(tptbox_channel="exception").log(level_name(ltype), message)


def _install_excepthook() -> None:
    """Route uncaught exceptions through Loguru (opt-in; replaces ``sys.excepthook``)."""

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.opt(exception=(exc_type, exc_value, exc_tb)).bind(tptbox_channel="exception").log(
            level_name(Log_Type.FAIL), "Uncaught exception"
        )

    sys.excepthook = _hook


_original_showwarning = None


def _install_warnings_hook() -> None:
    """Route ``warnings.warn(...)`` through the TPTBox logger (idempotent).

    Overrides :func:`warnings.showwarning` so every warning is emitted as a WARNING-level
    line via the module-wide default logger AND as a structured Loguru record on the
    ``"warning"`` channel (for user sinks that ``.add(..., serialize=True)``).

    Call :func:`_restore_warnings_hook` to undo.
    """
    global _original_showwarning  # noqa: PLW0603
    import warnings

    from TPTBox.logger.log_file import get_default_logger

    if _original_showwarning is not None:
        return
    _original_showwarning = warnings.showwarning

    def _show(message, category, filename, lineno, file=None, line=None):  # noqa: ARG001
        text = f"{category.__name__}: {message}  ({filename}:{lineno})"
        get_default_logger().on_warning(text)
        if _configured:
            logger.bind(tptbox_channel="warning").log(level_name(Log_Type.WARNING), text)

    warnings.showwarning = _show


def _restore_warnings_hook() -> None:
    """Undo :func:`_install_warnings_hook`. No-op if the hook is not installed."""
    global _original_showwarning  # noqa: PLW0603
    import warnings

    if _original_showwarning is None:
        return
    warnings.showwarning = _original_showwarning
    _original_showwarning = None


# Configure eagerly at import: strips Loguru's built-in default handler (id 0) so
# structured records (`emit_file` / `emit_exception`) don't double-print through
# Loguru's `time | LEVEL | file:func:line - msg` format. Sinks a host application
# added *before* importing TPTBox survive. Set TPTBOX_LOGGER_TAKEOVER=0 before
# importing TPTBox to keep Loguru's default handler in place.
configure()
