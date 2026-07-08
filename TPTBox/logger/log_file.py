from __future__ import annotations

import time
import traceback
import warnings
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from TPTBox.logger import _loguru_backend as _backend
from TPTBox.logger.log_constants import (
    Log_Type,
    _clean_all_color_from_text,
    _convert_seconds,
    color_log_text,
    datatype_to_string,
    format_time_short,
    get_time,
    type2bcolors,
)

if TYPE_CHECKING:
    from TPTBox import BIDS_FILE

indentation_level: int = 0


class Logger_Interface(Protocol):
    """Structural protocol defining the logging interface for medical imaging pipelines.

    All concrete logger classes (``Logger``, ``No_Logger``, ``String_Logger``,
    ``Reflection_Logger``) implement this protocol.  Client code should type-hint
    against ``Logger_Interface`` to remain independent of the concrete
    implementation.
    """

    prefix: str | None = None

    def print(
        self,
        *text,
        end="\n",
        ltype=Log_Type.TEXT,
        verbose: bool | None = None,
        ignore_prefix: bool = False,
    ) -> None:
        """Print text to the logger and optionally to the terminal.

        Args:
            *text: Text to be printed/logged.
            end: End char (default: newline).
            ltype: Log type (Text, Warning,...). If it is contained in *text and not here, it will still work.
            verbose: true/false: prints to terminal (if None, uses default verbose).
            ignore_prefix: If False, will set a prefix character based on Log_Type (e.g. [*], [!], ...).
        """
        if verbose is None:
            verbose = getattr(self, "default_verbose", False)
        if len(text) == 0 or text in ([""], "") or text is None:
            ignore_prefix = True
            string: str = ""
        else:
            log_type_in_text = [t for t in text if isinstance(t, Log_Type)]
            if len(log_type_in_text) > 0 and ltype == Log_Type.TEXT:
                ltype = log_type_in_text[0]

            string = self._preprocess_text(text, ltype=ltype, ignore_prefix=ignore_prefix)
        if verbose:
            print_to_terminal(string, end=end, ltype=ltype)
        self._log(_clean_all_color_from_text(string), end=end, ltype=ltype)

    def _preprocess_text(self, text: tuple[str, ...], ltype=Log_Type.TEXT, ignore_prefix: bool = False) -> str:
        """Processes given text parts, converting manually specified datatypes, and adds the prefix and ltype corresponding color.

        Layout: ``[prefix] YYYY-MM-DD HH:MM:SS body`` — the ``[]`` prefix always comes
        first, then the timestamp, then the message body. No level name is emitted.

        Args:
            text (tuple[str, ...]): _description_
            type (_type_, optional): _description_. Defaults to Log_Type.TEXT.
            ignore_prefix (bool, optional): _description_. Defaults to False.

        Returns:
            str: _description_
        """
        text_list: list[str] = [datatype_to_string(t, ltype) for t in text if not isinstance(t, Log_Type)]
        string = str.join(" ", text_list)

        if "[" not in string[:3] and not ignore_prefix:
            prefix = self._get_logger_prefix(ltype)
            string = prefix + " " + _backend._timestamp() + " " + string

        return string

    def __enter__(self):
        _set_indent(True)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        _set_indent(False)

    def _prefix_indentation_level(self) -> str:
        """Returns the indentation as processed string.

        Returns:
            str: indentantion string
        """
        string = ""
        if indentation_level == 0:
            return string
        string = " " + "-" * ((indentation_level * 3) - 2) + " "
        return string

    def _get_logger_prefix(self, ltype: Log_Type = Log_Type.TEXT):
        """Returns the prefix based on indentation level and log_type.

        Args:
            type (Log_Type, optional): _description_. Defaults to Log_Type.TEXT.

        Returns:
            _type_: _description_
        """
        indent: str = self._prefix_indentation_level()
        if self.prefix is not None:
            return indent + f"[{self.prefix}]"
        return indent + type2bcolors[ltype][1]

    def _log(self, text: str, end: str = "\n", ltype=Log_Type.TEXT): ...

    def close(self) -> None:
        """Release any resources held by this logger."""
        ...

    def flush(self) -> None:
        """Flush any buffered log content."""
        ...

    def flush_sub_logger(self, sublogger: String_Logger, closed: bool = False) -> None:
        """Flush or close a sub-logger attached to this logger."""
        ...

    def add_sub_logger(self, name: str, default_verbose: bool = False) -> String_Logger | No_Logger:
        """Creates a sub-logger that only logs to string. Will be appended in this loggers log file as sub-logger.

        Args:
            name: name of the sub-logger
            default_verbose: default_verbose attribute for the sub-logger

        Returns:
            sub_logger: String_Logger
        """
        if hasattr(self, "sub_loggers"):
            sub_logger = String_Logger.as_sub_logger(head_logger=self, default_verbose=default_verbose)
            self.sub_loggers.append(sub_logger)  # type: ignore
            sub_logger.print(
                "Sub-logger: ",
                name,
                verbose=False,
                ignore_prefix=False,
                ltype=Log_Type.LOG,
            )
            return sub_logger
        else:
            return self  # type: ignore

    def print_error(self, **args) -> None:
        """Log the current exception traceback with ``Log_Type.FAIL`` severity.

        Also emits a structured Loguru record carrying the active exception so user sinks
        (e.g. ``logger.add(..., serialize=True)``) can capture it; the human-readable
        traceback text below is unchanged.
        """
        self.print(traceback.format_exc(), ltype=Log_Type.FAIL, **args)
        _backend.emit_exception("Logged exception")

    logging_state = None

    def log_statistic(
        self, key: str, value: float, key2: str | int | None = None, verbose: bool = True, round_print: int | None = 5
    ) -> None:
        """Record a scalar metric and log it as a neutral message.

        Values are accumulated under ``key`` / ``key2`` and can later be
        summarized with :meth:`print_statistic`.

        Args:
            key: Primary metric name (e.g. ``"dice"``).
            value: Numeric value to record.
            key2: Secondary key (e.g. sample name). Defaults to the current
                count for ``key`` when None.
            verbose: If True, prints the entry immediately.
            round_print: Number of decimal places for display. Set to None to
                disable rounding.
        """
        if self.logging_state is None:
            self.logging_state = {}
        if key not in self.logging_state:
            self.logging_state[key] = {}
        if key2 is None:
            key2 = len(self.logging_state[key])

        self.logging_state[key][key2] = value
        if round_print is not None and isinstance(value, (float, np.floating)):
            value = np.round(value, decimals=round_print)
        self.on_neutral(f"{key2:10}: {key:17} = {value}", verbose=verbose)

    def _print_by_logger(self, *text, end="\n", verbose: bool | None = None, **qargs):
        self.print(*text, end=end, ltype=Log_Type.LOG, verbose=verbose, **qargs)

    def print_statistic(self) -> None:
        """Print a summary table of all accumulated statistics to the log.

        Computes and logs the mean, standard deviation, median, and count for
        every key previously recorded via :meth:`log_statistic`.
        """
        if self.logging_state is None:
            self._print_by_logger("??? No Accumulated Statistics ???")
            return
        self._print_by_logger("############## Accumulated Statistics ##############")
        self._print_by_logger(f"{'key':17} {'mean':8}{'std':9} {'median':8} {'  count':7}")
        for k, v in self.logging_state.items():
            values = np.array(list(v.values()))
            mean = f"{np.mean(values):.3}"
            std = f"{np.std(values):.3}"
            median = f"{np.median(values):.3}"
            count = len(values)
            self._print_by_logger(f"{k:17} {mean:8}±{std:8} {median:8} {count:7}")
        self._print_by_logger("####################################################")

    def on_fail(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a message with ``Log_Type.FAIL`` severity."""
        self.print(*text, end=end, ltype=Log_Type.FAIL, verbose=verbose, **qargs)

    def on_log(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a message with ``Log_Type.LOG`` severity."""
        self.print(*text, end=end, ltype=Log_Type.LOG, verbose=verbose, **qargs)

    def on_bold(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a message with ``Log_Type.BOLD`` formatting."""
        self.print(*text, end=end, ltype=Log_Type.BOLD, verbose=verbose, **qargs)

    def on_save(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a message with ``Log_Type.SAVE`` severity."""
        self.print(*text, end=end, ltype=Log_Type.SAVE, verbose=verbose, **qargs)

    def on_debug(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a message with ``Log_Type.STRANGE`` (debug) severity."""
        self.print(*text, end=end, ltype=Log_Type.STRANGE, verbose=verbose, **qargs)

    def on_ok(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a message with ``Log_Type.OK`` severity."""
        self.print(*text, end=end, ltype=Log_Type.OK, verbose=verbose, **qargs)

    def on_neutral(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a message with ``Log_Type.NEUTRAL`` severity."""
        self.print(*text, end=end, ltype=Log_Type.NEUTRAL, verbose=verbose, **qargs)

    def on_warning(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a message with ``Log_Type.WARNING`` severity."""
        self.print(*text, end=end, ltype=Log_Type.WARNING, verbose=verbose, **qargs)

    def on_text(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a message with default ``Log_Type.TEXT`` severity."""
        self.print(*text, end=end, ltype=Log_Type.TEXT, verbose=verbose, **qargs)

    # same logging as the python loger for drop in replacement
    def warning(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log a warning message (drop-in for Python's ``logging.warning``)."""
        return self.on_warning(*text, end=end, verbose=verbose, **qargs)

    def error(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log an error message (drop-in for Python's ``logging.error``)."""
        return self.on_fail(*text, end=end, verbose=verbose, **qargs)

    def info(self, *text, end: str = "\n", verbose: bool | None = None, **qargs) -> None:
        """Log an informational message (alias for :meth:`on_text`)."""
        return self.on_text(*text, end=end, verbose=verbose, **qargs)


class Logger(Logger_Interface):
    """Defines a logger object, that automatically creates a logs folder and file in it. Logs logger.print() calls to this file."""

    def __init__(
        self,
        path: Path | str,
        log_filename: str | dict[str, str],
        default_verbose: bool = False,
        log_arguments=None,
        prefix: str | None = None,
        rotation=None,
        retention=None,
        enqueue: bool = False,
    ):
        """Initialise a file-backed logger, creating the log directory and file automatically.

        Args:
            path: Path to the folder that needs logging (usual dataset with raw/der in it).
            log_filename: The filename or the bids-conform key-value pairs as dict.
            default_verbose: Default verbose behavior when not specified in calls.
            log_arguments: If set, will print the contents in a "run with arguments" section.
            prefix: If set, will use this string as prefix instead of the automatically chosen one.
            rotation: Optional Loguru rotation policy (e.g. ``"10 MB"``, ``"00:00"``). When set
                (or ``retention``), Loguru owns the file and rotates it.
            retention: Optional Loguru retention policy (e.g. ``"10 days"``, ``5``).
            enqueue: If True, file writes go through a background thread (thread/process-safe).
        """
        path = Path(path)  # ensure pathlib object
        # Get Start time
        self.start_time = get_time()
        start_time_short = format_time_short(self.start_time)

        self.prefix = prefix

        # Processes log_filename
        log_filename_processed = ""
        if isinstance(log_filename, dict):
            for k, v in log_filename.items():
                log_filename_processed += k + "-" + v + "_"
        else:
            log_filename_processed = log_filename + "_"
        log_filename_full = start_time_short + "_" + log_filename_processed + "log.log"

        # Creates logs folder if not existent
        log_path = Path(path).joinpath("logs")
        if not Path.exists(log_path):
            Path.mkdir(log_path)
        # Open log file -> route writes through a Loguru sink.
        self._filepath = log_path.joinpath(log_filename_full)
        self._file_key = id(self)
        self._closed = False
        if rotation is None and retention is None:
            # Default: own the handle (flush() works, `end` honored, atexit-safe duration line).
            self.f = open(self._filepath, "w")  # noqa: SIM115
            self._sink_id = _backend.add_file_stream_sink(self.f, self._file_key, enqueue=enqueue)
        else:
            # Loguru owns the file -> rotation/retention available (lines are \n-terminated).
            self.f = None
            self._sink_id = _backend.add_file_sink(self._filepath, self._file_key, rotation=rotation, retention=retention, enqueue=enqueue)
        # calls close() at interpreter exit (matches the previous atexit cleanup)
        self._finalizer = weakref.finalize(self, self.close)
        self.default_verbose = default_verbose
        # Log file always start with their name and start log time
        self.print(log_filename_processed[:-1], verbose=False, ltype=Log_Type.LOG)
        self.print(f"Log started at: {start_time_short}\n", ltype=Log_Type.LOG)

        if log_arguments is not None:
            if not isinstance(log_arguments, dict):
                self.print("Run with arguments", log_arguments, "\n", ltype=Log_Type.LOG)
            else:
                self.print("Run with arguments:", ltype=Log_Type.LOG)
                for k, v in log_arguments.items():
                    self.print("-", k, "=", v, Log_Type.LOG, ignore_prefix=True)
                self.print(ignore_prefix=True)

        self.sub_loggers: list[String_Logger] = []

    @classmethod
    def create_from_bids(
        cls,
        bids_file: BIDS_FILE,
        log_filename: str | dict[str, str],
        default_verbose: bool = False,
        override_prefix: str | None = None,
    ):
        """Creates a logger object based on metadata from a BIDS_FILE.

        Args:
            bids_file (BIDS_FILE): _description_
            log_filename (str | dict[str, str]): _description_
            default_verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        path = bids_file.dataset
        return Logger(path, log_filename, default_verbose=default_verbose, prefix=override_prefix)

    def _log(self, text: str, end: str = "\n", ltype=Log_Type.TEXT) -> None:
        """Write a plain-text line to the log file (via the Loguru file sink)."""
        _backend.emit_file(str(text), self._file_key, ltype, end)

    def flush(self) -> None:
        """Flush the underlying file buffer to disk (Loguru file sinks flush per write)."""
        if self.f is not None:
            self.f.flush()

    def flush_sub_logger(self, sublogger: String_Logger, closed: bool = False) -> None:
        """Append a sub-logger's accumulated content into this log file.

        Args:
            sublogger: The sub-logger whose content should be flushed.
            closed: If True, also removes ``sublogger`` from the tracked list.
        """
        if sublogger in self.sub_loggers:
            self.sub_loggers.remove(sublogger) if closed else None
            self.print("Flushed sub logger:", verbose=False, ltype=Log_Type.LOG)
            self.print(sublogger.log_content, verbose=False)
            # Clear Sublogger
            sublogger.log_content = ""

    def remove(self) -> None:
        """Trigger the finalizer to close and release the log file."""
        self._finalizer()

    def close(self) -> None:
        """Flush all sub-loggers, write timing information, and close the log file."""
        if not self._closed:
            self._closed = True
            self.sub_loggers = [s for s in self.sub_loggers if s.log_content != ""]
            if len(self.sub_loggers) > 0:
                self.print(ignore_prefix=True)
                self.print(
                    f"Found {len(self.sub_loggers)} sub logger:",
                    verbose=False,
                    ltype=Log_Type.LOG,
                )
                for tl in self.sub_loggers:
                    self.print(tl.log_content, verbose=False)

            end_time = get_time()
            duration = time.mktime(end_time) - time.mktime(self.start_time)
            self.print(ignore_prefix=True)
            self.print("Program duration:", _convert_seconds(duration), ltype=Log_Type.LOG)
            self.print(
                f"Log ended at: {format_time_short(end_time)}",
                verbose=False,
                ltype=Log_Type.LOG,
            )
            # remove the sink AFTER the closing lines were emitted, then release our handle
            _backend.remove_sink(self._sink_id)
            if self.f is not None:
                self.f.flush()
                self.f.close()

    @property
    def removed(self) -> bool:
        """True if the log file has been closed and its finalizer is no longer alive."""
        return not self._finalizer.alive


class No_Logger(Logger_Interface):
    """Does not create any logs, but instead verbose defaults to true, printing calls to the terminal."""

    def __init__(
        self,
        print_log_started: bool = False,
        prefix: str | None = None,
    ):
        self.default_verbose = True
        self.prefix = prefix

        if print_log_started:
            self.start_time = get_time()
            start_time_short = format_time_short(self.start_time)
            self.print(f"Log started at: {start_time_short}\n", ltype=Log_Type.LOG)

    def _log(self, text: str, end: str = "\n", ltype: Log_Type = Log_Type.TEXT) -> None:
        """No-op: No_Logger does not persist log entries."""
        # self.print()

    def flush(self) -> None:
        """No-op: No_Logger has no buffer to flush."""

    def flush_sub_logger(self, sublogger: String_Logger, closed: bool = False) -> None:
        """No-op: No_Logger does not track sub-loggers."""

    def close(self) -> None:
        """No-op: No_Logger has no resources to release."""

    def add_sub_logger(self, name: str, default_verbose: bool = False) -> Logger_Interface:  # noqa: ARG002
        """Return self as No_Logger does not support sub-loggers."""
        return self


class Reflection_Logger(No_Logger):
    """Logger that prints to the terminal and optionally delegates to another Logger_Interface."""

    def print(
        self,
        *text,
        end="\n",
        ltype=Log_Type.NEUTRAL,
        verbose: bool | Logger_Interface | None = True,
        ignore_prefix: bool = False,
    ) -> None:
        """Print to terminal or delegate to a Logger_Interface passed as ``verbose``.

        Args:
            *text: Text to be printed/logged.
            end: End char.
            ltype: Log type (Text, Warning,...).
            verbose: true/false or another Logger; prints to terminal (if None, uses default verbose).
            ignore_prefix: If False, will set a prefix character based on Log_Type (e.g. [*], [!], ...).
        """
        if isinstance(verbose, bool) or verbose is None:
            super().print(*text, end=end, ltype=ltype, verbose=verbose, ignore_prefix=ignore_prefix)
        else:
            verbose.print(*text, end=end, ltype=ltype, ignore_prefix=ignore_prefix)


class String_Logger(Logger_Interface):
    """Logger that logs only to a string object "log_content"."""

    def __init__(
        self,
        default_verbose: bool = False,
        finalize: bool = True,
        prefix: str | None = None,
    ):
        self.default_verbose = default_verbose
        self.prefix = prefix
        self.log_content = ""
        self.log_content_colored = ""
        self.sub_loggers: list[String_Logger] = []
        self.start_time = get_time()
        self.head_logger: Logger_Interface | None = None
        if finalize:
            self._finalizer = weakref.finalize(self, self.close)

    @classmethod
    def as_sub_logger(cls, head_logger: Logger_Interface, default_verbose: bool = False) -> String_Logger:
        """Create a ``String_Logger`` that is wired to a parent ``Logger_Interface``.

        When flushed or closed the content is forwarded to ``head_logger``.

        Args:
            head_logger: Parent logger that will receive this sub-logger's content.
            default_verbose: Default verbosity for calls on the sub-logger.

        Returns:
            A new ``String_Logger`` instance with ``head_logger`` set.
        """
        sub_logger = String_Logger(default_verbose=default_verbose, finalize=False)
        sub_logger.head_logger = head_logger
        return sub_logger

    def _log(self, text: str, end: str = "\n", ltype: Log_Type = Log_Type.TEXT) -> None:
        """Append a log entry to both the plain-text and colored string buffers."""
        self.log_content += text
        self.log_content += end
        self.log_content_colored += color_log_text(ltype=ltype, text=text + end)

    def flush(self) -> None:
        """Forward accumulated content to the head logger without closing."""
        if self.head_logger is not None:
            self.head_logger.flush_sub_logger(self, closed=False)

    def close(self) -> tuple[str, str]:
        """Finalize the sub-logger and forward its content to the head logger.

        Returns:
            A 2-tuple of ``(log_content, log_content_colored)`` strings.
        """
        if len(self.sub_loggers) > 0:
            self.print()
            self.print(
                f"Found {len(self.sub_loggers)} sub logger:",
                verbose=None,
                ltype=Log_Type.LOG,
                ignore_prefix=True,
            )
            for tl in self.sub_loggers:
                self.print(tl.log_content, verbose=None)
                tl.close()

        end_time = get_time()
        duration = time.mktime(end_time) - time.mktime(self.start_time)
        self.print("Sub-process duration:", _convert_seconds(duration), ltype=Log_Type.LOG)
        # self.print(f"Log ended at: {format_time_short(end_time)}", verbose=False, type=Log_Type.LOG)
        if self.head_logger is not None:
            self.head_logger.flush_sub_logger(self, closed=True)
            self.head_logger.flush()
        return self.log_content, self.log_content_colored

    def flush_sub_logger(self, sublogger: String_Logger, closed: bool = False) -> None:
        """No-op: String_Logger does not flush nested sub-loggers."""


#####################################
# Utils
#####################################


def print_to_terminal(text: str, end: str, ltype: Log_Type = Log_Type.TEXT) -> None:
    r"""Print colored text to the terminal, or emit a warning for WARNING_THROW types.

    Args:
        text: The message string to display.
        end: Line ending character (e.g. ``"\n"``).
        ltype: Log type that controls the ANSI color applied to ``text``.
    """
    if ltype == Log_Type.WARNING_THROW:
        warnings.warn(color_log_text(ltype=ltype, text=text), Warning, stacklevel=3)
    else:
        # Route through Loguru (native level-driven coloring). The facade already built the
        # exact prefixed text; the backend applies the matching color and honors `end`.
        _backend.emit_terminal(text, ltype, end)


def sub_log_call_func(name: str, logger: Logger, function, default_verbose: bool | None = None, **kwargs) -> object:
    """Call a function inside a sub-logger context and return its result.

    Creates a sub-logger from ``logger``, then calls ``function`` passing
    ``name`` and the sub-logger as keyword arguments.

    Args:
        name: Label for the sub-logger.
        logger: Parent logger from which the sub-logger is created.
        function: Callable to invoke. Must accept ``name`` and ``logger`` as
            keyword arguments.
        default_verbose: Verbosity for the sub-logger. Inherits from ``logger``
            when None.
        **kwargs: Additional keyword arguments forwarded to ``function``.

    Returns:
        Whatever ``function`` returns.
    """
    if default_verbose is None:
        default_verbose = logger.default_verbose
    sub_logger = logger.add_sub_logger(name, default_verbose=default_verbose)
    return function(name=name, logger=sub_logger, **kwargs)


def _set_indent(indent_change: int | bool):
    """Changes the indentation level.

    Args:
        indent_change (int | bool): If bool, true/false == +1/-1, if int, sets to that int
    """
    global indentation_level  # noqa: PLW0603
    if isinstance(indent_change, bool):
        indentation_level = indentation_level + 1 if indent_change else max(0, indentation_level - 1)
    else:
        indentation_level = indent_change
    return indentation_level


_default_logger: Logger_Interface = No_Logger()


def _set_default_logger(logger: Logger_Interface) -> None:
    """Install a module-wide default logger. Prefer :func:`configure_logging`."""
    global _default_logger  # noqa: PLW0603
    _default_logger = logger


def get_default_logger() -> Logger_Interface:
    """Return the module-wide default logger."""
    return _default_logger


def _set_logger_silence() -> None:
    """Silence every TPTBox printout.

    default logger becomes a non-verbose ``No_Logger`` and the global terminal severity threshold is raised above every real level.
    """
    from TPTBox.logger import _loguru_backend as _backend_mod

    _set_default_logger(No_Logger())
    _default_logger.default_verbose = False  # type: ignore[attr-defined]
    _backend_mod._set_verbosity_level(_backend_mod.SILENT_LEVEL)


class _LoggerProxy:
    """Attribute-forwarding proxy: every access resolves to the current default logger.

    Import once at module top-level:

        from TPTBox.logger import logger

    Then use it exactly like a ``Logger``:

        logger.on_ok("done")
        logger.on_fail("bad")
        logger.print("plain", ltype=Log_Type.LOG)

    ``logger.<name>`` always looks up ``<name>`` on whatever :func:`get_default_logger`
    currently returns — so :func:`configure_logging` reconfigures every callsite at
    once, without library code having to be re-imported or refactored.
    """

    __slots__ = ()

    def __getattr__(self, name: str):
        return getattr(_default_logger, name)

    def __repr__(self) -> str:
        return f"<TPTBox logger → {_default_logger!r}>"


logger = _LoggerProxy()
# `log` is kept as a backwards-compatible alias for the older `log.print(...)` idiom.
log = logger


def _capture_exceptions(fn):
    """Decorator: log any exception raised by ``fn`` through the default logger, then re-raise.

    The uncaught exception is captured via :func:`Logger_Interface.print_error`, which
    emits a colored traceback to the terminal AND a structured Loguru record so
    user-added sinks receive the exception too.
    """
    import functools

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            get_default_logger().print_error()
            raise

    return _wrapped


def configure_logging(
    file=None,
    filename=None,
    level=None,
    verbose=None,
    silent: bool = False,
    capture_warnings: bool | None = None,
    capture_exceptions: bool | None = None,
    loguru_sink=None,
) -> dict:
    """One-call TPTBox logging setup — the single entry point for configuring output.

    Every argument is optional; unset args leave the corresponding aspect untouched.
    Call with no arguments to inspect the current effective configuration.

    Args:
        file: Path to a log file, or a directory / dataset root (in which case a
            timestamped ``<start>_<filename>_log.log`` is created inside a ``logs/``
            subfolder). ``None`` leaves the current default logger in place.
        filename: The log file's stem when ``file`` is a directory / dataset root.
            Ignored when ``file`` is a full path or ``None``.
        level: Global terminal severity threshold. Accepts an int (loguru severity),
            a string (``"WARNING"``, ``"TPTBOX_OK"``, ``"INFO"``, ...), or a
            :class:`Log_Type`.
        verbose: Sets ``default_verbose`` on the default logger. When True, every
            ``logger.print(..., verbose=None)`` call inside TPTBox reaches the terminal;
            when False, only file sinks receive them. Independent of ``level``.
        silent: Shortcut for ``verbose=False`` plus terminal threshold above every
            real level. Nothing prints.
        capture_warnings: If True, install a hook so ``warnings.warn(...)`` is routed
            through the logger. If False, restore the previous behavior.
        capture_exceptions: If True, replace ``sys.excepthook`` so uncaught exceptions
            are routed through Loguru. If False, restore ``sys.__excepthook__``.
        loguru_sink: Optional dict of kwargs forwarded to ``loguru_logger.add(...)`` —
            e.g. ``{"sink": "run.jsonl", "serialize": True, "level": "TPTBOX_WARNING"}``.

    Returns:
        A dict with the resulting configuration: ``file``, ``level``, ``verbose``,
        ``capture_warnings``, ``capture_exceptions``.
    """
    import sys

    from TPTBox.logger import _loguru_backend as _backend_mod

    if file is not None:
        _set_default_logger(_build_logger(file, filename))

    if silent:
        _set_logger_silence()
    else:
        if level is not None:
            _backend_mod._set_verbosity_level(level)
        if verbose is not None:
            _default_logger.default_verbose = verbose  # type: ignore[attr-defined]

    if capture_warnings is True:
        _backend_mod._install_warnings_hook()
    elif capture_warnings is False:
        _backend_mod._restore_warnings_hook()

    if capture_exceptions is True:
        _backend_mod._install_excepthook()
    elif capture_exceptions is False:
        sys.excepthook = sys.__excepthook__

    if loguru_sink is not None:
        from TPTBox.logger._loguru_backend import logger as _loguru

        _loguru.add(**loguru_sink)

    return {
        "file": getattr(_default_logger, "_filepath", None),
        "level": _backend_mod.current_level(),
        "verbose": getattr(_default_logger, "default_verbose", None),
        "capture_warnings": _backend_mod._original_showwarning is not None,
        "capture_exceptions": sys.excepthook is not sys.__excepthook__,
    }


def _build_logger(file, filename) -> Logger:
    """Instantiate a :class:`Logger` from ``file`` (path-or-directory) and ``filename``.

    - ``file`` is a directory: use it as-is as the log dataset root, with ``filename`` as
      the log stem (defaults to ``"run"``).
    - ``file`` is a file path: use its parent as the root and its stem as the log stem
      (``filename`` overrides the stem if provided).
    """
    from pathlib import Path

    file = Path(file)
    if file.exists() and file.is_dir():
        return Logger(file, filename or "run")
    return Logger(file.parent or Path("."), filename or file.stem or "run")
