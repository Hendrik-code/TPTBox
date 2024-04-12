from __future__ import annotations

import time
import traceback
import warnings
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

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
    """General Logger Interface"""

    prefix: str | None = None

    def print(
        self,
        *text,
        end="\n",
        ltype=Log_Type.TEXT,
        verbose: bool | None = None,
        ignore_prefix: bool = False,
    ):
        """
        Args:
            *text: Text to the printed/logged
            end: end char (default: newline)
            ltype: log type (Text, Warning,...). If it is contained in *text and not here, it will still work
            verbose: true/false: prints to terminal (If none, uses default verbose)
            ignore_prefix: If false, will set a prefix character based on Log_Type (e.g. [*], [!], ...)


        logger.print() without arguments will result in an empty newline
        """
        if verbose is None:
            verbose = getattr(self, "default_verbose", False)
        if len(text) == 0 or text == [""] or text == "" or text is None:
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
        """Processes given text parts, converting manually specified datatypes, and adds the prefix and ltype corresponding color

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
            string = prefix + " " + string

        return string

    def __enter__(self):
        _set_indent(True)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        _set_indent(False)

    def _prefix_indentation_level(self) -> str:
        """Returns the indentation as processed string

        Returns:
            str: indentantion string
        """
        global indentation_level  # noqa: PLW0602
        string = ""
        if indentation_level == 0:
            return string
        string = " " + "-" * ((indentation_level * 3) - 2) + " "
        return string

    def _get_logger_prefix(self, ltype: Log_Type = Log_Type.TEXT):
        """Returns the prefix based on indentation level and log_type

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

    def close(self): ...

    def flush(self): ...

    def flush_sub_logger(self, sublogger: String_Logger, closed=False): ...

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

    def print_error(self, **args):
        self.print(traceback.format_exc(), ltype=Log_Type.FAIL, **args)

    logging_state = None

    def log_statistic(self, key, value, key2=None, verbose=True, round_print: int | None = 5):
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

    def print_statistic(self):
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

    def on_fail(self, *text, end="\n", verbose: bool | None = None, **qargs):
        self.print(*text, end=end, ltype=Log_Type.FAIL, verbose=verbose, **qargs)

    def on_save(self, *text, end="\n", verbose: bool | None = None, **qargs):
        self.print(*text, end=end, ltype=Log_Type.SAVE, verbose=verbose, **qargs)

    def on_debug(self, *text, end="\n", verbose: bool | None = None, **qargs):
        self.print(*text, end=end, ltype=Log_Type.STRANGE, verbose=verbose, **qargs)

    def on_ok(self, *text, end="\n", verbose: bool | None = None, **qargs):
        self.print(*text, end=end, ltype=Log_Type.OK, verbose=verbose, **qargs)

    def on_neutral(self, *text, end="\n", verbose: bool | None = None, **qargs):
        self.print(*text, end=end, ltype=Log_Type.NEUTRAL, verbose=verbose, **qargs)

    def on_warning(self, *text, end="\n", verbose: bool | None = None, **qargs):
        self.print(*text, end=end, ltype=Log_Type.WARNING, verbose=verbose, **qargs)

    def on_text(self, *text, end="\n", verbose: bool | None = None, **qargs):
        self.print(*text, end=end, ltype=Log_Type.TEXT, verbose=verbose, **qargs)


class Logger(Logger_Interface):
    """
    Defines a logger object, that automatically creates a logs folder and file in it. Logs logger.print() calls to this file.
    """

    def __init__(
        self,
        path: Path | str,
        log_filename: str | dict[str, str],
        default_verbose: bool = False,
        log_arguments=None,
        prefix: str | None = None,
    ):
        """

        Args:
            path: path to the folder that needs logging (usual dataset with raw/der in it)
            log_filename: the filename or the bids-conform key-value pairs as dict
            default_verbose: default verbose behavior of not specified in calls
            log_arguments: if set, will print the contents in a "run with arguments" section
            prefix: if set, will use this string as prefix instead of the automatically chosen one based on log_type
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
        # Open log file
        self.f = open(log_path.joinpath(log_filename_full), "w")  # noqa: SIM115
        # calls close() if program terminates
        self._finalizer = weakref.finalize(self.f, self.close)
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
        """Creates a logger object based on metadata from a BIDS_FILE

        Args:
            bids_file (BIDS_FILE): _description_
            log_filename (str | dict[str, str]): _description_
            default_verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        path = bids_file.dataset
        return Logger(path, log_filename, default_verbose=default_verbose, prefix=override_prefix)

    def _log(self, text: str, end: str = "\n", ltype=Log_Type.TEXT):  # noqa: ARG002
        self.f.write(str(text))
        self.f.write(end)

    def flush(self):
        self.f.flush()

    def flush_sub_logger(self, sublogger: String_Logger, closed=False):
        if sublogger in self.sub_loggers:
            self.sub_loggers.remove(sublogger) if closed else None
            self.print("Flushed sub logger:", verbose=False, ltype=Log_Type.LOG)
            self.print(sublogger.log_content, verbose=False)
            # Clear Sublogger
            sublogger.log_content = ""

    def remove(self):
        self._finalizer()

    def close(self):
        if not self.f.closed:
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
            self.f.flush()
            self.f.close()

    @property
    def removed(self):
        return not self._finalizer.alive


class No_Logger(Logger_Interface):
    """
    Does not create any logs, but instead verbose defaults to true, printing calls to the terminal
    """

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

    def _log(self, text: str, end: str = "\n", ltype=Log_Type.TEXT):
        pass  # self.print()

    def flush(self):
        pass

    def flush_sub_logger(self, sublogger: String_Logger, closed=False):
        pass

    def close(self):
        pass

    def add_sub_logger(self, name: str, default_verbose: bool = False) -> Logger_Interface:  # noqa: ARG002
        return self


class Reflection_Logger(No_Logger):
    """
    Does not create any logs, but instead verbose defaults to true, printing calls to the terminal. Accepts verbose to be a Logger_Interface. In that case, the given Logger_Interface calls their print function
    """

    def print(
        self,
        *text,
        end="\n",
        ltype=Log_Type.NEUTRAL,
        verbose: bool | Logger_Interface | None = True,
        ignore_prefix: bool = False,
    ):
        """
        Args:
            *text: Text to the printed/logged
            end: end char
            type: log type (Text, Warning,...)
            verbose: true/false or other Logger: prints to terminal (If none, uses default verbose)
            ignore_prefix: If false, will set a prefix character based on Log_Type (e.g. [*], [!], ...)

        Returns:

        """
        if isinstance(verbose, bool) or verbose is None:
            super().print(*text, end=end, ltype=ltype, verbose=verbose, ignore_prefix=ignore_prefix)
        else:
            verbose.print(*text, end=end, ltype=ltype, ignore_prefix=ignore_prefix)


class String_Logger(Logger_Interface):
    """
    Logger that logs only to a string object "log_content".
    """

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
        sub_logger = String_Logger(default_verbose=default_verbose, finalize=False)
        sub_logger.head_logger = head_logger
        return sub_logger

    def _log(self, text: str, end: str = "\n", ltype=Log_Type.TEXT):
        self.log_content += text
        self.log_content += end
        self.log_content_colored += color_log_text(ltype=ltype, text=text + end)

    def flush(self):
        if self.head_logger is not None:
            self.head_logger.flush_sub_logger(self, closed=False)

    def close(self):
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

    def flush_sub_logger(self, sublogger, closed=False):
        pass


#####################################
# Utils
#####################################


def print_to_terminal(text: str, end: str, ltype=Log_Type.TEXT) -> None:
    if ltype == Log_Type.WARNING_THROW:
        warnings.warn(color_log_text(ltype=ltype, text=text), Warning, stacklevel=3)
    else:
        print(color_log_text(ltype=ltype, text=text), end=end)


def sub_log_call_func(name, logger: Logger, function, default_verbose: bool | None = None, **kwargs):
    if default_verbose is None:
        default_verbose = logger.default_verbose
    sub_logger = logger.add_sub_logger(name, default_verbose=default_verbose)
    return function(name=name, logger=sub_logger, **kwargs)


def _set_indent(indent_change: int | bool):
    """Changes the indentation level

    Args:
        indent_change (int | bool): If bool, true/false == +1/-1, if int, sets to that int
    """
    global indentation_level  # noqa: PLW0603
    if isinstance(indent_change, bool):
        indentation_level = indentation_level + 1 if indent_change else max(0, indentation_level - 1)
    else:
        indentation_level = indent_change
    return indentation_level
