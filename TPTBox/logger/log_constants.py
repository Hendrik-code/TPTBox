from __future__ import annotations

import datetime
import time
from enum import Enum, auto
from time import struct_time


class Log_Type(Enum):
    """The different types of Logs supported."""

    TEXT = auto()
    NEUTRAL = auto()
    SAVE = auto()
    WARNING = auto()
    WARNING_THROW = auto()
    LOG = auto()
    OK = auto()
    FAIL = auto()
    Yellow = auto()
    STRANGE = auto()
    UNDERLINE = auto()
    ITALICS = auto()
    BOLD = auto()
    DOCKER = auto()
    TOTALSEG = auto()
    STAGE = auto()


class bcolors:
    """Terminal color symbols."""

    # Front Colors
    BLACK = "\033[30m"
    PINK = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    Yellow2 = "\033[33m"  # "\033[33m" <-- Yellow
    GRAY = "\033[37m"
    LightGray = "\033[37m"
    DarkGray = "\033[90m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    # Modes
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DISABLE = "\033[02m"
    STRIKETHROUGH = "\033[09m"
    REVERSE = "\033[07m"
    ITALICS = "\033[3m"
    # Background Colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_ORANGE = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_PURPLE = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_GRAY = "\033[47m"
    # End of line (cleans color)
    ENDC = "\033[0m"


# Defines for each Log_Type the corresponding color to be used as well as its prefix
# TODO make bcolors enum more fancy
type2bcolors: dict[Log_Type, tuple[str, str]] = {
    Log_Type.TEXT: (bcolors.ENDC, "[*]"),
    Log_Type.NEUTRAL: (bcolors.ENDC, "[ ]"),
    Log_Type.SAVE: (bcolors.CYAN, "[*]"),
    Log_Type.WARNING: (bcolors.YELLOW, "[?]"),
    Log_Type.WARNING_THROW: (bcolors.YELLOW, "[?]"),
    Log_Type.LOG: (bcolors.BLUE, "[#]"),
    Log_Type.OK: (bcolors.GREEN, "[+]"),
    Log_Type.FAIL: (bcolors.RED, "[!]"),
    Log_Type.Yellow: (bcolors.Yellow2, "[*]"),
    Log_Type.STRANGE: (bcolors.PINK, "[-]"),
    Log_Type.UNDERLINE: (bcolors.UNDERLINE, "[_]"),
    Log_Type.ITALICS: (bcolors.ITALICS, "[ ]"),
    Log_Type.BOLD: (bcolors.BOLD, "[*]"),
    Log_Type.DOCKER: (bcolors.ITALICS, "[Docker]"),
    Log_Type.TOTALSEG: (bcolors.ITALICS, "[TOTALSEG]"),
    Log_Type.STAGE: (bcolors.BG_BLUE, "[*]"),
}


def datatype_to_string(text: object, log_type: Log_Type) -> str:
    """Convert an arbitrary value to its loggable string representation.

    Dicts receive special colored formatting via :func:`_dict_to_string`; all
    other types are converted with ``str()``.

    Args:
        text: The value to convert.
        log_type: Log type used for dict colorization.

    Returns:
        A human-readable string representation of ``text``.
    """
    if isinstance(text, dict):
        return _dict_to_string(text, log_type)
    return str(text)


def _dict_to_string(u_dict: dict, ltype: Log_Type = Log_Type.TEXT) -> str:
    """Convert a dictionary into a colored, human-readable string for logging."""
    text = ""
    text += "{"
    for key, value in u_dict.items():
        if isinstance(key, str):
            key = f"'{key}'"  # noqa: PLW2901
        if isinstance(value, str):
            value = f"'{value}'"  # noqa: PLW2901
        text += " " + color_log_text(Log_Type.UNDERLINE, str(key), end=ltype) + ": " + str(value) + ";  "
    text += "}"
    return text


def get_formatted_time() -> str:
    """Return the current local time as a short formatted string."""
    return format_time_short(get_time())


def get_time() -> struct_time:
    """Return the current local time as a :class:`time.struct_time`."""
    t = time.localtime()
    return t


def _format_time(t: struct_time) -> str:
    """Return a human-readable representation of a ``struct_time`` value."""
    return time.asctime(t)


def format_time_short(t: struct_time) -> str:
    """Format a ``struct_time`` as a compact ``date-YYYY-M-D_time-H-M-S`` string.

    Args:
        t: Local time struct to format.

    Returns:
        A string such as ``"date-2024-1-15_time-9-30-0"``.
    """
    return (
        "date-"
        + str(t.tm_year)
        + "-"
        + str(t.tm_mon)
        + "-"
        + str(t.tm_mday)
        + "_time-"
        + str(t.tm_hour)
        + "-"
        + str(t.tm_min)
        + "-"
        + str(t.tm_sec)
    )


def _convert_seconds(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable ``H:MM:SS h:mm:ss`` string."""
    return str(datetime.timedelta(seconds=seconds)) + " h:mm:ss"


def color_log_text(ltype: Log_Type, text: str, end: Log_Type = Log_Type.TEXT) -> str:
    """Wrap ``text`` in the ANSI color codes corresponding to ``ltype``.

    Args:
        ltype: Log type that determines the opening color code.
        text: The string to colorize.
        end: Log type whose color code is applied after ``text``.
            Defaults to ``Log_Type.TEXT`` (reset to default terminal color).

    Returns:
        The colorized string with ANSI escape sequences.
    """
    return _color_text(color_char=type2bcolors[ltype][0], text=text, end=type2bcolors[end][0])


def _color_text(text: str, color_char: str, end: str = bcolors.ENDC) -> str:
    """Wrap text between an opening color code and a closing reset code."""
    return f"{color_char}{text}{bcolors.ENDC}{end}"


def _clean_all_color_from_text(text: str) -> str:
    """Strip all ANSI escape sequences from a string."""
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    return text


if __name__ == "__main__":
    text = "Hello World"
    colored_text = color_log_text(Log_Type.OK, text)
    uncolored_text = _clean_all_color_from_text(colored_text)
    print(colored_text)
    print(uncolored_text)
