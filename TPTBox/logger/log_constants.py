from __future__ import annotations

import datetime
import time
from enum import Enum, auto
from time import struct_time


class Log_Type(Enum):
    """The different types of Logs supported"""

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
    """Terminal color symbols"""

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


def datatype_to_string(text, log_type: Log_Type):
    """Processes given text into a readable string

    Args:
        text (str): _description_
        log_type (Log_Type): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(text, dict):
        return _dict_to_string(text, log_type)
    return str(text)


def _dict_to_string(u_dict: dict, ltype: Log_Type = Log_Type.TEXT):
    """Converts a dictionary into a readable string

    Args:
        u_dict (dict): dictionary to be logged
        ltype (Log_Type, optional): Log_Type. Defaults to Log_Type.TEXT.

    Returns:
        _type_: string version of the dictionary
    """
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


def get_formatted_time():
    return format_time_short(get_time())


def get_time() -> struct_time:
    t = time.localtime()
    return t


def _format_time(t: struct_time):
    return time.asctime(t)


def format_time_short(t: struct_time) -> str:
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


def _convert_seconds(seconds: float):
    return str(datetime.timedelta(seconds=seconds)) + " h:mm:ss"


def color_log_text(ltype: Log_Type, text: str, end: Log_Type = Log_Type.TEXT):
    """Colors text(str) based on given Log_Type

    Args:
        ltype (Log_Type): Log_Type (defines the color being used)
        text (str): Text to be colored
        end (Log_Type, optional): What color should come after this text. Defaults to Log_Type.TEXT. (no color)

    Returns:
        _type_: _description_
    """
    return _color_text(color_char=type2bcolors[ltype][0], text=text, end=type2bcolors[end][0])


def _color_text(text: str, color_char, end=bcolors.ENDC):
    return f"{color_char}{text}{bcolors.ENDC}{end}"


def _clean_all_color_from_text(text: str):
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
