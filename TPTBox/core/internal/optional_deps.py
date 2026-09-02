"""Helpers for degrading gracefully when an optional dependency is absent.

TPTBox's core is deliberately installable without the heavy optional stacks
(DICOM conversion, nnU-Net/SPINEPS inference, deepali registration). Importing a
sub-package must therefore succeed even when its backend is missing - only
*using* an entry point should fail, and it should fail with an actionable
message instead of a bare ``ModuleNotFoundError`` from three frames deep.

``TPTBox.registration`` grew the original version of this pattern; these two
factories are the reusable form of it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["missing_dependency_class", "missing_dependency_func"]


def _message(name: str, extra: str, packages: str, original_error: str, call: str) -> str:
    return (
        f"`{name}{call}` requires optional dependencies that are not installed.\n"
        f"    pip install 'TPTBox[{extra}]'\n"
        f"or install them directly:\n"
        f"    pip install {packages}\n"
        f"Original import error was: {original_error}"
    )


def missing_dependency_func(name: str, exc: BaseException, extra: str, packages: str) -> Callable[..., Any]:
    """Return a callable stub that raises a helpful ``ImportError`` when called.

    Args:
        name: Name of the entry point being replaced.
        exc: The original ``ImportError``, quoted in the message.
        extra: The ``pip install 'TPTBox[...]'`` extra that provides the backend.
        packages: Space-separated package names, for a direct pip install.
    """
    original_error = str(exc) or exc.__class__.__name__

    def _stub(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError(_message(name, extra, packages, original_error, "()"))

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub._tptbox_missing_extra = extra  # type: ignore[attr-defined]
    _stub._tptbox_import_error = original_error  # type: ignore[attr-defined]
    return _stub


def missing_dependency_class(name: str, exc: BaseException, extra: str, packages: str) -> type:
    """Return a class placeholder that raises on instantiation or attribute access.

    ``isinstance``/``issubclass`` checks stay safe - the stub is a plain class.
    """
    original_error = str(exc) or exc.__class__.__name__

    class _Missing:
        __name__ = name
        __qualname__ = name
        _tptbox_missing_extra = extra
        _tptbox_import_error = original_error

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError(_message(name, extra, packages, original_error, "()"))

        def __class_getitem__(cls, item):  # keep type annotations happy
            return cls

        def __getattr__(self, item):
            raise ImportError(_message(f"{name}.{item}", extra, packages, original_error, ""))

    _Missing.__name__ = name
    _Missing.__qualname__ = name
    return _Missing
