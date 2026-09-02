from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Some of the registration entry points require ``hf-deepali`` (and therefore
# also PyTorch). ``hf-deepali`` is an *optional* dependency: importing this
# package must succeed even when it is missing, and only *using* one of the
# deepali-backed classes should surface an error.
#
# For each optional class we try to import it. On failure we replace it with a
# stub that raises a clear ``ImportError`` the first time the caller touches it
# (instantiation *or* attribute access). Users then get "install hf-deepali
# (also needs PyTorch)" instead of a vague ``NameError`` from Python.
# ---------------------------------------------------------------------------


def _make_missing_deepali_stub(name: str, exc: BaseException):
    """Return a class placeholder for a deepali-backed entry point.

    Any instantiation or attribute access raises an ``ImportError`` explaining
    that ``hf-deepali`` (and PyTorch) must be installed. ``isinstance``/subclass
    checks are safe – the stub is a plain class.
    """
    original_error = str(exc) or exc.__class__.__name__

    class _MissingDeepali:
        __name__ = name
        __qualname__ = name
        _tptbox_optional_dep = "hf-deepali"
        _tptbox_import_error = original_error

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: D401
            raise ImportError(
                f"`{name}` requires the optional dependency `hf-deepali` "
                f"(which in turn requires PyTorch). Install both with:\n"
                f"    pip install torch hf-deepali\n"
                f"Original import error was: {original_error}"
            )

        def __class_getitem__(cls, item):  # keep type-annotations happy
            return cls

        def __getattr__(self, item):
            raise ImportError(
                f"`{name}.{item}` requires `hf-deepali` (and PyTorch). "
                f"Install with:  pip install torch hf-deepali\n"
                f"Original import error was: {original_error}"
            )

    _MissingDeepali.__name__ = name
    _MissingDeepali.__qualname__ = name
    return _MissingDeepali


def _make_missing_deepali_func(name: str, exc: BaseException):
    """Return a callable stub for a deepali-backed helper function."""
    original_error = str(exc) or exc.__class__.__name__

    def _stub(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError(
            f"`{name}()` requires the optional dependency `hf-deepali` "
            f"(which in turn requires PyTorch). Install both with:\n"
            f"    pip install torch hf-deepali\n"
            f"Original import error was: {original_error}"
        )

    _stub.__name__ = name
    _stub.__qualname__ = name
    return _stub


# --- SITK point registration (no deepali needed) ---------------------------
try:
    from ._ridged_points.point_registration import (
        Point_Registration,
        ridged_points_from_poi,
        ridged_points_from_subreg_vert,
    )
except ImportError as _e_sitk:  # SimpleITK missing - very unlikely, still guard.
    Point_Registration = _make_missing_deepali_stub("Point_Registration", _e_sitk)  # type: ignore[misc,assignment]
    ridged_points_from_poi = _make_missing_deepali_func("ridged_points_from_poi", _e_sitk)  # type: ignore[assignment]
    ridged_points_from_subreg_vert = _make_missing_deepali_func("ridged_points_from_subreg_vert", _e_sitk)  # type: ignore[assignment]

# --- Deepali closed-form point registration --------------------------------
try:
    from ._ridged_points.deepali_point_registration import (
        Deepali_Point_Registration,
        ridged_points_from_poi_deepali,
        ridged_points_from_subreg_vert_deepali,
    )
except ImportError as _e_deep_pt:
    Deepali_Point_Registration = _make_missing_deepali_stub("Deepali_Point_Registration", _e_deep_pt)  # type: ignore[misc,assignment]
    ridged_points_from_poi_deepali = _make_missing_deepali_func("ridged_points_from_poi_deepali", _e_deep_pt)  # type: ignore[assignment]
    ridged_points_from_subreg_vert_deepali = _make_missing_deepali_func(  # type: ignore[assignment]
        "ridged_points_from_subreg_vert_deepali", _e_deep_pt
    )

# --- General deepali image registration ------------------------------------
try:
    from ._deepali.deepali_model import General_Registration
except ImportError as _e_gen:
    General_Registration = _make_missing_deepali_stub("General_Registration", _e_gen)  # type: ignore[misc,assignment]

try:
    from TPTBox.registration._deformable.deformable_reg import Deformable_Registration
except ImportError as _e_def:
    Deformable_Registration = _make_missing_deepali_stub("Deformable_Registration", _e_def)  # type: ignore[misc,assignment]

try:
    from ._deepali.spine_rigid_elements_reg import Rigid_Elements_Registration
except ImportError as _e_rer:
    Rigid_Elements_Registration = _make_missing_deepali_stub("Rigid_Elements_Registration", _e_rer)  # type: ignore[misc,assignment]

# --- Template registrations (deformable + optional pre-reg) ----------------
try:
    from TPTBox.registration._deformable.multilabel_segmentation import Template_Registration
except ImportError as _e_tpl:
    Template_Registration = _make_missing_deepali_stub("Template_Registration", _e_tpl)  # type: ignore[misc,assignment]

try:
    from ._deformable.multilabel_segmentation import Template_Registration2
except ImportError as _e_tpl2:
    Template_Registration2 = _make_missing_deepali_stub("Template_Registration2", _e_tpl2)  # type: ignore[misc,assignment]


__all__ = [
    "Deepali_Point_Registration",
    "Deformable_Registration",
    "General_Registration",
    "Point_Registration",
    "Rigid_Elements_Registration",
    "Template_Registration",
    "Template_Registration2",
    "ridged_points_from_poi",
    "ridged_points_from_poi_deepali",
    "ridged_points_from_subreg_vert",
    "ridged_points_from_subreg_vert_deepali",
]
