from __future__ import annotations

from TPTBox.core.internal.optional_deps import missing_dependency_class, missing_dependency_func

# ---------------------------------------------------------------------------
# Most registration entry points require ``hf-deepali`` (and therefore PyTorch),
# which is the optional ``reg`` extra. Importing this package must succeed even
# when it is missing; only *using* a deepali-backed entry point should fail, and
# it should say what to install. The stub factories live in
# TPTBox.core.internal.optional_deps and are shared with TPTBox.segmentation and
# TPTBox.core.dicom.
# ---------------------------------------------------------------------------

_REG_EXTRA = "reg"
_REG_PACKAGES = "torch hf-deepali"


def _make_missing_deepali_stub(name: str, exc: BaseException):
    """Class placeholder for a deepali-backed entry point."""
    return missing_dependency_class(name, exc, _REG_EXTRA, _REG_PACKAGES)


def _make_missing_deepali_func(name: str, exc: BaseException):
    """Callable stub for a deepali-backed helper function."""
    return missing_dependency_func(name, exc, _REG_EXTRA, _REG_PACKAGES)


# --- SITK point registration (no deepali needed) ---------------------------
try:
    from ._ridged_points.point_registration import (
        Point_Registration,
        ridged_points_from_poi,
        ridged_points_from_subreg_vert,
    )
except ImportError as _e_sitk:  # SimpleITK is a hard dependency - very unlikely, still guard.
    Point_Registration = missing_dependency_class("Point_Registration", _e_sitk, None, "SimpleITK")  # type: ignore[misc,assignment]
    ridged_points_from_poi = missing_dependency_func("ridged_points_from_poi", _e_sitk, None, "SimpleITK")  # type: ignore[assignment]
    ridged_points_from_subreg_vert = missing_dependency_func(  # type: ignore[assignment]
        "ridged_points_from_subreg_vert", _e_sitk, None, "SimpleITK"
    )

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
