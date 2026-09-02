"""Segmentation integrations (SPINEPS, VibeSeg, nnU-Net, ribs).

These entry points need the optional ``seg`` extra (``torch``, ``nnunetv2``,
``acvl_utils``, ``batchgenerators``) and, for SPINEPS, the ``spineps`` package.
Importing this sub-package must not require any of them - otherwise a plain
``import TPTBox.segmentation`` fails on a clean install - so each entry point is
replaced by a stub that explains what to install if its backend is missing.
"""

from __future__ import annotations

from TPTBox.core.internal.optional_deps import missing_dependency_func

_SEG_PACKAGES = "torch nnunetv2 acvl_utils batchgenerators"

try:
    from TPTBox.segmentation.rib.add_ribs import add_ribs_to_vert_spine
except ImportError as _e_rib:
    add_ribs_to_vert_spine = missing_dependency_func("add_ribs_to_vert_spine", _e_rib, "seg", _SEG_PACKAGES)  # type: ignore[assignment]

# `spineps` has no TPTBox extra on purpose: it declares a dependency on TPTBox
# itself, so `TPTBox[spineps]` would be circular and Poetry would refuse to solve
# it. Pass extra=None so the hint says `pip install spineps` rather than pointing
# at an extra that does not contain it.
try:
    from TPTBox.segmentation.spineps import _run_spineps_all, get_outpaths_spineps, run_spineps
except ImportError as _e_spineps:
    _run_spineps_all = missing_dependency_func("_run_spineps_all", _e_spineps, None, "spineps")  # type: ignore[assignment]
    get_outpaths_spineps = missing_dependency_func("get_outpaths_spineps", _e_spineps, None, "spineps")  # type: ignore[assignment]
    run_spineps = missing_dependency_func("run_spineps", _e_spineps, None, "spineps")  # type: ignore[assignment]

try:
    from TPTBox.segmentation.VibeSeg.vibeseg import (
        extract_vertebra_bodies_from_VibeSeg,
        run_inference_on_file,
        run_nnunet,
        run_vibeseg,
    )
except ImportError as _e_vibe:
    extract_vertebra_bodies_from_VibeSeg = missing_dependency_func(  # noqa: N816  # type: ignore[assignment]
        "extract_vertebra_bodies_from_VibeSeg", _e_vibe, "seg", _SEG_PACKAGES
    )
    run_inference_on_file = missing_dependency_func("run_inference_on_file", _e_vibe, "seg", _SEG_PACKAGES)  # type: ignore[assignment]
    run_nnunet = missing_dependency_func("run_nnunet", _e_vibe, "seg", _SEG_PACKAGES)  # type: ignore[assignment]
    run_vibeseg = missing_dependency_func("run_vibeseg", _e_vibe, "seg", _SEG_PACKAGES)  # type: ignore[assignment]

__all__ = [
    "_run_spineps_all",
    "add_ribs_to_vert_spine",
    "extract_vertebra_bodies_from_VibeSeg",
    "get_outpaths_spineps",
    "run_inference_on_file",
    "run_nnunet",
    "run_spineps",
    "run_vibeseg",
]
