"""The measurement registry shared by the speed and the memory benchmark.

Both benches call :func:`build_measurements` so the two JSON files always cover
exactly the same keys, and so a comparison never accidentally lines up different
work under the same name.

Everything here has to survive being pointed at **two different versions of
TPTBox**: the workflow runs this same file against the PR head and against the
PR base commit. Three things make that work.

* :func:`bind` pre-filters keyword arguments against the signature actually
  present, so a kwarg added in the PR is simply dropped on the baseline side
  instead of raising ``TypeError``. (Same trick as
  ``benchmarks/benchmark_nnunet_inference.py``.)
* :func:`accepts` lets a measurement opt *out* entirely when the feature it
  exists to measure is absent, rather than silently measuring something else.
* Setup and calls are individually guarded. A failure drops the affected keys
  and is reported on a ``skipped:`` / ``setup:`` line; it never aborts the run.
  The comparison tools list keys present on only one side as ``new`` /
  ``removed`` and never gate on them.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import io
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# This directory is added to sys.path so `workloads` resolves whether the bench
# scripts are run as files or this module is imported directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from workloads import Case  # noqa: E402

from TPTBox.core import poi as poi_module  # noqa: E402
from TPTBox.core.nii_wrapper import NII  # noqa: E402
from TPTBox.core.poi import POI  # noqa: E402

# Resolved by name rather than imported: a module-level `from ... import foo`
# would raise ImportError on a baseline commit that predates `foo`, taking the
# whole harness down instead of dropping one measurement.
calc_centroids = getattr(poi_module, "calc_centroids", None)
calc_poi_from_subreg_vert = getattr(poi_module, "calc_poi_from_subreg_vert", None)

#: Number of points fed to POI.local_to_global_arr.
_ARR_POINTS = 10_000


@dataclass
class Measurement:
    """A single measured call, already bound to its arguments."""

    key: str
    call: Callable[[], Any]


@contextlib.contextmanager
def silenced():
    """Swallow the chatty ``print``-based progress output of NII/POI operations.

    Entered *outside* the measured region so it costs nothing in the numbers.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _parameters(fn) -> dict[str, inspect.Parameter] | None:
    try:
        return dict(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


def accepts(fn, name: str) -> bool:
    """True if ``fn`` in *this* version of TPTBox takes a ``name`` keyword."""
    params = _parameters(fn)
    return bool(params) and name in params


def bind(fn, *args, **kwargs) -> Callable[[], Any]:
    """Pre-bind a zero-argument call, dropping kwargs this version does not accept.

    Resolved once here rather than at call time, so the signature introspection
    never lands inside a measured region.
    """
    params = _parameters(fn)
    if params is not None and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return functools.partial(fn, *args, **kwargs)


def _other_orientation(nii: NII):
    return ("R", "A", "S") if tuple(nii.orientation) == ("P", "I", "R") else ("P", "I", "R")


def _pad_target(case: Case) -> tuple[int, ...]:
    """Grow the in-plane axes by 8 voxels, leaving a singleton axis singleton."""
    return tuple(s + 8 if s > 1 else s for s in case.shape)


def build_measurements(case: Case, tmp: Path) -> tuple[list[Measurement], dict[str, float], list[str]]:
    """Return the measurements for ``case``, its ``metric_*`` context values, and setup notes.

    ``tmp`` is a scratch directory for the save/load measurements; it is written
    to once here (untimed) so the load measurements have something real to read.
    """
    img, vert, subreg = case.img, case.vert, case.subreg
    notes: list[str] = []
    ok = object()

    def resolve(label: str, target):
        """Turn ``(obj, "method")`` into a bound callable, or None if absent here.

        Methods are referred to by *name* rather than attribute access so that a
        method which does not exist on the baseline commit drops one measurement
        instead of raising AttributeError while the registry is being built.
        """
        if isinstance(target, tuple):
            obj, name = target
            fn = getattr(obj, name, None)
            if fn is None:
                notes.append(f"{label}: {type(obj).__name__}.{name} does not exist in this version")
            return fn
        if target is None:
            notes.append(f"{label}: not available in this version")
        return target

    def setup(label: str, target, *args, **kwargs):
        """Run one piece of untimed setup, recording rather than raising on failure.

        Returns the call's result, or ``None`` if it was unavailable or failed.
        Calls that return nothing on success (the ``save`` ones) get the ``ok``
        sentinel instead, so success stays distinguishable from failure.
        """
        fn = resolve(label, target)
        if fn is None:
            return None
        try:
            with silenced():
                result = bind(fn, *args, **kwargs)()
        except Exception as exc:  # noqa: BLE001 -- an absent API must not abort the run
            notes.append(f"{label} ({type(exc).__name__}: {exc})")
            return None
        return ok if result is None else result

    orient_to = _other_orientation(vert)
    labels = setup("unique", (vert, "unique"))
    labels = list(labels) if isinstance(labels, (list, tuple, np.ndarray)) else []
    first_label = labels[0] if labels else 1
    grid_2mm = setup("rescale-grid", (img, "rescale"), (2, 2, 2), verbose=False)
    crop = setup("compute-crop", (vert, "compute_crop"), raise_error=False)
    poi = setup("calc_centroids", calc_centroids, vert)
    region_map = None
    if poi is not None:
        regions = setup("poi-keys_region", (poi, "keys_region"))
        if regions is not None:
            region_map = {int(k): int(k) + 100 for k in regions}

    img_file = tmp / f"{case.name}_img.nii.gz"
    seg_file = tmp / f"{case.name}_seg.nii.gz"
    poi_file = tmp / f"{case.name}_poi.json"
    out_img = tmp / f"{case.name}_roundtrip.nii.gz"
    have_img = setup("write-img", (img, "save"), img_file, verbose=False) is not None
    have_seg = setup("write-seg", (vert, "save"), seg_file, verbose=False) is not None
    have_poi_file = poi is not None and setup("write-poi", (poi, "save"), poi_file, verbose=False) is not None

    label_map = {int(a): int(a) + 100 for a in labels}
    coords = np.asarray([[i % 97, (i * 7) % 89, (i * 13) % 83] for i in range(_ARR_POINTS)], dtype=float)
    pad_to = _pad_target(case)

    m: list[Measurement] = []

    def add(key: str, target, *args, **kwargs) -> None:
        fn = resolve(key, target)
        if fn is not None:
            m.append(Measurement(key, bind(fn, *args, **kwargs)))

    # --- NII: IO -----------------------------------------------------------
    if have_img:
        add("nii_load_img", lambda: NII.load(img_file, seg=False).get_array())
    if have_seg:
        add("nii_load_seg", lambda: NII.load(seg_file, seg=True).get_seg_array())
    add("nii_save", (img, "save"), out_img, verbose=False)

    # --- NII: array access / dtype ----------------------------------------
    add("nii_get_array", (img, "get_array"))
    add("nii_set_dtype", (img, "set_dtype"), np.float32)

    # --- NII: geometry -----------------------------------------------------
    add("nii_reorient", (img, "reorient"), orient_to)
    add("nii_rescale", (img, "rescale"), (2, 2, 2), verbose=False)
    add("nii_rescale_seg", (vert, "rescale"), (2, 2, 2), verbose=False)
    if grid_2mm is not None:
        add("nii_resample_from_to", (img, "resample_from_to"), grid_2mm, verbose=False)
    add("nii_compute_crop", (vert, "compute_crop"), raise_error=False)
    if crop is not None:
        add("nii_apply_crop", (vert, "apply_crop"), crop)
    add("nii_pad_to", (img, "pad_to"), pad_to)

    # --- NII: statistics ---------------------------------------------------
    add("nii_unique", (vert, "unique"))
    add("nii_volumes", (vert, "volumes"))
    add("nii_center_of_masses", (vert, "center_of_masses"))

    # --- NII: label ops ----------------------------------------------------
    add("nii_extract_label", (vert, "extract_label"), first_label)
    add("nii_map_labels", (vert, "map_labels"), label_map, verbose=False)

    # --- NII: morphology / connected components ----------------------------
    add("nii_dilate_msk", (vert, "dilate_msk"), 2, verbose=False)
    add("nii_erode_msk", (vert, "erode_msk"), 1, verbose=False)
    add("nii_fill_holes", (vert, "fill_holes"))
    add("nii_get_connected_components", (vert, "get_connected_components"), connectivity=3)
    add("nii_filter_connected_components", (vert, "filter_connected_components"), min_volume=8, connectivity=3)

    # --- POI ---------------------------------------------------------------
    add("poi_calc_centroids", calc_centroids, vert)
    if not case.is_heavy and calc_centroids is not None and accepts(calc_centroids, "_crop"):
        # The _crop=False path is a per-label scipy loop kept around as an A/B
        # against the single-pass np_center_of_mass. On a 400^3 volume it is
        # minutes per repeat and would dominate the whole run, so large cases
        # skip it. Where the kwarg does not exist there is nothing to compare,
        # and measuring it anyway would just duplicate poi_calc_centroids.
        add("poi_calc_centroids_nocrop", calc_centroids, vert, _crop=False)
    if poi is not None:
        add("poi_reorient", (poi, "reorient"), orient_to)
        add("poi_rescale", (poi, "rescale"), (2, 2, 2), verbose=False)
        add("poi_to_global", (poi, "to_global"))
        add("poi_local_to_global_arr", (poi, "local_to_global_arr"), coords)
        add("poi_save", (poi, "save"), poi_file, verbose=False)
        if region_map is not None:
            add("poi_map_labels", (poi, "map_labels"), label_map_region=region_map)
        if grid_2mm is not None:
            add("poi_resample_from_to", (poi, "resample_from_to"), grid_2mm)
    if have_poi_file:
        add("poi_load", (POI, "load"), poi_file)
    if case.anatomical:
        # The heavy strategy pipeline. Only meaningful on real spine data, and
        # nonsense on a single slice or on synthetic cuboids.
        add("poi_calc_poi_from_subreg_vert", calc_poi_from_subreg_vert, vert, subreg, verbose=False)

    metrics = {
        "metric_voxels": float(case.voxels),
        "metric_labels": float(len(labels)),
        "metric_foreground_pct": float(round(100.0 * float(np.count_nonzero(vert.get_seg_array())) / max(case.voxels, 1), 3)),
    }
    return m, metrics, notes
