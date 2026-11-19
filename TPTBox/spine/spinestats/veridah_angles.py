"""VERIDAH-aware lordosis / kyphosis variants.

Two extra angle computations on top of the standard ``curv`` block in
:mod:`TPTBox.spine.spinestats.angles`:

- ``anomaly``: relabel the POI using the VERIDAH ``orig_label -> fpath``
  map (so a supernumerary T13 lands on label 28 instead of shifting the
  entire lumbar enumeration down by one) and recompute all three
  regional angles (cervical / thoracic / lumbar).
- ``k4 / k5 / k6 / k7``: force the last ``k`` present vertebrae above
  the sacrum to be interpreted as L1..Lk, drop everything cranial, and
  compute **only** ``lumbar_lordosis``. Independent of VERIDAH.

Both variants live under the top-level JSON key ``curv_veridah`` (see
``_run_all.py``); the standard ``curv`` output is untouched.

:func:`plot_veridah_variants` writes a matching snapshot JPG. It re-uses
the already-computed POI points (no re-segmentation, no re-POI-derivation)
by driving the label re-key through :meth:`POI.map_labels` — which also
carries the registered ``label_name`` and vector-field metadata along —
and applies the same integer mapping to the vertebra NIfTI via
:meth:`NII.map_labels` so the overlay matches the relabeled points.
"""

from __future__ import annotations

import json
from pathlib import Path

from TPTBox.core.nii_wrapper import NII, Image_Reference, to_nii
from TPTBox.core.poi import POI, POI_Descriptor
from TPTBox.core.vert_constants import Vertebra_Instance
from TPTBox.spine.snapshot2D.snapshot_modular import create_snapshot
from TPTBox.spine.spinestats.angles import compute_lordosis_and_kyphosis, plot_compute_lordosis_and_kyphosis


def _load_veridah(path: Path) -> dict | None:
    """Return the single dict inside a VERIDAH ``_stat.json``, or ``None`` on error."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def _relabel_poi(poi: POI, mapping: dict[int, int]) -> POI:
    return poi.map_labels(label_map_region=mapping)


def _last_k_relabel(poi: POI, k: int) -> POI:
    """Keep the ``k`` most caudal non-sacral vertebrae; relabel them L1..Lk.

    Uses :meth:`Vertebra_Instance.order` to walk cranio-caudal, ignores
    sacrum members, and keeps the last ``k`` present labels. The bottom
    one becomes ``Lk`` (label ``20 + k - 1``), stepping up by one to
    ``L1`` (label 20).
    """
    order = Vertebra_Instance.order()
    sacrum_vals = {v.value for v in Vertebra_Instance.sacrum()}
    present = {r for r, _s, _c in poi.centroids.items()}
    non_sacral_in_order = [v.value for v in order if v.value in present and v.value not in sacrum_vals]
    if len(non_sacral_in_order) < k:
        return poi.copy(centroids=POI_Descriptor())
    last_k = non_sacral_in_order[-k:]  # cranio → caudal
    # last_k[0] should be L1 (20), last_k[-1] should be Lk (20+k-1).
    mapping = {orig: 20 + i for i, orig in enumerate(last_k)}
    return _relabel_poi(poi, mapping)


def _compute_anomaly_variant(poi: POI, veridah_json_path: Path | None, project_2D: bool = False) -> dict[str, float | None] | None:
    """Recompute the three regional angles after applying the VERIDAH label correction.

    ``project_2D`` must match the setting used for the standard ``curv`` block
    in :mod:`_run_all` (default ``False``, i.e. 3D angles). Returns ``None``
    when the VERIDAH file is missing or unusable.
    """
    if veridah_json_path is None:
        return None
    veridah = _load_veridah(Path(veridah_json_path))
    if veridah is None:
        return None
    orig = veridah.get("orig_label")
    fpath = veridah.get("fpath")
    if not isinstance(orig, list) or not isinstance(fpath, list) or len(orig) != len(fpath):
        return None
    mapping = {int(o): int(f) for o, f in zip(orig, fpath) if int(o) != int(f)}
    return compute_lordosis_and_kyphosis(_relabel_poi(poi, mapping), project_2D=project_2D)


def _compute_k_variant(poi: POI, k: int, project_2D: bool = False) -> dict[str, float | None]:
    """Compute lumbar-lordosis only, with the last ``k`` vertebrae treated as L1..Lk."""
    relabeled = _last_k_relabel(poi, k)
    if not relabeled.centroids:
        return {"lumbar_lordosis": None}
    full = compute_lordosis_and_kyphosis(relabeled, project_2D=project_2D)
    return {"lumbar_lordosis": full.get("lumbar_lordosis"), "lumbar_lordosis_apex": full.get("lumbar_lordosis_apex")}


def _veridah_region_map(veridah_json_path: Path) -> dict[int, int] | None:
    """Return the ``orig_label -> fpath`` region mapping from a VERIDAH stat json."""
    veridah = _load_veridah(Path(veridah_json_path))
    if veridah is None:
        return None
    orig = veridah.get("orig_label")
    fpath = veridah.get("fpath")
    if not isinstance(orig, list) or not isinstance(fpath, list) or len(orig) != len(fpath):
        return None
    return {int(o): int(f) for o, f in zip(orig, fpath) if int(o) != int(f)}


def plot_veridah_variants(
    jpg_path: str | Path | None,
    poi: POI,
    img: Image_Reference,
    seg_vert: Image_Reference,
    veridah_json_path: Path | str | None,
    line_len: int = 100,
    project_2D: bool = False,
) -> tuple[dict[str, float | None] | None, str | None]:
    """Render the VERIDAH-corrected lordosis / kyphosis snapshot next to the standard one.

    Re-uses the *already computed* POI points -- the angle numbers are cheap
    trig on those points, and no re-segmentation / re-``calc_poi_from_subreg_vert``
    happens. The re-key is driven through :meth:`POI.map_labels`, so any
    registered ``label_name`` / vector-field metadata is carried along
    automatically. The vertebra segmentation is remapped with the same
    integer table via :meth:`NII.map_labels` so the overlay matches.

    Args:
        jpg_path: Output path for the snapshot JPG. If ``None``, the image is not saved.
        poi: The (already computed) POI object.
        img: The reference image on which to plot.
        seg_vert: Vertebra segmentation NIfTI, remapped alongside the POI.
        veridah_json_path: Path to the VERIDAH ``_stat.json`` giving
            ``{orig_label, fpath}`` lists.
        line_len: Length of the direction lines drawn on the snapshot.
        project_2D: Compute the angles as 2D sagittal projections.

    Returns:
        ``(anomaly_angles, saved_path)`` -- ``anomaly_angles`` is the same
        dict :func:`compute_lordosis_and_kyphosis` returns (or ``None``
        when the VERIDAH file is missing/malformed), ``saved_path`` is
        the resolved output path (or ``None`` when nothing was written).
    """
    if veridah_json_path is None:
        return None, None
    mapping = _veridah_region_map(Path(veridah_json_path))
    if mapping is None:
        return None, None

    relabeled_poi = _relabel_poi(poi, mapping)
    if not relabeled_poi.centroids:
        return None, None
    # Remap the vertebra segmentation with the same integer table so the overlay
    # numbering matches the (already-remapped) POI. NII.map_labels leaves labels
    # that are not listed in ``mapping`` unchanged.
    seg_nii = seg_vert if isinstance(seg_vert, NII) else to_nii(seg_vert, seg=True)
    seg_relabeled = seg_nii.map_labels(mapping, verbose=False)  # type: ignore[arg-type]

    angles, _frame = plot_compute_lordosis_and_kyphosis(
        None,
        relabeled_poi,
        img,
        seg_relabeled,
        line_len=line_len,
        project_2D=project_2D,
    )
    saved: str | None = None
    if jpg_path is not None:
        create_snapshot(jpg_path, [_frame])
        saved = str(jpg_path)
    return angles, saved


def compute_veridah_variants(poi: POI, veridah_json_path: Path | str | None, project_2D: bool = False) -> dict:
    """Return the ``curv_veridah`` block for one subject.

    Shape::

        {
          "anomaly": {"cervical_lordosis", "thoracic_kyphosis", "lumbar_lordosis"} | None,
          "k4": {"lumbar_lordosis": …},
          "k5": {...}, "k6": {...}, "k7": {...},
        }
    """
    veridah_path = Path(veridah_json_path) if veridah_json_path is not None else None
    out: dict = {"anomaly": _compute_anomaly_variant(poi, veridah_path, project_2D=project_2D)}
    for k in (4, 5, 6, 7):
        out[f"k{k}"] = _compute_k_variant(poi, k, project_2D=project_2D)
    return out
