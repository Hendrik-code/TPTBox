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
"""

from __future__ import annotations

import json
from pathlib import Path

from TPTBox.core.poi import POI, POI_Descriptor
from TPTBox.core.vert_constants import Vertebra_Instance
from TPTBox.spine.spinestats.angles import compute_lordosis_and_kyphosis


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
    """Return a copy of ``poi`` whose region ids are re-keyed via ``mapping``.

    Regions absent from ``mapping`` are dropped. Non-vertebra regions
    (i.e. those already outside the vertebra label range) fall through
    with their original id — but in practice a POI produced by
    :func:`calc_poi_from_subreg_vert` only carries vertebra regions.
    """
    new_centroids = POI_Descriptor()
    for region, subregion, coord in poi.centroids.items():
        if region in mapping:
            new_centroids[(mapping[region], subregion)] = coord
    return poi.copy(centroids=new_centroids)


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


def _compute_anomaly_variant(poi: POI, veridah_json_path: Path | None) -> dict[str, float | None] | None:
    """Recompute the three regional angles after applying the VERIDAH label correction.

    Returns ``None`` when the VERIDAH file is missing or unusable.
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
    mapping = {int(o): int(f) for o, f in zip(orig, fpath)}
    return compute_lordosis_and_kyphosis(_relabel_poi(poi, mapping))


def _compute_k_variant(poi: POI, k: int) -> dict[str, float | None]:
    """Compute lumbar-lordosis only, with the last ``k`` vertebrae treated as L1..Lk."""
    relabeled = _last_k_relabel(poi, k)
    if not relabeled.centroids:
        return {"lumbar_lordosis": None}
    full = compute_lordosis_and_kyphosis(relabeled)
    return {"lumbar_lordosis": full.get("lumbar_lordosis")}


def compute_veridah_variants(poi: POI, veridah_json_path: Path | str | None) -> dict:
    """Return the ``curv_veridah`` block for one subject.

    Shape::

        {
          "anomaly": {"cervical_lordosis", "thoracic_kyphosis", "lumbar_lordosis"} | None,
          "k4": {"lumbar_lordosis": …},
          "k5": {...}, "k6": {...}, "k7": {...},
        }
    """
    veridah_path = Path(veridah_json_path) if veridah_json_path is not None else None
    out: dict = {"anomaly": _compute_anomaly_variant(poi, veridah_path)}
    for k in (4, 5, 6, 7):
        out[f"k{k}"] = _compute_k_variant(poi, k)
    return out
