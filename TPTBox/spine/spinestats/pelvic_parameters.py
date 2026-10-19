"""Pelvic sagittal parameters (PI / PT / SS / PI-LL) from fullbody POIs.

Reads the per-subject fullbody POI json produced by the ``TReg`` fullbody
registration pipeline
(``.../derivatives-fullbody-poi/{pfx}/{sub}/vibe/sub-{sub}_..._seg-fullbody_poi.json``)
and returns the three classical pelvic parameters plus their pairwise
mismatches with lumbar lordosis.

Multiple variants are computed side by side (see ``variants`` below). The
idea is not to *pick* one here but to expose all reasonable definitions
so they can be compared against each other (and against literature) in
downstream QC.

Definitions
-----------
- **Pelvic Incidence (PI)**: angle between (a) the line from the
  bi-coxo-femoral axis (midpoint of the two femoral head centers) to
  the center of the S1 superior endplate, and (b) the perpendicular to
  the S1 superior endplate. Measured in the sagittal plane. Position-
  invariant (anatomical constant per subject).
- **Sacral Slope (SS)**: angle between the S1 superior endplate line
  and horizontal, in the sagittal plane. Position-dependent.
- **Pelvic Tilt (PT)**: angle between the line from the bi-coxo-femoral
  axis to the center of the S1 superior endplate and the vertical.
  Signed positive when the sacrum is *posterior* to the hip axis
  (retroverted pelvis). Position-dependent.
- Fundamental relationship: **PI = PT + SS** (up to sign convention).
- **PI-LL mismatch**: PI - lumbar_lordosis. A value close to zero is
  associated with balanced spinopelvic alignment; large positive
  mismatches with sagittal decompensation.

Coordinate system
-----------------
The fullbody POI json is written in ``nib`` (nibabel world) coordinates,
so ``[x, y, z] = [right, anterior, superior]`` in mm. All computations
here therefore project onto the ``(y, z)`` sagittal plane.

Limitations
-----------
1. **Supine vs. standing.** These metrics are conventionally measured on
   standing lateral radiographs. All NAKO acquisitions are **supine**
   MRI. Under gravity the sacrum tilts anteriorly; in supine SS is
   systematically ~10-15° lower and PT ~10-15° higher than in the same
   subject standing. **PI is anatomical and position-invariant**, so
   only PI (and PI-LL when using a supine LL) are directly comparable
   to standing-image reference ranges.
2. **Endplate proxy.** The S1 superior endplate is not landmarked as a
   contour — it is reconstructed from two point landmarks per variant.
   The ``poi_ap`` variant uses ``Sacral_Crest_S1`` (posterior) and
   ``Anterior_Longitudinal_Medial`` (anterior). The anterior ligament
   attachment point can drift inferior with age / degeneration, biasing
   the endplate normal.
3. **Bi-femoral axis.** Uses the atlas-registered ``PELVIS_CENTER``
   landmark (which despite the name lives inside each femur landmark
   group and marks the femoral head center). The registration is
   template-based; large hip pathology can distort this point.
4. **PI-LL uses the supine LL** produced by the existing spine
   pipeline. This is by definition smaller than a standing LL and the
   mismatch numbers cannot be interpreted the same way as Schwab-style
   thresholds derived from standing radiographs.
5. **No axial pelvic obliquity correction.** The sagittal plane is
   taken as the world ``(y, z)`` plane. If the subject is rotated in
   the scanner (obliquity around the SI axis), the projection is off
   by that angle. In practice supine MRI subjects are close to aligned.

The function is total: on missing landmarks / json each variant's block
contains ``None``s plus an ``error`` message; the outer function never
raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.linalg import norm

# Fullbody POI json coordinate convention: nibabel world = (R, A, S).
_IDX_R, _IDX_A, _IDX_S = 0, 1, 2


def _sag(v: np.ndarray) -> np.ndarray:
    """Project a nibabel-world (R, A, S) point onto the sagittal (anterior, superior) plane."""
    return v[[_IDX_A, _IDX_S]]


def _pelvic_from_endplate_pair(S_post: np.ndarray, S_ant: np.ndarray, FH_R: np.ndarray, FH_L: np.ndarray) -> dict[str, float]:
    """Compute PI, PT, SS from posterior/anterior S1 endplate points and the two femoral head centers.

    Sign convention (Legaye / Schwab):
      - SS positive when the S1 endplate tilts down anteriorly (typical).
      - PT positive when the sacrum center is posterior of the hip axis
        (retroverted pelvis). PT is negative for anteverted pelves.
      - PI = PT + SS exactly (both in signed degrees).

    All three angles are computed as signed values via ``atan2`` in the
    sagittal (anterior=Y, superior=Z) plane; ``abs()`` is avoided so the
    fundamental invariant holds.
    """
    S = 0.5 * (S_post + S_ant)
    F = 0.5 * (FH_R + FH_L)
    S2, F2 = _sag(S), _sag(F)
    e = _sag(S_ant) - _sag(S_post)  # posterior -> anterior in (Y, Z)
    if norm(e) == 0:
        return {"pi_deg": None, "pt_deg": None, "ss_deg": None, "error": "degenerate S1 endplate direction"}
    d = S2 - F2  # F -> S in (Y, Z)
    if norm(d) == 0:
        return {"pi_deg": None, "pt_deg": None, "ss_deg": None, "error": "S1 center coincides with hip axis"}
    # SS (signed): angle by which endplate tips down anteriorly.
    # e = (e_y, e_z). If anterior end is inferior (e_z < 0), SS > 0.
    SS = float(np.degrees(np.arctan2(-e[1], e[0])))
    # PT (signed): angle by which the F->S line tips posterior from vertical.
    # d = (d_y, d_z). If S is posterior of F (d_y < 0), PT > 0 (retroverted).
    PT = float(np.degrees(np.arctan2(-d[0], d[1])))
    PI = SS + PT
    return {
        "pi_deg": round(PI, 3),
        "pt_deg": round(PT, 3),
        "ss_deg": round(SS, 3),
        "hip_center_mm": [round(float(x), 2) for x in F],
        "s1_endplate_center_mm": [round(float(x), 2) for x in S],
    }


def compute_pelvic_parameters(
    fullbody_poi_json: Path | str | None,
    lumbar_lordosis_deg: float | None = None,
) -> dict[str, Any]:
    """Compute PI/PT/SS (and PI-LL) in multiple variants.

    Parameters
    ----------
    fullbody_poi_json : Path
        Path to ``sub-*_seg-fullbody_poi.json``. Two top-level list
        entries expected (``meta``, ``body``); ``body`` maps bone → sub-index → ``[x, y, z]``.
    lumbar_lordosis_deg : float, optional
        The subject's lumbar lordosis (from the existing spine pipeline,
        typically ``out["curv"]["lumbar_lordosis"]``). Used to compute
        the ``pi_ll_mismatch_deg`` fields.

    Returns:
    -------
    dict
        Keys:
        - ``variant`` = ``"poi_ap"``: canonical variant. Uses
          ``Sacral_Crest_S1`` (posterior of S1 top) and
          ``Anterior_Longitudinal_Medial`` (anterior of S1 top).
        - ``variant`` = ``"poi_ala"``: alternative using midpoint of
          ``Sacrum_Ala_Superior_Left/Right`` as the "anterior" reference
          instead of the ligament point. Included for QC comparison
          only; the ligament version is closer to the canonical
          endplate midline in most subjects.
        - ``pi_ll_mismatch_deg_poi_ap`` / ``..._poi_ala``: PI minus
          ``lumbar_lordosis_deg`` per variant (``None`` if LL missing).
        - ``fullbody_poi_json``: the resolved path used.
        - ``error``: top-level message when nothing could be computed.
    """
    out: dict[str, Any] = {"fullbody_poi_json": str(fullbody_poi_json) if fullbody_poi_json else None}
    if fullbody_poi_json is None:
        out["error"] = "no fullbody POI json path provided"
        return out
    p = Path(fullbody_poi_json)
    if not p.exists():
        out["error"] = f"fullbody POI json does not exist: {p}"
        return out
    import json

    try:
        with p.open() as f:
            payload = json.load(f)
    except Exception as e:
        out["error"] = f"failed to load fullbody POI json: {e}"
        return out
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], dict):
        out["error"] = "unexpected fullbody POI json shape"
        return out
    body = payload[1]

    def _pt(bone: str, idx: str) -> np.ndarray | None:
        try:
            return np.asarray(body[bone][idx], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None

    S1_post = _pt("sacrum", "1")  # Sacral_Crest_S1
    S1_ant = _pt("sacrum", "19")  # Anterior_Longitudinal_Medial
    ala_L = _pt("sacrum", "27")  # Sacrum_Ala_Superior_Left
    ala_R = _pt("sacrum", "28")  # Sacrum_Ala_Superior_Right
    FH_R = _pt("femur_right", "11")  # PELVIS_CENTER (right femoral head center)
    FH_L = _pt("femur_left", "11")  # PELVIS_CENTER (left  femoral head center)

    missing = []
    for k, v in (
        ("sacrum_1", S1_post),
        ("sacrum_19", S1_ant),
        ("femur_right_11", FH_R),
        ("femur_left_11", FH_L),
    ):
        if v is None:
            missing.append(k)
    if missing:
        out["error"] = f"missing required landmarks: {missing}"
        return out

    # Variant poi_ap.
    out["poi_ap"] = _pelvic_from_endplate_pair(S1_post, S1_ant, FH_R, FH_L)  # type: ignore[arg-type]

    # Variant poi_ala: use midpoint of ala_L/ala_R as the "anterior" reference.
    if ala_L is not None and ala_R is not None:
        ala_mid = 0.5 * (ala_L + ala_R)
        out["poi_ala"] = _pelvic_from_endplate_pair(S1_post, ala_mid, FH_R, FH_L)  # type: ignore[arg-type]
    else:
        out["poi_ala"] = {"pi_deg": None, "error": "ala landmarks missing"}

    # PI-LL mismatch per variant (when LL provided).
    for variant in ("poi_ap", "poi_ala"):
        v = out.get(variant, {})
        pi = v.get("pi_deg") if isinstance(v, dict) else None
        if pi is not None and lumbar_lordosis_deg is not None:
            v["pi_ll_mismatch_deg"] = round(float(pi) - float(lumbar_lordosis_deg), 3)
        elif isinstance(v, dict):
            v["pi_ll_mismatch_deg"] = None
    return out


def resolve_fullbody_poi_path(dataset_root: Path | str, nako_id: str) -> Path | None:
    """Return the expected fullbody POI json path for one NAKO subject, or ``None`` if it doesn't exist.

    Layout:
    ``<dataset_root>/derivatives-fullbody-poi/{pfx}/{sub}/vibe/sub-{sub}_sequ-stitched_acq-ax_part-water_seg-fullbody_poi.json``
    where ``pfx = sub[:3]`` and ``sub = nako_id.split("_")[0].removeprefix("sub-")``.
    """
    sub = str(nako_id).split("_")[0].replace("sub-", "")
    pfx = sub[:3]
    p = Path(dataset_root) / f"derivatives-fullbody-poi/{pfx}/{sub}/vibe/sub-{sub}_sequ-stitched_acq-ax_part-water_seg-fullbody_poi.json"
    return p if p.exists() else None
