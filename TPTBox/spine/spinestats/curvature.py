"""Spine curvature metrics beyond Cobb and regional lordosis/kyphosis.

All functions operate on a :class:`TPTBox.POI` in the internal
``("P", "I", "R")`` orientation (posterior/inferior/right axes 0/1/2),
which the pipeline already produces via ``poi.reorient_().rescale_()``.
Coordinates therefore live in millimetres.

Contents
--------
- :func:`compute_sva`
    Sagittal Vertical Axis (mm): horizontal distance in the sagittal
    plane between the C7 plumb line and the posterior-superior corner
    of S1. Positive = C7 is anterior of S1 (typical adult spine).
- :func:`compute_coronal_balance`
    Coronal Balance (mm): horizontal distance between the C7 plumb line
    and the Central Sacral Vertical Line (midpoint of the sacrum).
    Positive = C7 is to the patient's right of the CSVL.
- :func:`compute_wedge_metrics`
    Rewrites the raw x1..x6 heights already stored in
    ``vert_geometry``/``ivd_geometry`` into anterior/posterior and
    left/right wedge angles (in degrees) and wedge indices (unitless).
- :func:`compute_segmental_endplate_angles`
    Inter-vertebral (segmental) wedge angles between the inferior
    endplate of vertebra N and the superior endplate of vertebra N+1.
    Approximates disc wedging without needing the disc mesh.
- :func:`compute_axial_rotation`
    Per-vertebra axial rotation angle (degrees) between
    ``Vertebra_Direction_Right`` and the image-space right axis, in the
    axial (P-R) plane. Positive = rotation towards the patient's left.
- :func:`compute_curvature_profile`
    B-spline through the ``Vertebra_Corpus`` centroids. Returns arc
    length, chord length, tortuosity, sampled |κ|(s), and the arc
    positions plus |κ| values of the largest curvature peaks (apices).
- :func:`compute_multi_cobb`
    Multi-curve scoliotic Cobb detection based on the curvature profile:
    finds inflection points in the coronal projection and returns one
    Cobb angle per curve between neighbouring inflections.
- :data:`EXTENDED_CURVATURE_DEFINITIONS`
    Additional entries for ``angles.curvature_definition``:
    ``t1_slope``, ``c2_c7_angle``, ``cervical_sva_helper``. Merge into
    the default map when you want them in the lordosis/kyphosis output.

Design notes
------------
- Every function is total: on missing landmarks it returns a dict with
  the metric keys set to ``None``/``NaN`` and an ``"error"`` message.
- No side effects on the input POI beyond ``reorient_().rescale_()``
  (which the pipeline already does).
- The Cobb/lordosis code in ``angles.py`` stays untouched.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
from numpy.linalg import norm

from TPTBox import POI, Location, Vertebra_Instance
from TPTBox.spine.spinestats.angles import Def_Curvature, MoveTo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Internal POI axes after reorient/rescale: 0=P, 1=I, 2=R.
_AX_P, _AX_I, _AX_R = 0, 1, 2


def _prep(poi: POI) -> POI:
    """Return the POI in the internal ``(P, I, R)`` orientation at 1 mm scale."""
    return poi.reorient().rescale(verbose=False)


def _get(poi: POI, vert: int | Vertebra_Instance, loc: Location) -> np.ndarray | None:
    """Fetch a coordinate as ``np.array`` or ``None`` if not present."""
    v = vert.value if isinstance(vert, Vertebra_Instance) else vert
    key = (v, loc.value if isinstance(loc, Location) else loc)
    if key not in poi:
        return None
    return np.asarray(poi[key], dtype=float)


def _corpus(poi: POI, vert: int | Vertebra_Instance) -> np.ndarray | None:
    return _get(poi, vert, Location.Vertebra_Corpus)


def _endplate(poi: POI, vert: int | Vertebra_Instance, superior: bool) -> np.ndarray | None:
    """Return the (approximate) endplate center of ``vert``.

    Uses the standard ``Endplate`` subregion. If it's not there for both
    endplates individually, fall back to the corpus + inferior direction
    to estimate the endplate midpoint.
    """
    loc = Location.Vertebral_Body_Endplate_Superior if superior else Location.Vertebral_Body_Endplate_Inferior
    p = _get(poi, vert, loc)
    if p is not None:
        return p
    corp = _corpus(poi, vert)
    inf = _get(poi, vert, Location.Vertebra_Direction_Inferior)
    if corp is None or inf is None:
        return None
    d = inf - corp
    n = norm(d)
    if n == 0:
        return corp
    return corp + (d / n) * (5.0 if not superior else -5.0)


def _last_present(poi: POI, candidates: list[Vertebra_Instance]) -> Vertebra_Instance | None:
    for v in candidates:
        if _corpus(poi, v) is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# 1. Sagittal Vertical Axis (SVA)
# ---------------------------------------------------------------------------


_SVA_BASE_FALLBACKS = [Vertebra_Instance.S1, Vertebra_Instance.L5, Vertebra_Instance.L4]


def compute_sva(poi: POI, top_vert: Vertebra_Instance = Vertebra_Instance.C7) -> dict[str, Any]:
    """Sagittal Vertical Axis: signed horizontal offset (mm) in the sagittal plane.

    Definition
    ----------
    Drop a plumb line from the center of the ``top_vert`` (default C7)
    vertebra body downwards (image-inferior axis). Measure the signed
    distance along the posterior-anterior axis to the posterior-superior
    corner of S1 (falls back to L5, then L4, if S1 is absent). Positive
    values → top vertebra is anterior of the base reference (positive
    sagittal balance, typical adult).

    Returns:
    -------
    dict
        - ``sva_mm``: signed offset in mm (or ``None`` if landmarks missing)
        - ``top_vertebra`` / ``base_vertebra`` / ``base_landmark``
        - ``top_pi_coords`` / ``base_pi_coords``
        - ``error``: message if computation failed
    """
    poi = _prep(poi)
    top = _corpus(poi, top_vert)
    if top is None:
        return {"sva_mm": None, "error": f"{top_vert.name} corpus missing"}

    base_vert = None
    base_ref = None
    landmark = None
    for cand in _SVA_BASE_FALLBACKS:
        corp = _corpus(poi, cand)
        if corp is None:
            continue
        ep = _endplate(poi, cand, superior=True)
        base_vert = cand
        base_ref = ep if ep is not None else corp
        landmark = f"{cand.name}_endplate_superior" if ep is not None else f"{cand.name}_corpus"
        break
    if base_ref is None or base_vert is None:
        return {"sva_mm": None, "error": "no base vertebra (S1/L5/L4) available"}

    sva = float(base_ref[_AX_P] - top[_AX_P])
    return {
        "sva_mm": round(sva, 2),
        "top_vertebra": top_vert.name,
        "base_vertebra": base_vert.name,
        "base_landmark": landmark,
        "top_pi_coords": [round(float(top[_AX_P]), 2), round(float(top[_AX_I]), 2)],
        "base_pi_coords": [round(float(base_ref[_AX_P]), 2), round(float(base_ref[_AX_I]), 2)],
    }


# ---------------------------------------------------------------------------
# 2. Coronal Balance
# ---------------------------------------------------------------------------


def compute_coronal_balance(poi: POI, top_vert: Vertebra_Instance = Vertebra_Instance.C7) -> dict[str, Any]:
    """Coronal Balance: signed horizontal offset (mm) in the coronal plane.

    Definition
    ----------
    Distance along the patient's right axis between ``top_vert`` (default
    C7) and the Central Sacral Vertical Line, taken here as the
    R-coordinate of the S1 corpus (falls back to L5, then L4).
    Positive → top vertebra is to the patient's right of the CSVL.

    Returns:
    -------
    dict
        - ``coronal_balance_mm``
        - ``top_vertebra`` / ``base_vertebra``
        - ``top_r_coord`` / ``base_r_coord``
        - ``error``
    """
    poi = _prep(poi)
    top = _corpus(poi, top_vert)
    if top is None:
        return {"coronal_balance_mm": None, "error": f"{top_vert.name} corpus missing"}
    base = None
    base_vert = None
    for cand in _SVA_BASE_FALLBACKS:
        c = _corpus(poi, cand)
        if c is not None:
            base = c
            base_vert = cand
            break
    if base is None or base_vert is None:
        return {"coronal_balance_mm": None, "error": "no base vertebra (S1/L5/L4) available"}
    cb = float(top[_AX_R] - base[_AX_R])
    return {
        "coronal_balance_mm": round(cb, 2),
        "top_vertebra": top_vert.name,
        "base_vertebra": base_vert.name,
        "top_r_coord": round(float(top[_AX_R]), 2),
        "base_r_coord": round(float(base[_AX_R]), 2),
    }


# ---------------------------------------------------------------------------
# 3. + 5. Wedge metrics on top of the existing x1..x6 measurements
# ---------------------------------------------------------------------------


def compute_wedge_metrics(geometry: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """Wedge angles and indices derived from the already-computed x1..x6.

    Given a ``vert_geometry`` or ``ivd_geometry`` section (label →
    metrics with ``anterior_height_x1``, ``posterior_height_x2``,
    ``right_height_x3``, ``left_height_x4``, ``width_sagittal_x6``,
    ``width_lateral_x5``), this returns per label:

    - ``sagittal_wedge_deg`` = atan((x1 − x2) / x6)  (positive = anterior taller)
    - ``coronal_wedge_deg``  = atan((x3 − x4) / x5)  (positive = right taller)
    - ``sagittal_wedge_index`` = (x1 − x2) / mean(x1, x2)
    - ``coronal_wedge_index``  = (x3 − x4) / mean(x3, x4)

    Labels with missing / non-positive inputs get NaN for that metric.
    Non-destructive: returns a fresh dict.
    """
    out: dict[int, dict[str, float]] = {}
    for label, m in geometry.items():
        if not isinstance(m, dict):
            continue
        x1 = m.get("anterior_height_x1")
        x2 = m.get("posterior_height_x2")
        x3 = m.get("right_height_x3")
        x4 = m.get("left_height_x4")
        x5 = m.get("width_lateral_x5")
        x6 = m.get("width_sagittal_x6")

        def _finite(v):
            try:
                return v is not None and np.isfinite(float(v))
            except (TypeError, ValueError):
                return False

        row: dict[str, float] = {}
        if _finite(x1) and _finite(x2) and _finite(x6) and float(x6) > 0:
            row["sagittal_wedge_deg"] = round(float(np.degrees(np.arctan2(float(x1) - float(x2), float(x6)))), 3)
            avg = (float(x1) + float(x2)) / 2.0
            row["sagittal_wedge_index"] = round((float(x1) - float(x2)) / avg, 4) if avg > 0 else np.nan
        else:
            row["sagittal_wedge_deg"] = np.nan
            row["sagittal_wedge_index"] = np.nan

        if _finite(x3) and _finite(x4) and _finite(x5) and float(x5) > 0:
            row["coronal_wedge_deg"] = round(float(np.degrees(np.arctan2(float(x3) - float(x4), float(x5)))), 3)
            avg = (float(x3) + float(x4)) / 2.0
            row["coronal_wedge_index"] = round((float(x3) - float(x4)) / avg, 4) if avg > 0 else np.nan
        else:
            row["coronal_wedge_deg"] = np.nan
            row["coronal_wedge_index"] = np.nan

        out[label] = row
    return out


# ---------------------------------------------------------------------------
# 4. Segmental endplate angles (inter-vertebral wedge)
# ---------------------------------------------------------------------------


def compute_segmental_endplate_angles(poi: POI) -> dict[str, float]:
    """Angle (degrees) between adjacent vertebral endplates in the sagittal plane.

    For each pair of neighbouring vertebrae with both endplate normals
    available, computes the signed angle between (a) the inferior
    endplate line of vertebra N and (b) the superior endplate line of
    vertebra N+1, projected onto the sagittal plane. Approximates disc
    wedging without needing the disc mesh.

    Endplate line: perpendicular to ``Vertebra_Direction_Inferior``
    (which is normal to the endplate), in the sagittal plane.

    Returns:
    -------
    dict keyed by ``"<name_upper>-<name_lower>"``, e.g. ``"L4-L5"``.
    Positive value → anterior opening (typical lordotic disc).
    """
    poi = _prep(poi)
    order = Vertebra_Instance.order_dict()
    ordered = sorted(
        [v for v in Vertebra_Instance if _get(poi, v, Location.Vertebra_Direction_Inferior) is not None],
        key=lambda v: order.get(v.value, v.value),
    )
    out: dict[str, float] = {}
    for a, b in pairwise(ordered):
        if b.value - a.value not in (1,) and not (a.value < 25 and b.value < 25 and b.value == a.value + 1):
            # Only measure real neighbours (skip jumps like L5→S1 numbering gap).
            pass
        ca = _corpus(poi, a)
        cb = _corpus(poi, b)
        ia = _get(poi, a, Location.Vertebra_Direction_Inferior)
        ib = _get(poi, b, Location.Vertebra_Direction_Inferior)
        if ca is None or cb is None or ia is None or ib is None:
            continue
        # Endplate normal (inferior direction) → project onto sagittal (P-I) plane.
        na = np.array([ia[_AX_P] - ca[_AX_P], ia[_AX_I] - ca[_AX_I]])
        nb = np.array([ib[_AX_P] - cb[_AX_P], ib[_AX_I] - cb[_AX_I]])
        na_n = norm(na)
        nb_n = norm(nb)
        if na_n == 0 or nb_n == 0:
            continue
        na /= na_n
        nb /= nb_n
        # Signed angle between the two inferior normals in the sagittal plane.
        cross = na[0] * nb[1] - na[1] * nb[0]
        dot = float(np.clip(na @ nb, -1.0, 1.0))
        ang = float(np.degrees(np.arctan2(cross, dot)))
        out[f"{a.name}-{b.name}"] = round(ang, 3)
    return out


# ---------------------------------------------------------------------------
# 6. Axial rotation per vertebra
# ---------------------------------------------------------------------------


def compute_axial_rotation(poi: POI) -> dict[str, float]:
    """Per-vertebra axial rotation in the axial (P-R) plane, in degrees.

    Definition: signed angle between ``Vertebra_Direction_Right`` (from
    the vertebral body's centroid) and the image-space right axis,
    measured in the axial plane. 0° = right-direction aligned with image
    R axis, positive = rotation toward patient's left (counterclockwise
    when viewed from superior).

    Requires ``Vertebra_Direction_Right`` and ``Vertebra_Corpus`` in the POI.
    """
    poi = _prep(poi)
    out: dict[str, float] = {}
    for v in Vertebra_Instance:
        c = _corpus(poi, v)
        r = _get(poi, v, Location.Vertebra_Direction_Right)
        if c is None or r is None:
            continue
        d = r - c
        # Axial plane = (P, R) plane; drop the I component.
        planar = np.array([d[_AX_P], d[_AX_R]])
        n = norm(planar)
        if n == 0:
            continue
        planar /= n
        # Reference axis = image right = (0, 1) in (P, R).
        ang = float(np.degrees(np.arctan2(planar[0], planar[1])))
        out[v.name] = round(ang, 3)
    return out


# ---------------------------------------------------------------------------
# 7. Curvature profile via spline + apex detection
# ---------------------------------------------------------------------------


def _numerical_curvature(points: np.ndarray) -> np.ndarray:
    """|κ(s)| along an equidistantly-sampled curve (2D or 3D), via finite differences."""
    if len(points) < 3:
        return np.zeros(len(points))
    dp = np.gradient(points, axis=0)
    ddp = np.gradient(dp, axis=0)
    # 2D branch: signed curvature magnitude. 3D branch: cross-product norm.
    num = np.abs(dp[:, 0] * ddp[:, 1] - dp[:, 1] * ddp[:, 0]) if points.shape[1] == 2 else norm(np.cross(dp, ddp), axis=1)
    den = norm(dp, axis=1) ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(den > 0, num / den, 0.0)
    return k


def compute_curvature_profile(
    poi: POI,
    smoothness: int = 10,
    samples_per_poi: int = 20,
    top_k_apices: int = 6,
) -> dict[str, Any]:
    """Curvature profile of the spine, based on the internal ``fit_spline``.

    Fits a cubic B-spline through the ``Vertebra_Corpus`` centroids
    (sorted by ``Vertebra_Instance.order_dict()``) and returns:

    - ``arc_length_mm`` / ``chord_length_mm`` / ``tortuosity`` (arc/chord)
    - ``curvature_max_1_per_mm`` / ``curvature_mean_1_per_mm``
    - ``apices``: list of ``{arc_mm, kappa_1_per_mm}`` for the ``top_k_apices``
      local maxima of |κ|, sorted by arc position (superior → inferior).
    - ``sagittal_apices`` / ``coronal_apices``: same, but on the
      projected 2D curve in each plane (better matches the clinical
      notion of "the apex of a scoliotic curve").

    All units follow the POI: mm.
    """
    poi = _prep(poi)
    # fit_spline expects at least a handful of points.
    if len(poi.extract_subregion(Location.Vertebra_Corpus)) < 4:
        return {"error": "too few Vertebra_Corpus POIs to fit a spline"}
    try:
        pts, _der = poi.fit_spline(
            smoothness=smoothness,
            samples_per_poi=samples_per_poi,
            location=Location.Vertebra_Corpus,
            vertebra=True,
        )
    except Exception as e:
        return {"error": f"fit_spline failed: {e}"}

    seg_len = norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc_length = float(arc[-1])
    chord_length = float(norm(pts[-1] - pts[0]))
    tortuosity = arc_length / chord_length if chord_length > 0 else np.nan

    kappa_3d = _numerical_curvature(pts)
    kappa_sag = _numerical_curvature(pts[:, [_AX_P, _AX_I]])
    kappa_cor = _numerical_curvature(pts[:, [_AX_R, _AX_I]])

    apices_3d = _find_apices(arc, kappa_3d, top_k_apices)
    apices_sag = _find_apices(arc, kappa_sag, top_k_apices)
    apices_cor = _find_apices(arc, kappa_cor, top_k_apices)

    return {
        "arc_length_mm": round(arc_length, 2),
        "chord_length_mm": round(chord_length, 2),
        "tortuosity": round(float(tortuosity), 5) if np.isfinite(tortuosity) else None,
        "curvature_max_1_per_mm": round(float(kappa_3d.max()), 6),
        "curvature_mean_1_per_mm": round(float(kappa_3d.mean()), 6),
        "curvature_sagittal_max_1_per_mm": round(float(kappa_sag.max()), 6),
        "curvature_coronal_max_1_per_mm": round(float(kappa_cor.max()), 6),
        "apices": apices_3d,
        "sagittal_apices": apices_sag,
        "coronal_apices": apices_cor,
    }


def _find_apices(arc: np.ndarray, kappa: np.ndarray, top_k: int) -> list[dict[str, float]]:
    """Return up to ``top_k`` local maxima of ``kappa`` as ``{arc_mm, kappa_...}`` dicts."""
    if len(kappa) < 3:
        return []
    peak_mask = (kappa[1:-1] > kappa[:-2]) & (kappa[1:-1] > kappa[2:])
    peak_idx = np.flatnonzero(peak_mask) + 1
    if len(peak_idx) == 0:
        return []
    peak_idx = peak_idx[np.argsort(-kappa[peak_idx])[:top_k]]
    peak_idx = np.sort(peak_idx)
    return [{"arc_mm": round(float(arc[i]), 2), "kappa_1_per_mm": round(float(kappa[i]), 6)} for i in peak_idx]


# ---------------------------------------------------------------------------
# 8. Multi-curve Cobb via inflection points
# ---------------------------------------------------------------------------


def compute_multi_cobb(poi: POI, min_curve_length_mm: float = 30.0) -> dict[str, Any]:
    """Automatic multi-curve Cobb detection from the coronal spline projection.

    Fits the ``Vertebra_Corpus`` spline (as in :func:`compute_curvature_profile`)
    and projects it onto the coronal (R, I) plane. Finds inflection
    points as sign changes of the coronal signed curvature; between each
    pair of consecutive inflections, the local Cobb angle is measured as
    the angle between the tangent at the start and end of that segment.

    Curves shorter than ``min_curve_length_mm`` are dropped as noise.

    Returns:
    -------
    dict
        - ``curves``: list of dicts with ``arc_start_mm``, ``arc_end_mm``,
          ``apex_arc_mm``, ``length_mm``, ``cobb_deg``, and
          ``handedness`` (``"right"``/``"left"`` = direction of the
          curve's concavity).
        - ``max_cobb_deg``: maximum |Cobb| across all detected curves.
    """
    poi = _prep(poi)
    if len(poi.extract_subregion(Location.Vertebra_Corpus)) < 4:
        return {"error": "too few Vertebra_Corpus POIs"}
    try:
        pts, _der = poi.fit_spline(location=Location.Vertebra_Corpus, vertebra=True, smoothness=10, samples_per_poi=20)
    except Exception as e:
        return {"error": f"fit_spline failed: {e}"}

    # Project onto coronal plane (R, I): axis 2 = R (x), axis 1 = I (y).
    cor = pts[:, [_AX_R, _AX_I]]
    dp = np.gradient(cor, axis=0)
    ddp = np.gradient(dp, axis=0)
    signed_k = dp[:, 0] * ddp[:, 1] - dp[:, 1] * ddp[:, 0]
    seg_len = norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])

    sign = np.sign(signed_k)
    change = np.flatnonzero(sign[1:] != sign[:-1]) + 1
    boundaries = np.concatenate([[0], change, [len(pts) - 1]])

    curves: list[dict[str, Any]] = []
    for a, b in pairwise(boundaries):
        if arc[b] - arc[a] < min_curve_length_mm:
            continue
        # Tangents at each end.
        t0 = dp[a]
        t1 = dp[b]
        n0, n1 = norm(t0), norm(t1)
        if n0 == 0 or n1 == 0:
            continue
        cos = float(np.clip((t0 @ t1) / (n0 * n1), -1.0, 1.0))
        cobb = float(np.degrees(np.arccos(cos)))
        # Apex = |signed_k| max within the segment.
        seg_k = np.abs(signed_k[a : b + 1])
        apex_local = int(np.argmax(seg_k)) + a
        handedness = "right" if signed_k[apex_local] > 0 else "left"
        curves.append(
            {
                "arc_start_mm": round(float(arc[a]), 2),
                "arc_end_mm": round(float(arc[b]), 2),
                "apex_arc_mm": round(float(arc[apex_local]), 2),
                "length_mm": round(float(arc[b] - arc[a]), 2),
                "cobb_deg": round(cobb, 3),
                "handedness": handedness,
            }
        )
    max_cobb = max((c["cobb_deg"] for c in curves), default=0.0)
    return {"curves": curves, "max_cobb_deg": round(max_cobb, 3)}


# ---------------------------------------------------------------------------
# 8b. Additional curvature definitions (cervical / T1-slope)
# ---------------------------------------------------------------------------

EXTENDED_CURVATURE_DEFINITIONS: dict[str, Def_Curvature] = {
    # T1 slope: angle of the T1 superior endplate to horizontal — approximated
    # as the "lordosis" measured between T1 top and T1 bottom.
    "t1_slope": Def_Curvature(Vertebra_Instance.T1, MoveTo.TOP, Vertebra_Instance.T1, MoveTo.BOTTOM),
    # C2-C7 angle: the classic cervical Cobb between C2 inferior endplate
    # and C7 inferior endplate.
    "c2_c7_angle": Def_Curvature(Vertebra_Instance.C2, MoveTo.BOTTOM, Vertebra_Instance.C7, MoveTo.BOTTOM),
}
