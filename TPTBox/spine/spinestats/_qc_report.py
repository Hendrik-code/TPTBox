"""Generate a QC report for the aggregated NAKO Excel outputs.

Produces one .xlsx file next to the input tables with:
- one summary sheet listing coverage, missing/error counts, distributions, and
  the count of "outlier" values per metric (values outside a hard-coded
  physiological / plausible range);
- one sheet per outlier metric listing the subjects (or subject+label rows)
  that fall outside that range so they can be reviewed by hand.

Usage
-----
    python -m TPTBox.spine.spinestats._qc_report [<folder>]

If no folder is passed, defaults to
``/DATA/NAS/ongoing_projects/robert/test/NAKO-stats``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Outlier ranges (inclusive). A value outside this range is flagged for review.
# These are chosen wide enough to keep almost all real biological variation,
# so what remains is very likely a segmentation / landmark artifact.
# --------------------------------------------------------------------------

# per_subject metrics
OUTLIER_SUBJECT: dict[str, tuple[float, float]] = {
    "sva.sva_mm": (-80, 100),
    "coronal_balance.coronal_balance_mm": (-80, 80),
    "curvature_profile.arc_length_mm": (350, 700),
    "curvature_profile.tortuosity": (1.0, 1.20),
    "curvature_profile.curvature_max_1_per_mm": (0.0, 0.05),
    "multi_cobb.max_cobb_deg": (0, 60),
    "curv.cervical_lordosis": (0, 80),
    "curv.thoracic_kyphosis": (0, 80),
    "curv.lumbar_lordosis": (0, 80),
    "pelvic_parameters.poi_ap.pi_deg": (10, 90),
    "pelvic_parameters.poi_ap.pt_deg": (-30, 60),
    "pelvic_parameters.poi_ap.ss_deg": (-15, 75),
    "pelvic_parameters.poi_ap.pi_ll_mismatch_deg": (-40, 40),
    # NAKO T2w calibration shifts VBQ ~0.2 lower than clinical values from the
    # literature. Empirical NAKO distribution: median 0.28, IQR [0.24, 0.33].
    # Range chosen wide enough to keep all normal biological variation while
    # flagging clear segmentation / signal-normalization failures.
    "VBQ_score.VBQ_L1-L4": (0.1, 0.8),
}

# per_vertebra metrics (also apply to per_ivd where the column exists)
OUTLIER_LABEL: dict[str, tuple[float, float]] = {
    "axial_rotation_deg": (-30, 30),
    "endplate_internal_angle_deg": (0, 45),
    "sagittal_wedge_deg": (-30, 30),
    "coronal_wedge_deg": (-20, 20),
    "sagittal_wedge_index": (-0.6, 0.6),
    "coronal_wedge_index": (-0.6, 0.6),
    "segmental_endplate_angle_deg": (-30, 30),
}

TOP_OFFENDERS = 50


# --------------------------------------------------------------------------
# Report builders
# --------------------------------------------------------------------------


def _summary_row(df: pd.DataFrame, col: str, lo: float, hi: float) -> dict:
    v = pd.to_numeric(df[col], errors="coerce")
    n_total = len(df)
    n_valid = int(v.notna().sum())
    v_ok = v.dropna()
    mask_out = (v_ok < lo) | (v_ok > hi)
    n_out = int(mask_out.sum())
    if n_valid == 0:
        return {
            "column": col, "n_total": n_total, "n_valid": 0, "n_missing": n_total,
            "median": None, "iqr_low": None, "iqr_high": None, "min": None, "max": None,
            "outlier_range": f"[{lo}, {hi}]", "n_outliers": 0, "pct_outliers": None,
        }
    return {
        "column": col,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing": n_total - n_valid,
        "median": round(float(v_ok.median()), 3),
        "iqr_low": round(float(v_ok.quantile(0.25)), 3),
        "iqr_high": round(float(v_ok.quantile(0.75)), 3),
        "min": round(float(v_ok.min()), 3),
        "max": round(float(v_ok.max()), 3),
        "outlier_range": f"[{lo}, {hi}]",
        "n_outliers": n_out,
        "pct_outliers": round(100.0 * n_out / n_valid, 3),
    }


def _outlier_frame(
    df: pd.DataFrame, col: str, lo: float, hi: float, extra_cols: list[str]
) -> pd.DataFrame:
    v = pd.to_numeric(df[col], errors="coerce")
    mask = v.notna() & ((v < lo) | (v > hi))
    if not mask.any():
        return pd.DataFrame(columns=["subject", col, *extra_cols])
    dfo = df.loc[mask, [c for c in ["subject", "label", col, *extra_cols] if c in df.columns]].copy()
    dfo = dfo.sort_values(col, key=lambda s: pd.to_numeric(s, errors="coerce").abs(), ascending=False)
    return dfo.head(TOP_OFFENDERS)


def build_qc_report(folder: Path) -> Path:
    sub_p = folder / "per_subject.xlsx"
    vert_p = folder / "per_vertebra.xlsx"
    ivd_p = folder / "per_ivd.xlsx"
    out_p = folder / "qc_report.xlsx"

    print(f"loading {sub_p} ...", flush=True)
    sub = pd.read_excel(sub_p)
    print(f"loading {vert_p} ...", flush=True)
    vert = pd.read_excel(vert_p)
    print(f"loading {ivd_p} ...", flush=True)
    ivd = pd.read_excel(ivd_p)

    # ------------------------------------------------------------------
    # Header sheet
    # ------------------------------------------------------------------
    header = pd.DataFrame(
        [
            {"item": "n_subjects", "value": len(sub)},
            {"item": "per_subject_cols", "value": len(sub.columns)},
            {"item": "n_vertebra_rows", "value": len(vert)},
            {"item": "n_ivd_rows", "value": len(ivd)},
            {"item": "pelvic_error_rate_%",
             "value": round(100.0 * sub.get("pelvic_parameters.error", pd.Series([np.nan] * len(sub))).notna().sum() / len(sub), 3)},
        ]
    )

    # PI = PT + SS invariant check
    if "pelvic_parameters.poi_ap.pi_deg" in sub.columns:
        pi = pd.to_numeric(sub["pelvic_parameters.poi_ap.pi_deg"], errors="coerce")
        pt = pd.to_numeric(sub["pelvic_parameters.poi_ap.pt_deg"], errors="coerce")
        ss = pd.to_numeric(sub["pelvic_parameters.poi_ap.ss_deg"], errors="coerce")
        diff = (pi - (pt + ss)).abs()
        n_valid = int(pi.notna().sum())
        n_ok = int((diff < 0.1).sum())
        header = pd.concat(
            [
                header,
                pd.DataFrame(
                    [
                        {"item": "PI_valid", "value": n_valid},
                        {"item": "PI=PT+SS_within_0.1_deg", "value": n_ok},
                        {"item": "PI=PT+SS_violation_pct", "value": round(100.0 * (n_valid - n_ok) / max(n_valid, 1), 3)},
                    ]
                ),
            ],
            ignore_index=True,
        )

    # ------------------------------------------------------------------
    # Per-column summary
    # ------------------------------------------------------------------
    sub_summary = pd.DataFrame(
        [_summary_row(sub, c, lo, hi) for c, (lo, hi) in OUTLIER_SUBJECT.items() if c in sub.columns]
    )
    vert_summary = pd.DataFrame(
        [_summary_row(vert, c, lo, hi) for c, (lo, hi) in OUTLIER_LABEL.items() if c in vert.columns]
    )
    ivd_summary = pd.DataFrame(
        [_summary_row(ivd, c, lo, hi) for c, (lo, hi) in OUTLIER_LABEL.items() if c in ivd.columns]
    )

    # ------------------------------------------------------------------
    # Outlier rows
    # ------------------------------------------------------------------
    print("writing report ...", flush=True)
    with pd.ExcelWriter(out_p, engine="xlsxwriter") as w:
        header.to_excel(w, sheet_name="_header", index=False)
        sub_summary.to_excel(w, sheet_name="summary_subject", index=False)
        vert_summary.to_excel(w, sheet_name="summary_vertebra", index=False)
        ivd_summary.to_excel(w, sheet_name="summary_ivd", index=False)

        # Subject outliers
        for col, (lo, hi) in OUTLIER_SUBJECT.items():
            if col not in sub.columns:
                continue
            dfo = _outlier_frame(sub, col, lo, hi, extra_cols=[])
            if not dfo.empty:
                sn = _sheet_name(col, prefix="s_")
                dfo.to_excel(w, sheet_name=sn, index=False)

        # Per-vertebra outliers
        for col, (lo, hi) in OUTLIER_LABEL.items():
            if col not in vert.columns:
                continue
            dfo = _outlier_frame(vert, col, lo, hi, extra_cols=["label"])
            if not dfo.empty:
                sn = _sheet_name(col, prefix="v_")
                dfo.to_excel(w, sheet_name=sn, index=False)

        # Per-IVD outliers
        for col, (lo, hi) in OUTLIER_LABEL.items():
            if col not in ivd.columns:
                continue
            dfo = _outlier_frame(ivd, col, lo, hi, extra_cols=["label"])
            if not dfo.empty:
                sn = _sheet_name(col, prefix="i_")
                dfo.to_excel(w, sheet_name=sn, index=False)

    print(f"wrote {out_p} ({out_p.stat().st_size} bytes)")
    return out_p


def _sheet_name(col: str, prefix: str = "") -> str:
    r"""Excel sheet names must be <=31 chars and cannot contain []:*?/\ ."""
    s = prefix + col.replace(".", "_").replace(":", "_")
    for ch in "[]:*?/\\":
        s = s.replace(ch, "_")
    return s[:31]


if __name__ == "__main__":
    default = Path("/DATA/NAS/ongoing_projects/robert/test/NAKO-stats")
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    build_qc_report(folder)
