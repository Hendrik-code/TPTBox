"""Shared baseline-vs-head comparison: markdown table plus the regression gate.

``compare.py`` and ``compare_mem.py`` are thin wrappers around this; the only
differences are the measurements key (``measurements_ms`` / ``measurements_mb``)
and the unit label.

The gate is deliberately hard to trip. A measurement fails the build only when
*all three* hold:

1. the baseline is at least ``MIN_GATED`` (1 ms / 1 MiB) and the key is not a
   non-timing ``metric_*`` value — sub-unit jitter on a shared runner must never
   block a merge;
2. the regression is practically significant: ``|delta| >= --fail-on-regression-pct``;
3. it is statistically significant: Welch's t-test over the two samples gives
   ``p < --alpha``. With fewer than two samples per side there is no t-test, so
   it falls back to "head's best run is still worse than baseline's p90".

The rendered report shows only the ``TOP_N`` most-changed measurements per case
in the visible table; the rest go in a collapsible ``<details>`` block so the PR
comment stays scannable. Rows whose baseline is below ``NOISE_FLOOR`` get a
``(noise)`` tag: with only three repeats on a shared runner, a sub-unit
baseline can swing ±30% between runs without anything in the code having
changed, and that must be visually obvious.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

MIN_GATED = 1.0
#: Baselines below this are shown with a ``(noise)`` tag; sub-unit measurements
#: on a shared runner jitter by tens of percent between repeats.
NOISE_FLOOR = 3.0
#: How many rows to show up-front per case. The rest go into a <details> block.
TOP_N = 5


def _stats(entry: Any) -> dict[str, float] | None:
    """Normalise a measurement entry; a bare number is treated as min=median=p90."""
    if isinstance(entry, (int, float)):
        v = float(entry)
        return {"min": v, "median": v, "p90": v, "mean": v, "stddev": 0.0, "n": 1}
    if isinstance(entry, dict) and "median" in entry:
        out = {k: float(entry.get(k, entry["median"])) for k in ("min", "median", "p90", "mean")}
        out["stddev"] = float(entry.get("stddev", 0.0))
        out["n"] = int(entry.get("n", 1))
        return out
    return None


def _welch_p(b: dict[str, float], h: dict[str, float]) -> float | None:
    if b["n"] < 2 or h["n"] < 2 or (b["stddev"] == 0.0 and h["stddev"] == 0.0):
        return None
    try:
        from scipy.stats import ttest_ind_from_stats
    except ImportError:
        return None
    res = ttest_ind_from_stats(
        mean1=b["mean"], std1=b["stddev"], nobs1=b["n"], mean2=h["mean"], std2=h["stddev"], nobs2=h["n"], equal_var=False
    )
    p = float(res.pvalue)
    return None if math.isnan(p) else p


@dataclass
class Row:
    key: str
    base: dict[str, float]
    head: dict[str, float]

    @property
    def delta(self) -> float:
        return self.head["median"] - self.base["median"]

    @property
    def pct(self) -> float | None:
        return (self.delta / self.base["median"] * 100.0) if self.base["median"] > 0 else None

    @property
    def p_value(self) -> float | None:
        return _welch_p(self.base, self.head)

    @property
    def gateable(self) -> bool:
        return self.base["median"] >= MIN_GATED and not self.key.startswith("metric_")

    @property
    def is_noisy(self) -> bool:
        """True for rows whose baseline sits in the sub-unit / near-floor range.

        Δ% on these is dominated by runner jitter and should be read as such,
        not as a real change.
        """
        return not self.key.startswith("metric_") and self.base["median"] < NOISE_FLOOR

    def regressed(self, threshold_pct: float, alpha: float) -> bool:
        pct = self.pct
        if not self.gateable or pct is None or pct < threshold_pct:
            return False
        p = self.p_value
        if p is None:
            return self.head["min"] > self.base["p90"]
        return p < alpha


def _fmt(s: dict[str, float]) -> str:
    spread = max(s["p90"] - s["min"], 0.0)
    return f"{s['median']:.2f} ±{spread / 2:.2f}"


def _fmt_pct(row: Row, threshold_pct: float, alpha: float) -> str:
    pct = row.pct
    if pct is None:
        return "n/a"
    if row.regressed(threshold_pct, alpha):
        mark = " 🔴"
    elif row.gateable and pct <= -threshold_pct:
        mark = " 🟢"
    elif row.is_noisy:
        mark = " (noise)"
    else:
        mark = ""
    return f"{pct:+.1f}%{mark}"


def _rank_key(row: Row) -> tuple[int, float]:
    """Sort order for the visible top-N: real changes first, noise last, then |Δ%|.

    Noisy rows are pushed to the bottom of the ranking so a big-percentage swing
    on a sub-ms row does not push a real 15% regression on a 200 ms row into
    the collapsed block.
    """
    pct = row.pct if row.pct is not None else 0.0
    return (1 if row.is_noisy else 0, -abs(pct))


def _render_rows(rows: list[Row], unit: str, threshold_pct: float, alpha: float) -> list[str]:
    lines = [
        f"| Measurement | baseline {unit} (median ±½·range) | head {unit} (median ±½·range) | Δ % | p |",
        "| --- | ---: | ---: | :--- | ---: |",
    ]
    for r in rows:
        p = r.p_value
        lines.append(
            f"| `{r.key}` | {_fmt(r.base)} | {_fmt(r.head)} | {_fmt_pct(r, threshold_pct, alpha)} | {'—' if p is None else f'{p:.3f}'} |"
        )
    return lines


def compare(
    baseline: dict[str, Any],
    head: dict[str, Any],
    *,
    measurements_key: str,
    unit: str,
    threshold_pct: float,
    alpha: float,
    title: str,
) -> tuple[str, bool]:
    """Render the markdown report and report whether anything regressed."""
    base_cases = {c["name"]: c for c in baseline.get("cases", [])}
    head_cases = {c["name"]: c for c in head.get("cases", [])}

    lines: list[str] = [f"## {title}", ""]
    lines.append(
        f"baseline `{baseline.get('commit', '?')}` vs head `{head.get('commit', '?')}` · "
        f"python {head.get('python', '?')} · {head.get('repeats', '?')} repeats + {head.get('warmup', '?')} warmup"
        + (f" · sampler `{head.get('sampler')}`, isolation `{head.get('isolation')}`" if head.get("sampler") else "")
        + (f" · measurement floor ≈ {head['floor_mib']:.2f} MiB" if head.get("floor_mib") else "")
    )
    lines.append("")
    lines.append(
        f"_Showing the {TOP_N} most-changed measurements per case; the rest are collapsed. "
        f"Rows with a baseline below {NOISE_FLOOR:g} {unit} are tagged `(noise)` — runner jitter "
        f"on those swamps any real change._"
    )
    lines.append("")

    any_regression = False
    for name, head_case in head_cases.items():
        shape = tuple(head_case.get("shape", []))
        lines.append(f"### `{name}` — shape {shape}")
        lines.append("")
        base_case = base_cases.get(name)
        if base_case is None:
            lines.append("_new case, no baseline to compare against._")
            lines.append("")
            continue

        b_meas = base_case.get(measurements_key, {})
        h_meas = head_case.get(measurements_key, {})
        rows: list[Row] = []
        new_keys: list[str] = []
        for key, entry in h_meas.items():
            h = _stats(entry)
            b = _stats(b_meas.get(key)) if key in b_meas else None
            if h is None:
                continue
            if b is None:
                new_keys.append(key)
            else:
                rows.append(Row(key, b, h))
        removed_keys = [k for k in b_meas if k not in h_meas]

        rows.sort(key=_rank_key)
        for r in rows:
            any_regression = any_regression or r.regressed(threshold_pct, alpha)

        top = rows[:TOP_N]
        rest = rows[TOP_N:]
        lines.extend(_render_rows(top, unit, threshold_pct, alpha))
        lines.append("")
        if rest:
            lines.append(f"<details><summary>… {len(rest)} more measurements</summary>")
            lines.append("")
            lines.extend(_render_rows(rest, unit, threshold_pct, alpha))
            lines.append("")
            lines.append("</details>")
            lines.append("")
        if new_keys:
            lines.append(f"_new (no baseline): {', '.join(f'`{k}`' for k in new_keys)}_")
            lines.append("")
        if removed_keys:
            lines.append(f"_removed (baseline only): {', '.join(f'`{k}`' for k in removed_keys)}_")
            lines.append("")

    lines.append(
        f"Gate: a measurement fails when the baseline is ≥ {MIN_GATED:g} {unit}, the median grows by "
        f"≥ {threshold_pct:g}%, **and** Welch's t-test gives p < {alpha:g}. `metric_*` rows are context only."
    )
    lines.append("")
    return "\n".join(lines), any_regression


def build_parser(description: str, default_pct: float) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("baseline", help="baseline JSON produced by the bench script")
    p.add_argument("head", help="head JSON produced by the bench script")
    p.add_argument("--fail-on-regression-pct", type=float, default=default_pct, help=f"regression threshold (default {default_pct:g}%%)")
    p.add_argument("--alpha", type=float, default=0.05, help="Welch's t-test significance level (default 0.05)")
    return p
