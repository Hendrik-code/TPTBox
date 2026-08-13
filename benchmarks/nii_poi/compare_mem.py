#!/usr/bin/env python
"""Compare two ``bench_mem.py`` JSON files and emit a markdown report.

Writes the report to stdout and exits 1 if any measurement regressed past the
threshold (see ``_compare_core`` for the exact gate).

    python benchmarks/nii_poi/compare_mem.py baseline_mem.json head_mem.json --fail-on-regression-pct 50
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_json  # noqa: E402
from _compare_core import build_parser, compare  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = build_parser(__doc__, default_pct=50.0).parse_args(argv)
    report, regressed = compare(
        load_json(args.baseline),
        load_json(args.head),
        measurements_key="measurements_mb",
        unit="MiB",
        threshold_pct=args.fail_on_regression_pct,
        alpha=args.alpha,
        title="Memory (peak RSS growth per call)",
    )
    print(report)
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
