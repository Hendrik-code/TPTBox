#!/usr/bin/env python
"""Speed benchmark for the NII and POI hot paths.

Times every measurement in ``measurements.py`` across the six workloads in
``workloads.py`` and writes a JSON document that ``compare.py`` can diff against
a second run (typically the PR base commit).

    python benchmarks/nii_poi/bench_speed.py                       # 10 repeats + 1 warmup
    python benchmarks/nii_poi/bench_speed.py --quick --json head.json
    python benchmarks/nii_poi/bench_speed.py --cases ct_3d,ct_2d --repeats 3
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    add_common_args,
    commit_hash,
    print_case_table,
    python_version,
    resolve_large_size,
    resolve_repeats,
    summarize,
    write_json,
)
from measurements import build_measurements, silenced  # noqa: E402
from workloads import build_case, parse_case_names  # noqa: E402


def time_measurement(call, repeats: int, warmup: int) -> list[float]:
    """Return per-iteration wall times in milliseconds."""
    with silenced():
        for _ in range(warmup):
            call()
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        with silenced():
            t0 = time.perf_counter_ns()
            call()
            t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return samples


def run_case(name: str, large_size: int, repeats: int, warmup: int, tmp: Path) -> dict:
    case = build_case(name, large_size=large_size)
    measurements, metrics, setup_notes = build_measurements(case, tmp)

    summaries: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for meas in measurements:
        try:
            summaries[meas.key] = summarize(time_measurement(meas.call, repeats, warmup))
        except Exception as exc:  # noqa: BLE001 -- a missing/broken op must not abort the run
            skipped.append(f"{meas.key} ({type(exc).__name__}: {exc})")

    print_case_table(case.name, case.shape, summaries, "ms")
    if setup_notes:
        print(f"  setup failed: {'; '.join(setup_notes)}")
    if skipped:
        print(f"  skipped: {'; '.join(skipped)}")

    summaries.update(metrics)
    return {"name": case.name, "shape": list(case.shape), "voxels": case.voxels, "measurements_ms": summaries}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args(argv)
    repeats, warmup = resolve_repeats(args)
    large_size = resolve_large_size(args)
    names = parse_case_names(args.cases)

    print(
        f"speed benchmark | python {python_version()} | commit {commit_hash()} | repeats={repeats} warmup={warmup} large_size={large_size}"
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tptbox-bench-") as td:
        tmp = Path(td)
        cases = [run_case(n, large_size, repeats, warmup, tmp) for n in names]
    print(f"\ntotal wall time: {time.perf_counter() - started:.1f}s")

    write_json(
        args.json,
        {
            "python": python_version(),
            "commit": commit_hash(),
            "repeats": repeats,
            "warmup": warmup,
            "large_size": large_size,
            "cases": cases,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
