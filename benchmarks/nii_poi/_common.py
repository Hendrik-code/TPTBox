"""Statistics, JSON schema helpers and CLI plumbing shared by all four scripts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_REPEATS = 10
DEFAULT_WARMUP = 1
DEFAULT_LARGE_SIZE = 400
# --quick is what CI runs. A full pass over the 400^3 case is ~2 min, which twice
# over (head + baseline) and twice again (speed + memory) is far too slow to gate
# every PR on; 256^3 is ~4x cheaper and still a large 3D volume. Both sides of a
# comparison always use the same value, so this only changes sensitivity, never
# correctness.
QUICK_REPEATS = 3
QUICK_WARMUP = 1
QUICK_LARGE_SIZE = 256


def summarize(samples: list[float]) -> dict[str, float]:
    """Reduce raw per-iteration samples to the schema both comparison tools read."""
    a = np.asarray(samples, dtype=float)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "mean": float(a.mean()),
        "stddev": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def commit_hash() -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except Exception:
        return "unknown"
    return out.stdout.strip() or "unknown"


def python_version() -> str:
    return ".".join(str(v) for v in sys.version_info[:3])


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--quick", action="store_true", help=f"fewer repeats ({QUICK_REPEATS}) for CI")
    p.add_argument("--repeats", type=int, default=None, help=f"sampled iterations per measurement (default {DEFAULT_REPEATS})")
    p.add_argument("--warmup", type=int, default=None, help=f"discarded iterations per measurement (default {DEFAULT_WARMUP})")
    p.add_argument(
        "--large-size",
        type=int,
        default=None,
        help=f"edge length of the synthetic cases (default {DEFAULT_LARGE_SIZE}, {QUICK_LARGE_SIZE} with --quick)",
    )
    p.add_argument("--cases", type=str, default=None, help="comma-separated subset of case names")
    p.add_argument("--json", type=str, default=None, help="write machine-readable results here")


def resolve_repeats(args: argparse.Namespace) -> tuple[int, int]:
    repeats = args.repeats if args.repeats is not None else (QUICK_REPEATS if args.quick else DEFAULT_REPEATS)
    warmup = args.warmup if args.warmup is not None else (QUICK_WARMUP if args.quick else DEFAULT_WARMUP)
    return max(repeats, 1), max(warmup, 0)


def resolve_large_size(args: argparse.Namespace) -> int:
    if args.large_size is not None:
        return args.large_size
    return QUICK_LARGE_SIZE if args.quick else DEFAULT_LARGE_SIZE


def write_json(path: str | None, doc: dict[str, Any]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_case_table(case_name: str, shape, summaries: dict[str, dict[str, float]], unit: str) -> None:
    """Human-readable per-case table, slowest/largest first."""
    print(f"\n=== {case_name}  shape={tuple(shape)} ===")
    if not summaries:
        print("  (nothing measured)")
        return
    width = max(len(k) for k in summaries)
    for key, s in sorted(summaries.items(), key=lambda kv: -kv[1]["median"]):
        print(f"  {key:<{width}}  {s['median']:10.3f} {unit}   (min {s['min']:.3f}, p90 {s['p90']:.3f}, n={s['n']})")
