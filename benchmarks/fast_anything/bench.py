"""Check and time the Rust port of ``corner_bisection_search`` against the Python original, in one process.

Run from the repo root after ``fast-anything verify`` built the port into ``.fast/lib``::

    python -m benchmarks.fast_anything.bench [--rounds 7]

1. Exactness: every call the POI workload makes, plus seeded random cases (size-1 axes, start
   outside, delta 0/-inf/NaN, NaN/inf coordinates), must give bit-identical results (or both raise).
2. Kernel: time per call on the real calls, for the Python original (scipy ``RegularGridInterpolator``),
   a pure-Python reference without scipy (same arithmetic, ``direct_python`` below; not used by TPTBox)
   and the Rust port.
3. Pipeline: ``calc_poi_from_subreg_vert`` per sample, swapping only this one function, rotated rounds.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

os.environ["FAST_ANYTHING_DISABLE"] = "1"  # keep the Python original in the module; Rust is loaded below

import numpy as np

from TPTBox import calc_poi_from_subreg_vert, to_nii
from TPTBox.core.poi_fun import strategies

from .workload import REPO_ROOT, SAMPLES, all_poi_functions

PORT = "fa_tptbox_core_poi_fun_strategies_corner_bisection_search"
SUBREG_IDS = list(range(40, 51)) + sorted(all_poi_functions)


def load_rust():
    sys.path.append(str(REPO_ROOT / ".fast" / "lib"))
    try:
        return __import__(PORT).corner_bisection_search
    except ImportError:
        raise SystemExit("Rust port not built: run `fast-anything verify TPTBox.core.poi_fun.strategies.corner_bisection_search`") from None


def direct_python(region, start, v1, v2, delta):
    """The same search with scipy's linear interpolation written out on Python floats (length-3 inputs only)."""
    n = region.shape
    upper = [n[0] - 1, n[1] - 1, n[2] - 1]
    flat = region.tobytes()
    s0, s1, s2 = (float(x) for x in start)
    a0, a1, a2 = (float(x) for x in v1)
    b0, b1, b2 = (float(x) for x in v2)

    def inside(f, g):
        c = (s0 + f * a0 + g * b0, s1 + f * a1 + g * b1, s2 + f * a2 + g * b2)
        if c[0] < 0 or c[1] < 0 or c[2] < 0 or c[0] > upper[0] or c[1] > upper[1] or c[2] > upper[2]:
            return False
        if c[0] != c[0] or c[1] != c[1] or c[2] != c[2]:
            raise ValueError("NaN coordinate")
        idx, w = [], []
        for d in range(3):
            if n[d] == 1:
                i, y = 0, 0.0
                idx.append((0, 0))
            else:
                i = n[d] - 2 if c[d] >= upper[d] else math.floor(c[d])
                y = c[d] - i
                idx.append((i, i + 1))
            w.append((1.0 - y, y))
        value = 0.0
        for a in (0, 1):
            for b in (0, 1):
                for e in (0, 1):
                    v = flat[(idx[0][a] * n[1] + idx[1][b]) * n[2] + idx[2][e]]
                    if v:
                        value += v * (w[0][a] * w[1][b] * w[2][e])
        return value > 0.5

    factor1 = factor2 = 1.0
    while not inside(factor1 + 0.0, factor2 + 0.0):
        shrunk1, shrunk2 = factor1 * 0.9, factor2 * 0.7
        if shrunk1 == factor1 and shrunk2 == factor2:
            return None
        factor1, factor2 = shrunk1, shrunk2
    v1_n, v2_n, f1, f2 = 1.0, 0.4, 0.0, 0.0
    while delta < (v1_n + v2_n) / 2:
        state = (f1, f2, v1_n, v2_n)
        changed = False
        if inside(factor1 + (v1_n + f1), factor2 + f2):
            f1 += v1_n
            changed = True
        if inside(factor1 + f1, factor2 + (f2 + v2_n)):
            f2 += v2_n
            changed = True
        if not changed:
            v1_n /= 2
            v2_n /= 2
        if (f1, f2, v1_n, v2_n) == state:
            break
    f, g = factor1 + f1, factor2 + f2
    return np.array([s0 + f * a0 + g * b0, s1 + f * a1 + g * b1, s2 + f * a2 + g * b2])


def capture_real_calls(python_impl):
    calls = []

    def recording(*args):
        calls.append(tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args))
        return python_impl(*args)

    strategies.corner_bisection_search = recording
    try:
        for vert_path, subreg_path in SAMPLES.values():
            calc_poi_from_subreg_vert(to_nii(vert_path, True), to_nii(subreg_path, True), subreg_id=SUBREG_IDS, verbose=False)
    finally:
        strategies.corner_bisection_search = python_impl
    return calls


def random_cases(n, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for k in range(n):
        shape = tuple(int(x) for x in rng.integers(1, 30, 3))
        if k % 7 == 0:
            shape = (1, *shape[1:])  # size-1 axis
        grid = np.indices(shape).transpose(1, 2, 3, 0) - (np.array(shape) - 1) / 2
        radii = rng.uniform(1, 12, 3)
        region = ((grid / radii) ** 2).sum(-1) <= 1.0
        region = region.astype(np.uint8)
        start = rng.uniform(-2, np.array(shape) + 1) if k % 5 == 0 else (np.array(shape) - 1) / 2 + rng.normal(0, 1, 3)
        v1, v2 = rng.normal(0, 8, 3), rng.normal(0, 8, 3)
        delta = [5e-8, 5e-8, 5e-8, 1e-3, 0.0, -math.inf, math.nan][k % 7]
        if k % 11 == 0:
            start[k % 3] = [math.nan, math.inf, -math.inf][k % 3]
        cases.append((region, start, v1, v2, delta))
    return cases


def outcome(fn, args):
    try:
        out = fn(*args)
    except ValueError as e:
        return ("raises", type(e).__name__)
    return None if out is None else (out.shape, out.dtype.str, out.tobytes())


def check_exact(impls, cases, label):
    ref_name, ref = next(iter(impls.items()))
    for name, fn in impls.items():
        if fn is ref:
            continue
        bad = sum(outcome(ref, c) != outcome(fn, c) for c in cases)
        print(f"  {label}: {name} vs {ref_name}: {len(cases) - bad}/{len(cases)} bit-identical")
        if bad:
            raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    python_impl, rust_impl = strategies.corner_bisection_search, load_rust()
    impls = {"python (scipy RGI)": python_impl, "python (no scipy)": direct_python, "rust": rust_impl}

    print("exactness")
    calls = capture_real_calls(python_impl)
    check_exact(impls, calls, f"{len(calls)} real calls")
    check_exact({"python (scipy RGI)": python_impl, "rust": rust_impl}, random_cases(400), "400 random cases")

    print(f"\nkernel, {len(calls)} real calls, median of {args.rounds} rounds")
    per_call = {}
    for name, fn in impls.items():
        times = []
        for _ in range(args.rounds):
            t = time.perf_counter()
            for c in calls:
                fn(*c)
            times.append((time.perf_counter() - t) / len(calls))
        per_call[name] = statistics.median(times)
    for name, t in per_call.items():
        print(f"  {name:20s} {t * 1e6:10.1f} us/call   {per_call['python (scipy RGI)'] / t:7.1f}x")

    print(f"\ncalc_poi_from_subreg_vert (all {len(SUBREG_IDS)} POI ids), median of {args.rounds} rotated rounds")
    data = {k: (to_nii(v, True), to_nii(s, True)) for k, (v, s) in SAMPLES.items()}
    order = list(impls)
    times = {(k, name): [] for k in data for name in impls}
    for r in range(args.rounds):
        for name in order[r % 3 :] + order[: r % 3]:
            strategies.corner_bisection_search = impls[name]
            for k, (vert, subreg) in data.items():
                t = time.perf_counter()
                calc_poi_from_subreg_vert(vert, subreg, subreg_id=SUBREG_IDS, verbose=False)
                times[k, name].append(time.perf_counter() - t)
    strategies.corner_bisection_search = python_impl
    for k in data:
        base = statistics.median(times[k, "python (scipy RGI)"])
        for name in impls:
            m = statistics.median(times[k, name])
            print(f"  {k:4s} {name:20s} {m:7.3f} s  (min {min(times[k, name]):.3f})   {base / m:5.2f}x")


if __name__ == "__main__":
    main()
