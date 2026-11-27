"""Check and time the six array-primitive Rust ports against their Python originals, in one process.

Run from the repo root after building the ports into ``.fast/lib`` (see README.md)::

    python -m benchmarks.fast_anything.bench_seg [--rounds 7]

1. Exactness: every call the segmentation workload (``workload_seg.py``) makes to a ported kernel
   gives the same result with the Rust port: values bit for bit, dtype, shape and memory order, or
   the same exception. A second workload run on the Rust ports must see the same call sequence with
   the same results, so the whole workload is identical end to end.
2. Kernel: time per kernel on those real calls, and on one large input per kernel (the CT
   segmentation or image resampled to 0.5 mm, 219x141x219).
3. Pipeline: the workload per sample, Python kernels vs. Rust ports, rotated rounds; all steps and
   each step on its own (``--only``).
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import statistics
import sys
import time

os.environ["FAST_ANYTHING_DISABLE"] = "1"  # keep the Python originals in the modules; Rust is loaded below

import numpy as np
from scipy.ndimage import generate_binary_structure

from TPTBox import to_nii
from TPTBox.core import np_utils
from TPTBox.core.internal import nii_help

from .workload_seg import REPO_ROOT, SAMPLES, STEPS, per_vertebra, whole_volume

# kernel -> (module it lives in, workload step that exercises it)
PORTS = {
    "unique_integers": (np_utils, "unique"),
    "connected_component_labels": (np_utils, "cc"),
    "euclidean_distance_transform": (np_utils, "edt"),
    "binary_dilation_perimeter": (np_utils, "dilate"),
    "binary_erosion_radius1": (np_utils, "erode"),
    "affine_resample": (nii_help, "resample"),
}
PYTHON = {name: getattr(module, name) for name, (module, _) in PORTS.items()}


def load_rust() -> dict:
    sys.path.append(str(REPO_ROOT / ".fast" / "lib"))
    rust, missing = {}, []
    for name, (module, _) in PORTS.items():
        ext = "fa_" + f"{module.__name__}.{name}".lower().replace(".", "_")
        try:
            rust[name] = getattr(importlib.import_module(ext), name)
        except ImportError:
            missing.append(ext)
    if missing:
        raise SystemExit(f"Rust ports not built: {', '.join(missing)} (see benchmarks/fast_anything/README.md)")
    return rust


def use(impls: dict) -> None:
    """Make the modules call these implementations (callers look the kernels up as module globals)."""
    for name, fn in impls.items():
        setattr(PORTS[name][0], name, fn)


def run_workload(steps=STEPS, data=None) -> None:
    for img, vert, subreg in (data or load_samples()).values():
        whole_volume(img, vert, subreg, set(steps))
        per_vertebra(vert, subreg, set(steps))


def load_samples() -> dict:
    return {k: (to_nii(p[0], False), to_nii(p[1], True), to_nii(p[2], True)) for k, p in SAMPLES.items()}


def outcome(fn, args, kwargs):
    """A result as comparable plain data: arrays by shape, dtype, memory order and bytes."""

    def plain(v):
        if isinstance(v, np.ndarray):
            return ("array", v.shape, v.dtype.str, v.flags.c_contiguous, v.flags.f_contiguous, v.tobytes())
        if isinstance(v, (list, tuple)):
            return (type(v).__name__, tuple(plain(x) for x in v))
        return (type(v).__name__, v)

    try:
        return plain(fn(*copy.deepcopy(args), **copy.deepcopy(kwargs)))
    except Exception as e:
        return ("raises", type(e).__name__)


def capture(impls: dict) -> dict:
    """Every call the workload makes to each kernel, with its outcome."""
    calls = {name: [] for name in PORTS}

    def recording(name, fn):
        def wrapper(*args, **kwargs):
            calls[name].append((copy.deepcopy(args), copy.deepcopy(kwargs), outcome(fn, args, kwargs)))
            return fn(*args, **kwargs)

        return wrapper

    use({name: recording(name, fn) for name, fn in impls.items()})
    try:
        run_workload()
    finally:
        use(PYTHON)
    return calls


def large_inputs() -> dict:
    """One large call per kernel, built from the CT sample at 0.5 mm (219x141x219)."""
    ct, vert, _ = load_samples()["ct"]
    seg = vert.rescale((0.5, 0.5, 0.5)).get_seg_array()
    mask = seg > 0
    resample_args = {}

    def grab(*args):
        resample_args.setdefault(args[4], args)  # by order
        return PYTHON["affine_resample"](*args)

    use({"affine_resample": grab})
    try:
        vert.rescale((0.5, 0.5, 0.5))
        ct.rescale((0.5, 0.5, 0.5), order=1)
    finally:
        use(PYTHON)
    struct = generate_binary_structure(3, 3)
    return {
        "unique_integers": [((seg,), {})],
        "connected_component_labels": [((seg, 26), {})],
        "euclidean_distance_transform": [((~mask,), {"return_indices": True})],
        "binary_dilation_perimeter": [((mask, struct), {})],
        "binary_erosion_radius1": [((mask, struct, 1, 0), {})],
        "affine_resample": [(resample_args[0], {}), (resample_args[1], {})],
    }


def time_calls(fn, calls, rounds) -> float:
    times = []
    for _ in range(rounds):
        t = time.perf_counter()
        for args, kwargs in calls:
            fn(*args, **kwargs)
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def time_pipeline(impls_by_name: dict, steps, data, rounds) -> dict:
    times = {(k, name): [] for k in data for name in impls_by_name}
    order = list(impls_by_name)
    for r in range(rounds):
        for name in order[r % len(order) :] + order[: r % len(order)]:
            use(impls_by_name[name])
            for k, sample in data.items():
                t = time.perf_counter()
                run_workload(steps, {k: sample})
                times[k, name].append(time.perf_counter() - t)
    use(PYTHON)
    return {key: statistics.median(v) for key, v in times.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    rust = load_rust()

    print("exactness")
    py_calls = capture(PYTHON)
    rust_calls = capture(rust)
    failed = False
    for name in PORTS:
        same_rust = sum(outcome(rust[name], a, kw) == out for a, kw, out in py_calls[name])
        # the run on Rust ports made as many calls, and Python gives the same result for each of them
        seq_ok = len(py_calls[name]) == len(rust_calls[name]) and all(
            outcome(PYTHON[name], a, kw) == out for a, kw, out in rust_calls[name]
        )
        failed |= same_rust != len(py_calls[name]) or not seq_ok
        print(
            f"  {name:30s} {same_rust}/{len(py_calls[name])} real calls identical; workload on Rust: {'same calls and results' if seq_ok else 'DIFFERS'}"
        )
    if failed:
        raise SystemExit(1)

    print(f"\nkernel, median of {args.rounds} rounds")
    print(f"  {'kernel':30s} {'input':12s} {'python':>10s} {'rust':>10s} {'speedup':>8s}")
    big = large_inputs()
    for name in PORTS:
        calls = [(a, kw) for a, kw, _ in py_calls[name]]
        rows = [(f"{len(calls)} real", calls)] + [
            (f"large #{i + 1}" if len(big[name]) > 1 else "large", [c]) for i, c in enumerate(big[name])
        ]
        for label, cs in rows:
            p, r = time_calls(PYTHON[name], cs, args.rounds), time_calls(rust[name], cs, args.rounds)
            print(f"  {name:30s} {label:12s} {p * 1e3:8.2f}ms {r * 1e3:8.2f}ms {p / r:7.1f}x")

    print(f"\nworkload_seg per sample, median of {args.rounds} rotated rounds")
    data = load_samples()
    for steps in [STEPS, *[(s,) for s in STEPS]]:
        t = time_pipeline({"python": PYTHON, "rust": rust}, steps, data, args.rounds)
        label = "all steps" if steps == STEPS else f"--only {steps[0]}"
        for k in data:
            print(
                f"  {label:16s} {k:4s} python {t[k, 'python']:7.3f} s   rust {t[k, 'rust']:7.3f} s   {t[k, 'python'] / t[k, 'rust']:5.2f}x"
            )


if __name__ == "__main__":
    main()
