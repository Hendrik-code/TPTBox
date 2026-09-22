#!/usr/bin/env python
"""Memory benchmark for the NII and POI hot paths.

Same workloads and same measurement registry as ``bench_speed.py``, but the
number reported is **peak RSS growth per call, in MiB**.

Two details make the numbers mean something:

* **Sampling.** A daemon thread polls ``VmRSS`` from ``/proc/self/status`` every
  5 ms while the call runs, and ``resource.getrusage`` is read afterwards as a
  belt-and-braces high-water mark in case a short-lived spike fell between two
  polls. On a platform without ``/proc`` only the latter is used.
* **Isolation.** Each sampled iteration runs in a forked child. CPython's heap
  does not shrink, so a second in-process iteration of the same call typically
  reuses the arena freed by the first and reports ~0 MiB. A fresh address space
  per iteration removes that. The baseline RSS is taken *inside* the child after
  the fork, so the copy-on-write pages of the input volume are already accounted
  for and do not inflate the result.

    python benchmarks/nii_poi/bench_mem.py                      # 10 repeats + 1 warmup
    python benchmarks/nii_poi/bench_mem.py --quick --json head_mem.json
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

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

_STATUS = Path("/proc/self/status")
_HAS_PROC = _STATUS.exists()
_POLL_SECONDS = 0.005
_MIB = 1024.0 * 1024.0


def _rss_bytes() -> float:
    """Current resident set size in bytes, or 0.0 where /proc is unavailable."""
    try:
        for line in _STATUS.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    return 0.0


def _peak_rusage_bytes() -> float:
    """High-water RSS of this process.

    Linux does *not* reset ``ru_maxrss`` at fork — a child starts out carrying the
    parent's high-water mark — but it does track the child's own growth on top of
    it. Since the baseline is read inside the child, the difference is still the
    child's own peak, which is all this is used for.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return float(maxrss) * (1.0 if sys.platform == "darwin" else 1024.0)


class _PeakSampler(threading.Thread):
    """Poll VmRSS in the background and remember the maximum seen."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        # NB: not `_stop` — threading.Thread already uses that name for an
        # internal method, and shadowing it breaks join().
        self._done = threading.Event()
        self.peak = 0.0

    def run(self) -> None:
        while not self._done.is_set():
            self.peak = max(self.peak, _rss_bytes())
            time.sleep(_POLL_SECONDS)
        self.peak = max(self.peak, _rss_bytes())

    def stop(self) -> float:
        self._done.set()
        self.join(timeout=1.0)
        return self.peak


def _measure_in_process(call) -> float:
    """Peak RSS growth of one call, in MiB. Used inside the forked child, or as fallback."""
    gc.collect()
    base = max(_rss_bytes(), _peak_rusage_bytes())
    sampler = _PeakSampler() if _HAS_PROC else None
    if sampler is not None:
        sampler.start()
    try:
        with silenced():
            call()
    finally:
        peak = sampler.stop() if sampler is not None else 0.0
    peak = max(peak, _rss_bytes(), _peak_rusage_bytes())
    return max(peak - base, 0.0) / _MIB


def _child(conn, call) -> None:
    try:
        conn.send(("ok", _measure_in_process(call)))
    except Exception as exc:  # noqa: BLE001 -- surfaced to the parent as a skip
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def measure(call, ctx) -> float:
    """One isolated sample. Falls back to in-process measurement without fork()."""
    if ctx is None:
        return _measure_in_process(call)
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, call))
    proc.start()
    child.close()
    try:
        status, payload = parent.recv()
    except EOFError as exc:
        proc.join()
        raise RuntimeError(f"measurement child died (exit {proc.exitcode})") from exc
    finally:
        parent.close()
        proc.join()
    if status != "ok":
        raise RuntimeError(payload)
    return float(payload)


def _fork_context():
    if os.name != "posix" or "fork" not in mp.get_all_start_methods():
        return None
    return mp.get_context("fork")


def measure_floor(ctx, n: int = 5) -> float:
    """Peak RSS attributed to an empty call.

    A forked child pays a small constant cost of its own (the sampler thread, the
    gc pass, copy-on-write faults from refcount writes on inherited objects).
    It is the same on both sides of a comparison so it cancels out, but it is
    worth reporting so nobody reads a 1 MiB row as 1 MiB of real allocation.
    """
    try:
        return float(np.median([measure(lambda: None, ctx) for _ in range(n)]))
    except Exception:  # noqa: BLE001 -- diagnostic only
        return 0.0


def run_case(name: str, large_size: int, repeats: int, warmup: int, tmp: Path, ctx) -> dict:
    case = build_case(name, large_size=large_size)
    measurements, metrics, setup_notes = build_measurements(case, tmp)

    summaries: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for meas in measurements:
        try:
            # Warmup runs in the *parent*, on purpose. A forked child inherits
            # whatever the parent has already imported, so the point of a warmup
            # here is to pull in lazily-imported modules (scipy.ndimage, cc3d, …)
            # before the fork — otherwise every child pays for them and every
            # sample is inflated by the same import cost.
            with silenced():
                for _ in range(warmup):
                    meas.call()
            summaries[meas.key] = summarize([measure(meas.call, ctx) for _ in range(repeats)])
        except Exception as exc:  # noqa: BLE001 -- a missing/broken op must not abort the run
            skipped.append(f"{meas.key} ({type(exc).__name__}: {exc})")

    print_case_table(case.name, case.shape, summaries, "MiB")
    if setup_notes:
        print(f"  setup failed: {'; '.join(setup_notes)}")
    if skipped:
        print(f"  skipped: {'; '.join(skipped)}")

    summaries.update(metrics)
    return {"name": case.name, "shape": list(case.shape), "voxels": case.voxels, "measurements_mb": summaries}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args(argv)
    repeats, warmup = resolve_repeats(args)
    large_size = resolve_large_size(args)
    names = parse_case_names(args.cases)

    ctx = _fork_context()
    sampler = "proc-status" if _HAS_PROC else "getrusage"
    isolation = "fork" if ctx is not None else "in-process"
    floor = measure_floor(ctx)
    print(
        f"memory benchmark | python {python_version()} | commit {commit_hash()} | "
        f"repeats={repeats} warmup={warmup} large_size={large_size} | "
        f"sampler={sampler} isolation={isolation} floor={floor:.2f} MiB"
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tptbox-benchmem-") as td:
        tmp = Path(td)
        cases = [run_case(n, large_size, repeats, warmup, tmp, ctx) for n in names]
    print(f"\ntotal wall time: {time.perf_counter() - started:.1f}s")

    write_json(
        args.json,
        {
            "python": python_version(),
            "commit": commit_hash(),
            "repeats": repeats,
            "warmup": warmup,
            "large_size": large_size,
            "sampler": sampler,
            "isolation": isolation,
            "floor_mib": round(floor, 3),
            "cases": cases,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
