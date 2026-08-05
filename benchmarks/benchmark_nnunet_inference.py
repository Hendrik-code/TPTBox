#!/usr/bin/env python
"""Timing harness for the VibeSeg / nnU-Net inference pipeline.

Measures ``run_inference_on_file`` split into phases (model load, preprocess,
sliding-window forward, postprocess) with correct CUDA synchronisation, warmup,
repeats and peak-memory tracking. It is written so the *same file* runs against
any commit in the optimisation range: arguments that a given commit does not yet
support are silently dropped (see :func:`supported_kwargs`).

Two ways to use it:

1. Per-commit deltas (always-on changes: cuDNN/TF32, inference_mode,
   empty_cache, fold fix). Drive it with ``bench_across_commits.sh``.

2. Flag sweep at HEAD (opt-in changes: cache_model, tile_batch_size, max_folds,
   step_size). Drive it with ``bench_flag_sweep.sh`` or call ``run`` directly
   with the relevant flags.

Examples::

    # synthetic input (only the model weights are required), single config
    python benchmark_nnunet_inference.py run --dataset-id 100 --synthetic \
        --shape 320 320 96 --repeats 5 --json /tmp/head.json

    # real input(s); multi-channel models take several --input paths
    python benchmark_nnunet_inference.py run --dataset-id 100 \
        --input water.nii.gz fat.nii.gz --tile-batch-size 4 --cache-model

    # compare result JSONs produced by several runs/commits
    python benchmark_nnunet_inference.py compare results/*.json

Caveats: TPTBox must be importable from the *working tree* (an editable
``poetry install`` does this), otherwise checking out a commit will not change
the measured code. The first run may download model weights; that happens during
warmup and is excluded from the reported numbers.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- phase timers / counters populated by monkeypatching (no library changes) ---------------
TIMINGS: dict[str, float] = {}
COUNTERS: dict[str, object] = {}


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _reset_collectors() -> None:
    TIMINGS.clear()
    COUNTERS.clear()
    COUNTERS["forward_calls"] = 0
    COUNTERS["tiles"] = 0
    COUNTERS["folds"] = None
    COUNTERS["fold_status"] = "n/a"


def _timed(name: str, device: str):
    """Wrap a callable so its synchronised wall time accumulates into ``TIMINGS[name]``."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            _sync(device)
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                _sync(device)
                TIMINGS[name] = TIMINGS.get(name, 0.0) + (time.perf_counter() - t)

        return wrapper

    return deco


def install_patches(device: str) -> None:
    """Monkeypatch the inference stack to record per-phase timings and forward counts.

    Patches the names in the modules where they are actually *called* so the
    instrumentation works regardless of how each commit imports them.
    """
    from TPTBox.segmentation.nnUnet_utils import inference_api, predictor

    # model load (run_inference_on_file does `from inference_api import load_inf_model` at call
    # time, so patching the attribute here is picked up by that local import).
    inference_api.load_inf_model = _timed("load", device)(inference_api.load_inf_model)

    # postprocess: convert_predicted_logits_... is called via predictor's own namespace binding.
    predictor.convert_predicted_logits_to_segmentation_with_correct_shape = _timed("postprocess", device)(
        predictor.convert_predicted_logits_to_segmentation_with_correct_shape
    )

    P = predictor.nnUNetPredictor
    P.predict_single_npy_array = _timed("single_npy", device)(P.predict_single_npy_array)
    P.predict_logits_from_preprocessed_data = _timed("predict", device)(_fold_probe(P.predict_logits_from_preprocessed_data))
    P._internal_maybe_mirror_and_predict = _forward_counter(P._internal_maybe_mirror_and_predict)


def _fold_probe(fn):
    """Record fold count and whether the loaded_networks cache is correct (surfaces the #6 bug)."""

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        params = getattr(self, "list_of_parameters", None)
        COUNTERS["folds"] = len(params) if params is not None else None
        loaded = getattr(self, "loaded_networks", None)
        if loaded is None:
            COUNTERS["fold_status"] = "lazy-per-fold"  # correct: weights swapped per fold
        else:
            ids = {id(n) for n in loaded}
            COUNTERS["fold_status"] = "distinct" if len(ids) == len(loaded) else f"DUPLICATED({len(loaded)}->{len(ids)})"
        return fn(self, *a, **k)

    return wrapper


def _forward_counter(fn):
    @functools.wraps(fn)
    def wrapper(self, x, *a, **k):
        COUNTERS["forward_calls"] = COUNTERS.get("forward_calls", 0) + 1
        COUNTERS["tiles"] = COUNTERS.get("tiles", 0) + int(x.shape[0])
        return fn(self, x, *a, **k)

    return wrapper


# --- input handling --------------------------------------------------------------------------
def make_synthetic(shape: list[int], spacing: list[float], channels: int, seed: int, cache_dir: Path) -> list[str]:
    """Create (and cache) ``channels`` random NIfTI volumes with the given shape/spacing."""
    import nibabel as nib

    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    affine = np.diag([*spacing, 1.0]).astype(float)
    paths = []
    tag = f"{channels}ch_{'x'.join(map(str, shape))}_sp{'-'.join(str(s) for s in spacing)}_s{seed}"
    for c in range(channels):
        p = cache_dir / f"synthetic_{tag}_ch{c}.nii.gz"
        if not p.exists():
            arr = (rng.standard_normal(tuple(shape)).astype(np.float32) * 200.0) + 100.0
            nib.save(nib.Nifti1Image(arr, affine), str(p))
        paths.append(str(p))
    return paths


def resolve_channels(dataset_id: int | None, fallback: int) -> int:
    if dataset_id is None:
        return fallback
    try:
        from TPTBox.segmentation.VibeSeg.inference_nnunet import get_ds_info

        ds = get_ds_info(dataset_id, exit_one_fail=False)
        if ds and "channel_names" in ds:
            return len(ds["channel_names"])
    except Exception as e:  # noqa: BLE001 - best effort, fall back to the user value
        print(f"[warn] could not read dataset.json for channel count ({e}); using --channels={fallback}")
    return fallback


# --- core measurement ------------------------------------------------------------------------
def supported_kwargs(fn, kwargs: dict) -> dict:
    """Keep only kwargs the target accepts, so the harness runs on commits that lack newer flags."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    keep = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(keep))
    if dropped:
        print(f"[info] this commit ignores unsupported args: {dropped}")
    return keep


def run_once(idx, inputs: list[str], call_kwargs: dict, device: str) -> dict:
    from TPTBox import to_nii
    from TPTBox.segmentation.VibeSeg.inference_nnunet import run_inference_on_file

    _reset_collectors()
    niis = [to_nii(p) for p in inputs]  # reload each repeat so in-place ops never bleed across runs
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    t0 = time.perf_counter()
    run_inference_on_file(idx, niis, **supported_kwargs(run_inference_on_file, call_kwargs))
    _sync(device)
    total = time.perf_counter() - t0

    phases = {
        "load": TIMINGS.get("load", 0.0),
        "preprocess": max(0.0, TIMINGS.get("single_npy", 0.0) - TIMINGS.get("predict", 0.0) - TIMINGS.get("postprocess", 0.0)),
        "predict": TIMINGS.get("predict", 0.0),
        "postprocess": TIMINGS.get("postprocess", 0.0),
    }
    phases["other"] = max(0.0, total - sum(phases.values()))  # NII I/O, reorient, rescale-back, save
    return {
        "total": total,
        "phases": phases,
        "peak_mem_mb": (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0,
        "forward_calls": COUNTERS["forward_calls"],
        "tiles": COUNTERS["tiles"],
        "folds": COUNTERS["folds"],
        "fold_status": COUNTERS["fold_status"],
    }


def summarize(repeats: list[dict]) -> dict:
    """Median over steady-state repeats (repeat 0 kept separately as the cold/first call)."""
    steady = repeats[1:] if len(repeats) > 1 else repeats
    med = lambda key: statistics.median(r[key] for r in steady)  # noqa: E731
    phase_keys = repeats[0]["phases"].keys()
    return {
        "total_first": repeats[0]["total"],
        "total_median": med("total"),
        "phases_median": {k: statistics.median(r["phases"][k] for r in steady) for k in phase_keys},
        "load_first": repeats[0]["phases"]["load"],
        "load_median": statistics.median(r["phases"]["load"] for r in steady),
        "peak_mem_mb": max(r["peak_mem_mb"] for r in repeats),
        "forward_calls": repeats[0]["forward_calls"],
        "tiles": repeats[0]["tiles"],
        "folds": repeats[0]["folds"],
        "fold_status": repeats[0]["fold_status"],
    }


def commit_hash() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:  # noqa: BLE001
        return "unknown"


# --- subcommands -----------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> None:
    device = args.device
    install_patches(device)

    channels = resolve_channels(args.dataset_id if args.model_path is None else None, args.channels)
    if args.synthetic:
        inputs = make_synthetic(args.shape, args.spacing, channels, args.seed, Path(args.cache_dir))
    else:
        if not args.input:
            sys.exit("error: provide --input PATH [PATH ...] or use --synthetic")
        inputs = args.input
    idx = Path(args.model_path) if args.model_path else args.dataset_id

    call_kwargs = {
        "out_file": None,
        "override": True,
        "ddevice": device,
        "gpu": args.gpu,
        "verbose": False,
        "max_folds": args.max_folds,
        "step_size": args.step_size,
        "tile_batch_size": args.tile_batch_size,
        "cache_model": args.cache_model,
        "keep_size": args.keep_size,
        "padd": args.padd,
    }

    print(f"warmup ({args.warmup}) ...")
    for _ in range(max(0, args.warmup)):
        run_once(idx, inputs, call_kwargs, device)

    repeats = []
    for i in range(args.repeats):
        r = run_once(idx, inputs, call_kwargs, device)
        repeats.append(r)
        print(
            f"  repeat {i}: total={r['total']:.3f}s predict={r['phases']['predict']:.3f}s "
            f"load={r['phases']['load']:.3f}s peak={r['peak_mem_mb']:.0f}MB forwards={r['forward_calls']}"
        )

    summary = summarize(repeats)
    result = {
        "commit": commit_hash(),
        "device": device,
        "inputs": inputs,
        "config": {k: call_kwargs[k] for k in ("max_folds", "step_size", "tile_batch_size", "cache_model", "keep_size", "padd")},
        "repeats": repeats,
        "summary": summary,
    }
    _print_summary(result)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")


def _print_summary(result: dict) -> None:
    s = result["summary"]
    print(f"\n=== {result['commit']} | device={result['device']} | config={result['config']} ===")
    print(f"folds={s['folds']} fold_status={s['fold_status']} forward_calls={s['forward_calls']} tiles={s['tiles']}")
    print(f"peak_mem={s['peak_mem_mb']:.0f} MB")
    print(f"total: first={s['total_first']:.3f}s  steady-median={s['total_median']:.3f}s")
    print(f"load:  first={s['load_first']:.3f}s  steady-median={s['load_median']:.3f}s")
    print("phase medians (steady-state):")
    for k, v in s["phases_median"].items():
        print(f"    {k:<11s} {v:.3f}s")


def cmd_compare(args: argparse.Namespace) -> None:
    rows = []
    for f in args.files:
        d = json.loads(Path(f).read_text())
        s = d["summary"]
        rows.append((d["commit"], s, d.get("config", {})))

    hdr = f"{'commit':<10} {'total_med':>10} {'predict':>9} {'load_1st':>9} {'peak_MB':>9} {'fwd':>6} {'folds':>5}  fold_status"
    print(hdr)
    print("-" * len(hdr))
    base = None
    prev = None
    for commit, s, _cfg in rows:
        tm = s["total_median"]
        line = (
            f"{commit:<10} {tm:>10.3f} {s['phases_median']['predict']:>9.3f} {s['load_first']:>9.3f} "
            f"{s['peak_mem_mb']:>9.0f} {s['forward_calls']:>6} {s['folds']!s:>5}  {s['fold_status']}"
        )
        print(line)
        base = base if base is not None else tm
        if prev is not None:
            d_prev = (tm - prev) / prev * 100
            d_base = (tm - base) / base * 100
            print(f"{'':<10} {'':>10} {'':>9} {'':>9} {'':>9} {'':>6} {'':>5}  Δprev={d_prev:+.1f}%  Δbaseline={d_base:+.1f}%")
        prev = tm
    print(
        "\nNote: opt-in commits (cache_model, tile_batch_size) show ~0 here under the default config; "
        "measure them with bench_flag_sweep.sh at HEAD."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="benchmark one configuration")
    r.add_argument("--dataset-id", type=int, default=100)
    r.add_argument("--model-path", default=None, help="explicit model folder; overrides --dataset-id")
    r.add_argument("--input", nargs="+", default=None, help="input NIfTI path(s); one per model channel")
    r.add_argument("--synthetic", action="store_true", help="generate random input(s) instead of reading files")
    r.add_argument("--shape", type=int, nargs=3, default=[320, 320, 96])
    r.add_argument("--spacing", type=float, nargs=3, default=[1.40625, 1.40625, 3.0])
    r.add_argument("--channels", type=int, default=1, help="used only if dataset.json channel count is unavailable")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--cache-dir", default=str(Path(__file__).parent / ".bench_cache"))
    r.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    r.add_argument("--gpu", type=int, default=0)
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--warmup", type=int, default=1)
    # optimisation knobs (dropped automatically on commits that predate them)
    r.add_argument("--max-folds", type=int, default=None)
    r.add_argument("--step-size", type=float, default=0.5)
    r.add_argument("--tile-batch-size", type=int, default=1)
    r.add_argument("--cache-model", action="store_true")
    r.add_argument("--keep-size", action="store_true")
    r.add_argument("--padd", type=int, default=0)
    r.add_argument("--json", default=None, help="write structured results to this path")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="tabulate result JSONs in the given order")
    c.add_argument("files", nargs="+")
    c.set_defaults(func=cmd_compare)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
