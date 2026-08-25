"""Fit the nnUNet memory guard parameters from GPU probes.

Measures peak GPU RAM consumed by run_vibeseg across a grid of synthetic
input shapes, then fits the three memory parameters used by TPTBox's
check_mem guard:

    check_mem passes when:
        (n_voxels / 1e6 * memory_factor) + memory_base
            < clamp(0.80 * gpu_total, lo=memory_base, hi=memory_max)

    So the fitted curve must be a CONSERVATIVE UPPER BOUND of actual usage,
    not a mean, otherwise ~50 % of runs would be incorrectly skipped.
    This script fits via quantile regression (default q=0.95) so the curve
    sits above nearly all observations while remaining tight.

Usage
-----
    python estimate_nnunet_memory.py [--gpu 0] [--model-path /path/to/nnUNet]
                                     [--dataset-id 12] [--out-dir /tmp/nnunet_probe]
                                     [--shapes-csv shapes.csv]

The script prints recommended values for memory_base, memory_factor, and
memory_max at the end, saves a CSV + PNG summary plot, and patches the
dataset.json in the nnUNet model folder.
"""

import argparse
import csv
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from TPTBox import NII
from TPTBox.segmentation import run_vibeseg
from TPTBox.segmentation.VibeSeg.auto_download import download_weights

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_kwargs):  # type: ignore[no-redef]  # noqa: ANN201
        """Fallback no-op progress bar when tqdm is not installed."""
        return it


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _nvidia_smi_query(field: str, gpu: int) -> float:
    out = (
        subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits", f"--id={gpu}"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    return float(out[0])


def gpu_used_mb(gpu: int) -> float:
    """Return currently used GPU memory (MB) as reported by nvidia-smi."""
    return _nvidia_smi_query("memory.used", gpu)


def gpu_total_mb(gpu: int) -> float:
    """Return total GPU memory (MB) as reported by nvidia-smi."""
    return _nvidia_smi_query("memory.total", gpu)


def wait_for_gpu_idle(gpu: int, stable_seconds: float = 2.0, poll_interval: float = 0.5) -> float:
    """Poll until GPU memory is stable; return baseline usage in MB."""
    prev = gpu_used_mb(gpu)
    stable_since = time.time()
    while True:
        time.sleep(poll_interval)
        cur = gpu_used_mb(gpu)
        if abs(cur - prev) < 50:
            if time.time() - stable_since >= stable_seconds:
                return cur
        else:
            stable_since = time.time()
        prev = cur


class PeakPoller:
    """Context manager: polls nvidia-smi in a thread, records peak usage."""

    def __init__(self, gpu: int, poll_interval: float = 0.1):
        self.gpu = gpu
        self.poll_interval = poll_interval
        self._peak = 0.0
        self._stop = False
        self._thread = None

    def __enter__(self):
        import threading

        self._stop = False
        self._peak = gpu_used_mb(self.gpu)

        def _poll():
            while not self._stop:
                try:
                    v = gpu_used_mb(self.gpu)
                    self._peak = max(self._peak, v)
                except Exception:
                    pass
                time.sleep(self.poll_interval)

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop = True
        self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> float:
        """Peak GPU memory (MB) observed while the poller was active."""
        return self._peak


# ---------------------------------------------------------------------------
# Core probe
# ---------------------------------------------------------------------------


def derive_model_zoom(nnunet_path: Path, dataset_id: int) -> tuple[float, float, float] | None:
    """Return the (X, Y, Z) target spacing the model rescales inputs to.

    Mirrors the resolution logic in ``inference_nnunet.run_inference_on_file``:
    ``dataset.json['spacing']`` (reversed unless dataset 527) → ``resolution_range``
    → ``plans.json['configurations']['3d_fullres']['spacing']`` de-transposed.
    Returns None if none of these are present so the caller can fall back.
    """
    with open(nnunet_path / "dataset.json") as f:
        ds_info = json.load(f)
    plans_path = nnunet_path / "plans.json"
    plans_info = json.loads(plans_path.read_text()) if plans_path.exists() else None

    zoom = ds_info.get("spacing")
    if dataset_id not in [527] and zoom is not None:
        zoom = zoom[::-1]
    zoom = ds_info.get("resolution_range", zoom)
    if zoom is None and plans_info is not None:
        try:
            zoom_ = plans_info["configurations"]["3d_fullres"]["spacing"]
            transpose_backward = plans_info["transpose_backward"]
            zoom = [zoom_[transpose_backward[i]] for i in range(len(zoom_))][::-1]
        except (KeyError, IndexError):
            zoom = None
    if zoom is None:
        return None
    zoom = [float(z) for z in zoom]
    assert len(zoom) == 3, zoom
    return (zoom[0], zoom[1], zoom[2])


def probe_shape(
    shape: tuple,
    gpu: int,
    dataset_id: int,
    model_path: str,
    out_dir: Path,
    voxel_size: float | tuple[float, float, float] = 0.8,
) -> dict:
    """Run run_vibeseg on a synthetic volume and measure peak GPU RAM.

    memory_max is set to 999 GB so the check_mem guard never fires here.
    ``voxel_size`` accepts either an isotropic scalar or a 3-tuple. Passing the
    model's own zoom keeps run_vibeseg's internal rescale a no-op, so the shape
    that reaches nnUNet matches the shape we probed.
    """
    zoom_vec = (voxel_size, voxel_size, voxel_size) if isinstance(voxel_size, (int, float)) else tuple(voxel_size)
    affine = np.diag([float(zoom_vec[0]), float(zoom_vec[1]), float(zoom_vec[2]), 1.0])
    nii = NII.from_numpy(np.random.rand(*shape).astype(np.float32), affine=affine)
    out_path = str(out_dir / f"probe_{'x'.join(map(str, shape))}.nii.gz")

    baseline_mb = wait_for_gpu_idle(gpu)

    result = {
        "shape": shape,
        "n_voxels": int(np.prod(shape)),
        "peak_mb": None,
        "net_mb": None,
        "elapsed_s": None,
        "ok": False,
    }

    try:
        t0 = time.time()
        with PeakPoller(gpu) as poller:
            run_vibeseg(
                nii,
                out_path,
                gpu=gpu,
                dataset_id=dataset_id,
                model_path=model_path,
                memory_base=0,
                memory_factor=0,
                memory_max=999_000,  # disable guard during probing
                override=True,
                fail_on_missing_memory=True,
            )
        result["peak_mb"] = poller.peak_mb
        result["net_mb"] = max(0.0, poller.peak_mb - baseline_mb)
        result["elapsed_s"] = time.time() - t0
        result["ok"] = True
    except Exception as exc:
        print(f"  [WARN] shape {shape} failed: {exc}", file=sys.stderr)
        result["error"] = str(exc)

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    return result


# ---------------------------------------------------------------------------
# Fitting - quantile regression (upper envelope, not mean)
# ---------------------------------------------------------------------------


def fit_parameters(records: list, quantile: float = 0.95) -> dict:
    """Fit an upper-envelope line ``net_mb ≈ memory_base + n_voxels / 1e6 * memory_factor``.

    Uses quantile regression at `quantile` (default 0.95) so the predicted
    curve lies above ~95 % of observations.  This is intentional: the
    check_mem guard must never *wrongly skip* a run that would have fit in
    VRAM, so over-estimating slightly is safer than under-estimating.

    Falls back to OLS + residual-std shift if scipy is unavailable.
    """
    ok = [r for r in records if r["ok"] and r["net_mb"] is not None]
    if len(ok) < 2:
        raise ValueError("Need at least 2 successful probes to fit parameters.")

    x = np.array([r["n_voxels"] / 1e6 for r in ok])  # M voxels
    y = np.array([r["net_mb"] for r in ok])

    try:
        from scipy.optimize import linprog  # quantile regression via LP

        # min  q * sum(u) + (1-q) * sum(v)
        # s.t. y - (a + b*x) = u - v,   u,v >= 0
        # variables: [a, b, u_0..u_n, v_0..v_n]
        n = len(x)
        # Objective
        c = np.zeros(2 + 2 * n)
        c[2 : 2 + n] = quantile
        c[2 + n :] = 1.0 - quantile

        # Equality: a + b*x_i + u_i - v_i = y_i
        A_eq = np.zeros((n, 2 + 2 * n))
        A_eq[:, 0] = 1.0
        A_eq[:, 1] = x
        A_eq[np.arange(n), 2 + np.arange(n)] = 1.0
        A_eq[np.arange(n), 2 + n + np.arange(n)] = -1.0
        b_eq = y

        bounds = [(None, None), (None, None)] + [(0, None)] * (2 * n)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        memory_base = float(res.x[0])
        memory_factor = float(res.x[1])
        method_used = f"quantile regression (q={quantile})"

    except Exception:
        # Fallback: OLS + push intercept up by (1-quantile) sigma
        X = np.column_stack([np.ones(len(ok)), x])
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        memory_base = float(coeffs[0])
        memory_factor = float(coeffs[1])
        resid_std = float(np.std(y - X @ coeffs))
        from scipy.stats import norm

        memory_base += norm.ppf(quantile) * resid_std
        method_used = f"OLS + {quantile:.0%}-quantile shift (scipy.optimize unavailable)"

    # OLS fit — used both for R² diagnostics and as a slope floor.
    X2 = np.column_stack([np.ones(len(ok)), x])
    ols_coeffs, *_ = np.linalg.lstsq(X2, y, rcond=None)
    ols_base, ols_slope = float(ols_coeffs[0]), float(ols_coeffs[1])
    y_pred_ols = X2 @ ols_coeffs
    ss_res = np.sum((y - y_pred_ols) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Guard against an outlier flattening the quantile fit: take the LARGER of
    # the quantile-regression slope and the OLS slope. If OLS wins, re-fit the
    # intercept at the same quantile so the line still sits above ~q of points.
    if ols_slope > memory_factor:
        memory_factor = ols_slope
        memory_base = float(np.quantile(y - memory_factor * x, quantile))
        method_used += " + OLS slope floor (quantile slope was flattened)"

    # memory_max: largest observed net usage + headroom (caller applies factor)
    memory_max = float(max(y))

    return {
        "memory_base": max(0.0, memory_base),
        "memory_factor": max(0.0, memory_factor),
        "memory_max": memory_max,
        "r2": r2,
        "method": method_used,
        "n_points": len(ok),
        "ols_base": ols_base,
        "ols_slope": ols_slope,
    }


# ---------------------------------------------------------------------------
# Default shape grid
# ---------------------------------------------------------------------------

DEFAULT_SHAPES = [
    (96, 96, 96),
    (160, 160, 160),
    (192, 192, 192),
    (256, 256, 128),
    (256, 256, 256),
    (320, 320, 160),
    (320, 320, 320),
    (400, 400, 200),
    (400, 400, 400),
    (512, 512, 200),
    (512, 512, 400),
    (512, 512, 512),
]


def load_shapes_csv(path: str) -> list:
    """Load a shape grid from a CSV with columns ``d,h,w``."""
    with open(path) as f:
        return [(int(row["d"]), int(row["h"]), int(row["w"])) for row in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(records: list, fit: dict, out_path: Path) -> None:
    """Render the measured/fitted memory curves and residuals to ``out_path``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not available - skipping plot.")
        return

    ok = [r for r in records if r["ok"] and r["net_mb"] is not None]
    if not ok:
        return

    n_vox = np.array([r["n_voxels"] for r in ok])
    net_mb = np.array([r["net_mb"] for r in ok])
    x_M = n_vox / 1e6  # M voxels for plotting

    x_line = np.linspace(0, x_M.max(), 300)
    y_line = fit["memory_base"] + x_line * fit["memory_factor"]  # x already in M

    # OLS mean line for comparison
    X = np.column_stack([np.ones(len(ok)), x_M])
    ols, *_ = np.linalg.lstsq(X, net_mb, rcond=None)
    y_ols = ols[0] + x_line * ols[1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(x_M, net_mb, zorder=3, label="measured peak", color="steelblue")
    ax.plot(x_line, y_ols, "k:", linewidth=1.2, label="OLS mean")
    ax.plot(x_line, y_line, "r--", linewidth=1.8, label=f"upper envelope ({fit['method'].split('(')[1].rstrip(')')} )")
    ax.set_xlabel("Volume size (M voxels)")
    ax.set_ylabel("Net GPU RAM (MB)")
    ax.set_title("GPU RAM vs volume size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    y_pred = fit["memory_base"] + x_M * fit["memory_factor"]
    resid = net_mb - y_pred
    colors = ["tomato" if r > 0 else "steelblue" for r in resid]
    ax.bar(range(len(ok)), resid, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Probe index (sorted by size)")
    ax.set_ylabel("Residual (MB)  [+ve = under-predicted]")
    ax.set_title("Fit residuals  (negative = safe headroom)")
    ax.grid(True, alpha=0.3, axis="y")

    summary = (
        f"memory_base={fit['memory_base']:.0f} MB   "
        f"memory_factor={fit['memory_factor']:.2f}   "
        f"memory_max={fit['memory_max']:.0f} MB   "
        f"OLS R²={fit['r2']:.4f}   n={fit['n_points']}"
    )
    fig.suptitle("run_nnunet GPU memory parameter estimation", fontsize=13)
    fig.text(0.5, -0.01, summary, ha="center", fontsize=9, bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.6})

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the memory-estimation script."""
    p = argparse.ArgumentParser(description="Estimate run_nnunet memory parameters from GPU measurements.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--model-path",
        default=None,
        help="Base directory containing 'nnUNet_results'. If omitted, the TPTBox default weights path is used and missing models are auto-downloaded.",
    )
    p.add_argument("--dataset-id", type=int, default=12)
    p.add_argument("--out-dir", default="/tmp/run_nnunet_probe")
    p.add_argument("--shapes-csv", default=None, help="CSV with columns d,h,w for custom shape grid")
    p.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Isotropic voxel size for synthetic inputs. If omitted, the model's own "
        "target spacing is derived from dataset.json/plans.json so run_vibeseg's "
        "internal rescale is a no-op and the probed shape matches the shape reaching nnUNet.",
    )
    p.add_argument("--results-csv", default="memory_probes.csv")
    p.add_argument("--plot", default="memory_fit.png")
    p.add_argument("--headroom", type=float, default=1.20, help="Multiplier applied to max observed net MB → memory_max (default 1.20)")
    p.add_argument("--quantile", type=float, default=0.95, help="Quantile for upper-envelope fit (default 0.95)")
    return p.parse_args()


def main() -> None:
    """Entry point: probe shapes, fit memory parameters, patch dataset.json."""
    args = parse_args()

    # Resolve nnUNet model path (auto-download when using the default TPTBox path)
    if args.model_path is None:
        weights_dir = download_weights(args.dataset_id)
        model_path = weights_dir.parent  # <default>/nnUNet_results
        # When no --model-path is passed, run_vibeseg below must find the same weights,
        # so we hand it the same base directory instead of the outdated CLI default.
        args.model_path = str(model_path)
    else:
        model_path = Path(args.model_path)
        if model_path.name != "nnUNet_results":
            model_path = model_path / "nnUNet_results"
        model_path.mkdir(parents=True, exist_ok=True)
        # If the dataset isn't present under the supplied base, download into it.
        if not any(model_path.glob(f"*{args.dataset_id:03}*")):
            print(f"[INFO] Dataset {args.dataset_id:03} not found under {model_path}; downloading…")
            download_weights(args.dataset_id, model_path=model_path)
    assert model_path.exists(), model_path

    _key_ResEnc = "__nnUNet*ResEnc"

    def _resolve_nnunet_path():
        try:
            return next(next(iter(model_path.glob(f"*{args.dataset_id:03}*"))).glob(f"*{_key_ResEnc}*"))
        except StopIteration:
            return next(next(iter(model_path.glob(f"*{args.dataset_id:03}*"))).glob("*__nnUNetPlans*"))

    try:
        nnunet_path = _resolve_nnunet_path()
    except StopIteration:
        # Last-ditch: try one more download, then re-resolve.
        print(f"[INFO] No nnUNet configuration found under {model_path}/Dataset{args.dataset_id:03}; retrying download…")
        download_weights(args.dataset_id, model_path=model_path)
        try:
            nnunet_path = _resolve_nnunet_path()
        except StopIteration as e:
            raise RuntimeError(f"No nnUNet model found for dataset {args.dataset_id}") from e

    json_path = nnunet_path / "dataset.json"
    assert json_path.exists(), json_path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shape grid
    if args.shapes_csv:
        shapes = load_shapes_csv(args.shapes_csv)
        print(f"[INFO] Loaded {len(shapes)} shapes from {args.shapes_csv}")
    else:
        shapes = DEFAULT_SHAPES
        print(f"[INFO] Using default grid of {len(shapes)} shapes")

    total_gpu_mb = gpu_total_mb(args.gpu)
    print(f"[INFO] GPU {args.gpu}: {total_gpu_mb:.0f} MB total VRAM")
    print(f"[INFO] Fitting with quantile={args.quantile}  headroom={args.headroom}")

    # Resolve voxel_size once: prefer the model's own target spacing so run_vibeseg's
    # internal rescale is a no-op and the probed shape == the shape reaching nnUNet.
    if args.voxel_size is None:
        derived = derive_model_zoom(nnunet_path, args.dataset_id)
        if derived is None:
            voxel_size: float | tuple[float, float, float] = 0.8
            print("[WARN] Could not derive model zoom; falling back to isotropic 0.8 mm.")
        else:
            voxel_size = derived
            print(f"[INFO] Derived model zoom from configs: {derived} mm")
    else:
        voxel_size = args.voxel_size
        print(f"[INFO] Using explicit --voxel-size {voxel_size} mm (isotropic)")
    print()

    shapes = sorted(shapes, key=lambda s: np.prod(s))

    # ------------------------------------------------------------------ probes
    records = []
    results_csv = nnunet_path / args.results_csv

    # Warm-up: the first inference pays for CUDA context init, kernel autotune,
    # cuDNN benchmark, and one-time allocator growth. Discard it so the smallest
    # shape isn't systematically inflated (that outlier drags the slope down).
    print(f"[INFO] Warm-up pass on {shapes[0]} (discarded)")
    _ = probe_shape(
        shape=shapes[0],
        gpu=args.gpu,
        dataset_id=args.dataset_id,
        model_path=args.model_path,
        out_dir=out_dir,
        voxel_size=voxel_size,
    )

    with open(results_csv, "w", newline="") as csvfile:
        fieldnames = ["shape_d", "shape_h", "shape_w", "n_voxels", "peak_mb", "net_mb", "elapsed_s", "ok", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for shape in tqdm(shapes, desc="Probing shapes"):
            n_vox = int(np.prod(shape))
            print(f"\n→ shape={shape}  ({n_vox / 1e6:.2f} M voxels)")

            rec = probe_shape(
                shape=shape,
                gpu=args.gpu,
                dataset_id=args.dataset_id,
                model_path=args.model_path,
                out_dir=out_dir,
                voxel_size=voxel_size,
            )
            records.append(rec)

            writer.writerow(
                {
                    "shape_d": shape[0],
                    "shape_h": shape[1],
                    "shape_w": shape[2],
                    "n_voxels": n_vox,
                    "peak_mb": f"{rec['peak_mb']:.1f}" if rec["peak_mb"] else "",
                    "net_mb": f"{rec['net_mb']:.1f}" if rec["net_mb"] else "",
                    "elapsed_s": f"{rec['elapsed_s']:.1f}" if rec["elapsed_s"] else "",
                    "ok": rec["ok"],
                    "error": rec.get("error", ""),
                }
            )
            csvfile.flush()

            if rec["ok"]:
                print(f"   peak={rec['peak_mb']:.0f} MB  net={rec['net_mb']:.0f} MB  time={rec['elapsed_s']:.1f} s")
            else:
                print(f"   FAILED: {rec.get('error', '?')}")

    print(f"\n[INFO] Raw results saved → {results_csv}")

    # ------------------------------------------------------------------ fit
    try:
        fit = fit_parameters(records, quantile=args.quantile)
    except ValueError as e:
        print(f"[ERROR] Fitting failed: {e}")
        sys.exit(1)

    # Apply headroom to memory_max
    fit["memory_max"] = fit["memory_max"] * args.headroom

    # ------------------------------------------------------------------ report
    bar = "=" * 60
    print(f"\n{bar}")
    print("  FITTED MEMORY PARAMETERS")
    print(f"  method: {fit['method']}")
    print(bar)
    print(f"  memory_base   = {fit['memory_base']:>10.0f}   # MB  (fixed overhead)")
    print(f"  memory_factor = {fit['memory_factor']:>10.2f}   # n_voxels/1e6 * factor  MB")
    print(f"  memory_max    = {fit['memory_max']:>10.0f}   # MB  ({args.headroom:.0%} headroom on max observed)")
    print(f"  OLS R²        = {fit['r2']:>10.4f}   (diagnostic; fit targets upper envelope)")
    print(bar)
    print()
    print("  Suggested call:")
    print("  run_vibeseg(")
    print("      nii, out,")
    print(f"      memory_base={fit['memory_base']:.0f},")
    print(f"      memory_factor={fit['memory_factor']:.2f},")
    print(f"      memory_max={fit['memory_max']:.0f},")
    print("  )")
    print(bar)

    # ------------------------------------------------------------------ plot
    make_plot(records, fit, nnunet_path / args.plot)

    # --------------------------------------------------------- patch JSON
    backup_path = json_path.with_suffix(".json.bak")
    shutil.copy2(json_path, backup_path)
    try:
        with open(json_path) as f:
            data = json.load(f)
        data["memory_base"] = fit["memory_base"]
        data["memory_factor"] = fit["memory_factor"]
        tmp_path = json_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)
        tmp_path.replace(json_path)
        backup_path.unlink(missing_ok=True)
        print(f"[INFO] dataset.json patched → {json_path}")
    except Exception:
        if backup_path.exists():
            shutil.copy2(backup_path, json_path)
        raise


if __name__ == "__main__":
    main()
    # python estimate_vibeseg_memory.py \
    #   --gpu 1 \
    #   --model-path /DATA/NAS/FASTDATA/robert/nnUNet \
    #   --dataset-id 12 \
    #   --quantile 0.95 \
    #   --headroom 1.20
