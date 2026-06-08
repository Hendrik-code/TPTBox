# nnU-Net inference timing harness

Tools to measure the speed impact of the inference optimisations in
`TPTBox/segmentation/nnUnet_utils/` and `TPTBox/segmentation/VibeSeg/inference_nnunet.py`.

## Files

- `benchmark_nnunet_inference.py` — the measurement tool. Splits
  `run_inference_on_file` into phases (**load → preprocess → sliding-window
  predict → postprocess → other**) with CUDA synchronisation, warmup, repeats
  and peak-memory tracking. It instruments the pipeline by monkeypatching (no
  library changes) and drops any CLI flag a given commit does not yet support,
  so the *same file* runs against every commit.
- `bench_across_commits.sh` — checks out the baseline + each optimisation commit,
  runs one fixed config, and prints per-commit deltas.
- `bench_flag_sweep.sh` — stays on the current commit and sweeps the opt-in flags.

## Requirements

- An **editable** TPTBox install (`poetry install`) so that checking out a commit
  changes the imported code.
- Model weights for the chosen `--dataset-id` (default `100`). The first run may
  download them; that happens during warmup and is excluded from the numbers.
- A GPU for `--device cuda` (the default). `--synthetic` removes the need for a
  real input image — only the model is required.

## Quick start

```bash
# 1) per-commit deltas (always-on changes), synthetic input
benchmarks/bench_across_commits.sh --device cuda --repeats 5

# 2) opt-in flag effects at HEAD
benchmarks/bench_flag_sweep.sh --device cuda --repeats 5

# 3) one config by hand (e.g. on your real data)
python benchmarks/benchmark_nnunet_inference.py run \
    --dataset-id 100 --input water.nii.gz fat.nii.gz \
    --tile-batch-size 4 --repeats 5 --json /tmp/tb4.json
```

## Which tool measures which commit

| Commit | Change | How it shows up |
|---|---|---|
| `cuDNN/TF32` | autotune + TF32 | `bench_across_commits.sh`: `predict` drops at this commit (after warmup) |
| `inference_mode` | no_grad → inference_mode | `bench_across_commits.sh`: small `predict`/`peak_mem` drop |
| `empty_cache` | drop per-fold cache clears | `bench_across_commits.sh`: `predict` drop, larger with more folds |
| `fold fix` | repair `loaded_networks` | `fold_status` column flips from `DUPLICATED(...)` to `lazy-per-fold`/`distinct`. Time is ~unchanged (correctness fix) — the old code already paid for N passes. |
| `cache_model` | persistent model cache | `bench_flag_sweep.sh` → `sweep_cache_model.json`: `load: first` is large, `load: steady-median` ≈ 0 |
| `tile_batch_size` | batch tiles per forward | `bench_flag_sweep.sh`: `tile_batch_4/8` reduce `forward_calls` and `predict`, raise `peak_mem` |

Biggest raw-speed levers (quality trade-off): `--max-folds 1` (≈ folds×) and
`--step-size 0.7` (fewer tiles).

## Reading the output

- **`total: first` vs `steady-median`** — the first call pays lazy CUDA init,
  cuDNN autotuning and (without caching) model load; steady-median is the
  representative per-image cost.
- **`forward_calls`** — number of network forward passes = `folds × tiles` (÷ batch).
  Drops with `--tile-batch-size`, scales with folds.
- **`fold_status`** — `DUPLICATED(...)` means the ensemble was silently collapsed
  to one fold (pre-fix); `lazy-per-fold` (multi-fold) / `distinct` is correct.
- **`peak_mem_mb`** — `torch.cuda.max_memory_allocated`; watch this rise with
  `--tile-batch-size`.

## Notes / caveats

- Synthetic input is random noise with the model's channel count and a chosen
  shape/spacing — fine for *timing* (the network cost is content-independent), not
  for assessing segmentation quality. It is cached in a temp dir and reused across
  commits so the workload is identical.
- TF32 and the fold fix change the numeric output between commits, so outputs are
  not bit-identical across the range — expected, not a harness bug.
- `bench_across_commits.sh` restores your original branch on exit (even on error).
- Results and the synthetic cache (`results/`, `.bench_cache/`) are git-ignored.
