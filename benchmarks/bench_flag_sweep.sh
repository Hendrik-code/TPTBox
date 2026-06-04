#!/usr/bin/env bash
# Measure the OPT-IN optimisations at the current commit by sweeping their flags in one process.
# (cache_model needs >1 call to show amortisation, so it cannot be measured by the cross-commit
# driver -- that is what this script is for.)
#
# Usage:
#   benchmarks/bench_flag_sweep.sh [extra args forwarded to `benchmark ... run`]
#   benchmarks/bench_flag_sweep.sh --input water.nii.gz fat.nii.gz --device cuda
# With no extra args it uses a synthetic input.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
BENCH="$REPO_ROOT/benchmarks/benchmark_nnunet_inference.py"
RESULTS="$REPO_ROOT/benchmarks/results"
mkdir -p "$RESULTS"

if [ $# -eq 0 ]; then COMMON=(--synthetic); else COMMON=("$@"); fi

run() {  # name : extra flags...
  local name="$1"; shift
  echo "================ $name ================"
  python "$BENCH" run "${COMMON[@]}" "$@" --json "$RESULTS/sweep_${name}.json"
}

run baseline
run tile_batch_4   --tile-batch-size 4
run tile_batch_8   --tile-batch-size 8
run cache_model    --cache-model --repeats 6
run max_folds_1    --max-folds 1
run step_0p7       --step-size 0.7

echo
echo "################ COMPARISON ################"
python "$BENCH" compare \
  "$RESULTS/sweep_baseline.json" \
  "$RESULTS/sweep_tile_batch_4.json" \
  "$RESULTS/sweep_tile_batch_8.json" \
  "$RESULTS/sweep_cache_model.json" \
  "$RESULTS/sweep_max_folds_1.json" \
  "$RESULTS/sweep_step_0p7.json"
echo
echo "cache_model: compare 'load: first' vs 'load: steady-median' in sweep_cache_model.json --"
echo "the median load should collapse to ~0 once the model is cached."
