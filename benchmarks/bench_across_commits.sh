#!/usr/bin/env bash
# Replay the benchmark across the optimisation commits and tabulate per-commit deltas.
#
# Isolates the ALWAYS-ON changes (cuDNN/TF32, inference_mode, empty_cache, fold fix): each commit
# is checked out, run with one fixed config, and compared. The opt-in commits (cache_model,
# tile_batch_size) are no-ops under the default config and show ~0 here -- use bench_flag_sweep.sh
# for those.
#
# Usage:
#   benchmarks/bench_across_commits.sh [extra args forwarded to `benchmark ... run`]
#   BASELINE=<sha> benchmarks/bench_across_commits.sh --device cuda --repeats 5
# With no extra args it uses a synthetic input. Requires a clean working tree and an editable
# TPTBox install (so checking out a commit changes the imported code).
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is not clean. Commit or stash before running." >&2
  exit 1
fi

BASELINE="${BASELINE:-3906165}"
ORIG_REF="$(git symbolic-ref --short -q HEAD || git rev-parse HEAD)"
TMP_BENCH="$(mktemp -t bench_nnunet_XXXX.py)"
cp "$REPO_ROOT/benchmarks/benchmark_nnunet_inference.py" "$TMP_BENCH"
RESULTS="$REPO_ROOT/benchmarks/results"
mkdir -p "$RESULTS"

cleanup() { git checkout -q "$ORIG_REF"; rm -f "$TMP_BENCH"; }
trap cleanup EXIT

# baseline + every commit since it, in chronological order
mapfile -t COMMITS < <(printf '%s\n' "$BASELINE"; git rev-list --reverse "${BASELINE}..${ORIG_REF}")

if [ $# -eq 0 ]; then RUN_ARGS=(--synthetic); else RUN_ARGS=("$@"); fi

JSONS=()
i=0
for sha in "${COMMITS[@]}"; do
  short="$(git rev-parse --short "$sha")"
  out="$RESULTS/$(printf '%02d' "$i")_${short}.json"
  echo "================================================================"
  echo "[$i] $short : $(git log -1 --format=%s "$sha")"
  echo "================================================================"
  git checkout -q "$sha"
  python "$TMP_BENCH" run "${RUN_ARGS[@]}" --json "$out"
  JSONS+=("$out")
  i=$((i + 1))
done

git checkout -q "$ORIG_REF"
echo
echo "################ COMPARISON ################"
python "$TMP_BENCH" compare "${JSONS[@]}"
