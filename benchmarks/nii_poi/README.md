# NII / POI speed + memory benchmark

Regression detection for the two classes almost every TPTBox workflow runs
through: `NII` (`TPTBox/core/nii_wrapper.py`) and `POI` (`TPTBox/core/poi.py`,
`TPTBox/core/poi_fun/`).

`.github/workflows/benchmark.yml` runs this on every PR to `main`: once on the
PR head, once on the PR base commit (same runner, same harness), then posts a
comparison table as a PR comment and **fails the job** if anything regressed.

This is unrelated to `benchmarks/benchmark_nnunet_inference.py`, which is a
GPU-only inference harness.

## Why it exists

`TPTBox/tests/speedtests/` compares *candidate implementations against each
other*, by hand, locally. This compares *HEAD against main*, automatically, and
covers memory as well as time.

## Workloads

Six cases, spanning a small/large × 2D/3D grid, so a change that only hurts
large 3D volumes shows up as such instead of being averaged away:

| case | source | shape |
|---|---|---|
| `ct_3d` | `TPTBox/tests/sample_ct` | `(73, 47, 73)` |
| `ct_2d` | centre slice of `ct_3d` | `(73, 47, 1)` |
| `mri_3d` | `TPTBox/tests/sample_mri` | `(68, 52, 67)` |
| `mri_2d` | centre slice of `mri_3d` | `(68, 52, 1)` |
| `synth_3d` | generated | `(400, 400, 400)` |
| `synth_2d` | generated | `(400, 400, 1)` |

**2D means a singleton third axis, not a 2-element shape.** `NII` has no 2D code
path — orientation, `reorient` and `rescale` all assume three axes — so a real
2-D `Nifti1Image` would crash. A `(X, Y, 1)` volume is geometrically 2D and goes
through the normal machinery. The slice is taken with `NII.apply_crop`, which
updates the affine along with the data.

The synthetic volume is hollow labelled cuboids on a regular grid: hollow so
`fill_holes` has work to do, gridded so they never touch and the result is
byte-identical on every machine, ~20% foreground so `use_crop=True` cannot
trivialise the morphology measurements.

## What is measured

~33 measurements per case, defined once in `measurements.py` so the speed and
memory runs always cover exactly the same keys.

- **NII IO** — `load` (image and segmentation), `save`
- **NII arrays** — `get_array`, `set_dtype`
- **NII geometry** — `reorient`, `rescale` (image and segmentation),
  `resample_from_to`, `compute_crop`, `apply_crop`, `pad_to`
- **NII statistics** — `unique`, `volumes`, `center_of_masses`
- **NII labels** — `extract_label`, `map_labels`
- **NII morphology / cc3d** — `dilate_msk`, `erode_msk`, `fill_holes`,
  `get_connected_components`, `filter_connected_components`
- **POI** — `calc_centroids` (and the legacy `_crop=False` per-label path as a
  standing A/B), `reorient`, `rescale`, `to_global`, `local_to_global_arr`,
  `resample_from_to`, `map_labels`, `save`, `load`
- **POI pipeline** — `calc_poi_from_subreg_vert`, on the real CT/MRI 3D cases
  only; it is meaningless on a single slice or on synthetic cuboids

Two measurements are skipped where they would not be informative:
`poi_calc_centroids_nocrop` on the large cases (it is a per-label scipy loop and
would dominate the entire run) and `poi_calc_poi_from_subreg_vert` on anything
that is not real spine data.

Every call is wrapped in try/except. If an op raises, or does not exist on the
older baseline commit, the key is dropped and listed on a `skipped:` line rather
than aborting the run — that is what lets one harness measure two commits.

`metric_*` rows (voxel count, label count, foreground %) are context, not
measurements, and never gate.

## Running it

```bash
# defaults: 10 repeats + 1 warmup, 400^3 synthetic case  (~25 min)
python benchmarks/nii_poi/bench_speed.py --json head.json
python benchmarks/nii_poi/bench_mem.py   --json head_mem.json

# what CI runs: real CT/MRI cases only, 5 repeats + 1 warmup  (~1 min each)
python benchmarks/nii_poi/bench_speed.py --cases ct_3d,ct_2d,mri_3d,mri_2d --repeats 5 --json head.json

# quick local run including the 256^3 synthetic case  (~2 min each)
python benchmarks/nii_poi/bench_speed.py --quick --json head.json

# one case, fast iteration
python benchmarks/nii_poi/bench_speed.py --cases ct_3d,ct_2d --repeats 3

# compare two runs
python benchmarks/nii_poi/compare.py     baseline.json     head.json
python benchmarks/nii_poi/compare_mem.py baseline_mem.json head_mem.json
```

`--quick` lowers the synthetic edge length to 256 as well as the repeat count. A
full pass over 400³ is ~2 minutes; 256³ is ~4× cheaper and still a large 3D
volume. Both sides of any comparison always use the same value, so this changes
sensitivity, never correctness.

**CI skips the synthetic cases entirely** and only runs the real CT/MRI 3D/2D
cases: the synthetic 400³/256³ workloads dominate wall time (~30s per case per
run, four runs per PR) and are also the noisiest single knob in the workflow,
since they exercise code paths whose runtime scales with volume rather than with
what a spine workflow actually looks like. Regressions specific to very large
volumes are still catchable locally by omitting `--cases`.

To reproduce the baseline swap locally:

```bash
cp -a TPTBox /tmp/TPTBox-head
rm -rf TPTBox && git checkout main -- TPTBox
python benchmarks/nii_poi/bench_speed.py --quick --json baseline.json
rm -rf TPTBox && cp -a /tmp/TPTBox-head TPTBox
```

## Reading the output

Both comparison tools emit one table per case:

```
| Measurement | baseline ms (median ±½·range) | head ms (median ±½·range) | Δ % | p |
| `nii_dilate_msk` | 12.30 ±0.45 | 15.60 ±0.32 | +26.8% | 0.002 |
```

`±` is half the min→p90 range, i.e. how noisy that measurement was, not a
standard deviation. 🔴 marks a gated regression, 🟢 a gated improvement, and
`(noise)` flags a row whose baseline is below the display noise floor (3 ms /
3 MiB): its Δ% is dominated by shared-runner jitter, not by any real change.

Each per-case table shows only the **five most-changed rows**; the rest fold
into a `<details>` block so the PR comment stays scannable. Rank order is by
`|Δ%|`, with `(noise)` rows always sorted after real changes so a large swing
on a sub-ms measurement cannot push a real regression out of the headline.

### The gate

A measurement fails the build only when **all three** hold:

1. the baseline is at least **1 ms** / **1 MiB** and the key is not `metric_*` —
   sub-unit jitter on a shared runner must never block a merge;
2. the median grew by at least `--fail-on-regression-pct` (CI uses **50%**);
3. Welch's t-test over the two samples gives `p < --alpha` (default 0.05). With
   fewer than two samples per side there is no t-test, so it falls back to
   "head's best run is still worse than baseline's p90".

### Memory numbers specifically

The number is **peak RSS growth per call, in MiB**, not allocated bytes.

Each sampled iteration runs in a forked child. CPython's heap does not shrink,
so a second in-process iteration of the same call reuses the arena freed by the
first and reports ≈0 MiB; a fresh address space per iteration removes that. The
baseline RSS is read *inside* the child after the fork, so copy-on-write pages
of the input volume are already accounted for.

There is a constant **floor of ~1 MiB** per measurement (the sampler thread, the
gc pass, copy-on-write faults from refcount writes on inherited objects). It is
measured explicitly at startup and reported as `floor_mib` in the JSON and in
the comparison header. It cancels out between the two sides, and it sits at the
1 MiB gating threshold, so floor-level rows are excluded from the gate anyway.

Warmup for the memory benchmark runs in the **parent**, deliberately: a forked
child inherits whatever the parent has already imported, so the job of the
warmup here is to pull in lazily-imported modules (`scipy.ndimage`, `cc3d`, …)
before the fork, rather than making every child pay for them.

On a platform without `fork` or without `/proc`, the harness degrades to
in-process measurement and/or `getrusage` and records which, as `isolation` and
`sampler` in the JSON.

## Files

- `workloads.py` — case construction. Reads the sample data **by path**, not via
  `TPTBox.tests.test_utils`: the workflow swaps the whole `TPTBox/` tree to
  produce the baseline, so anything imported from inside it would silently
  become the old version.
- `measurements.py` — the shared measurement registry.
- `bench_speed.py` / `bench_mem.py` — the two harnesses.
- `compare.py` / `compare_mem.py` — thin CLI wrappers over `_compare_core.py`.
- `_common.py` — statistics, JSON schema helpers, shared CLI flags.
