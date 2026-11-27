# fast-anything: Rust ports

[fast-anything](https://github.com/EL4CTEO/fast-anything) (v0.8.8) profiles an app, records real calls to
a hot pure function, checks a Rust port against those calls, and swaps the port in through a marked block
appended to the Python module. This folder holds the workloads it was run on and the scripts to check and time the results.

| port | Python original | kernel speedup | workload step speedup |
|---|---|---:|---:|
| `poi_fun.strategies.corner_bisection_search` | scipy `RegularGridInterpolator` loop | ~2000× | POIs: 1.27-1.47× |
| `np_utils.unique_integers` | `np.bincount` / `np.unique` | 7.7-9.5× | 1.08-1.24× |
| `np_utils.connected_component_labels` | cc3d | 2.0-3.4× | 1.08-1.12× |
| `np_utils.euclidean_distance_transform` | scipy `distance_transform_edt` | 2.3× | 1.24× |
| `np_utils.binary_dilation_perimeter` | per-voxel Python loop | 92-109× | 7.2-10.3× |
| `np_utils.binary_erosion_radius1` | scipy `binary_erosion` | 3.6-11× | 1.15-1.27× |
| `internal.nii_help.affine_resample` | scipy `affine_transform`, order 0/1 | 5.3-15.6× | 4.3-5.7× |

The six array kernels together make the segmentation workload (`workload_seg.py`) 2.5× faster per sample.

Every port gives bit-identical results to its Python original (one exception, NaN sign bits for a NaN-and-inf
`sampling`, is described in section 2). When `.fast/lib` holds no built port (a fresh
clone, an installed wheel, CI), the blocks at the end of the modules fall back to the Python functions.
`FAST_ANYTHING_DISABLE=1` forces the fallback.

## 1. `corner_bisection_search` (POI pipeline)

### What was ported

Profiling the full POI pipeline (`workload.py`) put 15% of the run in `_find_corner_point`
(`TPTBox/core/poi_fun/strategies.py`), which `strategy_find_corner` uses for the corner POIs: a 2D bisection
search that evaluates scipy's `RegularGridInterpolator` one point at a time, ~80 times per call.

fast-anything only ports module-level functions on plain data, so the search loop was moved, unchanged, into
`corner_bisection_search(region: uint8 mask, start, v1, v2, delta)`. Two guards were added; each fires only
where the old loop never terminated:

- the start point is outside the mask: the shrinking factors get stuck at the smallest subnormal, so it now returns `None`;
- `delta` is at or below the floating-point resolution: steps get absorbed by rounding, so it now returns the point reached.

The Rust port (`.fast/ports/TPTBox.core.poi_fun.strategies.corner_bisection_search/src/lib.rs`) reproduces scipy's
linear interpolation in the same float-operation order, so results are bit-identical, not just within tolerance.

### Results

AMD EPYC 7413 (shared machine, load average ~35 during the runs), Python 3.11.15, numpy 2.2.6, scipy 1.17.1, rustc 1.100.0.

Exactness: 287/287 POIs of the workload are bit-identical to the unmodified code, both with the Rust port and with the
Python fallback. fast-anything verify: 295/295 calls (173 recorded, 43 held out, 79 derived). `bench.py`:
72/72 real calls and 400/400 seeded random cases bit-identical.

Kernel, per call on the 72 real calls (`bench.py`, median of 7 rounds):

| implementation | per call | speedup |
|---|---:|---:|
| Python, scipy `RegularGridInterpolator` (original) | 6108 µs | 1× |
| Python, interpolation written out (`direct_python` in `bench.py`, not used by TPTBox) | 155 µs | 39× |
| Rust port | 3.1 µs | ~2000× |

`calc_poi_from_subreg_vert` with all 57 POI ids (`bench.py`, median of 7 rotated rounds):

| sample | original | Rust port | pure-Python rewrite |
|---|---:|---:|---:|
| CT (73×47×73, 3 vertebrae) | 1.729 s | 1.366 s (1.27×) | 1.455 s (1.19×) |
| MRI (68×52×67, 3 vertebrae) | 0.681 s | 0.463 s (1.47×) | 0.454 s (1.50×) |

Whole workload process including interpreter start-up (`fast-anything bench --runs 7`): 9.74 s → 8.54 s, 1.14×
(A/A noise ±2.8%).

The pipeline gain is capped by the kernel's share of the pipeline (about 0.22 s per sample). The pure-Python
rewrite reaches the same pipeline speed within noise, without a Rust build. The Rust port only pays off over
that if the search is called far more often than in the POI pipeline.

## 2. Array primitives (segmentation post-processing)

### What was ported

The POI workload spends under 2% of its time in `np.unique`, connected components, distance transforms,
morphology and resampling, so `workload_seg.py` exercises them through the NII API on the same CT and MRI
samples: label lists, 1 mm and 0.75 mm rescales, resampling onto an image grid, erosion and dilation,
connected-component labelling and filtering, Euclidean dilation and erosion and a Voronoi partition, on the
whole volume and per vertebra.

Each primitive was moved into a module-level kernel with typed array arguments that returns new arrays
(fast-anything refuses functions that modify their arguments). The callers in `np_utils.py` and
`nii_help.py` go through the kernel:

| kernel | Python original | callers | Rust port |
|---|---|---|---|
| `unique_integers(arr)` | bincount (unsigned, max < 2**20) or `np.unique` | `np_unique`, `np_unique_withoutzero`, so every `_to_labels` | 8/16-bit: one-pass presence table; wider: min/max pass, then a table over the range or a sort |
| `connected_component_labels(arr, connectivity)` | `cc3d.connected_components` | `np_connected_components(_per_label)`, `np_filter_connected_components`, `_split_cc_mincut` | union-find over runs of equal values, numbered like cc3d (memory order, first voxel) with cc3d's uint16/uint32 choice |
| `euclidean_distance_transform(mask, sampling, ...)` | `scipy.ndimage.distance_transform_edt` | `np_erode_msk_euclid`, `np_dilate_msk_euclid`, `np_voronoi_labels`, `_expand_seeds_to_mask` | scipy's separable Voronoi feature transform with its float-operation order, so ties go to the same voxel |
| `binary_dilation_perimeter(image, selem)` | TPTBox's perimeter dilation, a Python loop per perimeter voxel | `_binary_dilation`, so `np_dilate_msk`, `_binary_erosion/closing/opening`, `_split_cc_mincut` | perimeter on 64-bit row words, element stamped onto perimeter voxels only |
| `binary_erosion_radius1(image, struct, iterations, border_value)` | `scipy.ndimage.binary_erosion` | `np_erode_msk` | AND of shifted rows, stops at a fixed point |
| `affine_resample(arr, matrix, offset, shape, order, mode, cval, dtype)` | `scipy.ndimage.affine_transform`, order 0/1, `nearest`/`constant` | `_resample_from_to`, so `NII.rescale` and `NII.resample_from_to` | scipy's coordinate, tap and rounding arithmetic in the same order; per-axis tap tables for axis-aligned matrices |

Scope and behaviour changes:

- `affine_resample` covers orders 0 and 1 with modes `nearest` and `constant`; `_resample_from_to` calls scipy
  directly for everything else. Image rescales default to order 3 and are not accelerated.
- `binary_erosion_radius1` accepts structuring elements of size 1 or 3 per axis, which covers every
  `generate_binary_structure` element TPTBox builds. For wider elements with a set `border_value`, scipy's
  iterated erosion is not plain repeated erosion next to the border.
- `connected_component_labels` copies views that are neither C- nor Fortran-contiguous before calling cc3d.
  cc3d 3.26 reads part of its bookkeeping from the buffer behind a view and raised IndexError or mislabelled
  about 1-2% of random strided views.
- `affine_resample` raises ValueError for an empty image with mode `nearest`, where scipy read outside the array.
- `_expand_seeds_to_mask` no longer computes distances it discarded. `np_unique_withoutzero` is now `np_unique`
  without 0, and `np_erode_msk` erodes boolean masks instead of label copies; both give the same results as before.

fast-anything ranks only the 40 slowest functions of a run, and cc3d is fast enough to miss that list, so
`connected_component_labels` was scanned and recorded with `workload_seg.py --only cc`.

### Results

Same machine, shared, load average 17-44 during the runs; cc3d 3.26.1.

Exactness. fast-anything verify replays the recorded calls, held-out calls and derived edge cases
(axis-0 slices including empty arrays, concatenations up to 4096 rows, special ints, floats and strings):

| kernel | verify | per call (verify) |
|---|---:|---:|
| `unique_integers` | 460/460 | 319 → 34 µs |
| `connected_component_labels` | 280/280 | 152 → 85 µs |
| `euclidean_distance_transform` | 291/291 | 2.6 → 0.97 ms |
| `binary_dilation_perimeter` | 701/701 | 3.8 ms → 35 µs |
| `binary_erosion_radius1` | 605/605 | 109 → 30 µs |
| `affine_resample` | 179/179 | 16.8 → 1.8 ms |

`bench_seg.py`: every call the workload makes (86, 26, 39, 166, 106 and 12 per kernel) is identical in values,
dtype, shape and memory order, and a workload run on the Rust ports makes the same calls with the same
results. During development each port was also checked against its original on 2,000-4,000 seeded random and
edge cases (byte-swapped, Fortran-order and strided arrays, every dtype). Those scripts are not in the repo;
`unit_tests/test_np_kernels.py` keeps a smaller set. The one known difference: with a `sampling` that mixes NaN
and inf, some NaN distances differ from scipy in the sign bit.

Kernel (`bench_seg.py`, median of 7 rounds). "Real" is the sum over the workload's calls; "large" is one call
on the CT sample at 0.5 mm (219×141×219):

| kernel | real calls, Python → Rust | large, Python → Rust |
|---|---:|---:|
| `unique_integers` | 28.8 → 3.0 ms (9.5×) | 15.1 → 2.0 ms (7.7×) |
| `connected_component_labels` | 4.3 → 2.2 ms (2.0×) | 6.5 → 2.0 ms (3.4×) |
| `euclidean_distance_transform` | 220 → 94 ms (2.3×) | 537 → 231 ms (2.3×) |
| `binary_dilation_perimeter` | 651 → 6.0 ms (109×) | 563 → 6.1 ms (92×) |
| `binary_erosion_radius1` | 11.9 → 3.3 ms (3.6×) | 50.9 → 4.6 ms (11×) |
| `affine_resample` | 524 → 73 ms (7.2×) | order 0: 140 → 8.9 ms (15.6×); order 1: 333 → 63 ms (5.3×) |

`workload_seg.py` per sample (`bench_seg.py`, median of 7 rotated rounds; all six kernels swapped):

| steps | CT | MRI |
|---|---:|---:|
| all | 1.100 → 0.431 s (2.55×) | 1.002 → 0.396 s (2.53×) |
| `--only dilate` | 0.390 → 0.038 s (10.3×) | 0.376 → 0.052 s (7.2×) |
| `--only resample` | 0.278 → 0.049 s (5.7×) | 0.275 → 0.063 s (4.3×) |
| `--only edt` | 0.410 → 0.332 s (1.24×) | 0.355 → 0.286 s (1.24×) |
| `--only erode` | 0.030 → 0.024 s (1.27×) | 0.046 → 0.040 s (1.15×) |
| `--only unique` | 0.010 → 0.008 s (1.24×) | 0.022 → 0.021 s (1.08×) |
| `--only cc` | 0.025 → 0.023 s (1.12×) | 0.039 → 0.036 s (1.08×) |

Whole workload process including interpreter start-up (`fast-anything bench --runs 7`): 7.19 s → 3.35 s,
2.1× (A/A noise ±1.4%).

Where it pays: dilation, because the original is a Python loop, and order-0/1 resampling. For dilation alone,
`scipy.ndimage.binary_dilation` gives the same result for `generate_binary_structure` elements and is 4.4-9×
faster than the original (4.8 vs 44 ms on 73×47×73, 124 vs 551 ms on 219×141×219), without a Rust build; the
Rust port is another 11-20× faster than that. The
distance transform is 2.3× faster, but the Voronoi tie search around it is plain Python and NumPy, so the EDT
step gains 1.24×. cc3d, scipy's erosion and `np_unique` were already compiled code: their kernels are 2-12×
faster, but the steps around them gain at most 1.3×, as fast-anything's scan predicted (all three under its 2%
threshold).

## Reproduce

Requires Rust >= 1.88 (edition 2024). From the repo root, build every port into `.fast/lib` with plain cargo. The
ports are built with `target-cpu=native`, so build them on the machine that runs them.

```sh
for crate in .fast/ports/*/; do
  ext=$(sed -n 's/^name = "\(.*\)"$/\1/p' "$crate/Cargo.toml" | head -n1)
  (cd "$crate" && CARGO_TARGET_DIR=../../target cargo build --release)
  mkdir -p .fast/lib && cp ".fast/target/release/lib$ext.so" ".fast/lib/$ext.abi3.so"
done

python -m benchmarks.fast_anything.bench              # corner search: exactness + kernel + pipeline (~1-2 min)
python -m benchmarks.fast_anything.bench_seg          # array primitives: exactness + kernels + pipeline (~2 min)
python -m pytest unit_tests/test_corner_search.py unit_tests/test_np_kernels.py   # active implementation
FAST_ANYTHING_DISABLE=1 python -m pytest unit_tests/test_corner_search.py unit_tests/test_np_kernels.py
```

`fast-anything build` fails on a fresh clone ("nothing ported yet"): it reads the port registry from
`.fast/state.json`, which is per-machine (it records the local interpreter path) and not committed. `scan` followed
by `port` recreates it; `port` re-records the calls but keeps the committed `src/lib.rs`. It does rewrite
`Cargo.toml` and adds the `numpy` dependency only when a parameter annotation names a typed array, so after porting
`unique_integers` or `connected_component_labels` add `numpy = "0.30"` back under `[dependencies]`.

The full fast-anything loop, as it was run here:

```sh
cargo install fast-anything --version 0.8.8                          # or: npm i -g @el4cteo/fast-anything
fast-anything scan -- python -m benchmarks.fast_anything.workload   # -m, so TPTBox counts as importable
fast-anything port TPTBox.core.poi_fun.strategies.corner_bisection_search
fast-anything verify TPTBox.core.poi_fun.strategies.corner_bisection_search
fast-anything enable TPTBox.core.poi_fun.strategies.corner_bisection_search

fast-anything scan -- python -m benchmarks.fast_anything.workload_seg
for f in unique_integers euclidean_distance_transform binary_dilation_perimeter binary_erosion_radius1; do
  fast-anything port TPTBox.core.np_utils.$f
done
fast-anything port TPTBox.core.internal.nii_help.affine_resample
fast-anything scan -- python -m benchmarks.fast_anything.workload_seg --only cc
fast-anything port TPTBox.core.np_utils.connected_component_labels
# then for each port: fast-anything verify <id> && fast-anything enable <id>
fast-anything scan -- python -m benchmarks.fast_anything.workload_seg   # bench times the last scanned command
fast-anything bench --runs 7
```

`check_pois.py --save ref.json` / `--compare ref.json` compares every POI the POI workload computes against a saved reference.
