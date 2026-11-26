# fast-anything: Rust port of `corner_bisection_search`

[fast-anything](https://github.com/EL4CTEO/fast-anything) (v0.8.8) profiles an app, records real calls to
a hot pure function, checks a Rust port against those calls, and swaps the port in through a marked block
appended to the Python module. This folder holds the workload it was run on and the scripts to check and time the result.

## What was ported

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

When `.fast/lib` holds no built port (a fresh clone, an installed wheel, CI), the block at the end of
`strategies.py` falls back to the Python function. `FAST_ANYTHING_DISABLE=1` forces the fallback.

## Reproduce

Requires Rust >= 1.88 (edition 2024). From the repo root:

```sh
cargo install fast-anything --version 0.8.8         # or: npm i -g @el4cteo/fast-anything
fast-anything build                                 # release-build the port into .fast/lib

python -m benchmarks.fast_anything.bench            # exactness + kernel + pipeline timings (~1-2 min)
python -m pytest unit_tests/test_corner_search.py   # runs against the active implementation
FAST_ANYTHING_DISABLE=1 python -m pytest unit_tests/test_corner_search.py
```

`fast-anything build` passes the interpreter recorded in `.fast/state.json` to PyO3; on a machine where that path
does not exist, run the `scan` below once first. To build without fast-anything, run `CARGO_TARGET_DIR=../../target cargo build --release` inside the port's folder,
then copy `.fast/target/release/libfa_tptbox_core_poi_fun_strategies_corner_bisection_search.so` to
`.fast/lib/fa_tptbox_core_poi_fun_strategies_corner_bisection_search.abi3.so`. The port is built with
`target-cpu=native`, so build it on the machine that runs it.

The full fast-anything loop, as it was run here:

```sh
fast-anything scan -- python -m benchmarks.fast_anything.workload   # -m, so TPTBox counts as importable
fast-anything port TPTBox.core.poi_fun.strategies.corner_bisection_search
fast-anything verify TPTBox.core.poi_fun.strategies.corner_bisection_search
fast-anything enable TPTBox.core.poi_fun.strategies.corner_bisection_search
fast-anything bench --runs 7
```

`check_pois.py --save ref.json` / `--compare ref.json` compares every POI the workload computes against a saved reference.

## Results

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
