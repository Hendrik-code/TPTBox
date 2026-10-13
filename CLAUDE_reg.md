# TPTBox – Registration subsystem notes

This file complements `CLAUDE.md` with a targeted overview of the
`TPTBox/registration/` sub-package. It is meant as a working memory for anyone
extending or debugging the registration code paths, plus a running log of the
changes that were made in the "Point-Reg" refactor branch.

Update this file *incrementally* whenever you touch the registration code – add
sections when you introduce new classes, and update the status table when you
add or finish tasks.

## Layout

```
TPTBox/registration/
├── __init__.py                      # Aggregates public API. Optional imports guarded.
├── script_ax2sag.py                 # CLI helper (unchanged).
├── _ridged_points/
│   ├── point_registration.py        # SITK closed-form rigid (VersorRigid3D) landmark fit.
│   └── deepali_point_registration.py  # NEW: DeepALI equivalent of the above.
├── _ridged_intensity/
│   └── affine_deepali.py            # Rigid intensity-based registration used by Rigid_Elements.
├── _deepali/
│   ├── deepali_model.py             # General_Registration wrapper around DeepaliPairwiseImageTrainer.
│   ├── deepali_trainer.py           # Multi-resolution pyramid training loop.
│   └── spine_rigid_elements_reg.py  # Per-vertebra rigid registration + weighted blending.
└── _deformable/
    ├── deformable_reg.py            # Wraps General_Registration for BSpline / SVFFD.
    └── multilabel_segmentation.py   # Template_Registration + Template_Registration2 (NEW).
```

## Public entry points

| Class                          | File                                                              | Purpose                                                                          |
| ------------------------------ | ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `Point_Registration`           | `_ridged_points/point_registration.py`                            | SITK `VersorRigid3D` landmark-based rigid registration. Serialisable.            |
| `Deepali_Point_Registration`   | `_ridged_points/deepali_point_registration.py` **(NEW)**          | Kabsch/Horn SVD fit on paired POI landmarks, wrapped as a DeepALI `HomogeneousTransform`. |
| `General_Registration`         | `_deepali/deepali_model.py`                                       | Generic DeepALI pairwise image registration (rigid / affine / SVFFD / …).        |
| `Deformable_Registration`      | `_deformable/deformable_reg.py`                                   | Thin wrapper enforcing a non-rigid transform on `General_Registration`.          |
| `Template_Registration`        | `_deformable/multilabel_segmentation.py`                          | Two-stage rigid-then-deformable atlas → target alignment (POI-based rigid).      |
| `Template_Registration2`       | `_deformable/multilabel_segmentation.py` **(NEW)**                | Same idea, but accepts a `Deepali_Point_Registration` as pre-registration to skip the SITK resample. |
| `Rigid_Elements_Registration`  | `_deepali/spine_rigid_elements_reg.py`                            | Per-vertebra rigid registration + inverse-distance blending field.               |

## Key invariants

* **Coordinate conventions.** POIs and NIIs store voxel coords. `local_to_global(x, itk=True)` converts to LPS (ITK / DeepALI) world coords. `local_to_global(x)` (default) yields RAS (NIfTI) world coords. Never mix conventions.
* **Rigid transform direction.** For resampling (SITK `Resample` **and** DeepALI `TransformImage`) the *forward* direction of the transform is `fixed → moving`. Landmark fits usually estimate `moving → fixed`; invert once and stick with fixed→moving thereafter.
* **DeepALI transform tensor space.** `HomogeneousTransform.tensor()` returns a `(N, D, D+1)` matrix expressed in **target-grid cube coordinates** (`Axes.CUBE_CORNERS` if `align_corners=True`). When the fit is done in LPS world coords, convert via `M = A^-1 @ W @ A` with `A = target.transform(CUBE_CORNERS, WORLD)`. The moving-grid conversion is done by `SampleImage` at sampling time and must not be baked in – see `_build_deepali_transform` in `deepali_point_registration.py`.
* **`SampleImage(target, source)` vs `TransformImage(target, source)`.** The existing `_warp_image` in `deepali_model.py` uses `source=target_grid`, which silently *requires* the moving image to already live on the fixed grid. This is where the "same-space" assumption of `General_Registration` comes from.

## In-flight changes ("Point-Reg" branch)

Status legend: 🟡 in progress · ✅ done · ⬜ pending

| # | Task | Status |
| - | ---- | ------ |
| 1 | New `Deepali_Point_Registration` (closed-form rigid via DeepALI) | ✅ |
| 2 | `General_Registration` accepts fixed/moving on different grids (new `same_space=True` flag) | ✅ |
| 3 | `General_Registration` accepts `POI` / `POI_Global` landmark sets and matches shared IDs automatically | ✅ |
| 4 | New `Template_Registration2` that consumes a `Deepali_Point_Registration` as pre-registration | ✅ |
| 5 | Unit tests + speed / memory sanity checks | ✅ |
| 6 | This file + `CLAUDE.md` cross-reference | ✅ |
| 7 | Speed benchmark Deepali vs SimpleITK on CPU (recorded below) | ✅ |

## Testing / running

* Conda env: `/home/robert/anaconda3/envs/py3.12/bin/python` – DeepALI (`hf-deepali`) is installed there.
* Sample data:
  * `TPTBox/tests/sample_ct/` and `TPTBox/tests/sample_mri/` – tiny NIfTIs + segmentations, checked into the repo.
  * `tutorials/tutorial_data_processing/` – DICOM + PixelPandemonium MR pair, downloaded by the tutorial.
* Existing unit tests live in `unit_tests/`. New registration tests should follow the same pattern (no GPU-only paths in default tests; guard CUDA imports).
* `test_deformable_stage_improves_over_rigid_only` needs `elasticdeform`. **Known issue: the PyPI wheel is built against NumPy 1.x and fails at import under NumPy 2.x.** Install the source tarball instead so it recompiles locally:

  ```
  pip install https://github.com/gvtulder/elasticdeform/archive/refs/tags/v0.5.1.tar.gz
  ```

  Confirmed working with NumPy 2.4.1 / SciPy 1.17.0. If `elasticdeform` is unavailable the test is skipped cleanly (`_HAS_ELASTIC = False`).

## Benchmark: Deepali vs SimpleITK on CPU

Two benchmarks were run in `py3.12`:

**Tiny CT (73×47×73, 3 landmarks):**

| step  | SimpleITK | Deepali   | notes |
| ----- | --------- | --------- | ----- |
| fit   | ~7.3 ms   | ~1.4 ms   | Kabsch SVD is trivially cheap |
| warp  | ~9.6 ms   | ~19.7 ms  | SITK BSplineResampler is fast on this size |
| accuracy (mean-abs voxel err on identity round-trip)| **54.2** HU | **0.003** HU | SITK BSpline shows heavy ringing on CT |
| peak mem | – | 0.5 MiB (`tracemalloc`) | |

**Tutorial MR volume (270×220×72, 6 landmarks):**

| step  | SimpleITK | Deepali   | notes |
| ----- | --------- | --------- | ----- |
| fit   | ~87 ms    | ~1.4 ms   | ~60× faster – closed-form SVD stays constant with landmark count |
| warp  | ~131 ms   | ~243 ms   | SITK still edges out on CPU for pure resampling |
| accuracy | 6.53 | **0.0002** | Deepali linear sampler is drastically more accurate |
| peak mem | 32.6 MiB | 32.6 MiB | Same order of magnitude |

Short answer to the user's mid-run question: **Deepali is meaningfully more accurate and its fit is ~5–60× faster on CPU, but SITK is still ~1.5–2× faster on the actual image warp step on CPU**. Deepali becomes clearly faster once a GPU is used or once the warp is followed by more DeepALI work (no extra CPU→GPU copies). See `unit_tests/test_registration_deepali.py::TestSpeedAndMemory` – it prints a summary line every run.

## Optional-dependency handling

`hf-deepali` (and its prerequisite PyTorch) is an *optional* install.
`TPTBox.registration/__init__.py` therefore imports each deepali-backed entry
point inside its own `try/except ImportError`. On failure the name is replaced
by a small class/function stub built by `_make_missing_deepali_stub` /
`_make_missing_deepali_func` – instantiating or calling the stub raises

    ImportError: `<Name>` requires the optional dependency `hf-deepali`
    (which in turn requires PyTorch). Install both with:
        pip install torch hf-deepali

so users get an actionable message instead of a bare `NameError`. The SITK
`Point_Registration` path stays fully usable when deepali is absent. See the
`TestOptionalDeepaliStubs` test for a regression guard.

## Design notes / gotchas

* The current `_warp_image` uses `TransformImage(target=target_grid, source=target_grid)`. Passing a source image that lives on the moving grid is only safe when moving grid == fixed grid. Task #2 addresses this properly by threading the moving grid through when `same_space=False`.
* When adding DeepALI landmark loss (`LandmarkPointDistance`), the trainer expects target/source landmarks as `(N, M, D)` tensors in `Axes.CUBE_CORNERS` on the transform's grid. The wrapper in `General_Registration` should convert from POI voxel coords to that space.
* `Template_Registration` mutates its inputs by resampling the atlas after each SITK point-reg attempt. Template_Registration2 avoids the resample by keeping the transform composable with the downstream `Deformable_Registration`.
