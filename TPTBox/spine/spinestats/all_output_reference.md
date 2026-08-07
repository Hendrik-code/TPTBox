# `_run_all.py` — Radiologist Reference

This document lists every key produced by `run_all(file_dict)` in
`_run_all.py`, together with its unit and important implementation details.
It is aimed at radiologists reviewing the numbers, so it focuses on
"what does this mean clinically" and "how was it computed", not on the
Python API.

## How the pipeline is organised

`run_all` writes a single json per subject with these top-level keys:

| Key | Source function | What it covers |
|---|---|---|
| `ivd_geometry` | `measure_ivd_and_vertebra_geometry(..., structure_label=100)` | intervertebral discs |
| `vert_geometry` | `measure_ivd_and_vertebra_geometry(..., structure_label=50)` | vertebral bodies |
| `VBQ_score` | `VBQ_score` | vertebral bone quality (T2 signal ratio) |
| `body_composition_score` | `body_composition_score` | axial CSA per tissue at chosen vertebral levels |
| `muscle_fat_infiltration` | `muscle_fat_infiltration` | Dixon fat-fraction based muscle-quality metrics |
| `torso_vat_sat_muscle_mass` | `torso_vat_sat_muscle_mass` | whole-torso VAT / SAT / muscle volume |
| `cobb`, `curv` | `plot_cobb_and_lordosis_and_kyphosis` | only when called with `cobb=True` |

Angles are in **degrees**, lengths in **millimetres**, areas in **mm²**,
volumes in **mm³**, fat fractions are **unitless** in `[0, 1]`, MR signal
values are in **arbitrary units (a.u.)** and only meaningful as ratios.

Caching: `run_all(..., override=False)` (the default) reuses the json
when it exists, is newer than every input segmentation file, and
contains all of the required top-level keys. Pass `override=True` to
force recomputation.

### Input requirements

`run_all` assumes whole-body-style acquisitions:

- **T2w image** — must cover the **full spine** (cervical through
  sacrum). Curvature angles and per-vertebra geometry silently return
  `None`/`NaN` for any level that is cropped away, and the VBQ ranges
  (`C3-C6`, `T5-T8`, `L1-L1`) need every vertebra in the range to be
  visible.
- **VIBE water/fat images** — must cover the **full torso**. Fat
  fraction and muscle CSA are computed on whatever axial slices are
  present, so a cropped VIBE silently biases per-region CSA and IMAT
  volumes.
- **Segmentations** (`vert`, `spine`, `vibeseg100`, `roi`) — must
  match the extent of their underlying image. In particular,
  `torso_vat_sat_muscle_mass` explicitly verifies that both the
  clavicula and the pelvis are present in the VIBESeg mask; if either
  is missing it aborts, returns `NaN` volumes and stores the reason in
  the `reason` key.

### How to produce the segmentations

All required segmentations can be produced from
`TPTBox.segmentation`:

- **`vert` / `spine`** — run **SPINEPS** on the T2w image
  (`run_spineps`, `get_outpaths_spineps`, `_run_spineps_all`).
- **`vibeseg100`** — run **VIBESegmentator** with
  `run_vibeseg(..., dataset_id=100)` on the VIBE stack. Dataset
  **100** (MR and CT) is what `run_all` targets; dataset **12**
  (0.8 mm iso CT) is also supported by the composition/infiltration
  functions via `dataset_id=12`.
- **`roi`** — run VIBESegmentator with dataset **278** on the VIBE
  stack. The raw dataset-278 ROI is **not perfect** and needs
  postprocessing before it is fed into `run_all`.

## Signal-based conventions used everywhere

Two things are worth understanding before reading the T2 signal keys:

1. **Peak-centered mean.** Ordinary mean signal inside a mask is
   sensitive to non-CSF voxels that leak into the spinal canal
   segmentation (nerve roots, vessel walls). The pipeline instead
   averages only voxels whose intensity falls in a window around the
   histogram peak. When both a peak-centered and a plain-mean version
   are stored, the plain-mean version is suffixed with `_old` for
   comparison. See `peak_centered_mean` in
   `TPTBox/spine/spinestats/torso_vat_sat.py`.
2. **Erosion.** Muscle and vertebral-body masks are eroded by one or
   two voxels before signal extraction to reduce partial-volume mixing
   at the boundary. Volume metrics are reported both after erosion
   (`*_volume_*`) and before erosion (`*_volume_no_erosion_*`) so the
   effect of the erosion is auditable.

---

## `ivd_geometry` and `vert_geometry`

Both keys hold `dict[label_id, dict[metric_name, value]]`, where each
`label_id` is one intervertebral disc (`ivd_geometry`) or vertebra
(`vert_geometry`). If a label fails to evaluate, its entry contains
`error` (message string) plus all metrics set to `NaN`.

| Key | Unit | Meaning |
|---|---|---|
| `volume_voxel` | mm³ | volume counted from the raw voxel mask |
| `volume_mesh` | mm³ | volume of the reconstructed surface mesh (same structure) |
| `height_center` | mm | height sampled through the structure's centroid |
| `mean_height` | mm | mean of the sampled heights over the structure surface |
| `max_height` | mm | maximum of the sampled heights |
| `lower_10_percent_height` | mm | 10th percentile of the sampled heights |
| `mean_diameter` | mm | diameter of the circle whose area equals the projected area |
| `anterior_height_x1` | mm | anterior height at the anterior point (x1) |
| `posterior_height_x2` | mm | posterior height at the posterior point (x2) |
| `right_height_x3` | mm | right-lateral height (x3) |
| `left_height_x4` | mm | left-lateral height (x4) |
| `width_lateral_x5` | mm | lateral width (x5) |
| `width_sagittal_x6` | mm | sagittal width (x6) |
| `signal` | unitless | peak-centered structure T2 signal / peak-centered spinal-canal T2 signal |
| `structure_signal` | a.u. | peak-centered T2 signal inside the eroded structure mask |
| `spinal_canal_signal` | a.u. | peak-centered T2 signal inside the eroded spinal canal reference |
| `signal_old` | unitless | same ratio computed with plain per-voxel means |
| `structure_signal_old` | a.u. | plain mean T2 signal inside the eroded structure mask |
| `spinal_canal_signal_old` | a.u. | plain mean T2 signal inside the eroded spinal canal |

Implementation notes:
- Structure orientation for IVDs is estimated from the disc's own voxel
  mask via PCA; vertebral orientation is read from precomputed POIs
  (except C2/dens, which falls back to PCA).
- x1–x6 are the six clinically standard directional heights/widths (see
  the geometry module docstring for the figure).
- Only labels present in the segmentation appear as keys.

## `VBQ_score`

`dict[str, float]`; one triple of entries per configured spinal range.
Default ranges are `C3-C6`, `T5-T8`, `L1-L1`.

| Key template | Unit | Meaning |
|---|---|---|
| `mean_signal_vertebra_<start>-<end>` | a.u. | mean T2 signal inside the eroded vertebral body mask over the range |
| `mean_signal_liquor_<start>-<end>` | a.u. | **peak-centered** mean T2 signal inside the spinal canal over the same S/I extent |
| `mean_signal_liquor_<start>-<end>_old` | a.u. | plain-mean version, kept for backward comparison |
| `VBQ_<start>-<end>` | unitless | vertebral signal divided by the peak-centered CSF signal |
| `VBQ_<start>-<end>_old` | unitless | same ratio using the plain-mean CSF signal |

Implementation notes:
- Vertebral body mask is eroded (default `n_erode=2`) to avoid the
  cortical rim.
- The spinal canal is cropped to the same superior–inferior slab as the
  vertebral bodies so the CSF reference matches the region of interest.
- Higher VBQ = darker vertebral bodies relative to CSF, associated in
  the literature with lower bone quality.

## `body_composition_score`

`dict[str, float]`; per-region axial cross-sectional-area statistics of
five tissue classes. Default regions are `T12-L1` and `L3-L3`.

Region tag: `{start.name}-{goal.name}` (e.g. `T12-L1`, `L3-L3`).

| Key template | Unit | Meaning |
|---|---|---|
| `mean_muscle_area_{region}` | mm² | mean skeletal muscle CSA across the region |
| `max_muscle_area_{region}` | mm² | maximum skeletal muscle CSA in the region |
| `mean_VAT_area_{region}` | mm² | mean visceral adipose tissue CSA |
| `max_VAT_area_{region}` | mm² | maximum visceral adipose tissue CSA |
| `mean_SAT_area_{region}` | mm² | mean subcutaneous adipose tissue CSA |
| `max_SAT_area_{region}` | mm² | maximum subcutaneous adipose tissue CSA |
| `mean_psoas_area_{region}` | mm² | mean psoas CSA (left + right) |
| `max_psoas_area_{region}` | mm² | maximum psoas CSA |
| `mean_autochthon_area_{region}` | mm² | mean autochthonous back-muscle CSA |
| `max_autochthon_area_{region}` | mm² | maximum autochthonous back-muscle CSA |
| `n_slices_{region}` | count | number of axial slices contributing to the muscle statistic |
| `muscle_index_{region}` | mm²/m² | `mean_muscle_area / height_m²`; only present when `height_m` is supplied |
| `muscle_fat_ratio_{region}` | unitless | `mean_muscle_area / (mean_VAT_area + mean_SAT_area)`; `NaN` if the denominator is zero |

Implementation notes:
- The superior–inferior extent of the vertebral bodies inside the region
  defines the slice range.
- Axial voxel area is derived from the VIBE geometry; slices with zero
  tissue are excluded before mean/max.
- If no vertebral body voxels are present for a region, that region is
  silently skipped (no keys emitted).

## `muscle_fat_infiltration`

`dict[str, float]`; per (region, muscle group) Dixon-based fat
infiltration metrics. Muscle groups (dataset_id=100) are:
`all_muscle`, `iliopsoas_left`, `iliopsoas_right`, `autochthon_left`,
`autochthon_right`, `muscle_other`. Suffix is `{region}_{muscle}`; when
no region is given, the region tag is `all`.

Fat fraction (FF) is computed voxel-wise as `FF = fat / (fat + water)`;
voxels with `FF >= threshold` (default 0.20) are IMAT, otherwise lean.

| Key template | Unit | Meaning |
|---|---|---|
| `mean_fat_fraction_{suffix}` | [0, 1] | mean FF over the eroded muscle mask |
| `median_fat_fraction_{suffix}` | [0, 1] | median FF over the eroded muscle mask |
| `mean_lean_fat_fraction_{suffix}` | [0, 1] | mean FF of lean voxels (FF < threshold) |
| `mean_IMAT_fat_fraction_{suffix}` | [0, 1] | mean FF of IMAT voxels (FF ≥ threshold) |
| `muscle_volume_{suffix}` | mm³ | muscle volume after erosion |
| `muscle_volume_no_erosion_{suffix}` | mm³ | muscle volume **before** erosion (raw segmentation volume) |
| `lean_muscle_volume_{suffix}` | mm³ | lean-muscle volume within the eroded mask |
| `lean_muscle_volume_no_erosion_{suffix}` | mm³ | lean-muscle volume within the un-eroded mask |
| `IMAT_volume_{suffix}` | mm³ | IMAT volume within the eroded mask |
| `IMAT_volume_no_erosion_{suffix}` | mm³ | IMAT volume within the un-eroded mask |
| `IMAT_fraction_{suffix}` | [0, 1] | IMAT voxel fraction within the eroded mask |

Implementation notes:
- Erosion iterations per muscle are configurable via the `erode` dict.
  Defaults: `all_muscle=1`, `iliopsoas_*=1`, `autochthon_*=2`,
  `muscle_other=1`.
- Fat-fraction statistics use the eroded mask; volumes are also
  reported for the un-eroded mask so the caller can inspect the effect
  of erosion.
- When `regions` are supplied, the analysis is restricted to the
  superior–inferior extent of the vertebral bodies in each range and
  the region tag becomes `{start.name}-{goal.name}`.

## `torso_vat_sat_muscle_mass`

`dict[str, float]`; whole-torso volumes restricted to the supplied ROI.
Only the results dict is stored in the json (the optional NII output is
dropped by `run_all` because it is not JSON-serializable).

| Key | Unit | Meaning |
|---|---|---|
| `VAT` | mm³ | visceral adipose tissue volume inside the ROI |
| `SAT` | mm³ | subcutaneous adipose tissue volume inside the ROI |
| `muscle_mass` | mm³ | skeletal muscle volume inside the ROI |
| `reason` | string | present only if the computation failed; explains why |

Implementation notes:
- The function checks that both the clavicula and the pelvis are
  present in the segmentation to make sure the full torso is covered.
  If either check fails, all three volumes are set to `NaN` and
  `reason` is populated.
- ROI labels (default 3–8) select which sub-regions of the torso count.

## `cobb` and `curv` (only when `cobb=True`)

- `cobb`: `list[tuple[float, int, int, int | None]]` from
  `compute_max_cobb_angle_multi` — one entry per detected scoliotic
  segment: `(max_angle_deg, from_vertebra_id, to_vertebra_id, apex_id_or_none)`.
  Angles are in **degrees**.
- `curv`: dict from `compute_lordosis_and_kyphosis`:
  - `cervical_lordosis` (deg) — computed between C2 and C7
  - `thoracic_kyphosis` (deg) — computed between T4 and the last thoracic vertebra
  - `lumbar_lordosis` (deg) — computed between L1 and the last lumbar vertebra

  Values can be `None` if the required vertebrae are missing from the
  POI.

---

## Excel collector

`ExcelCollector` in `all.py` runs a background process that turns each
finished json into two rolling Excel files in a configurable folder:

- `per_subject.xlsx` — one row per subject with every scalar top-level
  metric flattened to dotted keys
  (e.g. `VBQ_score.VBQ_L1-L1`, `torso_vat_sat_muscle_mass.VAT`).
- `per_vertebra.xlsx` — one row per (subject, label), populated from
  `vert_geometry` and `ivd_geometry`. The `source` column indicates
  which of the two sections the row came from.

Usage:

```python
collector = ExcelCollector(out_folder="/tmp/nako_summary")
collector.start()
for nako_id in ids:
    f = get_nako_paths(nako_id)
    run_all(f)  # writes the per-subject json
    collector.submit(nako_id, _final_json_path(f))
collector.close()  # flushes and joins
```

The collector re-writes the Excel files every `flush_every` submissions
(default 25) and once more at shutdown, so partial runs still produce
usable summaries.
