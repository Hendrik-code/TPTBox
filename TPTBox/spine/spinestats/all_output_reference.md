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
| `ivd_geometry` | `measure_ivd_and_vertebra_geometry(..., structure_label=100)` | intervertebral discs (per-label entries carry wedge angles/indices) |
| `vert_geometry` | `measure_ivd_and_vertebra_geometry(..., structure_label=50)` | vertebral bodies (per-label entries carry wedge angles/indices) |
| `VBQ_score` | `VBQ_score` | vertebral bone quality (T2 signal ratio) |
| `body_composition_score` | `body_composition_score` | axial CSA per tissue at chosen vertebral levels |
| `muscle_fat_infiltration` | `muscle_fat_infiltration` | Dixon fat-fraction based muscle-quality metrics |
| `torso_vat_sat_muscle_mass` | `torso_vat_sat_muscle_mass` | whole-torso VAT / SAT / muscle volume |
| `cobb`, `curv` | `plot_cobb_and_lordosis_and_kyphosis` | only when called with `cobb=True` |
| `sva`, `coronal_balance` | `curvature.compute_sva`, `compute_coronal_balance` | plumb-line balance offsets in mm (sagittal/coronal) |
| `axial_rotation` | `curvature.compute_axial_rotation` | per-vertebra axial rotation angle in the axial plane |
| `segmental_endplate_angles` | `curvature.compute_segmental_endplate_angles` | inter-vertebral (disc) wedge angle in the sagittal plane |
| `curvature_profile` | `curvature.compute_curvature_profile` | spline arc/chord/κ profile plus apex positions |
| `multi_cobb` | `curvature.compute_multi_cobb` | multi-curve Cobb from the coronal spline projection |
| `pelvic_parameters` | `pelvic_parameters.compute_pelvic_parameters` | PI / PT / SS + PI-LL mismatch, two variants |

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
  (`C3-C6`, `T5-T8`, `L1-L4`) need every vertebra in the range to be
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
Default ranges are `C3-C6`, `T5-T8`, `L1-L4`.

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

## `sva` (Sagittal Vertical Axis)

Signed horizontal offset (mm) in the sagittal plane between the top
vertebra (default C7) and a base reference (S1 → L5 → L4 fallback).
Positive = C7 anterior of the base (typical). Additional fields
`top_vertebra`, `base_vertebra`, `base_landmark`, `top_pi_coords`,
`base_pi_coords`. **Caveat:** supine MRI values differ from standing
X-ray by ~10 mm.

## `coronal_balance`

Signed horizontal offset (mm) in the coronal plane between C7 and the
CSVL (approximated by the base vertebra R-coordinate). Positive = C7
right of CSVL. Extra fields: `top_vertebra`, `base_vertebra`,
`top_r_coord`, `base_r_coord`.

## `axial_rotation`

`dict[vertebra_name, degrees]`. Signed angle between
`Vertebra_Direction_Right` and the image right axis in the axial plane.
Positive = rotation towards the patient's left.

## `segmental_endplate_angles`

`dict["<upper>-<lower>", degrees]`. Signed sagittal-plane wedge angle
between the inferior endplates of two adjacent vertebrae. Approximates
the disc wedge without needing the disc mesh.

## `curvature_profile`

Cubic B-spline through the `Vertebra_Corpus` centroids. Reports arc
length, chord length, tortuosity (arc/chord), and |κ| statistics in
3D as well as the two 2D projections. `apices` (3D), `sagittal_apices`,
`coronal_apices` are lists of up to 6 dicts `{arc_mm, kappa_1_per_mm}`
ordered by arc position (superior → inferior).

## `multi_cobb`

Multi-curve Cobb detection from the coronal spline projection. Sign
changes of the signed curvature act as inflection points; one Cobb
angle is emitted per segment.

- `curves`: `[{arc_start_mm, arc_end_mm, apex_arc_mm, length_mm,
  cobb_deg, handedness}]`
- `max_cobb_deg`: max absolute Cobb across curves

Handedness is `"right"` or `"left"` referring to the direction of the
curve's concavity.

## Wedge fields on `vert_geometry` / `ivd_geometry`

Added per label:

- `sagittal_wedge_deg` — `atan((x1 − x2) / x6)` (positive = anterior taller)
- `coronal_wedge_deg` — `atan((x3 − x4) / x5)` (positive = right taller)
- `sagittal_wedge_index` — `(x1 − x2) / mean(x1, x2)`
- `coronal_wedge_index` — `(x3 − x4) / mean(x3, x4)`

Genant-style fracture screening: `sagittal_wedge_index` below ≈ −0.4
corresponds to > 40 % anterior height loss.

## `pelvic_parameters`

Present when the fullbody-POI json for the subject exists under
`derivatives-fullbody-poi/…/vibe/sub-*_seg-fullbody_poi.json`. Two
variants side by side (compare in QC):

- `poi_ap` — canonical: uses `Sacral_Crest_S1` posterior + `Anterior_Longitudinal_Medial` anterior for the S1 endplate
- `poi_ala` — alternate: uses the midpoint of `Sacrum_Ala_Superior_L/R` as the "anterior" reference. Included as a robustness check; in practice it under-estimates PI compared to `poi_ap`

Per variant:

- `pi_deg` (unsigned, anatomical constant)
- `pt_deg` (signed, positive = sacrum posterior of hip axis)
- `ss_deg` (unsigned, endplate tilt from horizontal)
- `pi_ll_mismatch_deg` = `pi_deg − curv["lumbar_lordosis"]`; `None`
  if lumbar lordosis is missing
- `hip_center_mm`, `s1_endplate_center_mm` for QC

Relationship: `PI = PT + SS` (up to sign).

**Limitations:**

1. **Supine vs. standing.** SS/PT are position-dependent — supine SS
   is systematically lower than standing SS by ~10-15°, PT
   correspondingly higher. **PI is position-invariant** and the safe
   number to compare across cohorts. PI-LL uses the supine LL and is
   not directly comparable to Schwab-style standing thresholds.
2. S1 endplate is reconstructed from two point landmarks; the anterior
   ligament attachment can drift inferior with age / degeneration.
3. Bi-femoral axis uses the atlas-registered `PELVIS_CENTER` landmark
   under each femur (label 13/113 in the fullbody-POI mapping).
4. No axial-pelvic-obliquity correction; sagittal plane = world `(y, z)`.

---

## Excel collector

`ExcelCollector` in `_run_all.py` runs a background process that turns
each finished json into three rolling Excel files in a configurable
folder:

- `per_subject.xlsx` — one row per subject with every scalar top-level
  metric flattened to dotted keys
  (e.g. `VBQ_score.VBQ_L1-L4`, `torso_vat_sat_muscle_mass.VAT`).
  `ivd_geometry` and `vert_geometry` are excluded here.
- `per_vertebra.xlsx` — one row per (subject, label) from
  `vert_geometry` (vertebra bodies).
- `per_ivd.xlsx` — one row per (subject, label) from `ivd_geometry`
  (intervertebral discs).

The vertebra and IVD tables are split so that the full NAKO cohort stays
under Excel's per-sheet row limit (1 048 576 rows).

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
