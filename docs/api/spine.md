# Spine

Spine-specific utilities built on top of `NII` and `POI`:

- **2D snapshots** — sagittal / coronal / axial visualizations of CT and MRI volumes
  with vertebra masks, subregion overlays, centroids, MIPs, and colored-depth views.
- **Angles & curvature** — Cobb angle, lordosis and kyphosis measurements.
- **Vertebra & IVD geometry** — heights, widths, diameters, mesh-based measurements,
  and endplate reconstructions for individual vertebrae and intervertebral discs.
- **Body composition** — torso VAT / SAT / muscle statistics, VBQ score.
- **POI computation** — IVD landmarks, endplate points, facet joint (articularis)
  midpoints, plus body-quadrant partitioning of the vertebral body.

## 2D Snapshots

Modular building blocks for stitching together sagittal / coronal / axial views of
CT and MRI volumes. A snapshot is composed of one or more `Snapshot_Frame`s
(each defining image, segmentation, centroids, projection mode, and crop) and
rendered with `create_snapshot`. No 3D resampling is required — slices are
scaled to isotropic pixels before display.

::: TPTBox.spine.snapshot2D.snapshot_modular
    options:
      show_source: true
      filters: ["!^_"]

## Snapshot Templates

Ready-made snapshot layouts for common tasks (multi-panel CT MIP shots, fracture
rating views, virtual DXA / QCT panels). Each template wraps a set of
`Snapshot_Frame`s and writes the figure to disk.

::: TPTBox.spine.snapshot2D.snapshot_templates
    options:
      show_source: true
      filters: ["!^_"]

## Angles

Cobb angle, lordosis and kyphosis computations between vertebral endplates,
with plotting helpers to overlay the measurements on snapshots.

::: TPTBox.spine.spinestats.angles
    options:
      show_source: true
      filters: ["!^_"]

## Body Quadrants

Partition each vertebral body into 27 anatomically-oriented subregions (a
3×3×3 grid in vertebra-local coordinates). Robust to spinal curvature and
scan orientation because the local axes are derived from muscle-insertion
and median-body POIs.

::: TPTBox.spine.spinestats.body_quadrants
    options:
      show_source: true
      filters: ["!^_"]

## IVD & Vertebra Geometry

Geometric and signal measurements for intervertebral discs and vertebrae:
mesh-based volumes, principal axes, heights, widths, and orientation-aware
diameters (x1–x6). Works on both vertebra and IVD labels — the "up" axis is
read from POIs for vertebrae and estimated via PCA for discs.

::: TPTBox.spine.spinestats.measure_ivd_and_vertebra_geometry
    options:
      show_source: true
      filters: ["!^_"]

## Vertebra Anatomical Widths

Pairwise anatomical distances (mm) between landmark pairs on a vertebra —
e.g. endplate diameters, pedicle widths — stored back onto the `POI` for
downstream statistics.

::: TPTBox.spine.spinestats.vertebra_anatomical_widths
    options:
      show_source: true
      filters: ["!^_"]

## Torso VAT / SAT / Muscle

Body-composition metrics from torso segmentations: visceral / subcutaneous
adipose tissue, muscle mass, and the Vertebral Bone Quality (VBQ) score.
Includes `peak_centered_mean` for robust intensity estimation on noisy
tissue masks.

::: TPTBox.spine.spinestats.torso_vat_sat
    options:
      show_source: true
      filters: ["!^_"]

## IVD POIs

Compute intervertebral disc landmarks (superior / inferior extreme points,
disc centroid) via PCA-based normal estimation and ray casting through the
disc mask.

::: TPTBox.spine.spinestats.poi_fun.ivd_pois
    options:
      show_source: true
      filters: ["!^_"]

## Endplate POIs

Sample points on the superior and inferior endplate surfaces of a vertebra
using ray-triangle intersection against a mesh reconstruction. Useful for
endplate fitting, height measurement, and disc-space analysis.

::: TPTBox.spine.spinestats.poi_fun.endplates
    options:
      show_source: true
      filters: ["!^_"]

## Articularis Midpoint POIs

Detect facet joint (processus articularis) contact regions between adjacent
vertebrae and place a midpoint POI at each joint. Driven by a k-d tree
nearest-neighbor search on the two vertebrae's surface voxels.

::: TPTBox.spine.spinestats.poi_fun.articularis_midpoint
    options:
      show_source: true
      filters: ["!^_"]
