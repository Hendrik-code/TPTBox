# Mesh 3D

Build and visualize 3-D surface meshes derived from segmentation volumes and
Points of Interest.

The module is split into four cooperating pieces:

- [`mesh`][TPTBox.mesh3D.mesh] — extracts iso-surfaces from a segmentation
  `NII` via marching cubes and wraps them in a `Mesh3D`/`SegmentationMesh`
  container backed by a `pyvista.PolyData` object; supports save/load in PLY
  format.
- [`snapshot3D`][TPTBox.mesh3D.snapshot3D] — renders a segmentation as one or
  more 2-D PNG previews (`R`/`A`/`L`/`P`/`S`/`I` viewpoints) using an off-screen
  VTK/Fury pipeline; useful in headless environments via `Xvfb`.
- [`html_preview`][TPTBox.mesh3D.html_preview] — assembles interactive HTML
  previews of `NII`/`POI` objects for quick visual inspection.
- [`mesh_colors`][TPTBox.mesh3D.mesh_colors] — colour palette utilities keyed by
  vertebra/subregion label so meshes render with a consistent scheme.

## Snapshot 3D

::: TPTBox.mesh3D.snapshot3D
    options:
      show_source: true
      filters: ["!^_"]

## Mesh

::: TPTBox.mesh3D.mesh
    options:
      show_source: true
      filters: ["!^_"]

## Mesh Colors

::: TPTBox.mesh3D.mesh_colors
    options:
      show_source: true
      filters: ["!^_"]

## HTML Preview

::: TPTBox.mesh3D.html_preview
    options:
      show_source: true
      filters: ["!^_"]
