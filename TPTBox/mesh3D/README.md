# Mesh 3D (`TPTBox.mesh3D`)

3D surface mesh generation from segmentation NIfTI volumes and rendering of 3D snapshots.
Requires `pyvista`, `vtk`, `scikit-image` and (for `snapshot3D`) `fury`, `Pillow`
and `xvfbwrapper` (all included in the `dev` extras).

![Snapshot3D example](TPTBox/images/snp3D_example.jpg)

## Key symbols

| Symbol | Module | Description |
|---|---|---|
| `Mesh3D` | `mesh.py` | Thin wrapper around a `pyvista.PolyData` mesh with save / load / display helpers |
| `SegmentationMesh` | `mesh.py` | Generates a surface mesh from a segmentation array or `NII` via marching cubes |
| `POIMesh` | `mesh.py` | Glyph mesh (spheres) built from a `POI` container |
| `make_snapshot3D` | `snapshot3D.py` | Render a segmentation as one or more 3D views to a PNG file |
| `make_snapshot3D_parallel` | `snapshot3D.py` | Run `make_snapshot3D` in a process pool over many images |
| `Mesh_Color_List` | `mesh_colors.py` | Catalog of named `RGB_Color` constants (ITK palette) |
| `get_color_by_label(label)` | `mesh_colors.py` | Look up the `RGB_Color` for an integer label |
| `write_ctbl(path)` | `mesh_colors.py` | Export the palette as a 3D Slicer `.ctbl` color-table file |
| `make_html_preview(images, html_out)` | `html_preview.py` | Render `NII`/`POI` objects to an interactive HTML file |
| `Preview_Settings` | `html_preview.py` | Per-object visualization settings (color, opacity, offset) for `make_html_preview` |


## Installation

```bash
pip install pyvista vtk scikit-image fury Pillow xvfbwrapper
# or via the dev extras:
poetry install --with dev
```


## Examples

### Building meshes and saving them

```python
from TPTBox import NII
from TPTBox.mesh3D.mesh import SegmentationMesh

seg = NII.load("seg.nii.gz", seg=True)

# Build a single surface mesh over all labels
mesh = SegmentationMesh.from_segmentation_nii(seg)
mesh.save("seg.ply")
```

### Rendering a 3D snapshot

```python
from TPTBox.mesh3D.snapshot3D import make_snapshot3D

make_snapshot3D("seg.nii.gz", "snapshot3D.png", view=["A", "L"])
```

### Parallel snapshots across body systems

```python
from TPTBox.core.vert_constants import Full_Body_Instance
from TPTBox.mesh3D.snapshot3D import make_snapshot3D_parallel

path = "/path/to/folder"
seg = path / "seg-VIBESeg-11-lr_msk.nii.gz"
out_path = path / "snp3D.jpg"
out_path2 = path / "snp3D_2.jpg"
# We recommend using the parallel application of this function
# because it takes a minute, but does not need a lot of resources.
make_snapshot3D_parallel(
    [seg],
    [out_path],
    view=["A"],
    ids_list=[
        [a.value for a in Full_Body_Instance.bone()],
        [a.value for a in Full_Body_Instance.lung_system()],
        [a.value for a in Full_Body_Instance.organs()],
        [a.value for a in Full_Body_Instance.digestion()],
        [a.value for a in Full_Body_Instance.vessels()],
        [a.value for a in Full_Body_Instance.full_spine()],
        [a.value for a in Full_Body_Instance.muscle()],
        [a.value for a in Full_Body_Instance.body_comp()],
    ],
)
make_snapshot3D_parallel(
    [seg],
    [out_path2],
    view=["A", "R", "P", "L"],
    ids_list=[
        [a.value for a in Full_Body_Instance.bone()],
    ],
)
```

### Interactive HTML preview

```python
from TPTBox import NII, POI
from TPTBox.mesh3D.html_preview import Preview_Settings, make_html_preview

seg = NII.load("seg.nii.gz", seg=True)
poi = POI.load("poi.json", reference=seg)
make_html_preview(
    [seg, Preview_Settings(poi, opacity=1.0)],
    "preview.html",
    poi_size=2.0,
)
```
