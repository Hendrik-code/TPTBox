# Mesh 3D (`TPTBox.mesh3D`)

3D surface mesh generation from segmentation NIfTI volumes and rendering of 3D snapshots.
Requires `pyvista` and `vtk` (included in the `dev` extras).

![Snapshot3D example](TPTBox/images/snp3D_example.jpg)

## Key symbols

| Symbol | Module | Description |
|---|---|---|
| `Mesh` | `mesh.py` | Generates a surface mesh from a segmentation label using marching cubes |
| `create_snapshot3D` | `snapshot3D.py` | Renders a 3D snapshot from a list of meshes to a PNG file |
| `LABEL_COLORS` | `mesh_colors.py` | Default colour mapping for anatomical label IDs |
| `label_to_color(label_id)` | `mesh_colors.py` | Look up the RGB colour for a given label |
| `create_html_preview(meshes)` | `html_preview.py` | Generate an interactive HTML file with an embedded 3D viewer |


## Installation

```bash
pip install pyvista vtk
# or via the dev extras:
poetry install --with dev
```


## Example

```python
from TPTBox import NII
from TPTBox.mesh3D.mesh import Mesh
from TPTBox.mesh3D.snapshot3D import create_snapshot3D

seg = NII.load("seg.nii.gz", seg=True)

# Build meshes for all labels and render
meshes = [Mesh(seg, label=lbl) for lbl in seg.unique_labels()]
create_snapshot3D(meshes, to="snapshot3D.png")
```

More extensive:
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
