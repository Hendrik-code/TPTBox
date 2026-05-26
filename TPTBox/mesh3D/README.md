# Mesh 3D (`TPTBox.mesh3D`)

3D surface mesh generation from segmentation NIfTI volumes and rendering of 3D snapshots.
Requires `pyvista` and `vtk` (included in the `dev` extras).

## Key symbols

| Symbol | Module | Description |
|---|---|---|
| `Mesh` | `mesh.py` | Generates a surface mesh from a segmentation label using marching cubes |
| `create_snapshot3D` | `snapshot3D.py` | Renders a 3D snapshot from a list of meshes to a PNG file |
| `LABEL_COLORS` | `mesh_colors.py` | Default colour mapping for anatomical label IDs |
| `label_to_color(label_id)` | `mesh_colors.py` | Look up the RGB colour for a given label |
| `create_html_preview(meshes)` | `html_preview.py` | Generate an interactive HTML file with an embedded 3D viewer |

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

## Installation

```bash
pip install pyvista vtk
# or via the dev extras:
poetry install --with dev
```
