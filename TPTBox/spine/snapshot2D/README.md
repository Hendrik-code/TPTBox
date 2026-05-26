# 2D Snapshots (`spine/snapshot2D`)

Modular 2D image generation for NIfTI data.  Supports axial/sagittal/coronal slices,
maximum intensity projections (MIPs), and segmentation overlays.

## Key symbols

| Symbol | Module | Description |
|---|---|---|
| `create_snapshot` | `snapshot_modular.py` | Main entry point — renders a list of `Snapshot_Frame` objects to a PNG |
| `Snapshot_Frame` | `snapshot_modular.py` | Configuration for one image panel (image, overlay, view direction, …) |
| `Plane` | `snapshot_modular.py` | Enum: `Plane.axial`, `Plane.sagittal`, `Plane.coronal` |
| `to_image_nii` | `snapshot_modular.py` | Convert a NIfTI slice to a matplotlib-ready RGB array |
| Pre-built templates | `snapshot_templates.py` | Ready-to-use snapshot configurations for common spine workflows |

## Example

```python
from TPTBox.spine.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot, Plane

frames = [
    Snapshot_Frame(image=ct, segmentation=seg, mode="CT", plane=Plane.sagittal),
    Snapshot_Frame(image=ct, mode="CT", plane=Plane.axial),
]
create_snapshot(frames, to="output.png")
```
