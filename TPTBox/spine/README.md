# Spine (`TPTBox.spine`)

Spine-specific utilities built on top of the core `NII` and `POI` abstractions.
Contains two sub-modules: 2D snapshot generation and statistical spine measurements.

## Sub-modules

| Sub-module | Description |
|---|---|
| [`snapshot2D/`](https://github.com/Hendrik-code/TPTBox/blob/main/TPTBox/spine/snapshot2D/README.md) | Modular 2D image snapshot generation (slices, MIPs, overlays) |
| [`spinestats/`](https://github.com/Hendrik-code/TPTBox/blob/main/TPTBox/spine/spinestats/README.md) | Clinical measurements: distances, Cobb angles, IVD POIs, endplates |

## Quick Example

```python
from TPTBox import NII
from TPTBox.spine.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot

ct = NII.load("ct.nii.gz", seg=False)
seg = NII.load("seg.nii.gz", seg=True)

# Generate a 2D sagittal snapshot with a segmentation overlay
create_snapshot(
    "snapshot.png",
    [Snapshot_Frame(image=ct, segmentation=seg, mode="CT")],
)
```
