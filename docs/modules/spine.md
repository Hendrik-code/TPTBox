# Spine (`TPTBox.spine`)

Spine-specific utilities built on top of the core `NII` and `POI` abstractions.
Contains two sub-modules: 2D snapshot generation and statistical spine measurements.

## Sub-modules

| Sub-module | Description |
|---|---|
| [`snapshot2D/`](snapshot2d.md) | Modular 2D image snapshot generation (slices, MIPs, overlays) |
| [`spinestats/`](spinestats.md) | Clinical measurements: distances, Cobb angles, IVD POIs, endplates |

## Quick Example

```python
from TPTBox import NII, calc_centroids
from TPTBox.spine.snapshot2D.snapshot_modular import Snapshot_Frame, create_snapshot

ct = NII.load("ct.nii.gz", seg=False)
seg = NII.load("seg.nii.gz", seg=True)

# Generate a 2D sagittal snapshot with a segmentation overlay
create_snapshot(
    [Snapshot_Frame(image=ct, segmentation=seg, mode="CT")],
    to="snapshot.png",
)
```
