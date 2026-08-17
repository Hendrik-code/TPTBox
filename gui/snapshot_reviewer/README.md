# Snapshot Reviewer GUI

A PyQt6 desktop app for triaging point-of-interest snapshots produced from a
BIDS-organised medical-imaging dataset. Snapshots are grouped by subject,
tagged with a verdict (Good / Bad / Allowed Missing / …), and can be opened
directly in 3D Slicer for fixes.

Two files:

- `review_gui.py` — main window, review queue, verdict log, keyboard shortcuts,
  LRU image cache with background prefetch, SQLite-backed review log.
- `slicer_launcher.py` — resolves a snapshot's BIDS family, writes a Slicer
  startup script, launches Slicer, and watches touched files so a saved
  markup can trigger a re-processing hook.

## What the reviewer expects

The reviewer scans a folder under the dataset root (default:
`derivatives-VIBESeg-12-points-snp`, override with `--parent-dir`) for
`*_snp.jpg` and `*_snp.png` files. Each snapshot filename should contain the
usual BIDS entities (`sub-…`, `ses-…`, `sequ-…`) plus two configurable fields:

- **category key** (default `seg-…`) — used for the top-level "category"
  filter combo.
- **region key** (default `desc-…`) — the finer split shown in the second
  combo; falls back to the category if absent.

Both key names are editable at runtime via **⚙ Filter Keys…** and persisted
per-dataset in the SQLite settings table.

## Installation

The reviewer only needs PyQt6 on top of the core TPTBox install:

```bash
pip install poetry
poetry install --with dev            # from the repo root
pip install PyQt6                    # only extra runtime dependency
```

3D Slicer is optional; without it the review workflow still works — only the
"Open in Slicer" button is disabled.  Slicer download: <https://download.slicer.org>.

## Running

```bash
python gui/snapshot_reviewer/review_gui.py /path/to/BIDS_dataset
```

All CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `dataset` (positional) | — | Path to the BIDS dataset root |
| `--name` | `review_log` | Log file stem (SQLite + JSON backup) |
| `-p`, `--parent-dir` | `derivatives-VIBESeg-12-points-snp` | Folder scanned for `*_snp.jpg`/`*.png` |
| `--slicer-exe` | `$TPTBOX_SLICER_EXE` or `which Slicer` | Path to the 3D Slicer executable |
| `--derivatives` | see settings | Comma-separated derivatives folders to include in the BIDS scan |
| `--prefetch` | `20` | Number of upcoming snapshots to warm in the background |
| `--cache-size` | `250` | LRU cache of decoded pixmaps |
| `--buttons-per-row` | `4` | Layout of the action-button grid |

All of the above (except `dataset`, `--name`, `-p`) are also editable at
runtime under **⚙ Settings…** and are persisted per-dataset in the SQLite
`settings` table, so the CLI flags are only needed for the first run or when
you want to override a stored value.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `A` / `←` / `↑` | Previous snapshot |
| `D` / `→` / `↓` | Next snapshot |
| `S` | Skip (same as Next) |
| `G` | ✓ Good (upgrades a *Processed* entry to *Final Confirmed*) |
| `B` | ✗ Bad (points only) |
| `R` | ✗ Bad (segmentation) |
| `I` | ✗ Image issue (implant etc) |
| `M` / `F` | ~ Some points outside FOV |
| `P` | ✂ Remove auto-detected issues |
| `E` | ✂ Remove all points (FOV) |
| `L` | Open current snapshot in 3D Slicer |
| `V` | Show / hide verdict log |
| `Del` | Delete snapshot file from disk and advance |

## Data model

- Verdicts live in `<parent-dir>/.<name>.snp-review.sqlite` (WAL mode; primary
  store, crash-safe).
- A human-readable JSON snapshot is written to
  `<parent-dir>/.<name>.snp-review.json` roughly once a minute as a backup.
- The `settings` table in the same SQLite file stores GUI preferences (filter
  keys, Slicer path, derivatives search list, prefetch/cache size, …).

## Creating the snapshots the reviewer consumes

The reviewer only *reads* JPG/PNG snapshots — it does not generate them.  Two
in-repo modules are the usual sources:

### 2D snapshots — `TPTBox.spine.snapshot2D`

Renders CT/MRI slices with optional segmentation overlays and POI markers.

```python
from pathlib import Path
from TPTBox import calc_poi_from_subreg_vert
from TPTBox.spine.snapshot2D import Snapshot_Frame, Visualization_Type, create_snapshot

root = Path("/path/to/sub-001/ses-01")
img = root / "ct.nii.gz"
vert = root / "seg-vert_msk.nii.gz"
subreg = root / "seg-spine_msk.nii.gz"
poi = calc_poi_from_subreg_vert(vert, subreg)

# Write under the folder the reviewer scans; filename ends with _snp.jpg
out = root / "derivatives-VIBESeg-12-points-snp/sub-001_ses-01_sequ-01_seg-vert_desc-review_snp.jpg"
out.parent.mkdir(parents=True, exist_ok=True)

create_snapshot(
    out,
    [
        Snapshot_Frame(img, vert, poi, sagittal=True, coronal=True, mode="CT"),
        Snapshot_Frame(img, subreg, poi, sagittal=True, coronal=True, axial=True, mode="CTs", axial_heights=[0.20, 0.4, 0.6, 0.8]),
    ],
)
```

Full options are documented in `TPTBox/spine/snapshot2D/README.md`.

### 3D snapshots — `TPTBox.mesh3D`

Marching-cubes surface meshes with `pyvista`/`vtk`.  Best rendered with the
parallel helper so a batch of orientations runs on multiple cores:

```python
from pathlib import Path
from TPTBox.core.vert_constants import Full_Body_Instance
from TPTBox.mesh3D.snapshot3D import make_snapshot3D_parallel

root = Path("/path/to/sub-001/ses-01")
seg = root / "seg-VIBESeg-11-lr_msk.nii.gz"
out = root / "derivatives-VIBESeg-12-points-snp/sub-001_ses-01_sequ-01_seg-body_desc-3d_snp.jpg"
out.parent.mkdir(parents=True, exist_ok=True)

make_snapshot3D_parallel(
    [seg],
    [out],
    view=["A", "R", "P", "L"],
    ids_list=[
        [a.value for a in Full_Body_Instance.bone()],
        [a.value for a in Full_Body_Instance.organs()],
    ],
)
```

Full options are documented in `TPTBox/mesh3D/README.md`.

### Where the reviewer looks for these files

The scanner in `scan_snapshots()` walks `<dataset>/<parent-dir>/` recursively
and picks up `*_snp.jpg` and `*_snp.png`.  Any BIDS-style filename works — the
reviewer parses `sub`/`ses`/`sequ` for the header label and the configured
category/region keys for the filters.

## Opening a snapshot in 3D Slicer

Pressing **L** (or clicking **Slicer**) opens a dialog listing every image,
mask and `.mrk.json` markup file that shares BIDS entities with the current
snapshot.  Selected files are handed to Slicer through a generated startup
script.  Files opened this way are watched for modification, so saving an
edited markup in Slicer triggers the review window's callback (defaulting to
merging the new POI back into the source `.json` and invalidating the derived
`msk_seg-treg` output).

Configure the Slicer executable via one of:

- `--slicer-exe /path/to/Slicer`
- `TPTBOX_SLICER_EXE=/path/to/Slicer` environment variable
- **⚙ Settings…** dialog inside the app
- `Slicer` on `$PATH`

## Troubleshooting

- **"No snapshots to review"** — the scanned folder is empty or `--parent-dir`
  points somewhere else.  Regenerate the snapshots (see above) or pass the
  correct folder.
- **The filter combos are empty** — the snapshot filenames don't contain the
  configured BIDS field.  Open **⚙ Filter Keys…** and set them to whichever
  entities you actually use (e.g. `category=seg`, `region=desc`).
- **"Slicer not found"** — configure the executable via **⚙ Settings…**, the
  CLI flag, or `TPTBOX_SLICER_EXE`.
- **Verdict log tabs are empty after picking a category** — you're on the
  fixed build; the pre-fix version had a bug that hid all log entries when a
  specific category was selected.
