"""Segmentation post-processing workload for profiling the array primitives with fast-anything.

The POI workload (``workload.py``) spends under 2% of its time in ``np.unique``, connected
components, distance transforms, morphology and resampling, so it cannot rank them. This one runs
the NII operations built on them on the bundled CT and MRI samples, ``--repeats`` times each:

- whole volume: label lists, 1 mm and 0.75 mm rescales (segmentation order 0, image order 1),
  resampling the segmentation onto the image grid, erosion and dilation with connectivity 1 and 3,
  connected components, Euclidean dilation and erosion, a Voronoi partition;
- per vertebra, on its bounding box: the same morphology, connected-component and distance-transform
  steps a cleanup pass runs label by label.

``--only cc`` (or any of ``STEPS``) runs a subset. fast-anything ranks only the 40 slowest functions
of a run, and connected components (cc3d) are too fast to make that list in the full run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TPTBox import NII, to_nii  # noqa: E402

SAMPLES = {
    "ct": (
        REPO_ROOT / "TPTBox/tests/sample_ct/sub-ct_label-22_ct.nii.gz",
        REPO_ROOT / "TPTBox/tests/sample_ct/sub-ct_seg-vert_label-22_msk.nii.gz",
        REPO_ROOT / "TPTBox/tests/sample_ct/sub-ct_seg-subreg_label-22_msk.nii.gz",
    ),
    "mri": (
        REPO_ROOT / "TPTBox/tests/sample_mri/sub-mri_label-6_T2w.nii.gz",
        REPO_ROOT / "TPTBox/tests/sample_mri/sub-mri_seg-vert_label-6_msk.nii.gz",
        REPO_ROOT / "TPTBox/tests/sample_mri/sub-mri_seg-subreg_label-6_msk.nii.gz",
    ),
}


STEPS = ("unique", "resample", "erode", "dilate", "cc", "edt")


def whole_volume(img: NII, vert: NII, subreg: NII, steps: set[str]) -> None:
    if "unique" in steps:
        vert.unique()
        subreg.unique()
    if "resample" in steps:
        for spacing in ((1, 1, 1), (0.75, 0.75, 0.75)):
            vert.rescale(spacing)
            img_iso = img.rescale(spacing, order=1)
            subreg.resample_from_to(img_iso, verbose=False)
    for connectivity in (1, 3):
        if "erode" in steps:
            subreg.erode_msk(1, connectivity=connectivity, verbose=False)
        if "dilate" in steps:
            subreg.dilate_msk(1, connectivity=connectivity, verbose=False)
    if "cc" in steps:
        vert.get_connected_components()
        vert.filter_connected_components(max_count_component=1, keep_label=True)
    if "edt" in steps:
        subreg.dilate_msk_euclid(2, verbose=False)
        subreg.erode_msk_euclid(1, verbose=False)
        vert.voronoi_labels(max_distance=6, verbose=False)


def per_vertebra(vert: NII, subreg: NII, steps: set[str]) -> None:
    for label in vert.unique():
        if label == 0:
            continue
        crop = vert.extract_label(label).compute_crop(dist=4)
        region = subreg.apply_crop(crop) * vert.extract_label(label).apply_crop(crop)
        if "unique" in steps:
            region.unique()
        if "erode" in steps:
            region.erode_msk(2, connectivity=1, verbose=False)
        if "dilate" in steps:
            region.dilate_msk(2, connectivity=3, verbose=False)
        if "cc" in steps:
            region.get_connected_components()
            region.filter_connected_components(max_count_component=1, keep_label=True)
        if "edt" in steps:
            region.dilate_msk_euclid(2, verbose=False)
            region.erode_msk_euclid(1, verbose=False)
            region.voronoi_labels(max_distance=3, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", nargs="+", choices=STEPS, default=STEPS, help="run only these steps")
    args = parser.parse_args()
    steps = set(args.only)
    for name, paths in SAMPLES.items():
        img, vert, subreg = to_nii(paths[0], False), to_nii(paths[1], True), to_nii(paths[2], True)
        for _ in range(args.repeats):
            t = time.perf_counter()
            whole_volume(img, vert, subreg, steps)
            per_vertebra(vert, subreg, steps)
            print(f"{name}: {time.perf_counter() - t:.2f}s", flush=True)


if __name__ == "__main__":
    main()
