"""Representative TPTBox workload for profiling with fast-anything.

Computes every POI the spine pipeline knows (centroids of all subregions plus every
non-centroid POI registered in ``all_poi_functions``) on the bundled CT and MRI sample
segmentations, ``--repeats`` times each.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TPTBox import calc_poi_from_subreg_vert, to_nii  # noqa: E402
from TPTBox.core.poi_fun.vertebra_pois_non_centroids import all_poi_functions  # noqa: E402

SAMPLES = {
    "ct": (
        REPO_ROOT / "TPTBox/tests/sample_ct/sub-ct_seg-vert_label-22_msk.nii.gz",
        REPO_ROOT / "TPTBox/tests/sample_ct/sub-ct_seg-subreg_label-22_msk.nii.gz",
    ),
    "mri": (
        REPO_ROOT / "TPTBox/tests/sample_mri/sub-mri_seg-vert_label-6_msk.nii.gz",
        REPO_ROOT / "TPTBox/tests/sample_mri/sub-mri_seg-subreg_label-6_msk.nii.gz",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    subreg_ids = list(range(40, 51)) + sorted(all_poi_functions)
    for name, (vert_path, subreg_path) in SAMPLES.items():
        vert, subreg = to_nii(vert_path, True), to_nii(subreg_path, True)
        for _ in range(args.repeats):
            t = time.perf_counter()
            poi = calc_poi_from_subreg_vert(vert, subreg, subreg_id=subreg_ids, verbose=False)
            print(f"{name}: {len(poi)} points in {time.perf_counter() - t:.2f}s", flush=True)


if __name__ == "__main__":
    main()
