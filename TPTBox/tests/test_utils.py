import os
import sys
from pathlib import Path

if not os.path.isdir("test"):  # noqa: PTH112
    sys.path.append("..")
file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
import random  # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

import TPTBox.core.bids_files as bids  # noqa: E402
from TPTBox import Centroids  # noqa: E402
from TPTBox.core.nii_wrapper import NII  # noqa: E402
from TPTBox.core.poi import POI  # noqa: E402
from TPTBox.core.vert_constants import AX_CODES  # noqa: E402

repeats = 20


def get_tests_dir():
    return Path(__file__).parent


def get_test_ct() -> tuple[NII, NII, NII, int]:
    tests_path = get_tests_dir()
    ct_path = tests_path.joinpath("sample_ct")
    ct = NII.load(ct_path.joinpath("sub-ct_label-22_ct.nii.gz"), seg=False)
    subreg = NII.load(ct_path.joinpath("sub-ct_seg-subreg_label-22_msk.nii.gz"), seg=True)
    vert = NII.load(ct_path.joinpath("sub-ct_seg-vert_label-22_msk.nii.gz"), seg=True)
    return ct, subreg, vert, 22


def get_test_mri() -> tuple[NII, NII, NII, int]:
    tests_path = get_tests_dir()
    mri_path = tests_path.joinpath("sample_mri")
    mri = NII.load(mri_path.joinpath("sub-mri_label-6_T2w.nii.gz"), seg=False)
    subreg = NII.load(mri_path.joinpath("sub-mri_seg-subreg_label-6_msk.nii.gz"), seg=True)
    vert = NII.load(mri_path.joinpath("sub-mri_seg-vert_label-6_msk.nii.gz"), seg=True)
    return mri, subreg, vert, 6


def get_BIDS_test():
    ds = ["/media/robert/Expansion/dataset-Testset"]
    if not Path(ds[0]).exists():
        ds = []
    bids_global = bids.BIDS_Global_info(
        datasets=ds,
        parents=["sourcedata", "rawdata", "rawdata_ct", "rawdata_dixon", "derivatives"],
        additional_key=["sequ", "seg", "ovl", "e"],
    )
    if len(ds) == 0:
        for i in a:
            bids_global.add_file_2_subject(Path(i), "")
    return bids_global


def overlap(
    c1: tuple[float, float, float],
    w1: tuple[float, float, float],
    c2: tuple[float, float, float],
    w2: tuple[float, float, float],
):
    return all(sqr1d(a, b, c, d) for a, b, c, d in zip(c1, w1, c2, w2, strict=False))


def extract_affine(nii: NII):
    return {"origin": nii.origin, "shape": nii.shape, "rotation": nii.rotation}


def sqr1d(c1: float, w1: float, c2: float, w2: float):
    if (c1 + w1) < (c2 - w2):
        # print("case1", c1 + w1, "<", (c2 - w2))
        return False
    return not c1 - w1 > c2 + w2


def get_random_ax_code() -> AX_CODES:
    directions = [["R", "L"], ["S", "I"], ["A", "P"]]
    idx = [0, 1, 2]
    random.shuffle(idx)
    return tuple(directions[i][random.randint(0, 1)] for i in idx)  # type: ignore


def get_POI(x: tuple[int, int, int] = (50, 30, 40), num_point=3):
    out_points: dict[tuple[int, int], tuple[float, float, float]] = {}

    for idx in range(num_point):
        point = tuple(random.randint(1, a * 100) / 100.0 for a in x)
        out_points[idx + 1, 50] = point
    return POI(out_points, orientation=("R", "A", "S"), zoom=(1, 1, 1))


def get_poi(x: tuple[int, int, int] = (50, 30, 40), num_vert=3, num_subreg=1, rotation=True, min_subreg=1, max_subreg=255):
    out_points: dict[int, dict[int, tuple[float, float, float]]] = {}

    for idx in range(num_vert):
        out_points[idx + 1] = {}
        for _ in range(num_subreg):
            point = tuple(random.randint(1, a * 100) / 100.0 for a in x)
            subregion = random.randint(min_subreg, max_subreg)
            out_points[idx + 1][subregion] = point
    origin = tuple(random.randint(1, 100) for _ in range(3))
    if rotation:
        from scipy.spatial.transform import Rotation

        m = 30
        r = Rotation.from_euler("xyz", (random.randint(-m, m), random.randint(-m, m), random.randint(-m, m)), degrees=True)
        r = np.round(r.as_matrix(), decimals=5)
    else:
        r = np.eye(3)
    return POI(out_points, orientation=("R", "A", "S"), zoom=(1, 1, 1), shape=x, origin=origin, rotation=r)


def get_nii(x: tuple[int, int, int] | None = None, num_point=3, min_size: int = 1):  # type: ignore
    if x is None:
        x = (random.randint(30, 70), random.randint(30, 70), random.randint(30, 70))
    a = np.zeros(x, dtype=np.uint16)
    points = []
    out_points: dict[tuple[int, int], tuple[float, float, float]] = {}
    sizes = []
    idx = 1
    while True:
        if num_point == len(points):
            break
        point = tuple(random.randint(1, a - 1) for a in x)
        size = tuple(random.randint(min_size, min_size + a) for a in [5, 5, 5])
        if any(a - b < 0 for a, b in zip(point, size, strict=False)):
            continue
        if any(a + b > c - 1 for a, b, c in zip(point, size, x, strict=False)):
            continue
        skip = False
        for p2, s2 in zip(points, sizes, strict=False):
            if overlap(point, size, p2, s2):
                skip = True
                break
        if skip:
            continue
        a[
            point[0] - size[0] : point[0] + size[0] + 1,
            point[1] - size[1] : point[1] + size[1] + 1,
            point[2] - size[2] : point[2] + size[2] + 1,
        ] = idx

        points.append(point)
        sizes.append(size)
        out_points[(idx, 50)] = tuple(float(a) for a in point)

        idx += 1
    aff = np.eye(4)

    aff[0, 3] = random.randint(-100, 100)
    aff[1, 3] = random.randint(-100, 100)
    aff[2, 3] = random.randint(-100, 100)
    return NII(nib.nifti1.Nifti1Image(a, aff), seg=True), out_points, ("R", "A", "S"), sizes


a = [
    "sub-spinegan0026_ses-409_sequ-203_seg-subreg_ctd.json",
    "sub-spinegan0026_ses-409_sequ-203_seg-subreg_msk.nii.gz",
    "sub-spinegan0026_ses-409_sequ-203_seg-vert_msk.nii.gz",
    "sub-spinegan0026_ses-409_sequ-203_snp.png",
    "sub-spinegan0026_ses-411_sequ-204_seg-subreg_ctd.json",
    "sub-spinegan0026_ses-411_sequ-204_seg-subreg_msk.nii.gz",
    "sub-spinegan0026_ses-411_sequ-204_seg-vert_msk.nii.gz",
    "sub-spinegan0026_ses-411_sequ-204_snp.png",
    "sub-spinegan0026_ses-411_sequ-301_e-3_seg-subreg_ctd.json",
    "sub-spinegan0026_ses-411_sequ-302_e-2_seg-subreg_ctd.json",
    "sub-spinegan0026_ses-411_sequ-303_e-2_seg-subreg_ctd.json",
    "sub-spinegan0026_ses-411_sequ-305_seg-subreg_ctd.json",
    "sub-spinegan0026_ses-411_sequ-305_seg-subreg_msk.nii.gz",
    "sub-spinegan0026_ses-411_sequ-305_seg-vert_msk.nii.gz",
    "sub-spinegan0026_ses-411_sequ-305_snp.png",
    "sub-spinegan0042_ses-417_sequ-301_e-1_seg-subreg_ctd.json",
    "sub-spinegan0042_ses-417_sequ-302_e-1_seg-subreg_ctd.json",
    "sub-spinegan0042_ses-417_sequ-303_e-3_seg-subreg_ctd.json",
    "sub-spinegan0042_ses-417_sequ-406_seg-subreg_ctd.json",
    "sub-spinegan0042_ses-417_sequ-406_seg-subreg_msk.nii.gz",
    "sub-spinegan0042_ses-417_sequ-406_seg-vert_msk.nii.gz",
    "sub-spinegan0042_ses-417_sequ-406_snp.png",
    "sub-spinegan0042_ses-417_sequ-None_seg-subreg_ctd.json",
    "sub-spinegan0042_ses-417_sequ-None_seg-subreg_msk.nii.gz",
    "sub-spinegan0042_ses-417_sequ-None_seg-vert_msk.nii.gz",
    "sub-spinegan0042_ses-417_sequ-None_snp.png",
    "sub-spinegan0026_ses-409_sequ-203_ct.json",
    "sub-spinegan0026_ses-409_sequ-203_ct.nii.gz",
    "sub-spinegan0026_ses-411_sequ-204_ct.json",
    "sub-spinegan0026_ses-411_sequ-204_ct.nii.gz",
    "sub-spinegan0026_ses-411_sequ-305_ct.json",
    "sub-spinegan0026_ses-411_sequ-305_ct.nii.gz",
    "sub-spinegan0042_ses-417_sequ-406_ct.json",
    "sub-spinegan0042_ses-417_sequ-406_ct.nii.gz",
    "sub-spinegan0042_ses-417_sequ-None_ct.json",
    "sub-spinegan0042_ses-417_sequ-None_ct.nii.gz",
    "sub-spinegan0026_ses-409_sequ-205_ct.json",
    "sub-spinegan0026_ses-409_sequ-205_ct.nii.gz",
    "sub-spinegan0026_ses-409_sequ-205_snp.png",
    "sub-spinegan0026_ses-411_sequ-205_ct.json",
    "sub-spinegan0026_ses-411_sequ-205_ct.nii.gz",
    "sub-spinegan0026_ses-411_sequ-205_snp.png",
    "sub-spinegan0026_ses-411_sequ-301_e-1_dixon.json",
    "sub-spinegan0026_ses-411_sequ-301_e-1_dixon.nii.gz",
    "sub-spinegan0026_ses-411_sequ-301_e-1_ovl-ctd_snp.png",
    "sub-spinegan0026_ses-411_sequ-301_e-2_dixon.json",
    "sub-spinegan0026_ses-411_sequ-301_e-2_dixon.nii.gz",
    "sub-spinegan0026_ses-411_sequ-301_e-2_ovl-ctd_snp.png",
    "sub-spinegan0026_ses-411_sequ-301_e-3_dixon.json",
    "sub-spinegan0026_ses-411_sequ-301_e-3_dixon.nii.gz",
    "sub-spinegan0026_ses-411_sequ-301_e-3_msk.nii.gz",
    "sub-spinegan0026_ses-411_sequ-301_e-3_ovl-ctd_snp.png",
    "sub-spinegan0026_ses-411_sequ-302_e-1_dixon.json",
    "sub-spinegan0026_ses-411_sequ-302_e-1_dixon.nii.gz",
    "sub-spinegan0026_ses-411_sequ-302_e-1_ovl-ctd_snp.png",
    "sub-spinegan0026_ses-411_sequ-302_e-2_dixon.json",
    "sub-spinegan0026_ses-411_sequ-302_e-2_dixon.nii.gz",
    "sub-spinegan0026_ses-411_sequ-302_e-2_msk.nii.gz",
    "sub-spinegan0026_ses-411_sequ-302_e-2_ovl-ctd_snp.png",
    "sub-spinegan0026_ses-411_sequ-302_e-3_dixon.json",
    "sub-spinegan0026_ses-411_sequ-302_e-3_dixon.nii.gz",
    "sub-spinegan0026_ses-411_sequ-302_e-3_ovl-ctd_snp.png",
    "sub-spinegan0026_ses-411_sequ-303_e-1_dixon.json",
    "sub-spinegan0026_ses-411_sequ-303_e-1_dixon.nii.gz",
    "sub-spinegan0026_ses-411_sequ-303_e-1_ovl-ctd_snp.png",
    "sub-spinegan0026_ses-411_sequ-303_e-2_dixon.json",
    "sub-spinegan0026_ses-411_sequ-303_e-2_dixon.nii.gz",
    "sub-spinegan0026_ses-411_sequ-303_e-2_msk.nii.gz",
    "sub-spinegan0026_ses-411_sequ-303_e-2_ovl-ctd_snp.png",
    "sub-spinegan0026_ses-411_sequ-303_e-3_dixon.json",
    "sub-spinegan0026_ses-411_sequ-303_e-3_dixon.nii.gz",
    "sub-spinegan0026_ses-411_sequ-303_e-3_ovl-ctd_snp.png",
    "sub-spinegan0026_ses-411_sequ-305_ct.json",
    "sub-spinegan0026_ses-411_sequ-305_ct.nii.gz",
    "sub-spinegan0026_ses-411_sequ-305_snp.png",
    "sub-spinegan0042_ses-417_sequ-206_ct.json",
    "sub-spinegan0042_ses-417_sequ-206_ct.nii.gz",
    "sub-spinegan0042_ses-417_sequ-206_snp.png",
    "sub-spinegan0042_ses-417_sequ-301_e-1_dixon.json",
    "sub-spinegan0042_ses-417_sequ-301_e-1_dixon.nii.gz",
    "sub-spinegan0042_ses-417_sequ-301_e-1_msk.nii.gz",
    "sub-spinegan0042_ses-417_sequ-301_e-1_ovl-ctd_snp.png",
    "sub-spinegan0042_ses-417_sequ-301_e-2_dixon.json",
    "sub-spinegan0042_ses-417_sequ-301_e-2_dixon.nii.gz",
    "sub-spinegan0042_ses-417_sequ-301_e-2_ovl-ctd_snp.png",
    "sub-spinegan0042_ses-417_sequ-301_e-3_dixon.json",
    "sub-spinegan0042_ses-417_sequ-301_e-3_dixon.nii.gz",
    "sub-spinegan0042_ses-417_sequ-301_e-3_ovl-ctd_snp.png",
    "sub-spinegan0042_ses-417_sequ-302_e-1_dixon.json",
    "sub-spinegan0042_ses-417_sequ-302_e-1_dixon.nii.gz",
    "sub-spinegan0042_ses-417_sequ-302_e-1_msk.nii.gz",
    "sub-spinegan0042_ses-417_sequ-302_e-1_ovl-ctd_snp.png",
    "sub-spinegan0042_ses-417_sequ-302_e-2_dixon.json",
    "sub-spinegan0042_ses-417_sequ-302_e-2_dixon.nii.gz",
    "sub-spinegan0042_ses-417_sequ-302_e-2_ovl-ctd_snp.png",
    "sub-spinegan0042_ses-417_sequ-302_e-3_dixon.json",
    "sub-spinegan0042_ses-417_sequ-302_e-3_dixon.nii.gz",
    "sub-spinegan0042_ses-417_sequ-302_e-3_ovl-ctd_snp.png",
    "sub-spinegan0042_ses-417_sequ-303_e-1_dixon.json",
    "sub-spinegan0042_ses-417_sequ-303_e-1_dixon.nii.gz",
    "sub-spinegan0042_ses-417_sequ-303_e-1_ovl-ctd_snp.png",
    "sub-spinegan0042_ses-417_sequ-303_e-2_dixon.json",
    "sub-spinegan0042_ses-417_sequ-303_e-2_dixon.nii.gz",
    "sub-spinegan0042_ses-417_sequ-303_e-2_ovl-ctd_snp.png",
    "sub-spinegan0042_ses-417_sequ-303_e-3_dixon.json",
    "sub-spinegan0042_ses-417_sequ-303_e-3_dixon.nii.gz",
    "sub-spinegan0042_ses-417_sequ-303_e-3_msk.nii.gz",
    "sub-spinegan0042_ses-417_sequ-303_e-3_ovl-ctd_snp.png",
    "sub-spinegan0042_ses-417_sequ-406_ct.json",
    "sub-spinegan0042_ses-417_sequ-406_ct.nii.gz",
    "sub-spinegan0042_ses-417_sequ-406_snp.png",
]
