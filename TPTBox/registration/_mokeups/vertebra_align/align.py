from __future__ import annotations
import math
import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
from TPTBox.core.bids_files import BIDS_FILE
import numpy as np
from nii_utils import calc_centroids, centroids_to_dict_list
import nibabel as nib

if __name__ == "__main__":
    if False:
        a: nib.Nifti1Image = nib.load(
            "/media/data/dataset-spinegan/register_test/derivatives/spinegan0010/ses-20220426/sub-spinegan0010_ses-20220426_sequ-201_seg-subreg_msk.nii.gz"
        )
        a = a.slicer[100:500, 50:400, 400:800]
        nib.save(a, "vertebra_align/subreg.nii.gz")
        a: nib.Nifti1Image = nib.load(
            "/media/data/dataset-spinegan/register_test/derivatives/spinegan0010/ses-20220426/sub-spinegan0010_ses-20220426_sequ-201_seg-vert_msk.nii.gz",
        )
        a = a.slicer[100:500, 50:400, 400:800]
        nib.save(a, "vertebra_align/vert.nii.gz")
        a: nib.Nifti1Image = nib.load(
            "/media/data/dataset-spinegan/register_test/rawdata_ct/spinegan0010/ses-20220426/sub-spinegan0010_ses-20220426_sequ-201_ct.nii.gz",
        )
        a = a.slicer[100:500, 50:400, 400:800]
        nib.save(a, "vertebra_align/ct.nii.gz")
    subreg_nib: nib.Nifti1Image = nib.load("vertebra_align/subreg.nii.gz")
    vert_nib = nib.load("vertebra_align/vert.nii.gz")
    subreg_arr1 = subreg_nib.get_fdata()
    vert_arr1 = vert_nib.get_fdata()
    arr = np.zeros_like(subreg_arr1)

    for vert_id in np.unique(vert_arr1):
        # vert_id = 17
        print(vert_id)
        subreg_arr = subreg_arr1.copy()
        vert_arr = vert_arr1.copy()
        vert_arr[vert_arr != vert_id] = 0
        vert_arr[vert_arr == vert_id] = 1
        subreg_arr *= vert_arr

        v = nib.Nifti1Image(subreg_arr, subreg_nib.affine)
        # if len(np.unique(subreg_arr) < 2):
        #    continue
        try:
            # print(np.unique(subreg_arr))
            centroids = calc_centroids(v)
            # print(centroids)
            import raster_geometry as rg
            from tqdm import tqdm

            dc = {a[0]: a[1:] for a in centroids[1:]}
            if 50 not in dc:
                continue
            # print(dc)
            # 50 - main body
            # 44/43 oben
            # 48/47 unten
            # 42 Dornenfortsatz
            # 41 Umschluss

            print(f"Roll: {dc[44][2]-dc[43][2]} {dc[48][2]-dc[47][2]} {(dc[44][2]-dc[43][2])/(dc[48][2]-dc[47][2]):.3f}")

            def yan(a, b, c=None):
                if c is None:
                    dx = dc[a][0] - dc[b][0]
                    dy = dc[a][1] - dc[b][1]
                    out = math.atan(abs(dx / dy))
                else:
                    x2 = 0.5 * (dc[b][0] + dc[c][0])
                    y2 = 0.5 * (dc[b][1] + dc[c][1])
                    dx = dc[a][0] - x2
                    dy = dc[a][1] - y2
                    out = math.atan(abs(dx / dy))
                return round(out / math.pi * 90, 3)

            print(f"Yan: {yan(50,44,43)}, {yan(50,48,47)} ,{yan(50,41)}, {yan(50,42)}")
            # finde open
            subreg_arr_top = subreg_arr
            # 49
            x0, y0, z0 = dc[50]
            x, y, z = np.mgrid[0 : subreg_arr_top.shape[0] : 1, 0 : subreg_arr_top.shape[1] : 1, 0 : subreg_arr_top.shape[2] : 1]
            r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
            r[subreg_arr_top != 49] = 100000
            subset_idx = np.argmin(r)
            print(np.unravel_index(r.argmin(), subreg_arr_top.shape))
            centroids.append([100.0] + list(np.unravel_index(r.argmin(), subreg_arr_top.shape)))

            for key, x, y, z in tqdm(centroids[1:], maxinterval=len(centroids) - 1):
                print("\r", subreg_arr.shape, 10, (x, y, z), key)
                radius = 15

                x0, y0, z0 = (x, y, z)

                x, y, z = np.mgrid[0 : subreg_arr.shape[0] : 1, 0 : subreg_arr.shape[1] : 1, 0 : subreg_arr.shape[2] : 1]
                r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
                arr[r < radius] = key - 40

                nib.save(
                    nib.Nifti1Image(arr, subreg_nib.affine),
                    "vertebra_align/centroids.nii.gz",
                )
        except Exception:
            pass
