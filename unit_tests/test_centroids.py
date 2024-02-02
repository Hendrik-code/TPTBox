# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import sys
import tempfile
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
import random
import unittest

import nibabel as nib
import numpy as np

from TPTBox import Centroids, calc_centroids
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import (
    LABEL_MAX,
    VertebraCentroids,
    _poi_to_dict_list,
    calc_centroids_from_subreg_vert,
    calc_centroids_labeled_buffered,
    load_poi,
)
from TPTBox.core.vert_constants import *
from TPTBox.tests.test_utils import overlap, repeats, get_nii, get_centroids, get_random_ax_code


class Test_Centroids(unittest.TestCase):
    def test_overlap(self):
        a = overlap((13.0, 13.0, 5.0), (4, 2, 2), (10.0, 10.0, 5.0), (3, 4, 4))
        self.assertTrue(a)
        a = overlap((13.0, 13.0, 5.0), (4, 2, 2), (100.0, 100.0, 50.0), (3, 4, 4))
        self.assertFalse(a)

    def test_calc_centroids(self):
        for i in range(repeats):
            with self.subTest(dataset=f"{i} loop"):
                msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
                centroid = calc_centroids(msk)
                msg = "\n\n"
                for x, y in zip(cent, sizes, strict=False):
                    msg += f"{cent[x]},{y}\n"
                self.assertEqual(cent, centroid.centroids, msg=msg)

    def test_cdt_2_dict(self):
        c = Centroids({(1, 50): (10, 11, 12), (32, 50): (23, 23, 23)}, orientation=("R", "I", "P"))
        c.info["Tested"] = "now"
        add_info = {"Tested": "now"}
        cl, format_str = _poi_to_dict_list(c, add_info)

        self.assertTrue((1, 50) in c)
        self.assertTrue((32, 50) in c)
        self.assertFalse((20, 50) in c)
        self.assertEqual(c[1, 50], (10, 11, 12))
        self.assertEqual(c[32, 50], (23, 23, 23))
        self.assertNotEqual(c[1, 50], (10, 22, 12))
        self.assertNotEqual(c[1, 50], (23, 23, 23))
        self.assertEqual(cl[0]["direction"], ("R", "I", "P"))  # type: ignore
        self.assertEqual(cl[0]["Tested"], "now")  # type: ignore
        self.assertEqual(cl[1], {"label": 50 * LABEL_MAX + 1, "X": 10.0, "Y": 11.0, "Z": 12.0})
        self.assertEqual(cl[2], {"label": 50 * LABEL_MAX + 32, "X": 23.0, "Y": 23.0, "Z": 23.0})
        self.assertEqual(len(cl), 3)

    def test_save_load_centroids(self):
        p = get_centroids(num_point=90)

        file = Path(tempfile.gettempdir(), "test_save_load_centroids.json")
        p.save(file, verbose=False)
        c = load_poi(file)
        file.unlink(missing_ok=True)
        self.assertEqual(c, p)

    # Deprecated
    # def test_VertebraCentroids(self):
    #    p = get_centroids(num_point=33)
    #    vc = VertebraCentroids(p.centroids, p.orientation, (1, 1, 1)).sort()
    #    self.assertEqual(len(vc), len(p))
    #    for (k1, k2, _), j in zip(vc.items(), v_idx_order, strict=False):  # noqa: F405
    #        self.assertEqual(k1, j)
    #    self.assertEqual(
    #        [a for a, b in list(vc)],
    #        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 27],
    #    )

    def test_calc_centroids_labeled_bufferd(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        msk2, cent2, order2, sizes2 = get_nii(num_point=len(cent) - 1)

        cent = Centroids(cent, **msk._extract_affine())
        cent2 = Centroids(cent2, **msk2._extract_affine())
        file = Path(tempfile.gettempdir(), "test_save_load_centroids.json")
        file.unlink(missing_ok=True)
        out = calc_centroids_labeled_buffered(msk, None, out_path=file, verbose=False)
        self.assertEqual(out, cent)
        out = calc_centroids_labeled_buffered(msk2, None, out_path=file, verbose=False)
        self.assertEqual(out, cent)
        file.unlink(missing_ok=True)
        out = calc_centroids_labeled_buffered(msk2, None, out_path=file, verbose=False)
        self.assertEqual(out, cent2)
        file.unlink(missing_ok=True)

    def test_calc_centroids_labeled_bufferd2(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        msk2, cent2, order2, sizes2 = get_nii(num_point=len(cent) - 1)
        arr = msk.get_array()
        arr[arr != 0] = 50
        subreg = NII(nib.nifti1.Nifti1Image(arr, msk.affine), True)
        arr = msk2.get_array()
        arr[arr != 0] = 50
        subreg2 = NII(nib.nifti1.Nifti1Image(arr, msk2.affine), True)
        cent = Centroids(cent, **msk._extract_affine())
        cent2 = Centroids(cent2, **msk2._extract_affine())
        file = Path(tempfile.gettempdir(), "test_save_load_centroids.json")
        file.unlink(missing_ok=True)
        out = calc_centroids_labeled_buffered(msk, subreg, out_path=file, verbose=False)
        self.assertEqual(out, cent)
        out = calc_centroids_labeled_buffered(msk2, subreg2, out_path=file, verbose=False)
        self.assertEqual(out, cent)
        file.unlink(missing_ok=True)
        out = calc_centroids_labeled_buffered(msk2, subreg2, out_path=file, verbose=False)
        self.assertEqual(out, cent2)
        file.unlink(missing_ok=True)

    def test_calc_centroids_from_subreg_vert(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        arr = msk.get_array()
        arr[arr != 0] = 50
        subreg = NII(nib.nifti1.Nifti1Image(arr, msk.affine), True)
        out = calc_centroids_from_subreg_vert(msk, subreg, decimals=3)
        cent = Centroids(cent, **msk._extract_affine())
        self.assertEqual(out, cent)

        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        arr = msk.get_array()
        arr[arr != 0] += 40
        subreg = NII(nib.nifti1.Nifti1Image(arr, msk.affine), True)
        out = calc_centroids_from_subreg_vert(msk, subreg, decimals=3, subreg_id=list(range(41, 50)))
        cent_new = {}
        for idx, (k, v) in enumerate(cent.items(), 41):
            cent_new[k[0], idx] = v
        cent = Centroids(cent_new, **msk._extract_affine())
        self.assertEqual(out, cent)

    ######## CLASS FUNCTIONS ############
    def test_copy(self):
        for _ in range(repeats):
            p1 = get_centroids(num_point=29)
            p2 = get_centroids(num_point=28)
            self.assertEqual(p1, p1.copy())
            self.assertEqual(p2, p2.clone())
            self.assertEqual(p2, p1.copy(centroids=p2.centroids))
            p1.orientation = ("R", "A", "L")
            self.assertEqual(p1, p1.copy())
            self.assertNotEqual(p1, p2.copy(centroids=p2.centroids))
            self.assertNotEqual(p1, p2.copy())
            self.assertNotEqual(p2, p1.copy(centroids=p2.centroids))

    def test_crop_centroids(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 2))
            msk.seg = True
            cdt = Centroids(cent, order)
            ex_slice = msk.compute_crop()
            msk2 = msk.apply_crop(ex_slice)
            cdt2 = cdt.apply_crop(ex_slice)
            cdt2_alt = calc_centroids(msk2)
            self.assertEqual(cdt2.centroids, cdt2_alt.centroids)

    def test_crop_centroids_(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = Centroids(cent, order, zoom=msk.zoom, shape=msk.shape)
            ex_slice = msk.compute_crop()
            msk.apply_crop_(ex_slice)
            cdt.apply_crop_(ex_slice)
            cdt2_alt = calc_centroids(msk)
            self.assertEqual(cdt.centroids, cdt2_alt.centroids)

    def test_crop_centroids_trival(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = Centroids(cent, order, zoom=msk.zoom, shape=msk.shape)
            ex_slice = msk.compute_crop(minimum=-1)
            msk.apply_crop_(ex_slice)
            cdt.apply_crop_(ex_slice)
            cdt2_alt = calc_centroids(msk)
            self.assertEqual(cdt.centroids, cdt2_alt.centroids)

    def test_crop_centroids_ValueError(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        self.assertRaises(ValueError, msk.compute_crop, minimum=99)

    def test_reorient_centroids_to(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = Centroids(cent, orientation=order, zoom=msk.zoom, shape=msk.shape)
            axcode = get_random_ax_code()

            msk.reorient_(axcodes_to=axcode)
            self.assertEqual(msk.orientation, axcode)
            cdt.reorient_centroids_to_(msk, verbose=False)
            cdt_other = calc_centroids(msk, subreg_id=50)
            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assertEqual(cdt_other.orientation, axcode)

            axcode = get_random_ax_code()

            msk.reorient_(axcodes_to=axcode)
            self.assertEqual(msk.orientation, axcode)

            cdt = cdt.reorient_centroids_to(msk, verbose=False)
            cdt_other = calc_centroids(msk, subreg_id=Location.Vertebra_Corpus.value)
            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assertEqual(cdt_other.orientation, axcode)

    def test_reorient(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = Centroids(cent, orientation=order, zoom=msk.zoom, shape=msk.shape)
            axcode = get_random_ax_code()

            msk.reorient_(axcodes_to=axcode)
            self.assertEqual(msk.orientation, axcode)
            cdt = cdt.reorient(axcode, verbose=False)
            self.assertEqual(cdt.shape, msk.shape)
            cdt_other = calc_centroids(msk, subreg_id=Location.Vertebra_Corpus.value)
            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assertEqual(cdt_other.orientation, axcode)

            axcode = get_random_ax_code()

            msk.reorient_(axcodes_to=axcode)
            self.assertEqual(msk.orientation, axcode)

            cdt = cdt.reorient(axcode, verbose=False)
            cdt_other = calc_centroids(msk, subreg_id=Location.Vertebra_Corpus.value)
            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assertEqual(cdt_other.orientation, axcode)
            self.assertEqual(cdt.shape, msk.shape)
            self.assertEqual(cdt.zoom, msk.zoom)

    def test_rescale(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            msk.seg = True
            cdt = Centroids(cent, orientation=order, zoom=msk.zoom)
            voxel_spacing = (random.random() * 3, random.random() * 3, random.random() * 3)
            voxel_spacing2 = (1, 1, 1)  # tuple(1 / i for i in voxel_spacing)
            cdt2 = cdt.rescale(voxel_spacing=voxel_spacing, verbose=False, decimals=10)
            cdt2.rescale_(voxel_spacing=voxel_spacing2, verbose=False, decimals=10)
            for (k1, k2, v), (k1_2, k2_2, v2) in zip(cdt.items(), cdt2.items(), strict=False):
                self.assertEqual(k1, k1_2)
                self.assertEqual(k2, k2_2)
                for v, v2 in zip(v, v2, strict=False):
                    self.assertAlmostEqual(v, v2, places=3)

    def test_map(self):
        for _ in range(repeats):
            cdt = get_centroids(num_point=random.randint(2, 98))
            a = random.sample(range(1, 99), len(cdt))
            mapping = {i: k for i, k in enumerate(a, start=1)}
            cdt2 = cdt.map_labels(label_map_region=mapping)  # type: ignore
            self.assertEqual(len(cdt), len(cdt2))
            self.assertNotEqual(cdt, cdt2)
            for key in a:
                self.assertTrue((key, 50) in cdt2)
            cdt2 = cdt.map_labels_(label_map_region={v: k for k, v in mapping.items()})
            self.assertEqual(cdt, cdt2)

    def test_map2(self):
        for _ in range(repeats * 10):
            cdt = get_centroids(num_point=random.randint(2, 28))
            a = random.sample(range(1, 29), len(cdt))

            def r(i):
                if random.random() > 0.5:
                    return v_idx2name[i]
                if random.random() > 0.5:
                    return str(i)
                return i

            mapping = {r(i): r(k) for i, k in enumerate(a, start=1)}
            cdt2 = cdt.map_labels(label_map_region=mapping, verbose=False)  # type: ignore
            self.assertEqual(len(cdt), len(cdt2))
            self.assertNotEqual(cdt, cdt2)
            for key in a:
                self.assertTrue((key, 50) in cdt2)
            cdt2.map_labels_(label_map_region={v: k for k, v in mapping.items()})
            self.assertEqual(cdt, cdt2)


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
