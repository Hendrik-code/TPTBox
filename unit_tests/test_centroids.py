# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import random
import tempfile
import unittest
from pathlib import Path

import nibabel as nib

from TPTBox import POI, calc_centroids
from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import LABEL_MAX, _poi_to_dict_list, calc_centroids_from_two_masks, calc_poi_from_subreg_vert, load_poi
from TPTBox.core.vert_constants import Location
from TPTBox.tests.test_utils import get_nii, get_random_ax_code, overlap, repeats
from unit_tests.test_centroids_save import get_centroids


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
        c = POI({(1, 50): (10, 11, 12), (32, 50): (23, 23, 23)}, orientation=("R", "I", "P"))
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

    def test_calc_centroids_labeled_bufferd(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        msk2, cent2, order2, sizes2 = get_nii(num_point=len(cent) - 1)

        cent = POI(cent, **msk._extract_affine())
        cent2 = POI(cent2, **msk2._extract_affine())
        file = Path(tempfile.gettempdir(), "test_save_load_centroids.json")
        file.unlink(missing_ok=True)
        out = calc_centroids_from_two_masks(msk, None, out_path=file, verbose=False)
        self.assertEqual(out, cent)
        out = calc_centroids_from_two_masks(msk2, None, out_path=file, verbose=False)
        self.assertEqual(out, cent)
        file.unlink(missing_ok=True)
        out = calc_centroids_from_two_masks(msk2, None, out_path=file, verbose=False)
        self.assertEqual(out, cent2)
        file.unlink(missing_ok=True)

    def test_calc_centroids_labeled_bufferd2(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        msk2, cent2, order2, sizes2 = get_nii(x=msk.shape, num_point=len(cent) - 1)
        assert msk.shape == msk2.shape, (msk.shape, msk2.shape)
        msk2.origin = msk.origin
        arr = msk.get_array()
        arr[arr != 0] = 50
        subreg = NII(nib.nifti1.Nifti1Image(arr, msk.affine), True)
        arr = msk2.get_array()
        arr[arr != 0] = 50
        subreg2 = NII(nib.nifti1.Nifti1Image(arr, msk2.affine), True)
        cent = POI(cent, **msk._extract_affine())
        cent2 = POI(cent2, **msk2._extract_affine())
        file = Path(tempfile.gettempdir(), "test_save_load_centroids.json")
        file.unlink(missing_ok=True)
        out = calc_poi_from_subreg_vert(msk, subreg, buffer_file=file, save_buffer_file=True, verbose=False)
        self.assertEqual(out, cent)
        out = calc_poi_from_subreg_vert(msk2, subreg2, buffer_file=file, save_buffer_file=True, verbose=False)
        self.assertEqual(out, cent)
        file.unlink(missing_ok=True)
        out = calc_poi_from_subreg_vert(msk2, subreg2, buffer_file=file, save_buffer_file=True, verbose=False)
        self.assertEqual(out, cent2)
        file.unlink(missing_ok=True)

    def test_calc_centroids_from_subreg_vert(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        arr = msk.get_array()
        arr[arr != 0] = 50
        subreg = NII(nib.nifti1.Nifti1Image(arr, msk.affine), True)
        out = calc_poi_from_subreg_vert(msk, subreg, decimals=3)
        cent = POI(cent, **msk._extract_affine())
        self.assertEqual(out, cent)

        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        arr = msk.get_array()
        arr[arr != 0] += 40
        subreg = NII(nib.nifti1.Nifti1Image(arr, msk.affine), True)
        out = calc_poi_from_subreg_vert(msk, subreg, decimals=3, subreg_id=list(range(41, 50)))
        cent_new = {}
        for idx, (k, v) in enumerate(cent.items(), 41):
            cent_new[k[0], idx] = v
        cent = POI(cent_new, **msk._extract_affine())
        self.assertEqual(out, cent)

    ######## CLASS FUNCTIONS ############
    def test_copy(self):
        for _ in range(repeats):
            p1 = get_centroids(num_point=29)
            p2 = get_centroids(num_point=28)
            self.assertEqual(p1, p1.copy())
            self.assertEqual(p2, p2.clone())
            p1.orientation = ("R", "A", "L")
            self.assertEqual(p1, p1.copy())
            self.assertNotEqual(p1, p2.copy(centroids=p2.centroids))
            self.assertNotEqual(p1, p2.copy())
            self.assertNotEqual(p2, p1.copy(centroids=p2.centroids))

    def test_crop_centroids(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 2))
            msk.seg = True
            cdt = msk.make_empty_POI(cent)
            ex_slice = msk.compute_crop()
            msk2 = msk.apply_crop(ex_slice)
            assert msk2.origin != msk.origin
            cdt2 = cdt.apply_crop(ex_slice)

            cdt2_alt = calc_centroids(msk2)
            self.assertEqual(cdt2.centroids, cdt2_alt.centroids)

    def test_crop_centroids_(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = msk.make_empty_POI(cent)
            ex_slice = msk.compute_crop()
            msk.apply_crop_(ex_slice)
            cdt.apply_crop_(ex_slice)
            cdt2_alt = calc_centroids(msk)
            self.assertEqual(cdt.centroids, cdt2_alt.centroids)

    def test_crop_centroids_trival(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = POI(cent, orientation=order, zoom=msk.zoom, shape=msk.shape)
            ex_slice = msk.compute_crop(minimum=-1)
            msk.apply_crop_(ex_slice)
            cdt.apply_crop_(ex_slice)
            cdt2_alt = calc_centroids(msk)
            self.assertEqual(cdt.centroids, cdt2_alt.centroids)

    def test_crop_centroids_ValueError(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        self.assertRaises(ValueError, msk.compute_crop, minimum=99)

    def test_reorient(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = POI(cent, orientation=order, zoom=msk.zoom, shape=msk.shape)
            axcode = get_random_ax_code()

            msk.reorient_(axcodes_to=axcode)
            self.assertEqual(msk.orientation, axcode)
            cdt = cdt.reorient(axcode, verbose=False)
            self.assertEqual(cdt.shape, msk.shape)
            cdt_other = calc_centroids(msk, second_stage=Location.Vertebra_Corpus.value)
            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assertEqual(cdt_other.orientation, axcode)

            axcode = get_random_ax_code()

            msk.reorient_(axcodes_to=axcode)
            self.assertEqual(msk.orientation, axcode)

            cdt = cdt.reorient(axcode, verbose=False)
            cdt_other = calc_centroids(msk, second_stage=Location.Vertebra_Corpus.value)
            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assertEqual(cdt_other.orientation, axcode)
            self.assertEqual(cdt.shape, msk.shape)
            self.assertEqual(cdt.zoom, msk.zoom)

    def test_rescale(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            msk.seg = True
            cdt = POI(cent, orientation=order, zoom=msk.zoom)
            voxel_spacing = (random.random() * 3, random.random() * 3, random.random() * 3)
            voxel_spacing2 = (1, 1, 1)  # tuple(1 / i for i in voxel_spacing)
            cdt2 = cdt.rescale(voxel_spacing=voxel_spacing, verbose=False, decimals=10)
            cdt2.rescale_(voxel_spacing=voxel_spacing2, verbose=False, decimals=10)
            for (k1, k2, v), (k1_2, k2_2, v2) in zip(cdt.items(), cdt2.items(), strict=False):
                self.assertEqual(k1, k1_2)
                self.assertEqual(k2, k2_2)
                for v, v2 in zip(v, v2, strict=False):  # noqa: B020, PLW2901
                    self.assertAlmostEqual(v, v2, places=2)


class Test_Registation(unittest.TestCase):
    def test_reg(self):
        from TPTBox.registration.ridged_points.point_registration import ridged_points_from_poi

        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(4, 7))
            cdt = msk.make_empty_POI(cent)
            cdt_org = cdt.copy()
            cdt.origin = tuple(i + random.random() * 5 for i in cdt.origin)
            cdt = cdt.resample_from_to(cdt_org)

            cdt.rescale_((random.randint(1, 3), random.randint(1, 3), random.randint(1, 3)))
            registation_obj = ridged_points_from_poi(cdt_org, cdt)
            moved_poi = registation_obj.transform_poi(cdt).round(2)
            assert moved_poi == cdt_org


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
