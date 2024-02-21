# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import random
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox.core.nii_wrapper import NII
from TPTBox.core.poi import (
    LABEL_MAX,
    POI,
    Location,
    _poi_to_dict_list,
    calc_centroids,
    calc_poi_from_subreg_vert,
    calc_poi_labeled_buffered,
    load_poi,
)
from TPTBox.tests.test_utils import extract_affine, get_centroids, get_poi, get_random_ax_code, overlap, repeats, sqr1d


def get_nii(x: tuple[int, int, int] | None = None, num_point=3, rotation=True):  # type: ignore
    if x is None:
        x = (random.randint(30, 70), random.randint(30, 70), random.randint(30, 70))
    a = np.zeros(x, dtype=np.uint16)
    points = []
    out_points: dict[int, dict[int, tuple[float, float, float]]] = {}
    sizes = []
    idx = 1
    while True:
        if num_point == len(points):
            break
        point = tuple(random.randint(1, a - 1) for a in x)
        size = tuple(random.randint(1, 1 + a) for a in [5, 5, 5])
        if any(a - b < 0 for a, b in zip(point, size, strict=True)):
            continue
        if any(a + b > c - 1 for a, b, c in zip(point, size, x, strict=True)):
            continue
        skip = False
        for p2, s2 in zip(points, sizes, strict=True):
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
        out_points[idx] = {50: tuple(float(a) for a in point)}

        idx += 1
    aff = np.eye(4)

    aff[0, 3] = random.randint(-100, 100)
    aff[1, 3] = random.randint(-100, 100)
    aff[2, 3] = random.randint(-100, 100)
    if rotation:
        m = 30
        from scipy.spatial.transform import Rotation

        r = Rotation.from_euler("xyz", (random.randint(-m, m), random.randint(-m, m), random.randint(-m, m)), degrees=True)
        aff[:3, :3] = r.as_matrix()
    n = NII(nib.Nifti1Image(a, aff), seg=True)

    return n, out_points, n.orientation, sizes


class Test_POI(unittest.TestCase):
    def assert_affine(self, poi: POI, nii: NII | POI):
        assert poi.zoom is not None
        assert nii.zoom is not None
        self.assertTrue(np.isclose(poi.zoom, nii.zoom, atol=1e-1).all(), msg=f"poi {poi.zoom}, nii {nii.zoom}")
        self.assertEqual(poi.orientation, nii.orientation)
        s = poi.shape
        if isinstance(nii, NII):
            assert s is not None
            s = tuple(round(x, 0) for x in s)
        print(nii)
        self.assertEqual(s, nii.shape)
        self.assertTrue(poi.rotation is not None)
        self.assertTrue(nii.rotation is not None)
        assert poi.rotation is not None
        assert nii.rotation is not None
        self.assertTrue(np.isclose(poi.rotation, nii.rotation, atol=1e-1).all(), msg=f"poi {poi.rotation}, nii {nii.rotation}")
        self.assertTrue(poi.origin is not None)
        self.assertTrue(nii.origin is not None)
        assert nii.origin is not None
        assert poi.origin is not None
        [self.assertAlmostEqual(poi.origin[i], nii.origin[i], places=4) for i in range(len(poi.origin))]
        aff_equ = np.isclose(poi.affine, nii.affine, atol=1e-6).all()
        self.assertTrue(aff_equ, msg=f"{poi.affine}\n {nii.affine}")

    def test_overlap(self):
        a = overlap((13.0, 13.0, 5.0), (4, 2, 2), (10.0, 10.0, 5.0), (3, 4, 4))
        self.assertTrue(a)
        a = overlap((13.0, 13.0, 5.0), (4, 2, 2), (100.0, 100.0, 50.0), (3, 4, 4))
        self.assertFalse(a)

    def test_calc_POI(self):
        for i in range(repeats):
            with self.subTest(dataset=f"{i} loop"):
                msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
                poi = calc_centroids(msk)
                msg = "\n\n"
                for x, y in zip(cent, sizes, strict=True):
                    msg += f"{x} - {cent[x]},{y}\n"
                self.assertEqual(cent, poi.centroids.pois, msg=msg)
                self.assert_affine(poi, msk)

    def test_cdt_2_dict(self):
        c = POI({1: {99: (10, 11, 12)}, 32: {50: (23, 23, 23)}}, orientation=("R", "I", "P"))
        c.info["Tested"] = "now"
        add_info = {"Tested": "now"}
        cl, format_str = _poi_to_dict_list(c, add_info)

        self.assertTrue((1, 99) in c, msg=c)
        self.assertTrue((32, 50) in c)
        self.assertFalse((1, 50) in c)
        self.assertFalse((32, 99) in c)
        self.assertFalse((23, 23) in c)
        self.assertEqual(c[1, 99], (10, 11, 12))
        self.assertEqual(c[32, 50], (23, 23, 23))
        self.assertNotEqual(c[1, 99], (10, 22, 12))
        self.assertNotEqual(c[1, 99], (23, 23, 23))
        self.assertEqual(cl[0]["direction"], ("R", "I", "P"))  # type: ignore
        self.assertEqual(cl[0]["Tested"], "now")  # type: ignore
        self.assertEqual(cl[1], {"label": LABEL_MAX * 99 + 1, "X": 10.0, "Y": 11.0, "Z": 12.0})
        self.assertEqual(cl[2], {"label": LABEL_MAX * 50 + 32, "X": 23.0, "Y": 23.0, "Z": 23.0})
        self.assertEqual(len(cl), 3)

    def test_save_load_POI(self):
        for save_hint in [0, 2]:
            p = get_poi(num_vert=90, num_subreg=2)

            file = Path(tempfile.gettempdir(), "test_save_load_POI.json")
            p.save(file, verbose=False, save_hint=save_hint)
            c = load_poi(file)
            file.unlink(missing_ok=True)
            self.assertEqual(c, p)
            self.assert_affine(c, p)

    def test_calc_POI_labeled_bufferd(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        msk2, cent2, order2, sizes2 = get_nii(num_point=len(cent) - 1)

        cent = POI(cent, orientation=order, zoom=(1, 1, 1), **extract_affine(msk))
        cent2 = POI(cent2, orientation=order2, zoom=(1, 1, 1), **extract_affine(msk2))
        file = Path(tempfile.gettempdir(), "test_save_load_POI.json")
        file.unlink(missing_ok=True)
        out = calc_poi_labeled_buffered(msk, None, out_path=file, verbose=False)
        self.assertEqual(out, cent)
        self.assert_affine(out, msk)
        out = calc_poi_labeled_buffered(msk2, None, out_path=file, verbose=False)
        self.assertEqual(out, cent)
        self.assert_affine(out, msk)
        file.unlink(missing_ok=True)
        out = calc_poi_labeled_buffered(msk2, None, out_path=file, verbose=False)
        self.assert_affine(out, msk2)
        self.assertEqual(out, cent2)
        file.unlink(missing_ok=True)

    def test_calc_POI_from_subreg_vert(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        arr = msk.get_array()
        arr[arr != 0] = 50
        subreg = NII(nib.Nifti1Image(arr, msk.affine), True)  # type: ignore
        out = calc_poi_from_subreg_vert(msk, subreg, buffer_file=None, decimals=3)
        cent = POI(cent, orientation=order, zoom=(1, 1, 1), **extract_affine(msk))
        self.assertEqual(out, cent)
        self.assert_affine(out, cent)
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        arr = msk.get_array()
        arr[arr != 0] += 40
        subreg = NII(nib.Nifti1Image(arr, msk.affine), True)
        out = calc_poi_from_subreg_vert(msk, subreg, buffer_file=None, decimals=3, subreg_id=list(range(41, 50)))
        cent_new = {}
        for idx, (k, v) in enumerate(cent.items(), 41):
            cent_new[k, idx] = v[50]
        cent = POI(cent_new, orientation=order, zoom=msk.zoom, **extract_affine(msk))
        self.assertEqual(out, cent)
        self.assert_affine(out, cent)

    ######## CLASS FUNCTIONS ############
    def test_copy(self):
        for _ in range(repeats):
            p1 = get_poi(num_vert=29)
            p2 = get_poi(num_vert=28)
            p2.origin = p1.origin
            self.assertEqual(p1, p1.copy())
            self.assertEqual(p2, p2.clone())
            self.assertEqual(p2, p1.copy(centroids=p2.centroids, origin=p2.origin, rotation=p2.rotation))
            p1.orientation = ("R", "A", "L")
            self.assertEqual(p1, p1.copy())
            self.assertNotEqual(p1, p2.copy(centroids=p2.centroids, origin=p2.origin, rotation=p2.rotation))
            self.assertNotEqual(p1, p2.copy())
            self.assertNotEqual(p2, p1.copy(centroids=p2.centroids, origin=p2.origin, rotation=p2.rotation))
            self.assertNotEqual(p1, p2.copy(centroids=p2.centroids))
            self.assertNotEqual(p1, p2.copy())
            self.assertNotEqual(p2, p1.copy(centroids=p2.centroids))

    def test_crop_poi(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 2))
            msk.seg = True
            cdt = POI(cent, **msk._extract_affine())
            ex_slice = msk.compute_crop()
            msk2 = msk.apply_crop(ex_slice)
            cdt2 = cdt.apply_crop(ex_slice)
            cdt2_alt = calc_centroids(msk2)
            self.assertEqual(cdt2.centroids, cdt2_alt.centroids)
            self.assert_affine(cdt2, cdt2_alt)

    def test_crop_poi_(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = POI(cent, orientation=order, zoom=msk.zoom, **extract_affine(msk))
            ex_slice = msk.compute_crop()
            msk.apply_crop_(ex_slice)
            cdt.apply_crop_(ex_slice)
            cdt2_alt = calc_centroids(msk)
            self.assertEqual(cdt.centroids, cdt2_alt.centroids)
            self.assert_affine(cdt, cdt2_alt)

    def test_crop_POI_trivial(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = POI(cent, orientation=order, zoom=msk.zoom, **extract_affine(msk))
            ex_slice = msk.compute_crop(minimum=-1)
            msk.apply_crop_(ex_slice)
            cdt.apply_crop_(ex_slice)
            cdt2_alt = calc_centroids(msk)
            self.assertEqual(cdt.centroids, cdt2_alt.centroids)
            self.assert_affine(cdt, cdt2_alt)

    def test_crop_POI_ValueError(self):
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        self.assertRaises(ValueError, msk.compute_crop, minimum=99)

    def test_reorient_POI_to(self):
        mapp = {"A": "P", "P": "A", "S": "I", "I": "S", "R": "L", "L": "R"}
        np.set_printoptions(precision=4, linewidth=500)
        msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
        cdt = POI(cent, orientation=order, zoom=msk.zoom, **extract_affine(msk))
        msk.reorient_(axcodes_to=(order[0], order[1], order[2]))
        cdt.reorient_centroids_to_(msk, verbose=False)
        msk.reorient_(axcodes_to=(mapp[order[0]], order[1], order[2]))  # type: ignore
        cdt.reorient_centroids_to_(msk, verbose=False)
        msk.reorient_(axcodes_to=(order[1], order[0], order[2]))
        cdt.reorient_centroids_to_(msk, verbose=False)
        msk.reorient_(axcodes_to=(order[0], order[2], order[1]))
        cdt.reorient_centroids_to_(msk, verbose=False)
        self.assert_affine(cdt, msk)
        msk.reorient_(axcodes_to=("L", "A", "S"))
        cdt.reorient_centroids_to_(msk, verbose=False)
        msk.reorient_(axcodes_to=("A", "S", "R"))
        cdt.reorient_centroids_to_(msk, verbose=False)
        msk.reorient_(axcodes_to=("A", "S", "L"))
        cdt.reorient_centroids_to_(msk, verbose=False)
        msk.reorient_(axcodes_to=("S", "R", "A"))
        cdt.reorient_centroids_to_(msk, verbose=False)
        msk.reorient_(axcodes_to=("S", "L", "A"))
        cdt.reorient_centroids_to_(msk, verbose=False)
        msk.reorient_(axcodes_to=("I", "A", "R"))
        cdt.reorient_centroids_to_(msk, verbose=False)
        cdt.reorient_centroids_to_(msk, verbose=False)
        self.assert_affine(cdt, msk)
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = POI(cent, orientation=order, zoom=msk.zoom, **extract_affine(msk))
            axcode = get_random_ax_code()
            msk.reorient_(axcodes_to=axcode, verbose=True)
            self.assertEqual(msk.orientation, axcode)
            cdt.reorient_centroids_to_(msk, verbose=False)
            cdt_other = calc_centroids(msk, subreg_id=Location.Vertebra_Corpus.value)
            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assert_affine(cdt, cdt_other)
            self.assertEqual(cdt_other.orientation, axcode)

            axcode = get_random_ax_code()
            msk.reorient_(axcodes_to=axcode)
            self.assertEqual(msk.orientation, axcode)

            cdt = cdt.reorient_centroids_to(msk, verbose=False)
            cdt_other = calc_centroids(msk)

            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assert_affine(cdt, cdt_other)
            self.assertEqual(cdt_other.orientation, axcode)

    def test_reorient(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            cdt = POI(cent, **msk._extract_affine())
            axcode = get_random_ax_code()

            msk.reorient_(axcodes_to=axcode)
            self.assertEqual(msk.orientation, axcode)
            cdt = cdt.reorient(axcode, verbose=False)
            self.assertEqual(cdt.shape, msk.shape)
            cdt_other = calc_centroids(msk)
            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assertEqual(cdt_other.orientation, axcode)
            self.assert_affine(cdt, cdt_other)

            axcode = get_random_ax_code()

            msk.reorient_(axcodes_to=axcode)
            self.assertEqual(msk.orientation, axcode)

            cdt = cdt.reorient(axcode, verbose=False)
            cdt_other = calc_centroids(msk)
            self.assert_affine(cdt, cdt_other)
            self.assert_affine(cdt, msk)
            self.assertEqual(cdt.centroids, cdt_other.centroids)
            self.assert_affine(cdt, cdt_other)
            self.assertEqual(cdt_other.orientation, axcode)

    def test_rescale(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            axcode = get_random_ax_code()
            msk.seg = True
            cdt = POI(cent, orientation=order, zoom=msk.zoom, **extract_affine(msk))
            msk.reorient_(axcodes_to=axcode)
            cdt.reorient_(axcodes_to=axcode)
            voxel_spacing = (random.random() * 3, random.random() * 3, random.random() * 3)
            voxel_spacing2 = (1, 1, 1)  # tuple(1 / i for i in voxel_spacing)
            cdt2 = cdt.rescale(voxel_spacing=voxel_spacing, verbose=False, decimals=10)
            cdt2.rescale_(voxel_spacing=voxel_spacing2, verbose=False, decimals=10).round_(1)
            self.assertNotEqual(cdt2.shape, None)
            assert cdt2.shape is not None
            cdt2.shape = tuple(round(float(v), 0) for v in cdt2.shape)
            self.assert_affine(cdt, cdt2)
            for (k, s, v), (k2, s2, v2) in zip(cdt.items(), cdt2.items(), strict=True):
                self.assertEqual(k, k2)
                self.assertEqual(s, s2)
                for v3, v4 in zip(v, v2, strict=True):
                    self.assertAlmostEqual(v3, v4)

    def test_rescale_nii(self):
        for _ in range(repeats):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            axcode = get_random_ax_code()
            msk.seg = True
            cdt = POI(cent, **msk._extract_affine())
            msk.reorient_(axcodes_to=axcode)
            cdt.reorient_(axcodes_to=axcode)
            voxel_spacing = (random.random() * 3, random.random() * 3, random.random() * 3)
            cdt2 = cdt.rescale(voxel_spacing=voxel_spacing, verbose=False, decimals=10)
            msk.rescale_(voxel_spacing=voxel_spacing)
            self.assert_affine(cdt2, msk)

    def test_map_individual(self):
        for _ in range(repeats):
            cdt = get_poi(num_vert=random.randint(2, 98), num_subreg=random.randint(20, 30), max_subreg=50)
            tries = 10_0000
            in_list = []
            not_in_list = []
            while len(in_list) != 10 or len(not_in_list) != 10:
                a = random.randint(1, 98)
                b = random.randint(1, 50)
                if (a, b) in in_list:
                    continue
                if (a, b) in cdt:
                    if len(in_list) < 10:
                        in_list.append((a, b))
                elif (a, b) not in not_in_list and len(not_in_list) < 10:
                    not_in_list.append((a, b))
                tries -= 1
                assert tries != 0, (in_list, not_in_list)
            mapping = dict(zip(in_list, not_in_list, strict=True))
            cdt2 = cdt.map_labels(label_map_full=mapping)

            for a in in_list:
                self.assertNotIn(a, cdt2)
                self.assertIn(a, cdt)
            for a in not_in_list:
                self.assertIn(a, cdt2)
                self.assertNotIn(a, cdt)

    def test_map(self):
        for _ in range(repeats):
            num = random.randint(2, 98)
            cdt = get_poi(num_vert=num, num_subreg=random.randint(1, 5))
            a = random.sample(range(1, 99), num)
            mapping = dict(enumerate(a, start=1))
            cdt2 = cdt.map_labels(label_map_region=mapping)  # type: ignore
            self.assertEqual(len(cdt), len(cdt2))
            self.assertNotEqual(cdt, cdt2)
            for key in a:
                self.assertTrue(key in [k for k, k2 in cdt2.keys()])
            cdt2.map_labels_(label_map_region={v: k for k, v in mapping.items()})
            self.assertEqual(cdt, cdt2)
            self.assert_affine(cdt, cdt2)

    def test_map2(self):
        from TPTBox.core.vert_constants import subreg_idx2name

        for _ in range(repeats * 10):
            cdt = get_poi(num_vert=random.randint(4, 25), min_subreg=41, max_subreg=51, num_subreg=4)
            a = random.sample(range(40, 51), 10)

            def r(i):
                if random.random() > 0.5:
                    return subreg_idx2name[i]
                if random.random() > 0.5:
                    return i
                return i

            mapping = {r(k): i for i, k in enumerate(a, start=1)}
            cdt2 = cdt.map_labels(label_map_subregion=mapping, verbose=False)  # type: ignore
            self.assertEqual(len(cdt), len(cdt2))
            self.assertNotEqual(cdt, cdt2)
            for key in a:
                self.assertFalse(key in [k2 for k, k2 in cdt2.keys()], msg=f"{key}")
            cdt2.map_labels_(label_map_subregion={v: k for k, v in mapping.items()})
            self.assertEqual(cdt, cdt2)
            self.assert_affine(cdt, cdt2)

    def test_global(self):
        for _ in range(repeats * 10):
            msk, cent, order, sizes = get_nii(num_point=random.randint(1, 7))
            poi = POI(cent, **msk._extract_affine())
            msk.seg = True
            axcode = get_random_ax_code()
            msk.reorient_(axcodes_to=axcode)
            poi.reorient_(axcodes_to=axcode)
            glob_ctd = poi.to_global()
            poi2 = glob_ctd.to_other_nii(msk)
            for (k, s, v), (k2, s2, v2) in zip(poi.items(), poi2.items(), strict=True):
                self.assertEqual(k, k2)
                self.assertEqual(s, s2)
                for v3, v4 in zip(v, v2, strict=True):
                    self.assertAlmostEqual(v3, v4, places=6)

            self.assert_affine(poi2, msk)

            glob_ctd = poi.to_global()
            poi2 = glob_ctd.to_other_poi(poi)
            for (k, s, v), (k2, s2, v2) in zip(poi.items(), poi2.items(), strict=True):
                self.assertEqual(k, k2)
                self.assertEqual(s, s2)
                for v3, v4 in zip(v, v2, strict=True):
                    self.assertAlmostEqual(v3, v4, places=6)
            self.assert_affine(poi2, msk)

    def test_poi_to_global(self):
        cdt = POI({}, zoom=(1, 1, 1), shape=(1000, 1000, 1000), rotation=np.eye(3), origin=(0, 0, 0))
        cdt[5, 5] = (0, 9, 99)
        cdt.origin = (1, 1, 1)
        cdt2 = cdt.copy().reorient(("S", "A", "R"))
        assert (cdt.rotation == np.eye(3)).all(), cdt.rotation

        self.assertEqual(cdt2.global_to_local((10, 100, 1000)), (999, 99, 9))
        self.assertEqual(cdt2.local_to_global((5, 6, 7)), (8, 7, 6))
        self.assertEqual(cdt2.global_to_local(cdt2.local_to_global((8, 7, 6))), (8, 7, 6))
        cdt2.rescale_((0.5, 0.5, 0.5)).round(2)
        print("cdt2", cdt2)
        glob = cdt2.to_global()
        self.assertEqual(glob[5, 5], (1, 10, 100))
        self.assertEqual(glob.to_other_poi(cdt2), cdt2)
        print(glob)
        print()
        revert = glob.to_other_poi(cdt)
        print(glob)
        self.assertEqual(revert, cdt, msg=f"\n{revert}, \n{cdt}")

    def test_resample_from_to_simple(self):
        cdt = get_poi(num_vert=0, num_subreg=0, max_subreg=50)
        cdt[5, 5] = (100, 10, 1)
        cdt2 = cdt.copy().reorient(("I", "A", "R"))
        cdt2.rescale_((0.5, 2, 0.25)).round(2)
        cdt_res = cdt.resample_from_to(cdt2).round(2)
        self.assertEqual(cdt_res, cdt2, msg=f"'\n{cdt_res}, \n{cdt2}")

    def test_resample_from_to(self):
        for _ in range(repeats * 100):
            cdt = get_poi(num_vert=random.randint(1, 1), num_subreg=random.randint(1, 1), max_subreg=50)
            cdt2 = cdt.reorient(get_random_ax_code())
            cdt2.rescale_((random.random() * 5 + 0.3, random.random() * 5 + 0.3, random.random() * 5 + 0.3))  # .round(2)
            cdt_res = cdt.resample_from_to(cdt2)  # .round(1)
            # self.assertEqual(cdt_res, cdt2.round_(1), msg=f"\n{cdt_res}, \n{cdt2}")
            for idx, v in cdt_res.items_flatten():
                for c in range(3):
                    self.assertAlmostEqual(v[c], cdt2[idx][c], delta=1e-2)


if __name__ == "__main__":
    unittest.main()

    # @unittest.skipIf(condition, reason)
    # with self.subTest(i=i):
    # @unittest.skipIf(condition, reason)
    # with self.subTest(i=i):
    # @unittest.skipIf(condition, reason)
    # with self.subTest(i=i):
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
