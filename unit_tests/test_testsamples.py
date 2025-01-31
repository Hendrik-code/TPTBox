# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))

import unittest  # noqa: E402

from TPTBox import NII, Location, calc_poi_from_subreg_vert  # noqa: E402
from TPTBox.tests.test_utils import get_test_ct, get_test_mri, get_tests_dir  # noqa: E402


class Test_testsamples(unittest.TestCase):
    def test_load_ct(self):
        ct_nii, subreg_nii, vert_nii, label = get_test_ct()
        self.assertTrue(ct_nii.assert_affine(other=subreg_nii, raise_error=False))
        self.assertTrue(ct_nii.assert_affine(other=vert_nii, raise_error=False))

        l3 = vert_nii.extract_label(label)
        l3_subreg = subreg_nii.apply_mask(l3, inplace=False)
        self.assertEqual(l3.volumes()[1], sum(l3_subreg.volumes(include_zero=False).values()))

    def test_load_mri(self):
        mri_nii, subreg_nii, vert_nii, label = get_test_mri()
        self.assertTrue(mri_nii.assert_affine(other=subreg_nii, raise_error=False))
        self.assertTrue(mri_nii.assert_affine(other=vert_nii, raise_error=False))

        l3 = vert_nii.extract_label(label)
        l3_subreg = subreg_nii.apply_mask(l3, inplace=False)
        self.assertEqual(l3.volumes()[1], sum(l3_subreg.volumes(include_zero=False).values()))

    def test_make_snapshot_both(self, keep_images=False):
        from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

        mri_nii, subreg_nii, vert_nii, label = get_test_mri()
        ct_nii, ct_subreg_nii, ct_vert_nii, label = get_test_ct()
        mri_poi = calc_poi_from_subreg_vert(vert_nii, subreg_nii)
        ct_poi = calc_poi_from_subreg_vert(ct_vert_nii, ct_subreg_nii, subreg_id=Location.Spinosus_Process)
        mr_a = Snapshot_Frame(mri_nii, subreg_nii, mri_poi, axial=True, coronal=True, mode="MRI")
        mr_b = Snapshot_Frame(mri_nii, vert_nii, mri_poi, coronal=True, hide_centroid_labels=True)
        mr_b = Snapshot_Frame(mri_nii, vert_nii, mri_poi, coronal=True)
        ct_a = Snapshot_Frame(ct_nii, ct_vert_nii, ct_poi, mode="CT", coronal=True, axial=True)
        ct_b = Snapshot_Frame(ct_nii, ct_vert_nii, mode="CT", crop_msk=True)
        ct_c = Snapshot_Frame(ct_nii, ct_vert_nii, mode="CT", crop_img=True)
        ct_d = Snapshot_Frame(ct_nii, ct_vert_nii, mode="CT", alpha=0.9, ignore_cdt_for_centering=True)
        ct_e = Snapshot_Frame(ct_nii, ct_subreg_nii, mode="CT", only_mask_area=True, hide_segmentation=True)
        with tempfile.TemporaryDirectory() as td:
            if keep_images:
                td = Path(__file__).parent / "snaps"  # noqa: PLW2901
                if td.exists():
                    [a.unlink() for a in Path(td).iterdir() if a.name.endswith(".jpg") or a.name.endswith(".png")]
                td.mkdir(exist_ok=True)
            i = 0
            file = Path(td, f"poi{i}.jpg")
            i += 1
            create_snapshot(file, [mr_a, mr_b])
            self.assertTrue(file.exists(), file)
            file = Path(td, f"poi{i}.jpg")
            i += 1
            create_snapshot(str(file), [ct_a])
            self.assertTrue(file.exists(), file)
            file = Path(td, f"poi{i}.jpg")
            i += 1
            create_snapshot([file], [ct_b, ct_d, ct_e])
            self.assertTrue(file.exists(), file)
            file = Path(td, f"poi{i}.jpg")
            file2 = Path(td, f"poi100{i}.png")
            i += 1
            create_snapshot([file, str(file2)], [mr_a, mr_b, ct_b, ct_d, ct_c])
            self.assertTrue(file.exists(), file)
            self.assertTrue(file2.exists(), file2)

        self.assertFalse(Path(td).exists() and not keep_images)

    def make_POIs(self, vert_nii: NII, subreg_nii: NII, vert_id: int, ignore_list: list[Location], locs: None | list[Location] = None, n=5):
        for i in range(n):
            if locs is None:
                locs = [l for l in Location if l not in ignore_list and random.random() < (i + 1) / n * 3]
            poi = calc_poi_from_subreg_vert(vert_nii, subreg_nii, subreg_id=locs, verbose=False, _print_phases=True).extract_vert(vert_id)
            for l in locs:
                self.assertIn((vert_id, l.value), poi)
            poi.assert_affine(
                vert_nii,
                shape_tolerance=0.5,
            )

    def test_POIs_CT(self):
        _, subreg_nii, vert_nii, label = get_test_ct()
        ignore_list = [
            Location.Implant_Entry_Left,
            Location.Implant_Entry_Right,
            Location.Implant_Target_Left,
            Location.Implant_Target_Right,
            Location.Spinal_Canal,
            Location.Vertebra_Corpus_border,
            Location.Dens_axis,
            Location.Unknown,
            Location.Endplate,
            Location.Spinal_Cord,
            Location.Spinal_Canal,
            Location.Spinal_Canal_ivd_lvl,
            Location.Vertebral_Body_Endplate_Superior,
            Location.Vertebral_Body_Endplate_Inferior,
            Location.Rib_Left,
            Location.Rib_Right,
        ]
        self.make_POIs(vert_nii, subreg_nii, label, ignore_list)

    def test_POIs_MR(self):
        _, subreg_nii, vert_nii, label = get_test_mri()

        ignore_list = [
            Location.Implant_Entry_Left,
            Location.Implant_Entry_Right,
            Location.Implant_Target_Left,
            Location.Implant_Target_Right,
            Location.Spinal_Canal,
            Location.Vertebra_Corpus_border,
            Location.Dens_axis,
            Location.Unknown,
            Location.Spinal_Cord,
            Location.Endplate,
            Location.Vertebral_Body_Endplate_Superior,
            Location.Vertebral_Body_Endplate_Inferior,
            Location.Rib_Left,
            Location.Rib_Right,
        ]
        self.make_POIs(vert_nii, subreg_nii, label, ignore_list)

    def test_POIs_MR_disc(self):
        _, subreg_nii, vert_nii, label = get_test_mri()

        locs = [Location.Vertebra_Disc_Superior, Location.Vertebra_Disc_Inferior]
        self.make_POIs(vert_nii, subreg_nii, label, [], locs, n=1)

    def test_pad_crop(self):
        for _, _, vert_nii, label in [get_test_mri(), get_test_ct()]:
            vert_nii.extract_label_(label, keep_label=False)
            crop = vert_nii.compute_crop()
            cropped = vert_nii.apply_crop(crop)
            assert cropped.shape != vert_nii.shape
            assert cropped.origin != vert_nii.origin
            returned = cropped.pad_to(vert_nii)
            assert returned.origin == vert_nii.origin
            assert (returned.affine == vert_nii.affine).all()
            assert (returned.get_array() == vert_nii.get_array()).all()

    def test_angle(self):
        _, subreg_nii, vert_nii, label = get_test_mri()
        locs = [Location.Vertebra_Direction_Inferior]
        poi = calc_poi_from_subreg_vert(vert_nii, subreg_nii, subreg_id=locs, verbose=False)
        from TPTBox.spine.spinestats import angles

        a = angles.compute_angel_between_two_points_(poi, label, label + 1, "R", project_2D=True)
        assert a is not None
        assert abs(a - 5) < 0.1, a
        b = angles.compute_angel_between_two_points_(poi, label, label + 1, "R", project_2D=False)
        assert b is not None
        assert abs(b - 5.06) < 0.1, b
        a = angles.compute_angel_between_two_points_(poi, label, label + 1, "P", project_2D=True)
        assert a is not None
        assert abs(a - 6.7) < 0.1, a
        b = angles.compute_angel_between_two_points_(poi, label, label + 1, "P", project_2D=False)
        assert b is not None
        assert abs(b - 6.7) < 0.1, b

        _, subreg_nii, vert_nii, label = get_test_ct()
        locs = [Location.Vertebra_Direction_Inferior]
        poi = calc_poi_from_subreg_vert(vert_nii, subreg_nii, subreg_id=locs, verbose=False)
        a = angles.compute_angel_between_two_points_(poi, label, label + 1, "R", project_2D=True)
        assert a is not None
        assert abs(a - 6.77) < 0.1, a
        b = angles.compute_angel_between_two_points_(poi, label, label + 1, "R", project_2D=False)
        assert b is not None
        assert abs(b - 6.8) < 0.1, b
        a = angles.compute_angel_between_two_points_(poi, label, label + 1, "P", project_2D=True)
        assert a is not None
        assert abs(a - 16.8) < 0.1, a
        b = angles.compute_angel_between_two_points_(poi, label, label + 1, "P", project_2D=False)
        assert b is not None
        assert abs(b - 16.8) < 0.1, b

    def _test_deformable(self, dim=0, save=True):  # type: ignore
        try:
            import deepali
            import torch
        except Exception:
            return

        from TPTBox.registration.deformable.deformable_reg import Deformable_Registration

        test_save = get_tests_dir() / f"deformation_{dim}.pkl"
        mri, subreg_nii, vert_nii, label = get_test_mri()
        mov = mri.copy()
        mov[mov != 0] = 0
        if dim == 0:
            mov[:-3, :, :] = mov[3:, :, :]
        elif dim == 1:
            mov[:, :-3, :] = mov[:, 3:, :]
        else:
            mov[:, :, :-3] = mov[:, :, 3:]

        poi = mri.make_empty_POI()
        poi[123, 44] = (random.randint(0, mri.shape[0] - 1), random.randint(0, mri.shape[1] - 1), random.randint(0, mri.shape[2] - 1))
        poi[123, 45] = (random.randint(0, mri.shape[0] - 1), random.randint(0, mri.shape[1] - 1), random.randint(0, mri.shape[2] - 1))

        deform = Deformable_Registration(mov, mov, reference_image=mov)
        if save:
            deform.save(test_save)
            deform = Deformable_Registration.load(test_save)
        mov2 = mov.copy()
        mov2.seg = True
        mov2[mov > -10000] = 0
        mov2[int(poi[123, 44][0]), int(poi[123, 44][1]), int(poi[123, 44][2])] = 44
        mov2[int(poi[123, 45][0]), int(poi[123, 45][1]), int(poi[123, 45][2])] = 45
        out = deform.transform_nii(mov2)
        poi = deform.transform_poi(poi)

        for idx in [44, 45]:
            x = tuple([float(x.item()) for x in np.where(out == idx)])
            y = poi.round(1).resample_from_to(mov)[123, idx]
            assert x == y, (x, y)

        test_save.unlink(missing_ok=True)

    def test_deformable(self):
        self._test_deformable(dim=0, save=False)
        self._test_deformable(dim=1, save=False)
        self._test_deformable(dim=2, save=False)

    def test_deformable_saved(self):
        self._test_deformable(dim=0, save=True)

    def test_calc_POI_from_subreg_vert_(self):
        idxs = [
            Location.Vertebra_Full,
            Location.Arcus_Vertebrae,
            Location.Spinosus_Process,
            Location.Costal_Process_Left,
            Location.Costal_Process_Right,
            Location.Superior_Articular_Left,
            Location.Superior_Articular_Right,
            Location.Inferior_Articular_Left,
            Location.Inferior_Articular_Right,
            Location.Vertebra_Corpus,
            # Location.Dens_axis,
            # Location.Vertebral_Body_Endplate_Superior,
            # Location.Vertebral_Body_Endplate_Inferior,
            Location.Vertebra_Disc_Superior,
            Location.Vertebra_Disc_Inferior,
            Location.Vertebra_Disc,
            Location.Spinal_Cord,
            Location.Spinal_Canal,
            Location.Spinal_Canal_ivd_lvl,
            # Location.Endplate,
            Location.Muscle_Inserts_Spinosus_Process,
            Location.Muscle_Inserts_Transverse_Process_Left,
            Location.Muscle_Inserts_Transverse_Process_Right,
            Location.Muscle_Inserts_Vertebral_Body_Left,
            Location.Muscle_Inserts_Vertebral_Body_Right,
            Location.Muscle_Inserts_Articulate_Process_Inferior_Left,
            Location.Muscle_Inserts_Articulate_Process_Inferior_Right,
            Location.Muscle_Inserts_Articulate_Process_Superior_Left,
            Location.Muscle_Inserts_Articulate_Process_Superior_Right,
            # Location.Implant_Entry_Left,
            # Location.Implant_Entry_Right,
            # Location.Implant_Target_Left,
            # Location.Implant_Target_Right,
            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Median,
            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Median,
            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median,
            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median,
            Location.Additional_Vertebral_Body_Middle_Superior_Median,
            Location.Additional_Vertebral_Body_Posterior_Central_Median,
            Location.Additional_Vertebral_Body_Middle_Inferior_Median,
            Location.Additional_Vertebral_Body_Anterior_Central_Median,
            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Left,
            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Left,
            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Left,
            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Left,
            Location.Additional_Vertebral_Body_Middle_Superior_Left,
            Location.Additional_Vertebral_Body_Posterior_Central_Left,
            Location.Additional_Vertebral_Body_Middle_Inferior_Left,
            Location.Additional_Vertebral_Body_Anterior_Central_Left,
            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Superior_Right,
            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Superior_Right,
            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Right,
            Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Right,
            Location.Additional_Vertebral_Body_Middle_Superior_Right,
            Location.Additional_Vertebral_Body_Posterior_Central_Right,
            Location.Additional_Vertebral_Body_Middle_Inferior_Right,
            Location.Additional_Vertebral_Body_Anterior_Central_Right,
            Location.Ligament_Attachment_Point_Flava_Superior_Median,
            Location.Ligament_Attachment_Point_Flava_Inferior_Median,
            Location.Vertebra_Direction_Posterior,
            Location.Vertebra_Direction_Inferior,
            Location.Vertebra_Direction_Right,
        ]
        for idx in idxs:
            _, subreg_nii, vert_nii, label = get_test_mri()
            out = calc_poi_from_subreg_vert(vert_nii, subreg_nii, buffer_file=None, decimals=3, subreg_id=idx)
            if idx.value not in out.keys_subregion():
                raise ValueError(idx, "missing")
