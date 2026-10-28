"""Unit tests for `TPTBox.core.dicom`.

No DICOM sample ships with the repository, so the integration test generates one:
the sample CT is written out as a DICOM series with `nii2dicom` and read back with
`extract_dicom_folder`, which exercises the whole naming + conversion pipeline and
pins the BIDS path it produces. The pure string/dict/path helpers around it are
tested directly.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

HAS_PYDICOM = importlib.util.find_spec("pydicom") is not None and importlib.util.find_spec("dicom2nifti") is not None
PYDICOM_REASON = "requires the `dicom` extra (pydicom + dicom2nifti)"


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestNameHelpers(unittest.TestCase):
    def test_next_letter_suffix(self):
        from TPTBox.core.dicom.dicom_extract import _next_letter_suffix  # noqa: PLC0415

        cases = {"a": "b", "b": "c", "z": "aa", "aa": "ab", "az": "ba", "zz": "aaa"}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(_next_letter_suffix(value), expected)

    def test_next_letter_suffix_with_a_larger_increment(self):
        from TPTBox.core.dicom.dicom_extract import _next_letter_suffix  # noqa: PLC0415

        self.assertEqual(_next_letter_suffix("a", inc=2), "c")
        self.assertEqual(_next_letter_suffix("a", inc=26), "aa")

    def test_inc_key_appends_a_letter_suffix(self):
        from TPTBox.core.dicom.dicom_extract import _inc_key  # noqa: PLC0415

        keys = {"sub": "001", "sequ": "7"}
        _inc_key(keys)
        self.assertEqual(keys["sequ"], "7-a")
        _inc_key(keys)
        self.assertEqual(keys["sequ"], "7-b")

    def test_inc_key_keeps_going_until_the_path_is_free(self):
        from TPTBox.core.dicom.dicom_extract import _inc_key  # noqa: PLC0415

        taken = {"7-a", "7-b"}
        keys = {"sub": "001", "sequ": "7"}
        _inc_key(keys, path_exists=lambda k: k["sequ"] in taken)
        self.assertEqual(keys["sequ"], "7-c")

    def test_inc_key_on_a_custom_key(self):
        from TPTBox.core.dicom.dicom_extract import _inc_key  # noqa: PLC0415

        keys = {"sub": "001", "acq": "iso"}
        _inc_key(keys, k="acq")
        self.assertEqual(keys["acq"], "iso-a")

    def test_looks_like_dicom(self):
        from TPTBox.core.dicom.dicom_extract import _looks_like_dicom  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            text = folder / "notes.txt"
            text.write_text("hello", encoding="utf-8")
            self.assertFalse(_looks_like_dicom(text))

            dicom = folder / "slice.dcm"
            dicom.write_bytes(b"\0" * 128 + b"DICM" + b"\0" * 32)
            self.assertTrue(_looks_like_dicom(dicom))

    def test_with_echo_suffix_extends_the_part_entity(self):
        from TPTBox.core.dicom.dicom_extract import _with_echo_suffix  # noqa: PLC0415

        out = _with_echo_suffix(Path("/ds/sub-1_sequ-3_part-water_dixon.nii.gz"), 0)
        self.assertEqual(out.name, "sub-1_sequ-3_part-water-eco0_dixon.nii.gz")
        self.assertEqual(out.parent, Path("/ds"))

    def test_with_echo_suffix_falls_back_to_the_stem(self):
        from TPTBox.core.dicom.dicom_extract import _with_echo_suffix  # noqa: PLC0415

        for name, expected in (
            ("sub-1_sequ-3_mevibe.nii.gz", "sub-1_sequ-3_mevibe-eco2.nii.gz"),
            ("sub-1_sequ-3_mevibe.json", "sub-1_sequ-3_mevibe-eco2.json"),
        ):
            with self.subTest(name=name):
                self.assertEqual(_with_echo_suffix(Path("/ds") / name, 2).name, expected)

    def test_generate_bids_path_is_bids_shaped(self):
        from TPTBox.core.dicom.dicom_extract import _generate_bids_path  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            keys = {"sub": "TESTPAT", "ses": "20200101", "sequ": "7"}
            json_path, bids = _generate_bids_path(Path(tmp), keys, "ct", {"a": 1}, make_subject_chunks=0)
            rel = Path(json_path).relative_to(tmp).as_posix()
            self.assertEqual(rel, "rawdata/sub-TESTPAT/ses-20200101/ct/sub-TESTPAT_ses-20200101_sequ-7_ct.json")
            self.assertEqual(bids.bids_format, "ct")

    def test_generate_bids_path_make_subject_chunks(self):
        from TPTBox.core.dicom.dicom_extract import _generate_bids_path  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            json_path, _ = _generate_bids_path(Path(tmp), {"sub": "123456"}, "ct", {}, make_subject_chunks=3)
            self.assertIn("/123/sub-123456/", Path(json_path).as_posix())

    def test_generate_bids_path_sanitises_dots_in_the_subject(self):
        from TPTBox.core.dicom.dicom_extract import _generate_bids_path  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            _, bids = _generate_bids_path(Path(tmp), {"sub": "1.2.3"}, "ct", {}, make_subject_chunks=0)
            self.assertEqual(bids.get("sub"), "1-2-3")

    def test_filter_file_type_keeps_the_tokens_that_separate_sub_series(self):
        from TPTBox.core.dicom.dicom_extract import _filter_file_type  # noqa: PLC0415

        # Two ImageTypes in one series: "W" and "F" each appear once and therefore
        # identify their sub-series, while the shared tokens are dropped.
        out = _filter_file_type({"uid1": ["ORIGINAL,PRIMARY,W", "ORIGINAL,PRIMARY,F"]})
        self.assertEqual(sorted(out), ["uid1_ORIGINAL,PRIMARY,F", "uid1_ORIGINAL,PRIMARY,W"])
        self.assertEqual(out["uid1_ORIGINAL,PRIMARY,W"], ["W"])
        self.assertEqual(out["uid1_ORIGINAL,PRIMARY,F"], ["F"])

    def test_filter_file_type_skips_series_with_a_single_image_type(self):
        from TPTBox.core.dicom.dicom_extract import _filter_file_type  # noqa: PLC0415

        self.assertEqual(_filter_file_type({"uid1": ["ORIGINAL,PRIMARY"]}), {})


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestJsonHelpers(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        from TPTBox.core.dicom.dicom2nii_utils import load_json, save_json  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            payload = {"a": 1, "b": [1.5, 2.5], "c": "text"}
            self.assertFalse(save_json(payload, path))
            self.assertEqual(load_json(path), payload)

    def test_numpy_scalars_are_serialised(self):
        from TPTBox.core.dicom.dicom2nii_utils import load_json, save_json  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "np.json"
            save_json({"i": np.int64(3), "f": np.float32(1.5), "arr": np.arange(3)}, path)
            loaded = load_json(path)
            self.assertEqual(loaded["i"], 3)
            self.assertAlmostEqual(loaded["f"], 1.5, places=6)
            self.assertEqual(list(loaded["arr"]), [0, 1, 2])

    def test_override_false_keeps_the_existing_file(self):
        from TPTBox.core.dicom.dicom2nii_utils import load_json, save_json  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            save_json({"first": True}, path)
            self.assertTrue(save_json({"second": True}, path, override=False))
            self.assertEqual(load_json(path), {"first": True})

    def test_check_exist_raises_on_differing_content(self):
        from TPTBox.core.dicom.dicom2nii_utils import save_json  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            save_json({"first": True}, path)
            with self.assertRaises(FileExistsError):
                save_json({"second": True}, path, check_exist=True)

    def test_name_conflict_detects_only_real_differences(self):
        from TPTBox.core.dicom.dicom2nii_utils import save_json, test_name_conflict  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            self.assertFalse(test_name_conflict({"a": 1}, path))  # nothing there yet
            save_json({"a": 1}, path)
            self.assertFalse(test_name_conflict({"a": 1}, path))
            self.assertTrue(test_name_conflict({"a": 2}, path))

    def test_name_conflict_ignores_the_grid_key(self):
        from TPTBox.core.dicom.dicom2nii_utils import save_json, test_name_conflict  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            save_json({"a": 1, "grid": {"shape": [1, 2, 3]}}, path)
            self.assertFalse(test_name_conflict({"a": 1}, path))

    def test_name_conflict_replaces_unreadable_json(self):
        from TPTBox.core.dicom.dicom2nii_utils import test_name_conflict  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertFalse(test_name_conflict({"a": 1}, path))
            self.assertFalse(path.exists(), "a half-written json should be removed, not routed around")

    def test_replace_birthdate_with_age(self):
        from TPTBox.core.dicom.dicom2nii_utils import replace_birthdate_with_age  # noqa: PLC0415

        out = replace_birthdate_with_age({"00100030": {"Value": ["19800115"], "vr": "DA"}, "00080020": {"Value": ["20200114"], "vr": "DA"}})
        self.assertNotIn("00100030", out, "the birth date must be dropped")
        self.assertEqual(out["00101010"]["Value"], ["039Y"], "age on the day before the 40th birthday")

    def test_replace_birthdate_on_the_birthday_itself(self):
        from TPTBox.core.dicom.dicom2nii_utils import replace_birthdate_with_age  # noqa: PLC0415

        out = replace_birthdate_with_age({"00100030": {"Value": ["19800115"]}, "00080020": {"Value": ["20200115"]}})
        self.assertEqual(out["00101010"]["Value"], ["040Y"])

    def test_replace_birthdate_without_a_birthdate_is_a_noop(self):
        from TPTBox.core.dicom.dicom2nii_utils import replace_birthdate_with_age  # noqa: PLC0415

        payload = {"00080020": {"Value": ["20200115"]}}
        self.assertEqual(replace_birthdate_with_age(dict(payload)), payload)

    def test_replace_birthdate_with_unparsable_dates_is_a_noop(self):
        from TPTBox.core.dicom.dicom2nii_utils import replace_birthdate_with_age  # noqa: PLC0415

        payload = {"00100030": {"Value": ["not-a-date"]}, "00080020": {"Value": ["20200115"]}}
        self.assertEqual(replace_birthdate_with_age(dict(payload)), payload)


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestPlaneDetection(unittest.TestCase):
    """`get_plane_dicom` classifies an NII by its affine, with no DICOM involved."""

    @staticmethod
    def _nii(zoom):
        import nibabel as nib  # noqa: PLC0415

        from TPTBox import NII  # noqa: PLC0415

        return NII(nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.uint8), np.diag([*zoom, 1.0])), seg=False)

    def test_isotropic_and_anisotropic_planes(self):
        from TPTBox.core.dicom.dicom_header_to_keys import get_plane_dicom  # noqa: PLC0415

        cases = {
            (1.0, 1.0, 1.0): "iso",
            (0.5, 0.5, 6.0): "ax",
        }
        for zoom, expected in cases.items():
            with self.subTest(zoom=zoom):
                self.assertEqual(get_plane_dicom(self._nii(zoom)), expected)

    def test_each_thick_axis_gives_a_different_plane(self):
        from TPTBox.core.dicom.dicom_header_to_keys import get_plane_dicom  # noqa: PLC0415

        planes = {get_plane_dicom(self._nii(zoom)) for zoom in ((6.0, 0.5, 0.5), (0.5, 6.0, 0.5), (0.5, 0.5, 6.0))}
        self.assertEqual(len(planes), 3, planes)


def _save_sample_ct(folder: Path) -> Path:
    from TPTBox.tests.test_utils import get_test_ct  # noqa: PLC0415

    path = folder / "input.nii.gz"
    get_test_ct()[0].save(path, verbose=False)
    return path


def _meta() -> dict:
    return {
        "Modality": "CT",
        "PatientID": "TESTPAT",
        "SeriesDescription": "abdomen",
        "SeriesNumber": 7,
        "PatientBirthDate": "19800101",
        "StudyDate": "20200101",
    }


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestDicomRoundTrip(unittest.TestCase):
    """nii -> DICOM series -> BIDS nii, through the real conversion pipeline."""

    @classmethod
    def setUpClass(cls):
        from TPTBox.core.dicom.dicom_extract import extract_dicom_folder  # noqa: PLC0415
        from TPTBox.core.dicom.nii2dicom import nifti2dicom_1file  # noqa: PLC0415

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        nii_path = _save_sample_ct(root)
        (root / "input.json").write_text(json.dumps(_meta()), encoding="utf-8")
        cls.dcm_dir = root / "dcm"
        nifti2dicom_1file(nii_path, cls.dcm_dir)
        cls.slices = sorted(cls.dcm_dir.rglob("*.dcm"))

        cls.out_dir = root / "ds"
        cls.result = extract_dicom_folder(cls.dcm_dir, cls.out_dir, make_subject_chunks=0, verbose=False)
        cls.source = nii_path

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_one_dicom_slice_per_input_slice(self):
        from TPTBox import to_nii  # noqa: PLC0415

        self.assertEqual(len(self.slices), to_nii(self.source).shape[-1])

    def test_the_generated_series_is_readable_dicom(self):
        import pydicom  # noqa: PLC0415

        ds = pydicom.dcmread(str(self.slices[0]))
        self.assertEqual(ds.Modality, "CT")
        self.assertEqual(str(ds.PatientID), "TESTPAT")

    def test_extraction_produces_the_expected_bids_path(self):
        produced = sorted(p.relative_to(self.out_dir).as_posix() for p in self.out_dir.rglob("*") if p.is_file())
        self.assertIn("rawdata/sub-TESTPAT/ct/sub-TESTPAT_sequ-7_acq-iso_ct.nii.gz", produced)
        self.assertIn("rawdata/sub-TESTPAT/ct/sub-TESTPAT_sequ-7_acq-iso_ct.json", produced)

    def test_the_return_value_maps_series_uid_to_the_output(self):
        self.assertIsInstance(self.result, dict)
        self.assertEqual(len(self.result), 1)
        self.assertTrue(next(iter(self.result.values())).endswith(".nii.gz"))

    def test_the_round_tripped_volume_matches_the_input(self):
        from TPTBox import to_nii  # noqa: PLC0415

        source = to_nii(self.source)
        out = to_nii(self.out_dir / "rawdata/sub-TESTPAT/ct/sub-TESTPAT_sequ-7_acq-iso_ct.nii.gz")
        self.assertEqual(sorted(out.shape), sorted(source.shape))
        np.testing.assert_allclose(sorted(out.zoom), sorted(source.zoom), rtol=1e-3)
        self.assertAlmostEqual(float(out.max()), float(source.max()), delta=1.0)

    def test_the_sidecar_carries_grid_info_and_the_patient_age(self):
        sidecar = json.loads((self.out_dir / "rawdata/sub-TESTPAT/ct/sub-TESTPAT_sequ-7_acq-iso_ct.json").read_text())
        self.assertIn("grid", sidecar)
        flat = json.dumps(sidecar)
        self.assertNotIn("19800101", flat, "the birth date must not be written to disk")

    def test_a_second_extraction_is_a_cached_no_op(self):
        from TPTBox.core.dicom.dicom_extract import extract_dicom_folder  # noqa: PLC0415

        marker = list((self.out_dir / ".extract_cache").glob("*.json"))
        self.assertEqual(len(marker), 1)
        before = {p: p.stat().st_mtime_ns for p in self.out_dir.rglob("*.nii.gz")}
        extract_dicom_folder(self.dcm_dir, self.out_dir, make_subject_chunks=0, verbose=False)
        after = {p: p.stat().st_mtime_ns for p in self.out_dir.rglob("*.nii.gz")}
        self.assertEqual(before, after, "the extract cache should prevent a rewrite")

    def test_a_zipped_series_extracts_the_same_way(self):
        from TPTBox.core.dicom.dicom_extract import extract_dicom_folder  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "series.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for slice_path in self.slices:
                    zf.write(slice_path, arcname=slice_path.name)
            out = Path(tmp) / "ds"
            extract_dicom_folder(zip_path, out, make_subject_chunks=0, verbose=False)
            produced = [p.name for p in out.rglob("*.nii.gz")]
            self.assertEqual(produced, ["sub-TESTPAT_sequ-7_acq-iso_ct.nii.gz"])


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestDicomGrouping(unittest.TestCase):
    """Series/echo grouping works on in-memory datasets, no files involved."""

    @staticmethod
    def _ds(series_uid: str, echo: int | None = None, instance: int = 1):
        from pydicom.dataset import Dataset  # noqa: PLC0415

        ds = Dataset()
        ds.SeriesInstanceUID = series_uid
        ds.InstanceNumber = instance
        ds.ImagePositionPatient = [0.0, 0.0, float(instance)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        if echo is not None:
            ds.EchoNumbers = echo
        return ds

    def test_split_by_echo_numbers_returns_the_lowest_echo_for_stack_detection(self):
        from TPTBox.core.dicom.dicom_extract import _split_by_echo_numbers  # noqa: PLC0415

        data = [self._ds("1", echo=1, instance=1), self._ds("1", echo=1, instance=2), self._ds("1", echo=2, instance=3)]
        single, per_echo = _split_by_echo_numbers(data)
        self.assertEqual(len(single), 2, "the subset must be the lowest echo, one DICOM per slice")
        self.assertEqual(sorted(per_echo), [1, 2])
        self.assertEqual(sum(len(v) for v in per_echo.values()), len(data))

    def test_split_by_echo_numbers_is_a_noop_for_a_single_echo(self):
        from TPTBox.core.dicom.dicom_extract import _split_by_echo_numbers  # noqa: PLC0415

        for data in ([self._ds("1"), self._ds("1", instance=2)], [self._ds("1", echo=1), self._ds("1", echo=1, instance=2)]):
            with self.subTest(n_echo_tags=sum(hasattr(d, "EchoNumbers") for d in data)):
                single, per_echo = _split_by_echo_numbers(data)
                self.assertEqual(len(single), 2)
                self.assertEqual(per_echo, {})

    def test_split_by_echo_numbers_treats_a_missing_tag_as_echo_zero(self):
        from TPTBox.core.dicom.dicom_extract import _split_by_echo_numbers  # noqa: PLC0415

        data = [self._ds("1"), self._ds("1", echo=1, instance=2), self._ds("1", echo=2, instance=3)]
        _, per_echo = _split_by_echo_numbers(data)
        self.assertEqual(sorted(per_echo), [0, 1, 2])

    def test_filter_dicom_keeps_the_input_when_nothing_is_wrong(self):
        from TPTBox.core.dicom.dicom_extract import _filter_dicom  # noqa: PLC0415

        data = [self._ds("1"), self._ds("1", instance=2)]
        self.assertEqual(len(_filter_dicom(data)), 2)

    def test_clean_dicom_data_drops_pixel_data(self):
        from TPTBox.core.dicom.dicom2nii_utils import clean_dicom_data  # noqa: PLC0415

        ds = self._ds("1")
        ds.PatientID = "X"
        ds.PixelData = b"\x00\x01\x02\x03"
        out = clean_dicom_data(ds)
        self.assertNotIn("7FE00010", out, "PixelData must not survive into the sidecar")
        self.assertIn("00100020", out)  # PatientID
        self.assertTrue(hasattr(ds, "PixelData"), "the input dataset must not be modified")


if __name__ == "__main__":
    unittest.main()
