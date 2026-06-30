# Call 'python -m pytest unit_tests/test_dicom.py'
"""Unit tests for the five modules in ``TPTBox.core.dicom``.

No DICOM sample ships with the repository, so a synthetic axial series is generated
on the fly from the sample CT via :mod:`TPTBox.core.dicom.nii2dicom` and round-tripped
back through the extraction pipeline. The whole module is guarded on ``pydicom`` being
importable (the dicom submodules import ``pydicom``/``dicom2nifti`` at module level).
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from TPTBox import NII
from TPTBox.tests.test_utils import get_test_ct

try:
    import pydicom  # noqa: F401
    from pydicom.dataset import Dataset

    from TPTBox.core.dicom import dicom_extract, fix_brocken
    from TPTBox.core.dicom.dicom2nii_utils import __test_nii as _priv_test_nii
    from TPTBox.core.dicom.dicom2nii_utils import (
        _get_json_from_dicom,
        clean_dicom_data,
        get_json_from_dicom,
        load_json,
        replace_birthdate_with_age,
        save_json,
    )
    from TPTBox.core.dicom.dicom2nii_utils import test_name_conflict as _name_conflict  # aliased: avoid pytest collecting it
    from TPTBox.core.dicom.dicom_header_to_keys import extract_keys_from_json, get_plane_dicom
    from TPTBox.core.dicom.nii2dicom import nifti2dicom_1file, nifti2dicom_mfiles

    has_pydicom = True
except Exception:
    has_pydicom = False


# --------------------------------------------------------------------------------------
# Helpers (kept local to this test file; unit_tests/ is not part of the coverage metric)
# --------------------------------------------------------------------------------------
def _quiet():
    """Context manager that swallows the (chatty) Print_Logger output to stdout."""
    return contextlib.redirect_stdout(io.StringIO())


def _save_input_ct(directory: Path) -> Path:
    """Save the sample CT into *directory* as ``input.nii.gz`` and return the path."""
    in_nii = directory / "input.nii.gz"
    with _quiet():
        get_test_ct()[0].save(in_nii)
    return in_nii


def _default_meta() -> dict:
    """A tiny BIDSy sidecar (DICOM keyword keys) used to label the generated series."""
    return {
        "Modality": "CT",
        "PatientID": "TESTPAT",
        "SeriesDescription": "abdomen",
        "SeriesNumber": 7,
        "PatientBirthDate": "19800101",
        "StudyDate": "20200101",
    }


def _generate_dicom_series(directory: Path, meta: dict | None = None, no_json_ok: bool = False, **kwargs) -> Path:
    """Generate a DICOM series from the sample CT into ``directory/'dcm'``.

    A sidecar ``input.json`` is written when *meta* is given; otherwise the no-json path
    is exercised. Returns the directory that holds the ``.dcm`` slices.
    """
    in_nii = _save_input_ct(directory)
    if meta is not None:
        (directory / "input.json").write_text(json.dumps(meta))
    dcm_dir = directory / "dcm"
    with _quiet():
        nifti2dicom_1file(in_nii, dcm_dir, no_json_ok=no_json_ok, **kwargs)
    return dcm_dir


@unittest.skipIf(not has_pydicom, "requires pydicom")
class TestNii2Dicom(unittest.TestCase):
    def test_nifti2dicom_1file_with_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            dcm_dir = _generate_dicom_series(Path(tmp), _default_meta())
            files = sorted(dcm_dir.glob("*.dcm"))
            # one DICOM file is written per axial slice of the volume
            self.assertGreater(len(files), 3)
            d0 = pydicom.dcmread(files[0])
            self.assertEqual(d0.Modality, "CT")
            self.assertEqual(d0.PatientID, "TESTPAT")
            self.assertTrue(hasattr(d0, "ImageOrientationPatient"))

    def test_nifti2dicom_1file_no_json_secondary(self):
        with tempfile.TemporaryDirectory() as tmp:
            # no_json_ok + secondary + custom slice prefix; no sidecar written
            dcm_dir = _generate_dicom_series(Path(tmp), meta=None, no_json_ok=True, secondary=True, out_name="sl")
            files = sorted(dcm_dir.glob("sl*.dcm"))
            self.assertGreater(len(files), 3)
            d0 = pydicom.dcmread(files[0])
            # default modality is MR when no sidecar provides one
            self.assertEqual(d0.Modality, "MR")
            self.assertIn("DERIVED", str(d0.ImageType))

    def test_nifti2dicom_1file_missing_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_nii = _save_input_ct(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                nifti2dicom_1file(in_nii, Path(tmp, "out"), no_json_ok=False)

    def test_nifti2dicom_1file_explicit_json_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            in_nii = _save_input_ct(td)
            jp = td / "meta.json"
            jp.write_text(json.dumps({"Modality": "CT", "SeriesNumber": 3}))
            out_dir = td / "out"
            with _quiet():
                nifti2dicom_1file(in_nii, out_dir, json_path=jp)
            self.assertGreater(len(list(out_dir.glob("*.dcm"))), 3)

    def test_nifti2dicom_mfiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            mdir = td / "niftis"
            mdir.mkdir()
            with _quiet():
                get_test_ct()[0].save(mdir / "aa.nii.gz")
            (mdir / "aa.json").write_text(json.dumps({"Modality": "CT", "SeriesNumber": 1}))
            out_dir = td / "out"
            with _quiet():
                # nifti2dicom_mfiles concatenates strings -> pass str paths
                nifti2dicom_mfiles(str(mdir), str(out_dir))
            self.assertGreater(len(list(out_dir.rglob("*.dcm"))), 3)


@unittest.skipIf(not has_pydicom, "requires pydicom")
class TestDicom2NiiUtils(unittest.TestCase):
    def test_replace_birthdate_with_age(self):
        d = {"00100030": {"vr": "DA", "Value": ["19900615"]}, "00080020": {"vr": "DA", "Value": ["20200615"]}}
        out = replace_birthdate_with_age(dict(d))
        self.assertEqual(out["00101010"]["Value"][0], "030Y")
        self.assertEqual(out["00101010"]["vr"], "AS")
        self.assertNotIn("00100030", out)

    def test_replace_birthdate_no_birth(self):
        d = {"foo": 1}
        self.assertEqual(replace_birthdate_with_age(dict(d)), d)

    def test_replace_birthdate_invalid(self):
        d = {"00100030": {"Value": ["nota-date"]}}
        out = replace_birthdate_with_age(dict(d))
        self.assertIn("00100030", out)
        self.assertNotIn("00101010", out)

    def test_replace_birthdate_no_study_uses_today(self):
        out = replace_birthdate_with_age({"00100030": {"Value": ["19900615"]}})
        self.assertIsNotNone(re.fullmatch(r"\d{3}Y", out["00101010"]["Value"][0]))
        self.assertNotIn("00100030", out)

    def test_clean_dicom_data_strips_pixeldata(self):
        ds = Dataset()
        ds.PatientID = "P"
        ds.Modality = "CT"
        ds.PixelData = b"\x00\x01\x02\x03"
        out = clean_dicom_data(ds)
        self.assertNotIn("7FE00010", out)  # PixelData tag
        self.assertEqual(out["00100020"]["Value"][0], "P")

    def test_get_json_from_dicom_single_and_list(self):
        ds = Dataset()
        ds.PatientID = "P1"
        ds.Modality = "CT"
        ds.SeriesDescription = "abd"
        ds.PatientBirthDate = "19800101"
        ds.StudyDate = "20200101"
        single = get_json_from_dicom(ds)
        self.assertEqual(single["Modality"], "CT")
        self.assertEqual(single["PatientAge"], "040Y")
        self.assertNotIn("PatientBirthDate", single)
        # a list of slices uses only the first element
        self.assertEqual(get_json_from_dicom([ds, ds])["PatientID"], "P1")

    def test_get_json_from_dicom_nested_sequence(self):
        nested = {"00081140": {"vr": "SQ", "Value": [{"0020000E": {"vr": "UI", "Value": ["1.2.3"]}}]}}
        out = _get_json_from_dicom(nested)
        self.assertEqual(out["ReferencedImageSequence"][0]["SeriesInstanceUID"], "1.2.3")
        # empty-value and multi-value (non-dict) list branches
        self.assertEqual(_get_json_from_dicom({"00080060": {"Value": []}}), {"Modality": []})
        self.assertEqual(_get_json_from_dicom({"00280030": {"Value": [1.5, 2.5]}}), {"PixelSpacing": [1.5, 2.5]})
        # an unknown / non-keyword tag is skipped
        self.assertEqual(_get_json_from_dicom({"AABBCCDD": {"Value": [1]}}), {})

    def test_save_and_load_json_numpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "x.json")
            with _quiet():
                # numpy scalars must be converted by the custom default
                existed = save_json({"a": np.int64(3), "b": np.float64(1.5)}, p)
            self.assertFalse(existed)
            self.assertEqual(load_json(p), {"a": 3, "b": 1.5})

    def test_save_json_override_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "x.json")
            with _quiet():
                save_json({"a": 1}, p)
                # second write skipped because override=False and file exists
                self.assertTrue(save_json({"a": 2}, p, override=False))
            self.assertEqual(load_json(p), {"a": 1})

    def test_save_json_check_exist_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "x.json")
            with _quiet():
                save_json({"a": 1}, p)
                with self.assertRaises(FileExistsError):
                    save_json({"a": 999}, p, check_exist=True)

    def test_name_conflict_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, "g.json")
            p.write_text(json.dumps({"a": 1, "grid": {"shape": [1]}}))
            # the "grid" key is stripped before comparison
            self.assertFalse(_name_conflict({"a": 1}, p))
            self.assertTrue(_name_conflict({"a": 2}, p))
            self.assertFalse(_name_conflict({"a": 1}, Path(tmp, "missing.json")))


@unittest.skipIf(not has_pydicom, "requires pydicom")
class TestDicomHeaderToKeys(unittest.TestCase):
    def test_get_plane_dicom_nii(self):
        base = get_test_ct()[0].reorient(("R", "A", "S"))
        for plane, zoom in [("ax", (1, 1, 3)), ("sag", (3, 1, 1)), ("cor", (1, 3, 1))]:
            with self.subTest(plane=plane):
                nii = base.copy().rescale(zoom)
                self.assertEqual(get_plane_dicom(nii), plane)
        # an isotropic CT reports 'iso'
        self.assertEqual(get_plane_dicom(get_test_ct()[0]), "iso")

    def test_get_plane_dicom_failure_returns_none(self):
        self.assertIsNone(get_plane_dicom([object()]))
        self.assertIsNone(get_plane_dicom([]))

    def test_extract_keys_modalities(self):
        ct = get_test_ct()[0]
        cases = [
            ({"Modality": "CT", "SeriesDescription": "abd"}, "ct", ".nii.gz"),
            ({"Modality": "MR", "SeriesDescription": "t2_tse_sag"}, "T2w", ".nii.gz"),
            ({"Modality": "MR", "SeriesDescription": "t1_tse"}, "T1w", ".nii.gz"),
            ({"Modality": "PT", "SeriesDescription": "pet"}, "pet", ".nii.gz"),
            ({"Modality": "MR", "SeriesDescription": "localizer"}, "localizer", ".nii.gz"),
            ({"Modality": "MR", "SeriesDescription": "whatever"}, "mr", ".nii.gz"),
        ]
        for simp, fmt, ending in cases:
            with self.subTest(desc=simp["SeriesDescription"]):
                full = {"PatientID": "P", "SeriesNumber": 5, **simp}
                mri_format, keys, end = extract_keys_from_json(full, ct)
                self.assertEqual(mri_format, fmt)
                self.assertEqual(end, ending)
                self.assertEqual(keys["sub"], "P")

    def test_extract_keys_t1w_subtraction_and_contrast(self):
        ct = get_test_ct()[0]
        # T1w + "sub" in description -> part=subtraction; " km " -> contrast agent
        _, keys, _ = extract_keys_from_json({"Modality": "MR", "SeriesDescription": "t1_tse sub km ", "PatientID": "P"}, ct)
        self.assertEqual(keys.get("part"), "subtraction")
        self.assertEqual(keys.get("ce"), "ContrastAgent")

    def test_extract_keys_contrast_bolus(self):
        ct = get_test_ct()[0]
        _, keys, _ = extract_keys_from_json(
            {"Modality": "MR", "SeriesDescription": "t2", "PatientID": "P", "ContrastBolusTotalDose": 10}, ct
        )
        self.assertEqual(keys.get("ce"), "ContrastAgent")

    def test_extract_keys_xa_angiography(self):
        ct = get_test_ct()[0]
        cases = [
            ({"SeriesDescription": "Durchleuchtung"}, "fluroscopy"),
            ({"DerivationDescription": "subtraction", "PositionerMotion": "static"}, "DSA"),
            ({"PositionerMotion": "dynamic"}, "DSA3D"),
            ({"SeriesDescription": "angio run"}, "XA"),
        ]
        for extra, fmt in cases:
            with self.subTest(fmt=fmt):
                simp = {"Modality": "XA", "PatientID": "P", "ImageType": [], **extra}
                mri_format, _keys, _end = extract_keys_from_json(simp, ct)
                self.assertEqual(mri_format, fmt)

    def test_extract_keys_custom_mapping(self):
        # Exercises the custom map_series_description_to_file_format loop. NOTE (source quirk):
        # the internal `found` flag is never set True, so the default map always runs afterwards
        # and overrides the custom result -> a custom mapping never actually wins. We only assert
        # a valid format string is returned (current behavior), not that the custom value is used.
        ct = get_test_ct()[0]
        mri_format, _keys, _end = extract_keys_from_json(
            {"Modality": "MR", "SeriesDescription": "myspecial", "PatientID": "P"},
            ct,
            map_series_description_to_file_format={".*myspecial.*": "T2w"},
        )
        self.assertIsInstance(mri_format, str)

    def test_extract_keys_report_formats(self):
        ct = get_test_ct()[0]
        self.assertEqual(extract_keys_from_json({"Modality": "PDF", "PatientID": "P"}, ct)[::2], ("report", ".pdf"))
        self.assertEqual(
            extract_keys_from_json({"Modality": "SR", "PatientID": "P", "SeriesDescription": "rep"}, ct)[::2],
            ("report", ".txt"),
        )

    def test_extract_keys_unknown_modality_raises(self):
        with self.assertRaises(NotImplementedError):
            extract_keys_from_json({"Modality": "ZZ", "PatientID": "P"}, get_test_ct()[0])

    def test_extract_keys_no_patient_id_fallbacks(self):
        ct = get_test_ct()[0]
        # no PatientID -> StudyInstanceUID
        _, keys, _ = extract_keys_from_json({"Modality": "CT", "StudyInstanceUID": "1.2.3"}, ct)
        self.assertEqual(keys["sub"], "1-2-3")
        # no PatientID and no StudyInstanceUID -> composed from demographics
        _, keys2, _ = extract_keys_from_json({"Modality": "CT", "PatientSex": "M"}, ct)
        self.assertIn("M", keys2["sub"])

    def test_extract_keys_session_chunk_parts_override(self):
        ct = get_test_ct()[0]
        simp = {"Modality": "MR", "SeriesDescription": "t2_tse", "PatientID": "P", "StudyDate": "20200101", "SeriesNumber": 1}
        mri_format, keys, _ = extract_keys_from_json(
            simp, ct, session=True, parts=["fat", "water"], chunk=2, override_subject_name=lambda _j, _p: "OVERRIDE"
        )
        self.assertEqual(keys["sub"], "OVERRIDE")
        self.assertEqual(keys["ses"], "20200101")
        self.assertEqual(keys["part"], "fat-water")
        self.assertEqual(keys["chunk"], "2")

    def test_extract_keys_nako_pd(self):
        # NAKO study branch: only the 'PD' token is reachable (see test below / source note).
        simp = {
            "StudyDescription": "NAKO study",
            "PatientID": "123_ABC",
            "SeriesNumber": 42,
            "Modality": "MR",
            "SeriesDescription": "PD_FS_SPC_COR",
        }
        mri_format, keys, _ = extract_keys_from_json(simp, get_test_ct()[0])
        self.assertEqual(mri_format, "pd")
        self.assertEqual(keys["acq"], "iso")

    def test_extract_keys_nako_other_descriptions_raise(self):
        # Source quirk: _get() rewrites '_'->'-', so the 'T2_TSE'/'3D_GRE_TRA'/'ME_vibe'/'T2_HASTE'
        # substring checks never match -> these fall through to NotImplementedError. Documented here.
        for sd in ["T2_TSE_SAG_LWS", "3D_GRE_TRA_F", "ME_vibe_fatquant", "WEIRD"]:
            with self.subTest(sd=sd):
                simp = {"StudyDescription": "NAKO", "PatientID": "1", "SeriesNumber": 1, "Modality": "MR", "SeriesDescription": sd}
                with self.assertRaises(NotImplementedError):
                    extract_keys_from_json(simp, get_test_ct()[0])


@unittest.skipIf(not has_pydicom, "requires pydicom")
class TestDicomExtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.td = Path(cls._tmp.name)
        cls.dcm_dir = _generate_dicom_series(cls.td, _default_meta())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _series(self):
        return next(iter(dicom_extract._read_dicom_files(self.dcm_dir)[0].values()))

    def test_read_dicom_files(self):
        files_dict, parts = dicom_extract._read_dicom_files(self.dcm_dir)
        self.assertEqual(len(files_dict), 1)
        series = next(iter(files_dict.values()))
        self.assertGreater(len(series), 3)
        self.assertIsInstance(parts, dict)

    def test_classic_get_grouped_dicoms(self):
        series = self._series()
        grouped = dicom_extract._classic_get_grouped_dicoms(series)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(sum(len(g) for g in grouped), len(series))
        # tiny input (<=3 slices) collapses into the 'others' catch-all group
        small = [Dataset(), Dataset()]
        for i, d in enumerate(small):
            d.InstanceNumber = i
            d.ImagePositionPatient = [0.0, 0.0, float(i)]
        grouped_small = dicom_extract._classic_get_grouped_dicoms(small)
        self.assertEqual(len(grouped_small), 1)
        self.assertEqual(len(grouped_small[0]), 2)
        # two spatial stacks (5 slices along z at x=0, then 5 along z at x=100) -> split into 2 groups
        two_stack = []
        inst = 1
        for x in (0.0, 100.0):
            for z in range(5):
                d = Dataset()
                d.InstanceNumber = inst
                d.ImagePositionPatient = [x, 0.0, float(z)]
                two_stack.append(d)
                inst += 1
        grouped_two = dicom_extract._classic_get_grouped_dicoms(two_stack)
        self.assertEqual(len(grouped_two), 2)
        self.assertEqual(sorted(len(g) for g in grouped_two), [5, 5])

    def test_filter_dicom(self):
        series = self._series()
        self.assertEqual(len(dicom_extract._filter_dicom(series)), len(series))
        # single element is returned unchanged
        self.assertEqual(dicom_extract._filter_dicom([series[0]]), [series[0]])
        # multi-element drops datasets lacking ImageOrientationPatient
        with_o = Dataset()
        with_o.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        without = Dataset()
        self.assertEqual(dicom_extract._filter_dicom([with_o, without]), [with_o])

    def test_filter_file_type(self):
        out = dicom_extract._filter_file_type({"S1": ["ORIGINAL,PRIMARY,M", "ORIGINAL,PRIMARY,P"], "S2": ["ONLYONE"]})
        self.assertEqual(out["S1_ORIGINAL,PRIMARY,M"], ["M"])
        self.assertEqual(out["S1_ORIGINAL,PRIMARY,P"], ["P"])
        self.assertNotIn("S2_ONLYONE", out)

    def test_inc_key(self):
        k = {"sequ": "5"}
        dicom_extract._inc_key(k)
        self.assertEqual(k["sequ"], "6")
        k = {"sequ": "ax-3"}
        dicom_extract._inc_key(k)
        self.assertEqual(k["sequ"], "ax-4")
        k = {}
        dicom_extract._inc_key(k)
        self.assertEqual(k["sequ"], "1")

    def test_find_all_files(self):
        with _quiet():
            found = list(dicom_extract._find_all_files(self.dcm_dir, verbose=True))
        self.assertGreater(len(found), 1)

    def test_generate_bids_path(self):
        with tempfile.TemporaryDirectory() as out:
            json_name, fname = dicom_extract._generate_bids_path(Path(out), {"sub": "P1", "acq": "ax"}, "ct", {}, 0)
            self.assertTrue(str(json_name).endswith("_ct.json"))
            self.assertIn("sub-P1", str(json_name))
            self.assertEqual(fname.bids_format, "ct")

    def test_add_grid_info_to_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            nii_p = td / "grid.nii.gz"
            with _quiet():
                get_test_ct()[0].save(nii_p)
            json_p = td / "grid.json"
            with _quiet():
                out = dicom_extract._add_grid_info_to_json(nii_p, json_p)
                # second call short-circuits because "grid" is already present
                out2 = dicom_extract._add_grid_info_to_json(nii_p, json_p)
            self.assertIn("grid", out)
            self.assertEqual(set(out["grid"].keys()), {"shape", "spacing", "orientation", "rotation", "origin", "dims"})
            self.assertIn("grid", out2)

    def test_unzip_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            zip_path = td / "series.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for f in sorted(self.dcm_dir.glob("*.dcm")):
                    zf.write(f, f.name)
            out = dicom_extract._unzip_files(zip_path, td / "unz")
            self.assertGreater(len(list(Path(out).rglob("*.dcm"))), 3)

    def test_extract_dicom_folder_end_to_end(self):
        ct = get_test_ct()[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with _quiet():
                res = dicom_extract.extract_dicom_folder(self.dcm_dir, out, n_cpu=1, verbose=False)
            self.assertGreater(len(res), 0)
            niis = list(out.rglob("*.nii.gz"))
            jsons = list(out.rglob("*.json"))
            self.assertEqual(len(niis), 1)
            o = NII.load(niis[0], False)
            self.assertEqual(len(o.shape), 3)
            self.assertEqual(int(np.prod(o.shape)), int(np.prod(ct.shape)))
            sidecar = json.loads(jsons[0].read_text())
            self.assertIn("grid", sidecar)
            # birthdate was converted to age during extraction
            self.assertEqual(sidecar.get("PatientAge"), "040Y")
            self.assertNotIn("PatientBirthDate", sidecar)
            # re-running hits the "already exists" early-return branch
            with _quiet():
                res2 = dicom_extract.extract_dicom_folder(self.dcm_dir, out, n_cpu=1, verbose=False)
            self.assertEqual(len(list(out.rglob("*.nii.gz"))), 1)
            self.assertGreater(len(res2), 0)

    def test_extract_dicom_folder_from_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            zip_path = td / "series.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for f in sorted(self.dcm_dir.glob("*.dcm")):
                    zf.write(f, f.name)
            out = td / "dataset"
            with _quiet():
                # also exercise the validation-disable branches
                res = dicom_extract.extract_dicom_folder(
                    zip_path,
                    out,
                    n_cpu=1,
                    verbose=False,
                    validate_slicecount=False,
                    validate_orientation=False,
                    validate_slice_increment=False,
                )
            self.assertGreater(len(res), 0)
            self.assertEqual(len(list(out.rglob("*.nii.gz"))), 1)


@unittest.skipIf(not has_pydicom, "requires pydicom")
class TestFixBrocken(unittest.TestCase):
    def test_test_nii_good(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp, "good.nii.gz")
            with _quiet():
                get_test_ct()[0].save(good)
            self.assertTrue(fix_brocken.test_nii(good))
            # passing a string path is also accepted
            self.assertTrue(fix_brocken.test_nii(str(good)))

    def test_test_nii_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            good = td / "good.nii.gz"
            with _quiet():
                get_test_ct()[0].save(good)
            corrupt = td / "bad.nii.gz"
            corrupt.write_bytes(good.read_bytes()[:200])  # truncated gzip
            self.assertFalse(fix_brocken.test_nii(corrupt))

    def test_test_nii_missing_is_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(fix_brocken.test_nii(Path(tmp, "does_not_exist.nii.gz")))

    def test_private_test_nii_in_utils(self):
        # the duplicate private helper in dicom2nii_utils mirrors fix_brocken.test_nii
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            good = td / "good.nii.gz"
            with _quiet():
                get_test_ct()[0].save(good)
            self.assertTrue(_priv_test_nii(good))
            self.assertTrue(_priv_test_nii(str(good)))  # str path is accepted
            corrupt = td / "bad.nii.gz"
            corrupt.write_bytes(good.read_bytes()[:200])
            self.assertFalse(_priv_test_nii(corrupt))


if __name__ == "__main__":
    unittest.main()
