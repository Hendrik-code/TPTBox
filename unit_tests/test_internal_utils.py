"""Unit tests for `TPTBox.core.internal` helpers, `sitk_utils` and `optional_deps`.

These are the small plumbing layers the rest of the package builds on: dtype
coercion for the NIfTI-1 header, the backup-on-save decorator, the SimpleITK
metadata conversion, Slicer terminology strings, and the stubs that turn a missing
optional backend into an actionable ImportError.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox import NII
from TPTBox.core.internal.nii_help import (
    _arr_for_nifti1,
    _convert,
    _nifti_safe_dtype,
    save_json,
    secure_save,
)
from TPTBox.core.internal.optional_deps import _message, missing_dependency_class, missing_dependency_func
from TPTBox.core.internal.slicer_nrrd import (
    _generate_unique_segment_id,
    _terminology_entry_from_string,
    _terminology_entry_to_string,
    remove_not_supported_values,
)
from TPTBox.core.sitk_utils import (
    get_ras_affine_from_sitk,
    get_ras_affine_from_sitk_meta,
    get_sitk_metadata_from_ras_affine,
    nib_to_sitk,
    nii_to_sitk,
    sitk_to_nib,
    sitk_to_nii,
)


def _nii(arr=None, zoom=(1.5, 2.0, 3.0), origin=(10.0, -5.0, 2.0), seg=True) -> NII:
    arr = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6) if arr is None else arr
    affine = np.diag([*zoom, 1.0])
    affine[:3, 3] = origin
    return NII(nib.Nifti1Image(arr, affine), seg=seg)


class TestNiftiDtypeHelpers(unittest.TestCase):
    def test_supported_dtypes_pass_through(self):
        for dtype in (np.uint8, np.int16, np.int32, np.float32, np.float64):
            with self.subTest(dtype=np.dtype(dtype).name):
                self.assertEqual(_nifti_safe_dtype(dtype), np.dtype(dtype))

    def test_unsupported_dtypes_are_upcast(self):
        self.assertEqual(_nifti_safe_dtype(np.float16), np.dtype(np.float32))
        self.assertEqual(_nifti_safe_dtype(bool), np.dtype(np.uint8))

    def test_arr_for_nifti1_only_copies_when_needed(self):
        safe = np.zeros((2, 2, 2), dtype=np.uint8)
        self.assertIs(_arr_for_nifti1(safe), safe, "a supported dtype must not be copied")

        unsafe = np.zeros((2, 2, 2), dtype=bool)
        out = _arr_for_nifti1(unsafe)
        self.assertEqual(out.dtype, np.dtype(np.uint8))
        self.assertEqual(out.shape, unsafe.shape)

    def test_arr_for_nifti1_preserves_values(self):
        arr = np.array([[[True, False]]])
        np.testing.assert_array_equal(_arr_for_nifti1(arr), [[[1, 0]]])


class TestJsonConvert(unittest.TestCase):
    def test_numpy_scalars_and_arrays(self):
        self.assertEqual(_convert(np.int64(3)), 3)
        self.assertIsInstance(_convert(np.int64(3)), int)
        self.assertAlmostEqual(_convert(np.float32(1.5)), 1.5, places=6)
        self.assertEqual(_convert(np.arange(3)), [0, 1, 2])

    def test_paths_become_absolute_strings(self):
        out = _convert(Path("rel/path"))
        self.assertIsInstance(out, str)
        self.assertTrue(Path(out).is_absolute())

    def test_unsupported_types_raise(self):
        with self.assertRaises(TypeError):
            _convert(object())

    def test_save_json_writes_numpy_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            save_json(path, {"i": np.int32(1), "arr": np.zeros(2), "p": Path(tmp)})
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["i"], 1)
            self.assertEqual(loaded["arr"], [0.0, 0.0])

    def test_save_json_keeps_a_backup_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            save_json(path, {"first": 1})
            with self.assertRaises(TypeError):
                save_json(path, {"bad": object()})
            # The original content must survive a failed write.
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"first": 1})


class TestSecureSave(unittest.TestCase):
    class _Dummy:
        def __init__(self):
            self.calls = []

        @secure_save
        def save(self, file, payload="x"):
            Path(file).write_text(payload, encoding="utf-8")
            self.calls.append(file)

        @secure_save
        def failing_save(self, file, payload="x"):  # noqa: ARG002
            Path(file).write_text("partial", encoding="utf-8")
            raise RuntimeError("boom")

    def test_saves_through_to_the_wrapped_function(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.nii.gz"
            self._Dummy().save(target, payload="hello")
            self.assertEqual(target.read_text(encoding="utf-8"), "hello")
            self.assertEqual(list(Path(tmp).glob("*.backup")), [])

    def test_restores_the_previous_file_when_the_save_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.nii.gz"
            dummy = self._Dummy()
            dummy.save(target, payload="original")
            with self.assertRaises(RuntimeError):
                dummy.failing_save(target)
            self.assertEqual(target.read_text(encoding="utf-8"), "original", "the backup must be restored")
            self.assertEqual(list(Path(tmp).glob("*.backup")), [], "no backup file is left behind")

    def test_a_bids_file_without_a_supported_type_is_rejected(self):
        # The file_types list is only consulted for BIDS_FILE inputs; a plain path is
        # passed through whatever its extension is.
        from TPTBox.core.bids_files import BIDS_FILE  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "dataset-x"
            path = ds / "rawdata" / "sub-1" / "sub-1_ct.png"
            path.parent.mkdir(parents=True)
            path.touch()
            bids = BIDS_FILE(path, ds, verbose=False)
            with self.assertRaises(ValueError):
                self._Dummy().save(bids)

    def test_a_plain_path_with_any_extension_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.whatever"
            self._Dummy().save(target, payload="ok")
            self.assertEqual(target.read_text(encoding="utf-8"), "ok")


class TestSitkConversion(unittest.TestCase):
    def test_nii_to_sitk_round_trip_preserves_array_and_affine(self):
        for seg in (True, False):
            with self.subTest(seg=seg):
                nii = _nii(seg=seg)
                back = sitk_to_nii(nii_to_sitk(nii), seg=seg)
                np.testing.assert_array_equal(back.get_array(), nii.get_array())
                np.testing.assert_allclose(back.affine, nii.affine, atol=1e-6)
                self.assertEqual(back.orientation, nii.orientation)

    def test_nib_to_sitk_round_trip(self):
        nii = _nii()
        back = sitk_to_nib(nib_to_sitk(nii.nii))
        np.testing.assert_allclose(back.affine, nii.affine, atol=1e-6)

    def test_metadata_round_trip(self):
        nii = _nii()
        # Note the argument order: the getter returns (origin, spacing, direction) while
        # the builder takes (spacing, direction, origin).
        origin, spacing, direction = get_sitk_metadata_from_ras_affine(nii.affine)
        rebuilt = get_ras_affine_from_sitk_meta(spacing, direction, origin)
        np.testing.assert_allclose(rebuilt, nii.affine, atol=1e-6)

    def test_metadata_matches_what_simpleitk_reports(self):
        nii = _nii()
        origin, spacing, direction = get_sitk_metadata_from_ras_affine(nii.affine)
        image = nii_to_sitk(nii)
        np.testing.assert_allclose(image.GetOrigin(), origin, atol=1e-6)
        np.testing.assert_allclose(image.GetSpacing(), spacing, atol=1e-6)
        np.testing.assert_allclose(image.GetDirection(), direction, atol=1e-6)

    def test_an_unsupported_direction_length_raises(self):
        with self.assertRaises(NotImplementedError):
            get_ras_affine_from_sitk_meta((1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_affine_from_a_sitk_image(self):
        nii = _nii()
        np.testing.assert_allclose(get_ras_affine_from_sitk(nii_to_sitk(nii)), nii.affine, atol=1e-6)

    def test_spacing_matches_the_zoom(self):
        nii = _nii(zoom=(1.5, 2.0, 3.0))
        image = nii_to_sitk(nii)
        np.testing.assert_allclose(sorted(image.GetSpacing()), sorted(nii.zoom), atol=1e-6)

    def test_reoriented_input_round_trips_too(self):
        nii = _nii().reorient(("P", "I", "R"))
        back = sitk_to_nii(nii_to_sitk(nii), seg=True)
        np.testing.assert_array_equal(back.get_array(), nii.get_array())
        np.testing.assert_allclose(back.affine, nii.affine, atol=1e-6)


class TestSlicerTerminology(unittest.TestCase):
    ENTRY = (
        "Segmentation category and type - 3D Slicer General Anatomy list"
        "~SCT^123037004^Anatomical Structure~SCT^113197003^Rib~^^~Anatomic codes - DICOM master list"
        "~^^~^^"
    )

    def test_round_trip_through_the_string_form(self):
        parsed = _terminology_entry_from_string(self.ENTRY)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(_terminology_entry_to_string(parsed), self.ENTRY)

    def test_parsed_entry_exposes_the_type_name(self):
        parsed = _terminology_entry_from_string(self.ENTRY)
        self.assertIn("Rib", json.dumps(parsed))

    def test_generate_unique_segment_id_avoids_collisions(self):
        taken = {"Segment_1", "Segment_2"}
        new = _generate_unique_segment_id(taken)
        self.assertNotIn(new, taken)
        self.assertNotIn(_generate_unique_segment_id(taken | {new}), taken | {new})

    def test_generate_unique_segment_id_on_an_empty_set(self):
        self.assertTrue(_generate_unique_segment_id(set()))

    def test_remove_not_supported_values_mutates_in_place(self):
        payload = {"keep": 1, "space origin": None, "nested": {"a": None, "b": 2}}
        out = remove_not_supported_values(payload)
        self.assertIsNone(out, "the helper mutates in place and returns None")
        self.assertIn("keep", payload)


class TestOptionalDeps(unittest.TestCase):
    def test_message_names_the_extra_when_there_is_one(self):
        text = _message("do_thing", "reg", "torch hf-deepali", "No module named 'torch'", "()")
        self.assertIn("TPTBox[reg]", text)
        self.assertIn("pip install torch hf-deepali", text)
        self.assertIn("No module named 'torch'", text)

    def test_message_without_an_extra_points_at_the_packages(self):
        text = _message("do_thing", None, "spineps", "No module named 'spineps'", "()")
        self.assertNotIn("TPTBox[", text)
        self.assertIn("pip install spineps", text)

    def test_func_stub_raises_on_call_not_on_creation(self):
        stub = missing_dependency_func("run_thing", ImportError("No module named 'torch'"), "seg", "torch")
        with self.assertRaises(ImportError) as ctx:
            stub(1, keyword=2)
        self.assertIn("run_thing", str(ctx.exception))
        self.assertIn("TPTBox[seg]", str(ctx.exception))

    def test_class_stub_raises_on_instantiation(self):
        stub = missing_dependency_class("Thing", ImportError("No module named 'deepali'"), "reg", "hf-deepali")
        self.assertTrue(isinstance(stub, type))
        with self.assertRaises(ImportError) as ctx:
            stub()
        self.assertIn("Thing", str(ctx.exception))
        self.assertIn("hf-deepali", str(ctx.exception))

    def test_stub_message_without_an_extra(self):
        stub = missing_dependency_func("run_spineps", ImportError("nope"), None, "spineps")
        with self.assertRaises(ImportError) as ctx:
            stub()
        self.assertNotIn("TPTBox[", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
