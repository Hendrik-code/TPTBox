"""Unit tests for the XA (angiography) handling in `TPTBox.core.dicom`.

Covers: choosing DSA / fluoroscopy / still from the image content, key frames (`rec-keyframe`),
the reproducible series order, ignoring instance-specific tags when two copies of a series are
compared, the optional run numbering and the thread turn-taking used for `n_cpu > 1`.
"""

from __future__ import annotations

import importlib.util
import random
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

HAS_PYDICOM = importlib.util.find_spec("pydicom") is not None and importlib.util.find_spec("dicom2nifti") is not None
PYDICOM_REASON = "requires the `dicom` extra (pydicom + dicom2nifti)"


def _xa_header(image_type, *, derivation="", frames=10, motion="STATIC", description="Zerebral 2 B/s Niedrig", **extra) -> dict:
    header = {
        "Modality": "XA",
        "PatientID": "p1",
        "SeriesNumber": 3,
        "ImageType": list(image_type),
        "SeriesDescription": description,
        "NumberOfFrames": frames,
        "ProtocolName": "Kopf",
    }
    if derivation:
        header["DerivationDescription"] = derivation
    if motion is not None:
        header["PositionerMotion"] = motion
    header.update(extra)
    return header


def _keys(header: dict, n_objects: int = 1) -> tuple[str, dict]:
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    from TPTBox.core.dicom import dicom_header_to_keys as m  # noqa: PLC0415

    with mock.patch.object(m, "get_plane_dicom", return_value="ax"), contextlib.redirect_stdout(io.StringIO()):
        fmt, keys, _ = m.extract_keys_from_json(header, [object()] * n_objects)
    return fmt, keys


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestXaFormat(unittest.TestCase):
    def test_format_and_rec_from_the_image_content(self):
        bi_a = ["DERIVED", "PRIMARY", "BIPLANE A", "SINGLE A"]
        cases = [
            # name, header, expected format, expected rec
            (
                "subtracted run stored as 'Durchleuchtung' is DSA",
                _xa_header(bi_a, derivation="SUBTRACTION", description="Durchleuchtung - gespeichert"),
                "DSA",
                None,
            ),
            (
                "native live fluoroscopy",
                _xa_header(["ORIGINAL", "PRIMARY", "BIPLANE A"], description="Durchleuchtung - gespeichert"),
                "fluroscopy",
                None,
            ),
            (
                "DERIVED+PRIMARY without SUBTRACTION and several frames is DSA",
                _xa_header(["DERIVED", "PRIMARY", "SINGLE PLANE", "SINGLE A"], frames=19, description="Carotid."),
                "DSA",
                None,
            ),
            ("key frame of a subtracted run", _xa_header(bi_a, derivation="SUBTRACTION;FRAME_SELECTION", frames=6), "DSA", "keyframe"),
            (
                "one-frame key frame is a still as well",
                _xa_header(bi_a, derivation="SUBTRACTION;FRAME_SELECTION", frames=1),
                "DSA-still",
                "keyframe",
            ),
            ("single subtracted image is a still", _xa_header(bi_a, derivation="SUBTRACTION", frames=1), "DSA-still", None),
            ("single native fluoro image is a still", _xa_header(["ORIGINAL", "PRIMARY", "BIPLANE B"], frames=1), "fluroscopy-still", None),
            ("missing PositionerMotion counts as static", _xa_header(bi_a, derivation="SUBTRACTION", motion=None), "DSA", None),
            (
                "explicitly moving positioner stays a subtraction volume",
                _xa_header(bi_a, derivation="SUBTRACTION", motion="DYNAMIC"),
                "subtraction",
                None,
            ),
            ("3D reconstruction", _xa_header(["DERIVED", "PRIMARY", "VOLUME"], motion="DYNAMIC"), "DSA3D", None),
            ("3DRA", _xa_header(["ORIGINAL", "PRIMARY", "3DRA_PROP"], motion="DYNAMIC"), "3DRA", None),
            ("moving native run is not fluoroscopy", _xa_header(["ORIGINAL", "PRIMARY", "BIPLANE A"], motion="DYNAMIC"), "DSA3D", None),
        ]
        for name, header, fmt_expected, rec_expected in cases:
            with self.subTest(name):
                fmt, keys = _keys(header)
                self.assertEqual(fmt, fmt_expected)
                self.assertEqual(keys.get("rec"), rec_expected)

    def test_helper_objects_are_routed_to_xa_helper(self):
        for marker in ("SECONDARY", "REFIMAGE", "EXAM PROTOCOL"):
            with self.subTest(marker):
                fmt, _ = _keys(_xa_header(["DERIVED", marker]))
                self.assertEqual(fmt, "xa-helper")

    def test_a_stack_of_single_frame_objects_is_not_a_still(self):
        header = _xa_header(["DERIVED", "PRIMARY", "BIPLANE A", "SINGLE A"], derivation="SUBTRACTION", frames=1)
        self.assertEqual(_keys(header, n_objects=1)[0], "DSA-still")
        self.assertEqual(_keys(header, n_objects=5)[0], "DSA")

    def test_the_plane_is_still_taken_from_the_image_type(self):
        _, keys = _keys(_xa_header(["DERIVED", "PRIMARY", "BIPLANE B", "SINGLE B"], derivation="SUBTRACTION"))
        self.assertEqual(keys.get("acq"), "B")


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestConflictIgnoreKeys(unittest.TestCase):
    def test_instance_specific_tags_do_not_make_a_conflict(self):
        from TPTBox.core.dicom.dicom2nii_utils import save_json, test_name_conflict  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            save_json({"a": 1, "SOPInstanceUID": "1.2.3"}, path)
            other = {"a": 1, "SOPInstanceUID": "9.9.9"}
            self.assertTrue(test_name_conflict(other, path))
            self.assertFalse(test_name_conflict(other, path, ignore_keys=["SOPInstanceUID"]))
            # a real difference is still a conflict, and the stored file is left alone
            self.assertTrue(test_name_conflict({"a": 2, "SOPInstanceUID": "9.9.9"}, path, ignore_keys=["SOPInstanceUID"]))
            self.assertIn("SOPInstanceUID", path.read_text())

    def test_the_default_ignore_keys_are_the_ones_that_differ_per_object(self):
        from TPTBox.core.dicom.dicom_extract import DEFAULT_CONFLICT_IGNORE_KEYS  # noqa: PLC0415

        self.assertIn("SOPInstanceUID", DEFAULT_CONFLICT_IGNORE_KEYS)
        self.assertIn("ReferencedImageSequence", DEFAULT_CONFLICT_IGNORE_KEYS)
        self.assertNotIn("IrradiationEventUID", DEFAULT_CONFLICT_IGNORE_KEYS)


def _ds(**kw):
    from pydicom.dataset import Dataset  # noqa: PLC0415

    ds = Dataset()
    kw.setdefault("Modality", "XA")
    kw.setdefault("PatientID", "p1")
    for k, v in kw.items():
        setattr(ds, k, v)
    return ds


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestSeriesOrder(unittest.TestCase):
    def test_tm_seconds(self):
        from TPTBox.core.dicom.dicom_extract import _tm_seconds  # noqa: PLC0415

        self.assertAlmostEqual(_tm_seconds("130645.035"), 13 * 3600 + 6 * 60 + 45.035)
        self.assertAlmostEqual(_tm_seconds("13:06:45"), 13 * 3600 + 6 * 60 + 45)
        self.assertEqual(_tm_seconds("13"), 13 * 3600)
        self.assertIsNone(_tm_seconds(""))
        self.assertIsNone(_tm_seconds("xx"))

    def test_order_is_chronological_and_independent_of_the_input_order(self):
        from TPTBox.core.dicom.dicom_extract import _series_sort_key  # noqa: PLC0415

        items = [
            ("late", [_ds(AcquisitionDate="20220101", AcquisitionTime="110000", SeriesNumber=2)]),
            ("early", [_ds(AcquisitionDate="20220101", AcquisitionTime="100000", SeriesNumber=7)]),
            ("next_day", [_ds(AcquisitionDate="20220102", AcquisitionTime="090000", SeriesNumber=1)]),
        ]
        expected = ["early", "late", "next_day"]
        for seed in range(5):
            shuffled = items[:]
            random.Random(seed).shuffle(shuffled)
            self.assertEqual([k for k, _ in sorted(shuffled, key=_series_sort_key)], expected)

    def test_original_before_copy_and_full_run_before_key_frame(self):
        from TPTBox.core.dicom.dicom_extract import _series_sort_key  # noqa: PLC0415

        t = {"AcquisitionDate": "20220101", "AcquisitionTime": "100000", "SeriesNumber": 5}
        items = [
            ("key_frame", [_ds(**t, NumberOfFrames=1, DerivationDescription="SUBTRACTION;FRAME_SELECTION")]),
            ("copy", [_ds(**t, NumberOfFrames=16, DerivationDescription="SUBTRACTION")]),
            ("original", [_ds(**t, NumberOfFrames=16, DerivationDescription="SUBTRACTION", IrradiationEventUID="1.2.3")]),
        ]
        for seed in range(5):
            shuffled = items[:]
            random.Random(seed).shuffle(shuffled)
            self.assertEqual([k for k, _ in sorted(shuffled, key=_series_sort_key)], ["original", "copy", "key_frame"])

    def test_missing_time_sorts_last_and_does_not_raise(self):
        from TPTBox.core.dicom.dicom_extract import _series_sort_key  # noqa: PLC0415

        items = [("none", [_ds()]), ("t", [_ds(AcquisitionDate="20220101", AcquisitionTime="100000")])]
        self.assertEqual([k for k, _ in sorted(items, key=_series_sort_key)], ["t", "none"])


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestXaCopiesAndRuns(unittest.TestCase):
    def test_recomputed_copy_is_dropped_only_when_the_original_exists(self):
        from TPTBox.core.dicom.dicom_extract import _drop_recomputed_xa_copies  # noqa: PLC0415

        common = {"SeriesInstanceUID": "1.1", "ImageType": ["DERIVED", "PRIMARY"], "Rows": 8, "Columns": 8, "NumberOfFrames": 16}
        items = [
            ("orig", [_ds(**common, IrradiationEventUID="1.2")]),
            ("copy", [_ds(**common)]),
            ("lonely", [_ds(**{**common, "SeriesInstanceUID": "2.2"})]),  # no original -> must stay
            ("other_frames", [_ds(**{**common, "NumberOfFrames": 1})]),  # different object -> must stay
        ]
        self.assertEqual([k for k, _ in _drop_recomputed_xa_copies(items)], ["orig", "lonely", "other_frames"])

    def test_runs_follow_the_acquisition_time(self):
        from TPTBox.core.dicom.dicom_extract import _assign_xa_runs  # noqa: PLC0415

        def obj(time, plane="A", pid="p1", **kw):
            return _ds(
                PatientID=pid, AcquisitionDate="20220101", AcquisitionTime=time, ImageType=["DERIVED", "PRIMARY", f"BIPLANE {plane}"], **kw
            )

        items = [
            ("b2", [obj("100000.5", "B")]),
            ("a1", [obj("100000.0", "A")]),
            ("key", [obj("100002.0", "A")]),  # key frame a few seconds later: same run
            ("a2", [obj("101500.0", "A")]),
            ("other_patient", [obj("100000.0", "A", pid="p2")]),
            ("helper", [_ds(AcquisitionDate="20220101", AcquisitionTime="100000", ImageType=["DERIVED", "SECONDARY"])]),
        ]
        runs = _assign_xa_runs(items, tolerance_s=5.0)
        self.assertEqual({k: runs[k] for k in ("a1", "b2", "key", "a2")}, {"a1": 1, "b2": 1, "key": 1, "a2": 2})
        self.assertEqual(runs["other_patient"], 1)
        self.assertNotIn("helper", runs)


class TestNameOrder(unittest.TestCase):
    def test_workers_take_their_turn_in_ticket_order(self):
        from TPTBox.core.dicom.dicom_extract import _NameOrder, _Ticket  # noqa: PLC0415

        order = _NameOrder()
        claimed: list[int] = []
        lock = threading.Lock()

        def worker(i: int, delay: float):
            ticket = _Ticket(order, i)
            threading.Event().wait(delay)  # later tickets arrive first
            ticket.wait()
            with lock:
                claimed.append(i)
            ticket.release()

        threads = [threading.Thread(target=worker, args=(i, 0.05 * (5 - i))) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(claimed, list(range(6)))

    def test_a_failed_worker_does_not_block_the_others(self):
        from TPTBox.core.dicom.dicom_extract import _NameOrder, _run_with_ticket, _Ticket  # noqa: PLC0415

        order = _NameOrder()
        seen: list[int] = []

        def fn(key, _files, _parts, ticket):
            if key == 1:
                raise RuntimeError("boom")
            ticket.wait()
            seen.append(key)
            ticket.release()
            return key

        results: dict[int, object] = {}

        def run(i):
            try:
                results[i] = _run_with_ticket(fn, _Ticket(order, i), i, None, None)
            except RuntimeError as e:
                results[i] = e

        threads = [threading.Thread(target=run, args=(i,)) for i in (2, 1, 0)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(seen, [0, 2])
        self.assertIsInstance(results[1], RuntimeError)


if __name__ == "__main__":
    unittest.main()
