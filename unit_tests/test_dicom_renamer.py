"""Unit tests for `TPTBox.core.dicom.dicom_renamer` and `xa_pairing`.

Both modules rewrite an already-extracted dataset, and both offer a `dry_run`
mode that returns the full plan without touching disk - so the plan itself is
the assertion, and the destructive path is then checked against it on a copy.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

HAS_PYDICOM = importlib.util.find_spec("pydicom") is not None and importlib.util.find_spec("dicom2nifti") is not None
PYDICOM_REASON = "requires the dicom extra (pydicom + dicom2nifti)"

GRID = {"spacing": [0.5, 0.5, 6.0], "orientation": ["R", "A", "S"], "shape": [8, 8, 4]}


def _write_pair(folder: Path, stem: str, sidecar: dict) -> Path:
    """Write a `<stem>.nii.gz` + `<stem>.json` pair and return the json path."""
    folder.mkdir(parents=True, exist_ok=True)
    affine = np.diag([0.5, 0.5, 6.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 4), dtype=np.uint8), affine), folder / f"{stem}.nii.gz")
    path = folder / f"{stem}.json"
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def _sidecar(series_number: int = 7, study_date: str = "20200101", **extra) -> dict:
    out = {
        "grid": dict(GRID),
        "Modality": "MR",
        "SeriesNumber": series_number,
        "SeriesDescription": "t2_tse_sag",
        "StudyDate": study_date,
        "PatientID": "RAW_ID",
    }
    out.update(extra)
    return out


def _dataset(root: Path, subjects=("raw_1", "raw_2")) -> Path:
    ds = root / "dataset-x"
    for i, sub in enumerate(subjects, start=1):
        _write_pair(ds / "rawdata" / f"sub-{sub}" / "ses-20200101", f"sub-{sub}_ses-20200101_sequ-{i}_T2w", _sidecar(i))
    return ds


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestPlaneFromGrid(unittest.TestCase):
    def test_plane_from_the_thick_axis(self):
        from TPTBox.core.dicom.dicom_renamer import _plane_from_grid  # noqa: PLC0415

        cases = [
            ({"spacing": [0.5, 0.5, 6.0], "orientation": ["R", "A", "S"]}, "ax"),
            ({"spacing": [6.0, 0.5, 0.5], "orientation": ["R", "A", "S"]}, "sag"),
            ({"spacing": [0.5, 6.0, 0.5], "orientation": ["R", "A", "S"]}, "cor"),
            ({"spacing": [1.0, 1.0, 1.0], "orientation": ["R", "A", "S"]}, "iso"),
        ]
        for grid, expected in cases:
            with self.subTest(spacing=grid["spacing"]):
                self.assertEqual(_plane_from_grid(grid), expected)

    def test_hires_threshold_clamps_thin_slices(self):
        from TPTBox.core.dicom.dicom_renamer import _plane_from_grid  # noqa: PLC0415

        grid = {"spacing": [0.3, 0.3, 0.75], "orientation": ["R", "A", "S"]}
        # With the default 0.8 clamp all three zooms become equal -> isotropic.
        self.assertEqual(_plane_from_grid(grid), "iso")
        # Without the clamp the 0.75 axis is the thickest one.
        self.assertEqual(_plane_from_grid(grid, hires_threshold=None), "ax")

    def test_malformed_grids_return_none(self):
        from TPTBox.core.dicom.dicom_renamer import _plane_from_grid  # noqa: PLC0415

        for grid in ({}, None, {"spacing": [1.0]}, {"spacing": [1, 1, 1]}, {"orientation": ["R", "A", "S"]}):
            with self.subTest(grid=grid):
                self.assertIsNone(_plane_from_grid(grid))

    def test_zero_spacing_is_treated_as_one(self):
        from TPTBox.core.dicom.dicom_renamer import _plane_from_grid  # noqa: PLC0415

        self.assertEqual(_plane_from_grid({"spacing": [0.0, 0.0, 6.0], "orientation": ["R", "A", "S"]}), "ax")

    def test_agrees_with_get_plane_dicom_on_the_same_grid(self):
        from TPTBox import NII  # noqa: PLC0415
        from TPTBox.core.dicom.dicom_header_to_keys import get_plane_dicom  # noqa: PLC0415
        from TPTBox.core.dicom.dicom_renamer import _plane_from_grid  # noqa: PLC0415

        for zoom in ((0.5, 0.5, 6.0), (6.0, 0.5, 0.5), (1.0, 1.0, 1.0)):
            with self.subTest(zoom=zoom):
                nii = NII(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.uint8), np.diag([*zoom, 1.0])), seg=False)
                grid = {"spacing": list(zoom), "orientation": list(nii.orientation)}
                self.assertEqual(_plane_from_grid(grid), get_plane_dicom(nii))


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestEntityHelpers(unittest.TestCase):
    def test_sanitize_entity_value(self):
        from TPTBox.core.dicom.dicom_renamer import _sanitize_entity_value  # noqa: PLC0415

        cases = {"abc": "abc", "a_b": "ab", "__ab__": "ab", "-abc": "abc", "_": "unnamed", "": "unnamed", "-": "unnamed"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(_sanitize_entity_value(raw), expected)

    def test_sanitized_values_survive_bids_parsing(self):
        from TPTBox.core.bids_files import get_values_from_name  # noqa: PLC0415
        from TPTBox.core.dicom.dicom_renamer import _sanitize_entity_value  # noqa: PLC0415

        for raw in ("a_b", "-weird", "ok123"):
            with self.subTest(raw=raw):
                value = _sanitize_entity_value(raw)
                _, info, _, _ = get_values_from_name(f"sub-{value}_ct.nii.gz", verbose=False)
                self.assertEqual(info["sub"], value)

    def test_insert_run_entity_after_sequ(self):
        from TPTBox.core.dicom.dicom_renamer import _insert_run_entity  # noqa: PLC0415

        out = _insert_run_entity(Path("/ds/sub-1_ses-2_sequ-3_T2w.nii.gz"), 2)
        self.assertEqual(out.name, "sub-1_ses-2_sequ-3_run-2_T2w.nii.gz")

    def test_insert_run_entity_without_sequ_falls_back(self):
        from TPTBox.core.dicom.dicom_renamer import _insert_run_entity  # noqa: PLC0415

        out = _insert_run_entity(Path("/ds/sub-1_T2w.nii.gz"), 3)
        self.assertIn("run-3", out.name)

    def test_inserted_run_is_parsed_back(self):
        from TPTBox.core.bids_files import get_values_from_name  # noqa: PLC0415
        from TPTBox.core.dicom.dicom_renamer import _insert_run_entity  # noqa: PLC0415

        out = _insert_run_entity(Path("/ds/sub-1_ses-2_sequ-3_T2w.nii.gz"), 5)
        _, info, _, _ = get_values_from_name(out, verbose=False)
        self.assertEqual(info["run"], "5")
        self.assertEqual(info["sequ"], "3")

    def test_files_identical(self):
        from TPTBox.core.dicom.dicom_renamer import _files_identical  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            a, b, c, missing = (Path(tmp) / n for n in ("a", "b", "c", "missing"))
            a.write_bytes(b"same")
            b.write_bytes(b"same")
            c.write_bytes(b"different")
            self.assertTrue(_files_identical(a, b))
            self.assertFalse(_files_identical(a, c))
            self.assertFalse(_files_identical(a, missing))


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestSubjectMap(unittest.TestCase):
    @staticmethod
    def _folders(names):
        return [Path(f"/ds/rawdata/sub-{n}") for n in names]

    def test_sanitise_in_place_without_a_prefix(self):
        from TPTBox.core.dicom.dicom_renamer import _build_subject_map  # noqa: PLC0415

        out = _build_subject_map(self._folders(["ok1", "bad_id", "-lead"]), None, 3)
        self.assertEqual(out["ok1"], "ok1")
        self.assertEqual(out["bad_id"], "badid")
        self.assertEqual(out["-lead"], "lead")

    def test_prefix_mode_renumbers_in_order(self):
        from TPTBox.core.dicom.dicom_renamer import _build_subject_map  # noqa: PLC0415

        out = _build_subject_map(self._folders(["zzz", "aaa"]), "ID", 3)
        self.assertEqual(sorted(out.values()), ["ID001", "ID002"])
        self.assertEqual(len(set(out.values())), 2, "the mapping must stay injective")

    def test_prefix_itself_is_sanitised(self):
        from TPTBox.core.dicom.dicom_renamer import _build_subject_map  # noqa: PLC0415

        out = _build_subject_map(self._folders(["a"]), "I_D", 2)
        self.assertEqual(list(out.values()), ["ID01"])

    def test_rerunning_keeps_previously_assigned_ids(self):
        from TPTBox.core.dicom.dicom_renamer import _build_subject_map  # noqa: PLC0415

        first = _build_subject_map(self._folders(["aaa", "bbb"]), "ID", 3)
        # Second pass sees the already-renamed folders plus one new subject.
        second = _build_subject_map(self._folders([*first.values(), "ccc"]), "ID", 3, existing_map=first)
        for old, new in first.items():
            with self.subTest(old=old):
                self.assertEqual(second[old], new, "historic rows must be preserved")
                self.assertEqual(second[new], new, "an already-renamed folder maps to itself")
        self.assertEqual(second["ccc"], "ID003", "a fresh subject takes the next free slot")

    def test_existing_map_roundtrips_through_the_tsv(self):
        from TPTBox.core.dicom.dicom_renamer import _read_existing_subject_map  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(_read_existing_subject_map(root, "info"), {})
            info = root / "info"
            info.mkdir()
            with (info / "subject_map.tsv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["old_sub", "new_sub"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"old_sub": "raw_1", "new_sub": "ID001"})
            self.assertEqual(_read_existing_subject_map(root, "info"), {"raw_1": "ID001"})


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestRerunBidsNaming(unittest.TestCase):
    def test_dry_run_plans_moves_without_touching_disk(self):
        from TPTBox.core.dicom.dicom_renamer import rerun_bids_naming  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(Path(tmp))
            before = sorted(p.relative_to(ds).as_posix() for p in ds.rglob("*") if p.is_file())
            out = rerun_bids_naming(ds, subject_prefix="ID", dry_run=True, verbose=False)
            after = sorted(p.relative_to(ds).as_posix() for p in ds.rglob("*") if p.is_file())
            self.assertEqual(before, after, "a dry run must not move anything")
            self.assertTrue(out["moves"], "the plan should not be empty")
            for src, dst in out["moves"]:
                with self.subTest(src=Path(src).name):
                    self.assertTrue(Path(src).exists())
                    self.assertIn("sub-ID", Path(dst).as_posix())

    def test_applying_the_plan_produces_exactly_the_planned_files(self):
        from TPTBox.core.dicom.dicom_renamer import rerun_bids_naming  # noqa: PLC0415

        # Regression: the leftover sweep (pass 2) scans the disk, so under dry_run it
        # also reported every file pass 1 had only *planned* to move - with a different,
        # fallback destination. The plan listed 6 moves where the real run performed 4.
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(Path(tmp))
            plan = rerun_bids_naming(ds, subject_prefix="ID", dry_run=True, verbose=False)["moves"]
            applied = rerun_bids_naming(ds, subject_prefix="ID", dry_run=False, verbose=False)["moves"]
            self.assertEqual(len(plan), len(applied), "the dry run must plan exactly what the real run does")
            self.assertEqual(
                sorted(Path(d).name for _, d in plan),
                sorted(Path(d).name for _, d in applied),
            )
            for _, dst in plan:
                with self.subTest(dst=Path(dst).name):
                    self.assertTrue(Path(dst).exists(), dst)

    def test_the_translation_table_is_written(self):
        from TPTBox.core.dicom.dicom_renamer import rerun_bids_naming  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(Path(tmp))
            out = rerun_bids_naming(ds, subject_prefix="ID", dry_run=False, verbose=False)
            table = Path(out["mapping_file"])
            self.assertTrue(table.is_file())
            text = table.read_text(encoding="utf-8")
            self.assertIn("old_sub", text)
            self.assertIn("ID001", text)

    def test_a_second_run_is_stable(self):
        from TPTBox.core.dicom.dicom_renamer import rerun_bids_naming  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(Path(tmp))
            rerun_bids_naming(ds, subject_prefix="ID", dry_run=False, verbose=False)
            first = sorted(p.relative_to(ds).as_posix() for p in ds.rglob("*.nii.gz"))
            rerun_bids_naming(ds, subject_prefix="ID", dry_run=False, verbose=False)
            second = sorted(p.relative_to(ds).as_posix() for p in ds.rglob("*.nii.gz"))
            self.assertEqual(first, second, "re-running must be idempotent")

    def test_sanitise_mode_fixes_illegal_subject_ids(self):
        from TPTBox.core.dicom.dicom_renamer import rerun_bids_naming  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(Path(tmp), subjects=("bad_id",))
            rerun_bids_naming(ds, dry_run=False, verbose=False)
            subs = sorted(p.name for p in (ds / "rawdata").iterdir() if p.is_dir())
            self.assertEqual(subs, ["sub-badid"])

    def test_flat_dataset_without_a_parent_folder(self):
        from TPTBox.core.dicom.dicom_renamer import rerun_bids_naming  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "flat"
            _write_pair(ds / "sub-raw_1" / "ses-20200101", "sub-raw_1_ses-20200101_sequ-1_T2w", _sidecar(1))
            out = rerun_bids_naming(ds, parent=None, subject_prefix="ID", dry_run=True, verbose=False)
            self.assertTrue(out["moves"])

    def test_empty_dataset_plans_nothing(self):
        from TPTBox.core.dicom.dicom_renamer import rerun_bids_naming  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "dataset-empty"
            (ds / "rawdata").mkdir(parents=True)
            out = rerun_bids_naming(ds, dry_run=True, verbose=False)
            self.assertEqual(out["moves"], [])

    def test_prune_empty_folders(self):
        from TPTBox.core.dicom.dicom_renamer import _prune_empty_folders  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty" / "nested").mkdir(parents=True)
            (root / "full").mkdir()
            (root / "full" / "f.txt").write_text("x", encoding="utf-8")
            removed = _prune_empty_folders(root)
            self.assertGreaterEqual(removed, 2)
            self.assertFalse((root / "empty").exists())
            self.assertTrue((root / "full" / "f.txt").exists())


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestStemCollision(unittest.TestCase):
    def test_free_target_is_a_plain_move(self):
        from TPTBox.core.dicom.dicom_renamer import _resolve_stem_collision  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.json"
            src.write_text("{}", encoding="utf-8")
            target = Path(tmp) / "sub-1_sequ-3_T2w.json"
            final, action = _resolve_stem_collision(target, src)
            self.assertEqual((final, action), (target, "move"))

    def test_identical_content_is_reported_as_a_duplicate(self):
        from TPTBox.core.dicom.dicom_renamer import _resolve_stem_collision  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.json"
            target = Path(tmp) / "sub-1_sequ-3_T2w.json"
            src.write_text('{"a": 1}', encoding="utf-8")
            target.write_text('{"a": 1}', encoding="utf-8")
            self.assertEqual(_resolve_stem_collision(target, src)[1], "duplicate")

    def test_different_content_takes_a_run_slot(self):
        from TPTBox.core.dicom.dicom_renamer import _resolve_stem_collision  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.json"
            target = Path(tmp) / "sub-1_sequ-3_T2w.json"
            src.write_text('{"a": 1}', encoding="utf-8")
            target.write_text('{"a": 2}', encoding="utf-8")
            final, action = _resolve_stem_collision(target, src)
            self.assertEqual(action, "run")
            self.assertIn("run-2", final.name)

    def test_the_same_path_is_not_a_collision(self):
        from TPTBox.core.dicom.dicom_renamer import _resolve_stem_collision  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "sub-1_sequ-3_T2w.json"
            src.write_text('{"a": 1}', encoding="utf-8")
            self.assertEqual(_resolve_stem_collision(src, src)[1], "move")


@unittest.skipUnless(HAS_PYDICOM, PYDICOM_REASON)
class TestXaPairing(unittest.TestCase):
    @staticmethod
    def _xa_dataset(root: Path, b_sequ: int = 2) -> Path:
        """A biplane pair: both planes share (StudyInstanceUID, AcquisitionTime).

        The acq-A / acq-B entities already exist (the extractor writes them); this
        module's job is only to make the B side adopt the A side's `sequ`.
        """
        ds = root / "dataset-xa"
        folder = ds / "rawdata" / "sub-1" / "ses-20200101" / "XA"
        shared = {"StudyInstanceUID": "1.2.3", "AcquisitionTime": "120000.00", "Modality": "XA"}
        _write_pair(folder, "sub-1_ses-20200101_sequ-1_acq-A_XA", _sidecar(1, **shared))
        _write_pair(folder, f"sub-1_ses-20200101_sequ-{b_sequ}_acq-B_XA", _sidecar(b_sequ, **shared))
        return ds

    def test_norm_time_strips_subsecond_precision(self):
        from TPTBox.core.dicom.xa_pairing import _norm_time  # noqa: PLC0415

        self.assertEqual(_norm_time("120000.000000"), _norm_time("120000"))
        self.assertEqual(_norm_time(None), "")

    def test_bucket_key_groups_the_same_acquisition(self):
        from TPTBox.core.dicom.xa_pairing import _bucket_key  # noqa: PLC0415

        a = {"StudyInstanceUID": "1.2.3", "AcquisitionTime": "120000.00"}
        b = {"StudyInstanceUID": "1.2.3", "AcquisitionTime": "120000"}
        c = {"StudyInstanceUID": "9.9.9", "AcquisitionTime": "120000"}
        self.assertEqual(_bucket_key(a), _bucket_key(b))
        self.assertNotEqual(_bucket_key(a), _bucket_key(c))

    def test_sidecar_for_maps_nii_to_json(self):
        from TPTBox.core.dicom.xa_pairing import _sidecar_for  # noqa: PLC0415

        self.assertEqual(_sidecar_for(Path("/ds/a_xa.nii.gz")), Path("/ds/a_xa.json"))

    def test_flip_marker_is_idempotent(self):
        from TPTBox.core.dicom.xa_pairing import _flip_marker_set, _write_flip_marker  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "a_xa.json"
            sidecar.write_text(json.dumps({"a": 1}), encoding="utf-8")
            self.assertFalse(_flip_marker_set(sidecar))
            _write_flip_marker(sidecar)
            self.assertTrue(_flip_marker_set(sidecar))
            _write_flip_marker(sidecar)
            self.assertTrue(_flip_marker_set(sidecar))

    def test_flip_z_affine_only_changes_the_third_column_sign(self):
        from TPTBox.core.dicom.xa_pairing import _flip_z_affine  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a_xa.nii.gz"
            affine = np.diag([0.5, 0.5, 6.0, 1.0])
            arr = np.arange(8 * 8 * 4, dtype=np.uint8).reshape(8, 8, 4)
            nib.save(nib.Nifti1Image(arr, affine), path)
            _flip_z_affine(path)
            out = nib.load(path)
            np.testing.assert_array_equal(np.asanyarray(out.dataobj), arr, "voxel data must be untouched")
            new_affine = out.affine
            np.testing.assert_allclose(new_affine[:3, 0], affine[:3, 0])
            np.testing.assert_allclose(new_affine[:3, 1], affine[:3, 1])
            np.testing.assert_allclose(new_affine[:3, 2], -affine[:3, 2])

    def test_pair_biplane_dry_run_plans_without_touching_disk(self):
        from TPTBox.core.dicom.xa_pairing import pair_biplane_xa  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = self._xa_dataset(Path(tmp))
            before = {p: p.stat().st_mtime_ns for p in ds.rglob("*") if p.is_file()}
            out = pair_biplane_xa(ds, dry_run=True, verbose=False)
            after = {p: p.stat().st_mtime_ns for p in ds.rglob("*") if p.is_file()}
            self.assertEqual(before, after, "a dry run must not move anything")
            self.assertEqual(out["pairs_matched"], 1)
            self.assertTrue(out["renames"])

    def test_the_b_side_adopts_the_a_side_sequ(self):
        from TPTBox.core.dicom.xa_pairing import pair_biplane_xa  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = self._xa_dataset(Path(tmp))
            out = pair_biplane_xa(ds, dry_run=False, verbose=False)
            self.assertEqual(out["pairs_matched"], 1)
            names = sorted(p.name for p in ds.rglob("*.nii.gz"))
            self.assertEqual(
                names,
                ["sub-1_ses-20200101_sequ-1_acq-A_XA.nii.gz", "sub-1_ses-20200101_sequ-1_acq-B_XA.nii.gz"],
            )

    def test_the_renamed_file_stays_in_its_own_parent_folder(self):
        from TPTBox.core.dicom.xa_pairing import pair_biplane_xa  # noqa: PLC0415

        # Regression: the target path came from get_changed_bids(), whose `parent`
        # defaults to "derivatives", so renaming the B side moved it out of rawdata.
        # A rawdata-scoped scan then lost it entirely - including the flip_z pass.
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._xa_dataset(Path(tmp))
            pair_biplane_xa(ds, dry_run=False, verbose=False)
            parents = {p.relative_to(ds).parts[0] for p in ds.rglob("*") if p.is_file()}
            self.assertEqual(parents, {"rawdata"})

    def test_an_already_paired_bucket_needs_no_rename(self):
        from TPTBox.core.dicom.xa_pairing import pair_biplane_xa  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = self._xa_dataset(Path(tmp), b_sequ=1)
            out = pair_biplane_xa(ds, dry_run=True, verbose=False)
            self.assertEqual(out["pairs_matched"], 1)
            self.assertEqual(out["renames"], [])

    def test_a_lone_plane_is_not_a_pair(self):
        from TPTBox.core.dicom.xa_pairing import pair_biplane_xa  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "dataset-xa"
            folder = ds / "rawdata" / "sub-1" / "ses-20200101" / "XA"
            _write_pair(folder, "sub-1_ses-20200101_sequ-1_acq-A_XA", _sidecar(1, StudyInstanceUID="1.2.3"))
            out = pair_biplane_xa(ds, dry_run=True, verbose=False)
            self.assertEqual(out["pairs_matched"], 0)
            self.assertEqual(out["renames"], [])

    def test_flip_z_negates_the_affine_column_and_marks_the_sidecar(self):
        from TPTBox.core.dicom.xa_pairing import _flip_marker_set, pair_biplane_xa  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = self._xa_dataset(Path(tmp))
            out = pair_biplane_xa(ds, dry_run=False, flip_z=True, verbose=False)
            self.assertEqual(len(out["flipped"]), 2, "both planes must be flipped, not just the A side")
            for nii_path in ds.rglob("*.nii.gz"):
                with self.subTest(name=nii_path.name):
                    np.testing.assert_allclose(nib.load(nii_path).affine[:3, 2], [0.0, 0.0, -6.0])
                    self.assertTrue(_flip_marker_set(nii_path.with_name(nii_path.name.replace(".nii.gz", ".json"))))

            # A second pass must not flip the same file back.
            again = pair_biplane_xa(ds, dry_run=False, flip_z=True, verbose=False)
            self.assertEqual(again["flipped"], [])


if __name__ == "__main__":
    unittest.main()
