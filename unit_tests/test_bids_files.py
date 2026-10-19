"""Unit tests for `TPTBox.core.bids_files` and `TPTBox.core.bids_constants`.

`unit_tests/test_bids_file.py` covers the same module through `get_BIDS_test()`, but
all four of its tests are gated on a developer-local dataset path and therefore never
run. This file instead builds a small real BIDS tree (NIfTIs, JSON sidecars, a POI
json, a PNG) in a temporary directory, so the scan -> query -> family -> derived-path
pipeline is exercised end to end everywhere.

Two module-level containers in `bids_constants` are mutated as a side effect of
parsing (`entities_keys[key] = key` for unknown keys, `formats_relaxed.append(...)`
for unknown formats) and `BIDS_Global_info.__init__` registers its `additional_key`s
the same way. They are snapshotted and restored in place, because other modules bound
the same objects at import time.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from TPTBox import POI
from TPTBox.core import bids_constants
from TPTBox.core.bids_files import (
    BIDS_FILE,
    BIDS_Family,
    BIDS_Global_info,
    Buffered_BIDS_Global_info,
    Searchquery,
    get_values_from_name,
    validate_entities,
)

ZOOM = (1.5, 1.5, 3.0)
ORIGIN = (10.0, 20.0, 30.0)
SHAPE = (4, 5, 6)


def _affine() -> np.ndarray:
    aff = np.diag([*ZOOM, 1.0])
    aff[:3, 3] = ORIGIN
    return aff


def _save_nii(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(SHAPE, dtype=np.uint8), _affine()), path)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_dataset(root: Path) -> Path:
    """Create a two-subject BIDS dataset under ``root`` and return the dataset path.

    sub-001 has a CT with a derivatives family (two masks, a POI json, a snapshot) and a
    four-channel Dixon sequence with ImageType sidecars; sub-002 has a single T2w.
    """
    ds = root / "dataset-test"
    raw = ds / "rawdata" / "sub-001" / "ses-20210101"
    der = ds / "derivatives" / "sub-001" / "ses-20210101"

    _save_nii(raw / "sub-001_ses-20210101_sequ-301_ct.nii.gz")
    _save_json(raw / "sub-001_ses-20210101_sequ-301_ct.json", {"ImageType": ["ORIGINAL", "PRIMARY"], "EchoTime": 1.0})
    for part, image_type in (
        ("water", ["DERIVED", "W"]),
        ("fat", ["DERIVED", "F"]),
        ("inphase", ["ORIGINAL", "IP"]),
        ("outphase", ["ORIGINAL", "OP"]),
    ):
        _save_nii(raw / f"sub-001_ses-20210101_sequ-302_part-{part}_dixon.nii.gz")
        _save_json(raw / f"sub-001_ses-20210101_sequ-302_part-{part}_dixon.json", {"ImageType": image_type})

    _save_nii(der / "sub-001_ses-20210101_sequ-301_seg-subreg_msk.nii.gz")
    _save_nii(der / "sub-001_ses-20210101_sequ-301_seg-vert_msk.nii.gz")
    (der / "sub-001_ses-20210101_sequ-301_snp.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    poi = POI(
        {20: {50: (1.0, 2.0, 3.0)}, 21: {50: (2.0, 3.0, 4.0)}},
        orientation=("R", "A", "S"),
        zoom=ZOOM,
        shape=SHAPE,
        origin=ORIGIN,
        rotation=np.eye(3),
    )
    poi.save(der / "sub-001_ses-20210101_sequ-301_seg-subreg_ctd.json", verbose=False)

    raw2 = ds / "rawdata" / "sub-002" / "ses-20220202"
    _save_nii(raw2 / "sub-002_ses-20220202_sequ-401_T2w.nii.gz")
    _save_json(raw2 / "sub-002_ses-20220202_sequ-401_T2w.json", {"ImageType": ["ORIGINAL"]})
    return ds


class _BidsConstantsGuard(unittest.TestCase):
    """Restores the mutable module-level tables in `bids_constants` around each test."""

    def setUp(self):
        self._entities = dict(bids_constants.entities)
        self._entities_keys = dict(bids_constants.entities_keys)
        self._formats_relaxed = list(bids_constants.formats_relaxed)

    def tearDown(self):
        # Restore in place: other modules imported these objects by reference.
        bids_constants.entities.clear()
        bids_constants.entities.update(self._entities)
        bids_constants.entities_keys.clear()
        bids_constants.entities_keys.update(self._entities_keys)
        bids_constants.formats_relaxed[:] = self._formats_relaxed


class TestGetValuesFromName(_BidsConstantsGuard):
    def test_table(self):
        cases = [
            # name, bids_format, info, bids_key, file_type
            ("sub-001_ses-01_T1w.nii.gz", "T1w", {"sub": "001", "ses": "01"}, "sub-001_ses-01_T1w", "nii.gz"),
            ("sub-001_ct.nii.gz", "ct", {"sub": "001"}, "sub-001_ct", "nii.gz"),
            ("sub-001_ses-01_seg-subreg_msk.nii.gz", "msk", {"sub": "001", "ses": "01", "seg": "subreg"}, None, "nii.gz"),
            ("sub-x_ses-1_ctd.json", "ctd", {"sub": "x", "ses": "1"}, None, "json"),
            ("sub-x_ses-1_ctd.mrk.json", "ctd", {"sub": "x", "ses": "1"}, None, "mrk.json"),
            # value may itself contain a dash: only the first one splits key from value
            ("sub-001_part-water-fraction_dixon.nii.gz", "dixon", {"sub": "001", "part": "water-fraction"}, None, "nii.gz"),
        ]
        for name, fmt, info, key, ftype in cases:
            with self.subTest(name=name):
                got_fmt, got_info, got_key, got_ftype = get_values_from_name(name, verbose=False)
                self.assertEqual(got_fmt, fmt)
                self.assertEqual(got_info, info)
                self.assertEqual(got_ftype, ftype)
                self.assertEqual(got_key, key if key is not None else name[: -len(ftype) - 1])

    def test_accepts_path_and_str_identically(self):
        name = "sub-001_ses-01_sequ-7_ct.nii.gz"
        self.assertEqual(get_values_from_name(name, verbose=False), get_values_from_name(Path("/a/b") / name, verbose=False))

    def test_malformed_segments_are_skipped_not_fatal(self):
        # "broken" has no "-", so it cannot be a key-value pair and is dropped.
        fmt, info, _, ftype = get_values_from_name("sub-001_broken_ct.nii.gz", verbose=False)
        self.assertEqual((fmt, ftype), ("ct", "nii.gz"))
        self.assertEqual(info, {"sub": "001"})

    def test_duplicate_key_keeps_the_last_value(self):
        _, info, _, _ = get_values_from_name("sub-001_sub-002_ct.nii.gz", verbose=False)
        self.assertEqual(info, {"sub": "002"})

    def test_name_without_extension_raises(self):
        # Documents current behaviour: the name is split on "." without a fallback.
        with self.assertRaises(ValueError):
            get_values_from_name("sub-001_ct", verbose=False)

    def test_unknown_format_is_registered_only_when_verbose(self):
        before = list(bids_constants.formats_relaxed)
        get_values_from_name("sub-001_notaformat.nii.gz", verbose=False)
        self.assertEqual(bids_constants.formats_relaxed, before)
        get_values_from_name("sub-001_notaformat.nii.gz", verbose=True)
        self.assertIn("notaformat", bids_constants.formats_relaxed)


class TestValidateEntities(_BidsConstantsGuard):
    def test_valid_entities(self):
        for key, value in (("sub", "001"), ("ses", "20210101"), ("run", "7"), ("task", "rest1"), ("mt", "on"), ("hemi", "L")):
            with self.subTest(key=key, value=value):
                self.assertTrue(validate_entities(key, value, "name", verbose=True))

    def test_invalid_entities(self):
        cases = [
            ("run", "a1"),  # must be decimal
            ("echo", "1.5"),  # must be decimal
            ("task", "a_b"),  # must be alphanumeric
            ("mt", "maybe"),  # must be on/off
            ("hemi", "X"),  # must be L/R
            ("mod", "banana"),  # must be a known format
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.assertFalse(validate_entities(key, value, "name", verbose=True))

    def test_key_is_lowercased_before_lookup(self):
        self.assertTrue(validate_entities("SUB", "001", "name", verbose=True))

    def test_unknown_key_is_rejected_once_then_registered(self):
        self.assertNotIn("totallynewkey", bids_constants.entities_keys)
        self.assertFalse(validate_entities("totallynewkey", "v", "name", verbose=True))
        # Side effect: the key is remembered, so the second call passes.
        self.assertIn("totallynewkey", bids_constants.entities_keys)
        self.assertTrue(validate_entities("totallynewkey", "v", "name", verbose=True))

    def test_non_verbose_short_circuits_to_true(self):
        self.assertTrue(validate_entities("hemi", "X", "name", verbose=False))
        self.assertTrue(validate_entities("definitely not a key", "", "name", verbose=False))


class TestBidsConstants(unittest.TestCase):
    """Properties of the shipped constant tables.

    These assert the *source* tables, loaded fresh: `BIDS_Global_info.__init__` registers
    its `additional_key`s by writing `entities[k] = k`, so by the time a full-suite run
    reaches this class the live dicts have extra self-mapping entries.
    """

    @staticmethod
    def _pristine():
        import importlib.util  # noqa: PLC0415

        path = Path(bids_constants.__file__)
        spec = importlib.util.spec_from_file_location("_pristine_bids_constants", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_entities_keys_is_the_inverse_of_entities(self):
        c = self._pristine()
        self.assertEqual(len(c.entities_keys), len(c.entities), "value collision in `entities`")
        for human, short in c.entities.items():
            self.assertEqual(c.entities_keys[short], human)

    def test_entity_rule_groups_reference_known_keys(self):
        known = set(self._pristine().entities.values())
        for name in ("entity_alphanumeric", "entity_decimal", "entity_format", "entity_on_off", "entity_left_right", "entity_parts"):
            with self.subTest(group=name):
                self.assertTrue(set(getattr(bids_constants, name)).issubset(known))

    def test_formats_relaxed_extends_formats(self):
        c = self._pristine()
        self.assertTrue(set(c.formats).issubset(set(c.formats_relaxed)))

    def test_no_duplicate_formats_or_file_types(self):
        # `formats` carried copy-paste duplicates (T1w, T2w, T2star, sti x2).
        c = self._pristine()
        for name in ("formats", "formats_relaxed", "file_types"):
            with self.subTest(table=name):
                table = getattr(c, name)
                self.assertEqual(len(table), len(set(table)))


class _DatasetTestCase(_BidsConstantsGuard):
    """Base class providing a freshly scanned synthetic dataset per test class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ds = build_dataset(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        super().setUp()
        self.g = BIDS_Global_info([self.ds], parents=["rawdata", "derivatives"], verbose=False)
        self.subj = self.g.subjects["001"]

    def ct_file(self) -> BIDS_FILE:
        return next(b for b in self.subj.new_query(flatten=True).loop_list() if b.bids_format == "ct")


class TestBidsGlobalInfoScan(_DatasetTestCase):
    def test_subjects_and_sequences(self):
        self.assertEqual(len(self.g), 2)
        self.assertEqual(sorted(self.g.subjects), ["001", "002"])
        self.assertEqual(sorted(self.subj.sequences), ["20210101_301", "20210101_302"])

    def test_sequence_name_joins_the_splitting_keys(self):
        ct = self.ct_file()
        self.assertEqual(self.subj.get_sequence_name(ct), "20210101_301")

    def test_enumerate_subjects_sorted_and_shuffled_agree_as_sets(self):
        ordered = [n for n, _ in self.g.enumerate_subjects(sort=True)]
        self.assertEqual(ordered, sorted(ordered))
        shuffled = [n for n, _ in self.g.enumerate_subjects(shuffle=True)]
        self.assertEqual(sorted(shuffled), ordered)

    def test_iter_subjects_matches_enumerate_subjects(self):
        self.assertEqual(
            [n for n, _ in self.g.iter_subjects(sort=True)],
            [n for n, _ in self.g.enumerate_subjects(sort=True)],
        )

    def test_str_lists_parents_and_datasets(self):
        text = str(self.g)
        self.assertIn("rawdata", text)
        self.assertIn("dataset-test", text)

    def test_missing_dataset_raises(self):
        with self.assertRaises(FileNotFoundError):
            BIDS_Global_info([self.ds / "does-not-exist"], parents=["rawdata"], verbose=False)

    def test_additional_key_is_registered(self):
        BIDS_Global_info([self.ds], parents=["rawdata"], additional_key=["myextrakey"], verbose=False)
        self.assertIn("myextrakey", bids_constants.entities_keys)

    def test_filter_folder_can_exclude_a_subject(self):
        g = BIDS_Global_info(
            [self.ds],
            parents=["rawdata"],
            verbose=False,
            filter_folder=lambda p, lvl: lvl != 1 or p.name == "sub-001",
        )
        self.assertEqual(sorted(g.subjects), ["001"])

    def test_filter_folder_is_applied_to_nested_levels_too(self):
        # Regression: `_scan_tree` did not forward `filter_folder` to its recursive call,
        # so any filter targeting a level below the subject folder was silently ignored.
        g = BIDS_Global_info(
            [self.ds],
            parents=["rawdata"],
            verbose=False,
            filter_folder=lambda p, lvl: lvl != 2 or p.name != "ses-20210101",
        )
        self.assertEqual(sorted(g.subjects), ["002"])

    def test_additional_file_list_adds_files_without_scanning(self):
        extra = self.ds / "rawdata" / "sub-003" / "sub-003_ct.nii.gz"
        g = BIDS_Global_info([self.ds], parents=[], verbose=False, additional_file_list={self.ds: [extra]})
        self.assertIn("003", g.subjects)


class TestBidsFilePaths(_DatasetTestCase):
    def test_get_path_decomposed(self):
        ds, parent, mid, name = self.ct_file().get_path_decomposed(file_type="nii.gz")
        self.assertEqual(Path(ds), self.ds)
        self.assertEqual(parent, "rawdata")
        self.assertEqual(mid, "sub-001/ses-20210101")
        self.assertEqual(name, "sub-001_ses-20210101_sequ-301_ct.nii.gz")

    def test_properties(self):
        ct = self.ct_file()
        self.assertEqual(ct.parent, "rawdata")
        self.assertEqual(ct.bids_format, "ct")
        self.assertEqual(ct.info, {"sub": "001", "ses": "20210101", "sequ": "301"})
        self.assertEqual(ct.get("sequ"), "301")
        self.assertEqual(ct.get("nope", "dflt"), "dflt")
        self.assertEqual(dict(ct.loop_keys()), ct.info)
        self.assertTrue(ct.exists())

    def _rel(self, path) -> str:
        return Path(path).relative_to(self.ds).as_posix()

    def test_get_changed_path_variants(self):
        ct = self.ct_file()
        rel = self._rel
        cases = {
            rel(ct.get_changed_path()): "derivatives/sub-001/ses-20210101/sub-001_ses-20210101_sequ-301_ct.nii.gz",
            rel(ct.get_changed_path(bids_format="msk", info={"seg": "subreg"})): (
                "derivatives/sub-001/ses-20210101/sub-001_ses-20210101_sequ-301_seg-subreg_msk.nii.gz"
            ),
            rel(ct.get_changed_path(file_type="json")): "derivatives/sub-001/ses-20210101/sub-001_ses-20210101_sequ-301_ct.json",
            rel(ct.get_changed_path(additional_folder="X")): ("derivatives/sub-001/ses-20210101/X/sub-001_ses-20210101_sequ-301_ct.nii.gz"),
            rel(ct.get_changed_path(path="sub-{sub}/custom")): ("derivatives/sub-001/custom/sub-001_ses-20210101_sequ-301_ct.nii.gz"),
            rel(ct.get_changed_path(parent="rawdata2")): "rawdata2/sub-001/ses-20210101/sub-001_ses-20210101_sequ-301_ct.nii.gz",
        }
        for got, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(got, expected)

    def test_get_changed_path_entities_are_bids_sorted(self):
        # `run` must land after `sequ`, regardless of insertion order.
        p = self.ct_file().get_changed_path(info={"run": "2"})
        self.assertEqual(Path(p).name, "sub-001_ses-20210101_sequ-301_run-2_ct.nii.gz")

    def test_get_changed_path_make_parent_creates_the_folder(self):
        p = self.ct_file().get_changed_path(parent="derivatives_new", make_parent=True)
        self.assertTrue(Path(p).parent.is_dir())

    def test_get_changed_bids_wraps_the_same_path(self):
        ct = self.ct_file()
        bids = ct.get_changed_bids(bids_format="msk", info={"seg": "subreg"})
        self.assertIsInstance(bids, BIDS_FILE)
        self.assertEqual(bids.bids_format, "msk")
        self.assertEqual(bids.get("seg"), "subreg")

    def test_insert_info_into_path(self):
        ct = self.ct_file()
        self.assertEqual(ct.insert_info_into_path("{sub}/{ses}/x"), "001/20210101/x")
        self.assertIsNone(ct.insert_info_into_path(None))

    def test_set_remove_roundtrip(self):
        ct = self.ct_file()
        ct.set("run", "3")
        self.assertEqual(ct.get("run"), "3")
        self.assertEqual(ct.remove("run"), "3")
        self.assertIsNone(ct.get("run"))

    def test_get_file_and_open_json(self):
        ct = self.ct_file()
        self.assertTrue(ct.has_json())
        self.assertEqual(ct.get_file("json").name, "sub-001_ses-20210101_sequ-301_ct.json")
        self.assertIsNone(ct.get_file("nrrd", None))
        self.assertEqual(ct.open_json()["EchoTime"], 1.0)

    def test_open_nii_and_grid_info(self):
        ct = self.ct_file()
        self.assertTrue(ct.has_nii())
        nii = ct.open_nii()
        self.assertEqual(nii.shape, SHAPE)
        np.testing.assert_allclose(nii.zoom, ZOOM, rtol=1e-6)
        grid = ct.get_grid_info(add_grid_info_to_json=False)
        self.assertEqual(tuple(grid.shape), SHAPE)
        np.testing.assert_allclose(grid.zoom, ZOOM, rtol=1e-6)

    def test_open_nii_reorient(self):
        nii = self.ct_file().open_nii_reorient(("P", "I", "R"))
        self.assertEqual(nii.orientation, ("P", "I", "R"))

    def test_open_dispatch(self):
        ct = self.ct_file()
        self.assertEqual(ct.open("json")["EchoTime"], 1.0)
        self.assertEqual(ct.open("nii.gz").shape, SHAPE)

    def test_open_poi(self):
        fam = self.subj.get_sequence_files("20210101_301")
        poi = fam["ctd_seg-subreg"][0].open_poi()
        self.assertEqual(sorted(poi.keys()), [(20, 50), (21, 50)])
        np.testing.assert_allclose(poi[20, 50], (1.0, 2.0, 3.0), rtol=1e-6)

    def test_do_filter(self):
        ct = self.ct_file()
        self.assertTrue(ct.do_filter("format", "ct"))
        self.assertFalse(ct.do_filter("format", "msk"))
        self.assertTrue(ct.do_filter("sequ", lambda v: int(v) == 301))
        self.assertTrue(ct.do_filter("Filetype", "nii.gz"))

    def test_get_interpolation_order_is_3_for_images_and_0_for_masks(self):
        self.assertEqual(self.ct_file().get_interpolation_order(), 3)
        msk = next(b for b in self.subj.new_query(flatten=True).loop_list() if b.bids_format == "msk")
        self.assertEqual(msk.get_interpolation_order(), 0)

    def test_dunders(self):
        ct = self.ct_file()
        # A BIDS_FILE acts as a one-element sequence of itself.
        self.assertEqual(len(ct), 1)
        self.assertIs(ct[0], ct)
        self.assertEqual(list(iter(ct)), [ct])
        self.assertIn("ct", str(ct))
        self.assertEqual(ct, ct)
        self.assertNotEqual(ct, "not a bids file")
        self.assertEqual(hash(ct), hash(ct))

    def test_rename_and_symlink_files(self):
        ct = self.ct_file()
        target = self.ds / "derivatives_link" / "sub-001" / "linked_ct.nii.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        ct.symlink_files(target, ending=".nii.gz")
        self.assertTrue(target.exists())

        moved_from = ct.file["nii.gz"]
        renamed = self.ds / "rawdata" / "sub-001" / "ses-20210101" / "renamed_ct.nii.gz"
        ct.rename_files(renamed, ending=".nii.gz")
        self.assertTrue(renamed.exists())
        self.assertFalse(moved_from.exists())
        # restore for the other tests in this class (the dataset is class-scoped)
        renamed.rename(moved_from)


class TestSearchquery(_DatasetTestCase):
    def _flat(self) -> Searchquery:
        return self.subj.new_query(flatten=True)

    @staticmethod
    def _parts(query: Searchquery) -> list[str]:
        return sorted(b.get("part") or b.bids_format for b in query.loop_list())

    def test_flat_query_sees_every_file(self):
        self.assertEqual(len(self._flat().candidates), 9)

    def test_filter_format(self):
        q = self._flat()
        q.filter_format("ct")
        self.assertEqual(self._parts(q), ["ct"])

    def test_filter_is_monotone(self):
        q = self._flat()
        before = len(q.candidates)
        q.filter_format("dixon")
        self.assertLessEqual(len(q.candidates), before)
        self.assertEqual(self._parts(q), ["fat", "inphase", "outphase", "water"])

    def test_copy_is_independent(self):
        q = self._flat()
        c = q.copy()
        c.filter_format("ct")
        self.assertEqual(len(q.candidates), 9)
        self.assertEqual(len(c.candidates), 1)

    def test_flatten_unflatten_roundtrip(self):
        q = self.subj.new_query()
        keys_before = sorted(q.candidates)
        q.flatten()
        self.assertIsInstance(q.candidates, list)
        q.unflatten()
        self.assertEqual(sorted(q.candidates), keys_before)

    def test_filter_self_and_filter_json(self):
        q = self._flat()
        q.filter_self(lambda b: b.bids_format == "msk")
        self.assertEqual(self._parts(q), ["msk", "msk"])

        q = self._flat()
        q.filter_json(lambda j: "EchoTime" in j, required=True)
        self.assertEqual(self._parts(q), ["ct"])

    def test_filter_by_entity(self):
        q = self._flat()
        q.filter("sequ", lambda v: int(v) == 301)
        self.assertEqual(self._parts(q), ["ct", "ctd", "msk", "msk", "snp"])

    def test_filter_filetype(self):
        q = self._flat()
        q.filter_filetype("png")
        self.assertEqual(self._parts(q), ["snp"])

    def test_dixon_channel_filters_select_one_channel_each(self):
        # Regression: the ImageType filter required *all* spellings of a channel
        # ("W" and "WATER"), so every file with a sidecar was dropped and these
        # three filters always returned nothing.
        for channel, method in (("water", "filter_dixon_water"), ("fat", "filter_dixon_fat"), ("outphase", "filter_dixon_outphase")):
            with self.subTest(channel=channel):
                q = self._flat()
                q.filter_format("dixon")
                getattr(q, method)()
                self.assertEqual(self._parts(q), [channel])

    def test_filter_dixon_only_inphase(self):
        q = self._flat()
        q.filter_format("dixon")
        q.filter_dixon_only_inphase()
        self.assertEqual(self._parts(q), ["inphase"])

    def test_filter_non_existence(self):
        q = self._flat()
        q.filter_non_existence("seg", required=False)
        self.assertNotIn("msk", self._parts(q))

    def test_action_sets_a_key_when_the_json_matches(self):
        q = self._flat()
        q.filter_format("dixon")
        q.action(action_fun=lambda b: b.set("part", "real"), filter_fun=lambda j: "IP" in j["ImageType"], key="json", required=True)
        changed = [b for b in q.loop_list() if b.get("part") == "real"]
        self.assertEqual(len(changed), 1)

    def test_loop_dict_yields_families(self):
        q = self.subj.new_query()
        q.filter_format("dixon")
        fams = list(q.loop_dict())
        self.assertEqual(len(fams), 1)
        self.assertIsInstance(fams[0], BIDS_Family)
        self.assertEqual(sorted(fams[0].keys()), ["dixon"])

    def test_loop_list_sorted(self):
        q = self._flat()
        names = [b.get_path_decomposed()[3] for b in q.loop_list(sort=True)]
        self.assertEqual(names, sorted(names))

    def test_str_is_informative(self):
        self.assertIn("001", str(self._flat()))

    def test_from_bids_family_restricts_to_one_sequence(self):
        fam = self.subj.get_sequence_files("20210101_301")
        q = Searchquery.from_BIDS_Family(fam)
        self.assertEqual(sorted(q.candidates), ["20210101_301"])


class TestBidsFamily(_DatasetTestCase):
    def setUp(self):
        super().setUp()
        self.fam = self.subj.get_sequence_files("20210101_301")

    def test_keys_carry_the_naming_suffix(self):
        self.assertEqual(sorted(self.fam.keys()), ["ct", "ctd_seg-subreg", "msk_seg-subreg", "msk_seg-vert", "snp"])

    def test_contains_accepts_str_and_list(self):
        self.assertIn("ct", self.fam)
        self.assertTrue(["ct", "snp"] in self.fam)
        self.assertNotIn("nope", self.fam)

    def test_get_and_getitem(self):
        self.assertEqual(len(self.fam["ct"]), 1)
        self.assertEqual(len(self.fam.get("ct")), 1)
        self.assertIsNone(self.fam.get("nope"))
        self.assertEqual(self.fam.get("nope", []), [])
        self.assertEqual(self.fam.get(["nope", "ct"])[0].bids_format, "ct")

    def test_len_iter_items_keys_values(self):
        self.assertEqual(len(self.fam), 5)
        self.assertEqual(sorted(k for k, _ in self.fam), sorted(self.fam.keys()))
        self.assertEqual(len(list(self.fam.items())), 5)
        self.assertEqual(sum(len(v) for v in self.fam.values()), 5)

    def test_key_and_format_len(self):
        self.assertEqual(self.fam.get_key_len()["ct"], 1)
        self.assertEqual(self.fam.get_format_len()["msk"], (2, 2))

    def test_get_files_and_as_dict(self):
        self.assertEqual({k: len(v) for k, v in self.fam.get_files().items()}, dict.fromkeys(self.fam.keys(), 1))
        as_dict = self.fam.get_bids_files_as_dict(["ct", "msk_seg-vert"])
        self.assertEqual(sorted(as_dict), ["ct", "msk_seg-vert"])
        self.assertEqual(as_dict["ct"].bids_format, "ct")

    def test_identifier(self):
        self.assertEqual(self.fam.get_identifier(), "sub-001_ses-20210101_sequ-301")

    def test_sort_and_dunders(self):
        self.fam.sort()
        self.assertIn("ct", str(self.fam))
        self.assertIn("ct", repr(self.fam))
        self.assertEqual(hash(self.fam), hash(self.fam))

    def test_new_query_from_family(self):
        q = self.fam.new_query()
        self.assertEqual(sorted(q.candidates), ["20210101_301"])


class TestBufferedBidsGlobalInfo(_DatasetTestCase):
    def test_buffer_is_written_and_reused(self):
        cache = self.ds / "rawdata" / ".filepaths"
        self.assertFalse(cache.exists())
        first = Buffered_BIDS_Global_info([self.ds], parents=["rawdata", "derivatives"], verbose=False)
        self.assertTrue(cache.exists(), "the file-path cache should be written next to the scanned parent")

        # Second call reads the cache instead of walking the tree and must agree exactly.
        second = Buffered_BIDS_Global_info([self.ds], parents=["rawdata", "derivatives"], verbose=False)
        self.assertEqual(sorted(first.subjects), sorted(second.subjects))
        self.assertEqual(
            {k: sorted(v.sequences) for k, v in first.subjects.items()},
            {k: sorted(v.sequences) for k, v in second.subjects.items()},
        )

    def test_stale_buffer_is_recomputed(self):
        import os  # noqa: PLC0415

        Buffered_BIDS_Global_info([self.ds], parents=["rawdata"], verbose=False)
        cache = self.ds / "rawdata" / ".filepaths"
        os.utime(cache, (0, 0))  # 1970 -> older than max_age_days
        g = Buffered_BIDS_Global_info([self.ds], parents=["rawdata"], verbose=False, max_age_days=1)
        self.assertGreater(cache.stat().st_mtime, 0)
        self.assertEqual(sorted(g.subjects), ["001", "002"])

    def test_filter_file_restricts_the_cached_paths(self):
        g = Buffered_BIDS_Global_info(
            [self.ds],
            parents=["rawdata"],
            verbose=False,
            filter_file=lambda p: "sub-002" in str(p),
            recompute_parents=["rawdata"],
        )
        self.assertEqual(sorted(g.subjects), ["002"])


if __name__ == "__main__":
    unittest.main()
