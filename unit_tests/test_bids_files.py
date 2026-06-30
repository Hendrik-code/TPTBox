# Call 'python -m pytest unit_tests/test_bids_files.py'
# Drives TPTBox.core.bids_files without the real on-disk test dataset:
#   * the in-memory BIDS index from TPTBox.tests.test_utils.get_BIDS_test()
#   * a couple of tiny temporary BIDS datasets for the file/disk operations
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
import unittest.mock
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import TPTBox.core.bids_files as bids
from TPTBox.core.bids_files import (
    BIDS_FILE,
    BIDS_Family,
    BIDS_Global_info,
    Buffered_BIDS_Global_info,
    Searchquery,
    Subject_Container,
    _scan_tree,
    get_values_from_name,
    validate_entities,
)
from TPTBox.tests.test_utils import a, get_BIDS_test, get_nii, get_poi

# Register the dataset's additional (non-spec) entity keys so that directly
# constructed BIDS_FILEs (i.e. without a BIDS_Global_info) treat them as legal.
# This mirrors what BIDS_Global_info.__init__ does with `additional_key`.
for _k in ("sequ", "seg", "ovl", "e"):
    bids.entities.setdefault(_k, _k)
    bids.entities_keys.setdefault(_k, _k)

# A fake dataset root used by the no-disk path-manipulation tests. Nothing is
# written here; only string/relative-path arithmetic is exercised.
_DS = "/media/robert/Expansion/dataset-Testset"


@contextlib.contextmanager
def _silent():
    """Swallow the (very chatty) stdout that the BIDS machinery emits."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _bids() -> BIDS_Global_info:
    """A fresh in-memory BIDS_Global_info built from the filename list ``a``."""
    with _silent():
        return get_BIDS_test()


def _file(name: str, parent: str = "rawdata", sub: str = "sub-a/ses-1", verbose: bool = False) -> BIDS_FILE:
    """Construct a no-disk BIDS_FILE living under the fake dataset root."""
    return BIDS_FILE(f"{_DS}/{parent}/{sub}/{name}", _DS, verbose=verbose)


# ---------------------------------------------------------------------------
# module level functions
# ---------------------------------------------------------------------------
class Test_module_functions(unittest.TestCase):
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_validate_entities_valid(self, mock_stdout):
        for key, value in [("sub", "001"), ("ses", "20210101"), ("seg", "vert"), ("echo", "3"), ("hemi", "L"), ("mt", "on")]:
            with self.subTest(key=key):
                self.assertTrue(validate_entities(key, value, "name", verbose=True))
        self.assertEqual(mock_stdout.getvalue(), "")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_validate_entities_invalid(self, mock_stdout):
        # unknown key, non-decimal echo, non-alnum task, bad hemi, bad mt, bad mod-format
        cases = [
            ("thiskeydoesnotexist", "x"),
            ("echo", "abc"),
            ("task", "a-b"),
            ("hemi", "X"),
            ("mt", "maybe"),
            ("mod", "notaformat"),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                self.assertFalse(validate_entities(key, value, "name", verbose=True))
        self.assertNotEqual(mock_stdout.getvalue(), "")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_validate_entities_non_verbose(self, mock_stdout):
        # verbose=False is a hard short-circuit: always True, never prints
        self.assertTrue(validate_entities("totallybogus", "###", "name", verbose=False))
        self.assertEqual(mock_stdout.getvalue(), "")

    def test_get_values_from_name(self):
        fmt, info, key, ft = get_values_from_name("sub-spinegan0026_ses-409_sequ-203_seg-subreg_ctd.json", verbose=False)
        self.assertEqual(fmt, "ctd")
        self.assertEqual(ft, "json")
        self.assertEqual(key, "sub-spinegan0026_ses-409_sequ-203_seg-subreg_ctd")
        self.assertEqual(info, {"sub": "spinegan0026", "ses": "409", "sequ": "203", "seg": "subreg"})

        # additional key 'e' + nii.gz
        fmt, info, _, ft = get_values_from_name("sub-spinegan0026_ses-411_sequ-301_e-1_dixon.nii.gz", verbose=False)
        self.assertEqual((fmt, ft), ("dixon", "nii.gz"))
        self.assertEqual(info["e"], "1")

        # additional key 'ovl'
        fmt, info, _, ft = get_values_from_name("sub-spinegan0026_ses-411_sequ-301_e-3_ovl-ctd_snp.png", verbose=False)
        self.assertEqual((fmt, ft), ("snp", "png"))
        self.assertEqual(info["ovl"], "ctd")
        self.assertEqual(info["e"], "3")

        # sequ-None branch: the literal string 'None'
        fmt, info, _, _ = get_values_from_name("sub-spinegan0042_ses-417_sequ-None_ct.nii.gz", verbose=False)
        self.assertEqual(fmt, "ct")
        self.assertEqual(info["sequ"], "None")

    def test_get_values_from_name_covers_list(self):
        # parse every entry of `a`; the stem must re-assemble from key/values + format
        for name in a:
            with self.subTest(name=name):
                fmt, info, key, ft = get_values_from_name(name, verbose=False)
                self.assertEqual(name, f"{key}.{ft}")
                self.assertTrue(key.endswith(fmt))
                self.assertEqual(info.get("sub"), name.split("_")[0].split("-")[1])

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_get_values_from_name_verbose_warns(self, mock_stdout):
        # not starting with sub-, a bare token without KEY-VALUE -> warnings printed
        get_values_from_name("notsub-1_brokentoken_ct.json", verbose=True)
        self.assertNotEqual(mock_stdout.getvalue(), "")

    def test_scan_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.txt").write_text("x")
            (root / ".hidden").write_text("x")
            sub = root / "sub"
            sub.mkdir()
            (sub / "b.txt").write_text("x")
            names = sorted(e.name for e in _scan_tree(root))
            self.assertEqual(names, ["a.txt", "b.txt"])  # recursive, hidden skipped


# ---------------------------------------------------------------------------
# BIDS_Global_info
# ---------------------------------------------------------------------------
class Test_BIDS_Global_info(unittest.TestCase):
    def setUp(self):
        self.g = _bids()

    def test_len_and_str(self):
        self.assertEqual(len(self.g), 2)
        self.assertIn("BIDS_Global_info", str(self.g))
        self.assertIsInstance(self.g._global_bids_list, dict)

    def test_enumerate_and_iter(self):
        self.assertEqual(len(list(self.g.enumerate_subjects())), 2)
        names_sorted = [n for n, _ in self.g.enumerate_subjects(sort=True)]
        self.assertEqual(names_sorted, sorted(names_sorted))
        self.assertEqual(names_sorted, ["spinegan0026", "spinegan0042"])
        # shuffle returns the same set of subjects
        self.assertEqual({n for n, _ in self.g.enumerate_subjects(shuffle=True)}, set(names_sorted))
        self.assertEqual({n for n, _ in self.g.iter_subjects()}, set(names_sorted))
        self.assertEqual([n for n, _ in self.g.iter_subjects(sort=True)], names_sorted)
        self.assertEqual({n for n, _ in self.g.iter_subjects(shuffle=True)}, set(names_sorted))
        for _, subj in self.g.enumerate_subjects():
            self.assertIsInstance(subj, Subject_Container)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_add_file_2_subject_edge_cases(self, _mock_stdout):
        before = len(self.g.subjects)
        # DS_Store is skipped silently
        self.g.add_file_2_subject(Path(".DS_Store"), "")
        # a file without a '.'-type declaration is skipped
        self.g.add_file_2_subject(Path("file_without_a_type"), "")
        self.assertEqual(len(self.g.subjects), before)
        # a plain Path without a dataset raises
        with self.assertRaises(AssertionError):
            self.g.add_file_2_subject(Path("sub-z_ses-1_ct.nii.gz"), None)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_dataset_name_warning(self, mock_stdout):
        # a dataset name not starting with 'dataset-' triggers a warning, no raise
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d, "not-a-dataset-name")
            bad.mkdir()
            BIDS_Global_info([bad], parents=["rawdata"], additional_key=["sequ", "seg", "ovl", "e"], verbose=False)
        self.assertIn("dataset-", mock_stdout.getvalue())


# ---------------------------------------------------------------------------
# Subject_Container
# ---------------------------------------------------------------------------
class Test_Subject_Container(unittest.TestCase):
    def setUp(self):
        self.g = _bids()
        self.subj = self.g.subjects["spinegan0042"]

    def test_get_sequence_name_and_new_query(self):
        f = self.subj.sequences["417_406"][0]
        self.assertEqual(self.subj.get_sequence_name(f), "417_406")
        self.assertIsInstance(self.subj.new_query(), Searchquery)
        self.assertTrue(self.subj.new_query(flatten=True)._flatten)

    def test_get_sequence_files_default(self):
        fam = self.subj.get_sequence_files("417_406")
        self.assertIsInstance(fam, BIDS_Family)
        self.assertEqual(sorted(fam.keys()), ["ct", "ctd_seg-subreg", "msk_seg-subreg", "msk_seg-vert", "snp"])
        self.assertEqual(fam.family_id, "sub-spinegan0042_ses-417_sequ-406")

    def test_get_sequence_files_key_transform(self):
        def mapping(x: BIDS_FILE):
            if x.format == "ctd" and x.info.get("seg") == "subreg":
                return "other_key_word"
            return None

        fam = self.subj.get_sequence_files("417_406", key_transform=mapping)
        self.assertIn("other_key_word", fam)
        self.assertNotIn("ctd_seg-subreg", fam)

    def test_get_sequence_files_key_addendum(self):
        # 'e' becomes part of the family key -> the 3 dixon echoes split apart
        fam = self.subj.get_sequence_files("417_301", key_addendum=["e"])
        for k in ("dixon_e-1", "dixon_e-2", "dixon_e-3"):
            self.assertIn(k, fam)

    def test_get_sequence_files_alternative_list(self):
        alt = self.subj.sequences["417_406"][:2]
        fam = self.subj.get_sequence_files("417_406", alternative_sequ_list=alt)
        self.assertEqual(len(fam), 2)


# ---------------------------------------------------------------------------
# Searchquery (in-memory)
# ---------------------------------------------------------------------------
class Test_Searchquery(unittest.TestCase):
    def setUp(self):
        self.g = _bids()
        self.subj = self.g.subjects["spinegan0042"]

    def test_flatten_unflatten_roundtrip(self):
        q = self.subj.new_query()  # dict mode
        self.assertIsInstance(q.candidates, dict)
        q.flatten()
        self.assertIsInstance(q.candidates, list)
        flat_count = len(q.candidates)
        q.flatten()  # idempotent
        self.assertEqual(len(q.candidates), flat_count)
        q.unflatten()
        self.assertIsInstance(q.candidates, dict)
        q.unflatten()  # idempotent

    def test_filter_format_variants(self):
        q = self.subj.new_query(flatten=True)
        q.filter_format("ct")
        self.assertEqual(len(q.candidates), 3)
        for c in q.candidates:
            self.assertEqual(c.format, "ct")
        # list form
        q2 = self.subj.new_query(flatten=True)
        q2.filter_format(["ct", "dixon"])
        self.assertTrue(all(c.format in ("ct", "dixon") for c in q2.candidates))
        # callable form
        q3 = self.subj.new_query(flatten=True)
        q3.filter_format(lambda x: x == "snp")
        self.assertTrue(all(c.format == "snp" for c in q3.candidates))

    def test_filter_filetype_and_lambda(self):
        q = self.subj.new_query(flatten=True)
        q.filter_format("ct")
        q.filter_filetype(".json")  # leading dot stripped internally
        self.assertTrue(all("json" in c.file for c in q.candidates))
        # numeric-string lambda on the 'sequ' entity
        q2 = self.subj.new_query(flatten=True)
        q2.filter("sequ", lambda x: x != "None" and int(x) == 406, required=True)
        self.assertTrue(all(c.get("sequ") == "406" for c in q2.candidates))

    def test_filter_self(self):
        q = self.subj.new_query(flatten=True)
        q.filter_self(lambda b: b.format == "ct")
        self.assertEqual(len(q.candidates), 3)

    def test_filter_non_existence(self):
        q = self.subj.new_query()  # 6 sequence buckets
        n0 = len(q.candidates)
        q.filter_non_existence("format", "dixon")
        self.assertEqual(len(q.candidates), n0 - 3)  # 3 dixon sequences removed

    def test_copy_is_independent(self):
        q = self.subj.new_query(flatten=True)
        q.filter_format("ct")
        c = q.copy()
        self.assertIsNot(c, q)
        self.assertIsNot(c.candidates, q.candidates)
        self.assertEqual(len(c.candidates), len(q.candidates))

    def test_loop_list_and_loop_dict(self):
        q = self.subj.new_query()
        q.filter("format", "ct")
        fams = list(q.loop_dict(sort=True))
        self.assertTrue(all(isinstance(f, BIDS_Family) for f in fams))
        q.flatten()
        files = list(q.loop_list(sort=True))
        self.assertTrue(all(isinstance(f, BIDS_FILE) for f in files))
        # loop_list asserts flatten, loop_dict asserts not-flatten
        with self.assertRaises(AssertionError):
            list(q.loop_dict())

    def test_action(self):
        q = self.subj.new_query(flatten=True)
        seen: list[str] = []
        q.action(action_fun=lambda x: seen.append(x.format), key="format", filter_fun="ct")
        self.assertEqual(seen, ["ct", "ct", "ct"])
        # all_in_sequence requires unflatten mode
        q2 = self.subj.new_query()
        touched: list = []
        q2.action(action_fun=lambda x: touched.append(x), key="format", filter_fun="dixon", all_in_sequence=True)
        self.assertTrue(len(touched) > 0)
        with self.assertRaises(AssertionError):
            self.subj.new_query(flatten=True).action(action_fun=lambda x: x, all_in_sequence=True)

    def test_from_bids_family(self):
        fam = self.subj.get_sequence_files("417_301")
        q = Searchquery.from_BIDS_Family(fam)
        self.assertFalse(q._flatten)
        self.assertEqual(list(q.candidates.keys()), ["417_301"])

    def test_str(self):
        q = self.subj.new_query()
        self.assertIn("spinegan0042", str(q))
        q.flatten()
        self.assertIn("spinegan0042", str(q))


# ---------------------------------------------------------------------------
# Searchquery dixon filters (require json sidecars on disk)
# ---------------------------------------------------------------------------
class Test_Searchquery_dixon(unittest.TestCase):
    def setUp(self):
        import json

        self.tmp = tempfile.TemporaryDirectory()
        ds = Path(self.tmp.name, "dataset-Dixon")
        raw = ds / "rawdata" / "sub-x" / "ses-1"
        raw.mkdir(parents=True)
        nii, _, _, _ = get_nii(x=(8, 8, 8), num_point=1)
        # ImageType lists are crafted to satisfy the strict all()/membership checks
        echoes = {
            "1": ["ORIGINAL", "PRIMARY", "W", "WATER"],
            "2": ["ORIGINAL", "PRIMARY", "F", "FAT"],
            "3": ["ORIGINAL", "PRIMARY", "IP"],
        }
        for e, itype in echoes.items():
            stem = f"sub-x_ses-1_sequ-301_e-{e}_dixon"
            with _silent():
                nii.save(raw / (stem + ".nii.gz"), verbose=False)
            (raw / (stem + ".json")).write_text(json.dumps({"ImageType": itype, "FrameOfReferenceUID": "1.2.3.4.5"}))
        with _silent():
            self.g = BIDS_Global_info([ds], parents=["rawdata"], additional_key=["sequ", "seg", "ovl", "e"], verbose=False)
        self.subj = self.g.subjects["x"]

    def tearDown(self):
        self.tmp.cleanup()

    def _dixon_query(self):
        q = self.subj.new_query(flatten=True)
        q.filter_format("dixon")
        return q

    def test_dixon_water(self):
        q = self._dixon_query()
        q.filter_dixon_water()
        self.assertEqual(sorted(c.get("e") for c in q.candidates), ["1"])

    def test_dixon_fat(self):
        q = self._dixon_query()
        q.filter_dixon_fat()
        self.assertEqual(sorted(c.get("e") for c in q.candidates), ["2"])

    def test_dixon_outphase_none(self):
        q = self._dixon_query()
        q.filter_dixon_outphase()
        self.assertEqual(list(q.candidates), [])

    def test_dixon_only_inphase(self):
        q = self._dixon_query()
        q.filter_dixon_only_inphase()
        self.assertEqual(sorted(c.get("e") for c in q.candidates), ["3"])

    def test_dixon_water_requires_flatten(self):
        q = self.subj.new_query()  # unflatten
        with self.assertRaises(AssertionError):
            q.filter_dixon_water()


# ---------------------------------------------------------------------------
# BIDS_Family
# ---------------------------------------------------------------------------
class Test_BIDS_Family(unittest.TestCase):
    def setUp(self):
        self.g = _bids()
        self.subj = self.g.subjects["spinegan0026"]
        # dixon family: ctd_seg-subreg x1, dixon x3, snp x3, msk x1
        self.fam = self.subj.get_sequence_files("411_301")

    def test_getitem_get_and_keyerror(self):
        self.assertIsInstance(self.fam["dixon"], list)
        self.assertEqual(len(self.fam["dixon"]), 3)
        self.assertIsNotNone(self.fam.get(["missing", "dixon"]))
        self.assertEqual(self.fam.get("missing", default="DEF"), "DEF")
        with self.assertRaises(KeyError):
            _ = self.fam["does_not_exist"]

    def test_items_keys_values(self):
        self.assertEqual(len(list(self.fam.items())), len(self.fam.keys()))
        self.assertEqual(len(self.fam.values()), len(self.fam.keys()))
        self.assertEqual(set(dict(self.fam).keys()), set(self.fam.keys()))  # __iter__

    def test_contains_and_len(self):
        self.assertIn("dixon", self.fam)
        self.assertTrue(["dixon", "snp"] in self.fam)
        self.assertFalse(["dixon", "nope"] in self.fam)
        self.assertEqual(len(self.fam), 8)  # total underlying files

    def test_key_len_and_format_len(self):
        self.assertEqual(self.fam.get_key_len()["dixon"], 3)
        fl = self.fam.get_format_len()
        self.assertEqual(fl["dixon"], (1, 3))

    def test_get_files_and_multiples(self):
        self.assertEqual(len(self.fam.get_files("dixon")["dixon"]), 3)
        self.assertEqual(len(self.fam.get_files()), len(self.fam.keys()))  # all keys
        multiples = self.fam.get_files_with_multiples()
        self.assertEqual(set(multiples.keys()), {"dixon", "snp"})

    def test_sort_and_setitem(self):
        self.fam["zzz_extra"] = self.fam["dixon"]
        self.fam.sort()
        keys = list(self.fam.keys())
        self.assertEqual(keys, sorted(keys))

    def test_new_query(self):
        q = self.fam.new_query()
        self.assertFalse(q._flatten)
        self.assertEqual(list(q.candidates.keys()), ["411_301"])
        qf = self.fam.new_query(flatten=True)
        self.assertTrue(qf._flatten)

    def test_get_bids_files_as_dict(self):
        d = self.fam.get_bids_files_as_dict(["dixon", "snp"])
        self.assertEqual(set(d.keys()), {"dixon", "snp"})
        self.assertTrue(all(isinstance(v, BIDS_FILE) for v in d.values()))
        with self.assertRaises(KeyError):
            self.fam.get_bids_files_as_dict(["dixon", "missingkey"])

    def test_dunders(self):
        other = self.subj.get_sequence_files("409_203")
        self.assertEqual(self.fam < other, str(self.fam) < str(other))
        self.assertIsInstance(hash(self.fam), int)
        self.assertEqual(str(self.fam), repr(self.fam))
        self.assertIn("dixon", str(self.fam))


# ---------------------------------------------------------------------------
# BIDS_FILE without disk access (pure parsing / path arithmetic)
# ---------------------------------------------------------------------------
class Test_BIDS_FILE_nodisk(unittest.TestCase):
    def test_parse_and_accessors(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        self.assertEqual(f.format, "ct")
        self.assertEqual(f.bids_format, "ct")
        self.assertEqual(f.get("sub"), "a")
        self.assertIsNone(f.get("missing"))
        self.assertEqual(f.get("missing", default="d"), "d")
        f.set("seg", "vert")
        self.assertEqual(f.get("seg"), "vert")
        self.assertIn("seg", dict(f.loop_keys()))
        self.assertEqual(f.remove("seg"), "vert")
        self.assertNotIn("seg", f.info)
        with self.assertRaises(AssertionError):
            f.remove("sub")  # subject is protected

    def test_get_file(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        self.assertEqual(f.get_file("nii.gz"), f.file["nii.gz"])
        self.assertIsNone(f.get_file("json"))
        self.assertEqual(f.get_file("json", default="d"), "d")

    def test_mod_property(self):
        ct = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        self.assertEqual(ct.mod, "ct")
        msk = _file("sub-a_ses-1_sequ-2_mod-T1w_seg-vert_msk.nii.gz")
        self.assertEqual(msk.mod, "T1w")  # msk resolves to the 'mod' entity

    def test_path_decomposed_and_parent(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        ds, parent, subpath, filename = f.get_path_decomposed()
        self.assertEqual(str(ds), _DS)
        self.assertEqual(parent, "rawdata")
        self.assertEqual(subpath, "sub-a/ses-1")
        self.assertEqual(filename, "sub-a_ses-1_sequ-2_ct.nii.gz")
        self.assertEqual(f.parent, "rawdata")
        self.assertEqual(f.get_parent(), "rawdata")

    def test_get_changed_path(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        p = f.get_changed_path(file_type="json", bids_format="ctd", parent="derivatives", info={"seg": "subreg"})
        self.assertEqual(p.name, "sub-a_ses-1_sequ-2_seg-subreg_ctd.json")
        self.assertEqual(p.parts[-4], "derivatives")
        # from_info + a {key} template path + an additional folder
        p2 = f.get_changed_path(from_info=True, path="sub-{sub}", additional_folder="extra", bids_format="msk")
        self.assertIn("extra", p2.parts)
        self.assertIn("sub-a", p2.parts)

    def test_get_changed_path_auto_run_id(self):
        # target never exists -> the run loop returns on the first iteration
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        p = f.get_changed_path(auto_add_run_id=True, bids_format="ct")
        self.assertTrue(p.name.endswith("_ct.nii.gz"))

    def test_get_changed_path_non_strict(self):
        f = _file("weirdname_ct.nii.gz")  # BIDS_key does not start with sub
        self.assertFalse(f.BIDS_key.startswith("sub"))
        p = f.get_changed_path(bids_format="ct", non_strict_mode=True)
        self.assertTrue(p.name.endswith("_ct.nii.gz"))
        self.assertIn("sub-weirdname-ct", p.name)

    def test_get_changed_bids(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        nf = f.get_changed_bids(file_type="nii.gz", bids_format="msk", info={"seg": "vert"})
        self.assertIsInstance(nf, BIDS_FILE)
        self.assertEqual(nf.BIDS_key, "sub-a_ses-1_sequ-2_seg-vert_msk")

    def test_insert_info_into_path(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        self.assertEqual(f.insert_info_into_path("sub-{sub}/ses-{ses}"), "sub-a/ses-1")
        self.assertIsNone(f.insert_info_into_path(None))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(f.insert_info_into_path("x-{missingkey}"), "x-missingkey")

    def test_get_identifier(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        self.assertEqual(f.get_identifier(["ses", "sequ"]), "sub-a_ses-1_sequ-2")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_get_identifier_no_sub(self, _mock_stdout):
        f = _file("weirdname_ct.nii.gz")
        self.assertEqual(f.get_identifier(["ses"]), "sub-404")

    def test_dunders(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        g = _file("sub-b_ses-1_sequ-2_ct.nii.gz")
        self.assertTrue(f < g)
        self.assertEqual(len(f), 1)
        self.assertIs(f[0], f)
        self.assertEqual(list(f), [f])
        self.assertEqual(hash(f), hash(f.BIDS_key))
        # __eq__ compares BIDS_key; a json companion with same stem compares equal
        same = _file("sub-a_ses-1_sequ-2_ct.json")
        self.assertEqual(f, same)
        self.assertNotEqual(f, "not a bids file")
        self.assertIn("rawdata", str(f))
        self.assertEqual(str(f), repr(f))

    def test_do_filter(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        self.assertFalse(f.do_filter("", "x"))
        self.assertTrue(f.do_filter("format", "ct"))
        self.assertTrue(f.do_filter("format", ["ct", "msk"]))
        self.assertTrue(f.do_filter("format", lambda v: v == "ct"))
        self.assertTrue(f.do_filter("filetype", ".nii.gz"))  # dot stripped
        self.assertTrue(f.do_filter("parent", "rawdata"))
        self.assertTrue(f.do_filter("self", lambda b: isinstance(b, BIDS_FILE)))
        self.assertTrue(f.do_filter("sequ", "2"))
        # absent key: inverse of `required`
        self.assertTrue(f.do_filter("acq", "x", required=False))
        self.assertFalse(f.do_filter("acq", "x", required=True))

    def test_get_interpolation_order(self):
        self.assertEqual(_file("sub-a_ses-1_sequ-2_ct.nii.gz").get_interpolation_order(), 3)
        self.assertEqual(_file("sub-a_ses-1_sequ-2_seg-vert_msk.nii.gz").get_interpolation_order(), 0)
        self.assertEqual(_file("sub-a_ses-1_sequ-2_label-1_ct.nii.gz").get_interpolation_order(), 0)

    def test_add_file(self):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        f.add_file(Path(_DS + "/rawdata/sub-a/ses-1/sub-a_ses-1_sequ-2_ct.json"))
        self.assertIn("json", f.file)
        with self.assertRaises(AssertionError):
            f.add_file(Path(_DS + "/rawdata/sub-a/ses-1/sub-a_ses-1_sequ-99_ct.json"))

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_open_deprecated_and_npz(self, _mock_stdout):
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertIsNone(f.open("json"))  # not registered -> None
        self.assertFalse(f.has_npz())


# ---------------------------------------------------------------------------
# BIDS_FILE with a real (temporary) BIDS dataset on disk
# ---------------------------------------------------------------------------
class Test_BIDS_FILE_disk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ds = Path(self.tmp.name, "dataset-Test")
        self.dir = self.ds / "rawdata" / "sub-x" / "ses-1"
        self.dir.mkdir(parents=True)
        self.nii, _, _, _ = get_nii(x=(16, 16, 16), num_point=1)
        self.msk_path = self.dir / "sub-x_ses-1_sequ-1_seg-vert_msk.nii.gz"
        with _silent():
            self.nii.save(self.msk_path, verbose=False)

    def tearDown(self):
        self.tmp.cleanup()

    def _msk(self) -> BIDS_FILE:
        return BIDS_FILE(self.msk_path, self.ds, verbose=False)

    def test_exists_has_nii_open_nii(self):
        f = self._msk()
        self.assertTrue(f.exists())
        self.assertTrue(f.has_nii())
        self.assertEqual(f.get_nii_file(), self.msk_path)
        self.assertEqual(f.open_nii().shape, self.nii.shape)

    def test_open_nii_reorient(self):
        f = self._msk()
        reoriented = f.open_nii_reorient(("P", "I", "R"))
        self.assertEqual(reoriented.orientation, ("P", "I", "R"))

    def test_json_sidecar(self):
        import json

        json_path = self.dir / "sub-x_ses-1_sequ-1_seg-vert_msk.json"
        json_path.write_text(json.dumps({"hello": "world"}))
        f = self._msk()  # companion json auto-detected at construction
        self.assertTrue(f.has_json())
        self.assertEqual(f.open_json()["hello"], "world")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_get_grid_info(self, _mock_stdout):
        f = self._msk()
        self.assertFalse(f.has_json())
        grid = f.get_grid_info(add_grid_info_to_json=True)
        self.assertEqual(tuple(grid.shape), tuple(self.nii.shape))
        self.assertTrue(f.has_json())  # json was created on the fly

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_open_poi_full_metadata(self, _mock_stdout):
        ctd = self.dir / "sub-x_ses-1_sequ-1_seg-subreg_ctd.json"
        with _silent():
            get_poi().save(ctd, verbose=False, save_hint=2)
        f = BIDS_FILE(ctd, self.ds, verbose=False)
        poi = f.open_poi()
        self.assertIsNotNone(poi.zoom)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_open_poi_needs_nii(self, _mock_stdout):
        from TPTBox.core.poi import POI

        # a grid-less POI: shape is None on reload
        ctd = self.dir / "sub-x_ses-1_sequ-1_ctd.json"
        msk = self.dir / "sub-x_ses-1_sequ-1_msk.nii.gz"
        with _silent():
            self.nii.save(msk, verbose=False)
            POI({1: {50: (1.0, 2.0, 3.0)}}, orientation=("R", "A", "S")).save(ctd, verbose=False, save_hint=2)
        f = BIDS_FILE(ctd, self.ds, verbose=False)
        # auto-detect: '..._ctd.json' -> sibling '..._msk.nii.gz' fills the grid
        poi = f.open_poi()
        self.assertEqual(tuple(poi.shape), tuple(self.nii.shape))
        # explicitly handing in the reference works too
        poi2 = f.open_poi(nii=msk)
        self.assertIsNotNone(poi2.zoom)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_save_changed_path(self, _mock_stdout):
        f = self._msk()
        f.save_changed_path(parent="derivatives", info={"seg": "vert"}, bids_format="msk")
        out = list((self.ds / "derivatives").rglob("*_msk.nii.gz"))
        self.assertEqual(len(out), 1)

    def test_rename_files(self):
        f = self._msk()
        target = self.dir / "sub-x_ses-1_sequ-9_seg-vert_msk.nii.gz"
        f.rename_files(target, ending=".nii.gz")
        self.assertTrue(target.exists())
        self.assertFalse(self.msk_path.exists())

    def test_symlink_files(self):
        f = self._msk()
        target = self.dir / "sub-x_ses-1_sequ-8_seg-vert_msk.nii.gz"
        f.symlink_files(target, ending=".nii.gz", exist_ok=True)
        self.assertTrue(target.is_symlink())
        f.symlink_files(target, ending=".nii.gz", exist_ok=True)  # second call: no-op
        with self.assertRaises(AssertionError):
            f.symlink_files("wrong_suffix.json", ending=".nii.gz")

    def test_unlink(self):
        f = self._msk()
        self.assertTrue(f.exists())
        f.unlink()
        self.assertFalse(f.exists())

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_frame_of_reference_uid(self, _mock_stdout):
        import json

        f = self._msk()
        # no json -> falls back to the 'ses' entity
        self.assertEqual(f.get_frame_of_reference_uid(default="z"), "1")
        json_path = self.dir / "sub-x_ses-1_sequ-1_seg-vert_msk.json"
        json_path.write_text(json.dumps({"FrameOfReferenceUID": "1.2.3.4.5"}))
        f2 = self._msk()
        uid = f2.get_frame_of_reference_uid()
        self.assertEqual(len(uid), 8)
        self.assertTrue(uid.isalnum())

    def test_get_sequence_files_requires_subject(self):
        f = self._msk()  # constructed directly, never attached to a Subject_Container
        with self.assertRaises(AssertionError):
            f.get_sequence_files()


# ---------------------------------------------------------------------------
# Buffered_BIDS_Global_info
# ---------------------------------------------------------------------------
class Test_Buffered_BIDS_Global_info(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ds = Path(self.tmp.name, "dataset-Buf")
        raw = self.ds / "rawdata" / "sub-x" / "ses-1"
        raw.mkdir(parents=True)
        nii, _, _, _ = get_nii(x=(8, 8, 8), num_point=1)
        with _silent():
            nii.save(raw / "sub-x_ses-1_sequ-1_ct.nii.gz", verbose=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_buffer_create_then_read(self):
        with _silent():
            g1 = Buffered_BIDS_Global_info([self.ds], parents=["rawdata"], additional_key=["sequ", "seg", "ovl", "e"], verbose=False)
        self.assertTrue((self.ds / "rawdata" / ".filepaths").exists())  # cache written
        self.assertIn("x", g1.subjects)
        # second call reads the cache back
        with _silent():
            g2 = Buffered_BIDS_Global_info([self.ds], parents=["rawdata"], additional_key=["sequ", "seg", "ovl", "e"], verbose=False)
        self.assertIn("x", g2.subjects)

    def test_buffer_with_filter_file(self):
        with _silent():
            g = Buffered_BIDS_Global_info(
                self.ds,  # single (non-list) dataset path
                parents=["rawdata"],
                additional_key=["sequ", "seg", "ovl", "e"],
                verbose=False,
                filter_file=lambda p: p.name.endswith(".nii.gz"),
            )
        self.assertIn("x", g.subjects)

    def test_buffer_missing_parent(self):
        with _silent():
            g = Buffered_BIDS_Global_info([self.ds], parents=["does_not_exist"], additional_key=["sequ", "seg", "ovl", "e"], verbose=False)
        self.assertEqual(len(g), 0)


# ---------------------------------------------------------------------------
# extra branch coverage
# ---------------------------------------------------------------------------
class Test_extra_branches(unittest.TestCase):
    def setUp(self):
        self.g = _bids()
        self.subj = self.g.subjects["spinegan0042"]

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_find_changed_path(self, _mock_stdout):
        f = self.subj.sequences["417_406"][0]
        # the in-memory global list is empty, so this lookup returns None either way,
        # but both filename-assembly branches are exercised
        self.assertIsNone(f.find_changed_path(self.g, bids_format="msk", info={"seg": "vert"}))
        self.assertIsNone(f.find_changed_path(self.g, bids_format="ctd", from_info=True))

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_get_changed_path_non_strict_with_info(self, _mock_stdout):
        # a normal file with populated info, in non_strict_mode -> validate (not assert) path
        f = _file("sub-a_ses-1_sequ-2_ct.nii.gz")
        p = f.get_changed_path(bids_format="ct", non_strict_mode=True, info={"acq": "ax"})
        self.assertTrue(p.name.endswith("_ct.nii.gz"))

    def test_filter_non_existence_flatten(self):
        q = self.subj.new_query(flatten=True)
        before = len(q.candidates)
        q.filter_non_existence("format", "dixon")
        self.assertTrue(all(c.format != "dixon" for c in q.candidates))
        self.assertLess(len(q.candidates), before)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_add_file_2_subject_bids_file_infers_ds(self, _mock_stdout):
        # passing a BIDS_FILE with ds=None infers the dataset from the file itself
        bf = _file("sub-newsub_ses-1_sequ-2_ct.nii.gz")
        self.g.add_file_2_subject(bf, None)
        self.assertIn("newsub", self.g.subjects)

    def test_open_ctd_and_open_dispatch_and_exists(self):
        with tempfile.TemporaryDirectory() as d:
            ds = Path(d, "dataset-T")
            sub = ds / "rawdata" / "sub-x" / "ses-1"
            sub.mkdir(parents=True)
            nii, _, _, _ = get_nii(x=(12, 12, 12), num_point=1)
            ctd = sub / "sub-x_ses-1_sequ-1_seg-subreg_ctd.json"
            with _silent():
                get_poi().save(ctd, verbose=False, save_hint=2)
            f = BIDS_FILE(ctd, ds, verbose=False)
            # json-only file -> exists() takes the non-nii.gz branch
            self.assertTrue(f.exists())
            self.assertIsNotNone(f.open_ctd())  # alias of open_poi
            # deprecated open() dispatch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self.assertIsNotNone(f.open("json"))  # open() -> open_json()


if __name__ == "__main__":
    unittest.main()
