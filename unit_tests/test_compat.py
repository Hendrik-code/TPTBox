# Call 'python -m unittest' on this folder
# coverage run -m unittest
# coverage report
# coverage html
from __future__ import annotations

import re
import unittest
from pathlib import Path

from TPTBox.core.compat import _pairwise_py39, pairwise, zip_strict


class TestZipStrict(unittest.TestCase):
    def test_equal_length_lists(self):
        a = [1, 2, 3]
        b = ["a", "b", "c"]
        expected = [(1, "a"), (2, "b"), (3, "c")]
        result = list(zip_strict(a, b))
        self.assertEqual(result, expected)

    def test_unequal_length_lists(self):
        a = [1, 2, 3]
        b = ["a", "b"]
        with self.assertRaises(ValueError) as context:
            list(zip_strict(a, b))
        self.assertIn("Length mismatch", str(context.exception))

    def test_empty_iterables(self):
        a = []
        b = []
        result = list(zip_strict(a, b))
        self.assertEqual(result, [])

    def test_multiple_iterables(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        c = [7, 8, 9]
        expected = [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
        result = list(zip_strict(a, b, c))
        self.assertEqual(result, expected)

    def test_generator_iterables(self):
        a = (x for x in range(3))
        b = (chr(97 + x) for x in range(3))
        expected = [(0, "a"), (1, "b"), (2, "c")]
        result = list(zip_strict(a, b))
        self.assertEqual(result, expected)


class TestPairwise(unittest.TestCase):
    """`pairwise` is a 3.10+ stdlib function; TPTBox supports 3.9, so it is backported."""

    def test_pairs_of_list(self):
        self.assertEqual(list(pairwise([1, 2, 3, 4])), [(1, 2), (2, 3), (3, 4)])

    def test_too_short_inputs_are_empty(self):
        for seq in ([], [1]):
            with self.subTest(seq=seq):
                self.assertEqual(list(pairwise(seq)), [])

    def test_generator_input_is_consumed_lazily(self):
        self.assertEqual(list(pairwise(x * x for x in range(4))), [(0, 1), (1, 4), (4, 9)])

    def test_py39_fallback_matches_stdlib_path(self):
        # Both code paths must agree; on 3.10+ `pairwise` delegates to itertools.
        seq = ["a", "b", "c", "d"]
        self.assertEqual(list(_pairwise_py39(seq)), list(pairwise(seq)))

    def test_py39_fallback_empty(self):
        self.assertEqual(list(_pairwise_py39([])), [])


class TestNoPython310OnlyStdlibAtImport(unittest.TestCase):
    """Guards the 3.9 CI leg: a 3.10+ stdlib import at module scope kills the whole module."""

    def test_no_itertools_pairwise_import(self):
        root = Path(__file__).resolve().parents[1] / "TPTBox"
        compat = root / "core" / "compat.py"  # the one sanctioned place: the backport itself
        offenders = [
            f"{path.relative_to(root.parent)}:{i}"
            for path in root.rglob("*.py")
            if path != compat
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if "from itertools import" in line and "pairwise" in line
        ]
        self.assertEqual(offenders, [], "use TPTBox.core.compat.pairwise instead (python 3.9 support)")

    def test_no_zip_strict_keyword(self):
        # `zip(..., strict=...)` is 3.10+ and raises TypeError at call time on 3.9.
        root = Path(__file__).resolve().parents[1] / "TPTBox"
        offenders = [
            f"{path.relative_to(root.parent)}:{i}"
            for path in root.rglob("*.py")
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"\bzip\([^)]*\bstrict\s*=", line)
        ]
        self.assertEqual(offenders, [], "use TPTBox.core.compat.zip_strict instead (python 3.9 support)")


if __name__ == "__main__":
    unittest.main()
