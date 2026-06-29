"""Backwards-compatibility regression tests for the Loguru-backed logger.

These lock the public contract that must survive the Loguru migration: the terminal
color coding (the `type2bcolors` ANSI + `[*]/[!]/...` prefixes), carriage-return (`end`)
progress lines, `WARNING_THROW` -> warnings.warn, verbose gating, ANSI-free file output,
and exception capture into user sinks. Coloring is now Loguru-native, so we compare after
normalizing the (invisible) ANSI reset codes rather than byte-for-byte.
"""

from __future__ import annotations

import glob
import io
import os
import re
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path

from TPTBox.logger import Log_Type, Print_Logger
from TPTBox.logger.log_constants import color_log_text, type2bcolors
from TPTBox.logger.log_file import Logger, No_Logger


def _cap(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _norm(s: str) -> str:
    """Collapse runs of resets and drop a leading reset (all invisible) for render-equality."""
    s = re.sub(r"(\x1b\[0m)+", "\x1b[0m", s)
    return s.removeprefix("\x1b[0m")


class TestLoggerColorContract(unittest.TestCase):
    def test_each_log_type_renders_like_type2bcolors(self):
        for lt in Log_Type:
            if lt == Log_Type.WARNING_THROW:
                continue  # routed to warnings.warn, asserted separately
            got = _cap(lambda lt=lt: No_Logger().print("Hello World", ltype=lt))
            prefix = type2bcolors[lt][1]
            expected = color_log_text(lt, f"{prefix} Hello World") + "\n"  # the old reference rendering
            self.assertEqual(_norm(got), _norm(expected), f"color/format changed for {lt.name}")

    def test_exact_color_code_present(self):
        for lt in Log_Type:
            if lt in (Log_Type.WARNING_THROW, Log_Type.TEXT, Log_Type.NEUTRAL):
                continue  # throw-type and default (reset) colors
            got = _cap(lambda lt=lt: No_Logger().print("x", ltype=lt))
            self.assertTrue(got.startswith(type2bcolors[lt][0]), f"{lt.name} missing its ANSI color")

    def test_end_carriage_return_preserved(self):
        got = _cap(lambda: No_Logger().print("progress", ltype=Log_Type.SAVE, end="\r"))
        self.assertTrue(got.endswith("\r"))
        self.assertNotIn("\n", got)

    def test_verbose_false_suppresses_terminal(self):
        self.assertEqual(_cap(lambda: No_Logger().print("hidden", ltype=Log_Type.SAVE, verbose=False)), "")

    def test_warning_throw_emits_warning_not_stdout(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _cap(lambda: No_Logger().print("danger", ltype=Log_Type.WARNING_THROW))
        self.assertEqual(out, "")
        self.assertEqual(len(w), 1)

    def test_on_helpers_and_prefix_attr(self):
        self.assertIn("\x1b[92m[+] ok", _cap(lambda: No_Logger().on_ok("ok")))
        self.assertIn("\x1b[91m[!] bad", _cap(lambda: No_Logger().on_fail("bad")))

        def with_prefix():
            lg = No_Logger()
            lg.prefix = "API"
            lg.print("hi", ltype=Log_Type.SAVE)

        self.assertIn("[API] hi", _cap(with_prefix))

    def test_multi_arg_and_positional_ltype(self):
        got = _cap(lambda: No_Logger().print("Saved:", "/p/f", 42, Log_Type.SAVE))
        self.assertIn("[*] Saved: /p/f 42", got)
        self.assertTrue(got.startswith(type2bcolors[Log_Type.SAVE][0]))


class TestLoggerFileBackend(unittest.TestCase):
    def test_file_is_ansi_free_and_well_formed(self):
        d = tempfile.mkdtemp()
        lg = Logger(d, "unit", default_verbose=False)
        lg.print("Saved", "/p/f.nii.gz", ltype=Log_Type.SAVE)
        lg.print("oops", ltype=Log_Type.FAIL)
        lg.close()
        files = glob.glob(os.path.join(d, "logs", "*.log"))
        self.assertEqual(len(files), 1)
        self.assertTrue(os.path.basename(files[0]).endswith("_unit_log.log"))
        content = Path(files[0]).read_text()
        self.assertNotIn("\x1b", content)  # no ANSI in the file
        for needle in ["[#] Log started at:", "[*] Saved /p/f.nii.gz", "[!] oops", "[#] Program duration:"]:
            self.assertIn(needle, content)

    def test_two_loggers_isolated(self):
        d = tempfile.mkdtemp()
        a, b = Logger(d, "AAA"), Logger(d, "BBB")
        a.print("only-A", verbose=False)
        b.print("only-B", verbose=False)
        a.close()
        b.close()
        ca = Path(next(f for f in glob.glob(os.path.join(d, "logs", "*.log")) if "AAA" in f)).read_text()
        self.assertIn("only-A", ca)
        self.assertNotIn("only-B", ca)


class TestExceptionCapture(unittest.TestCase):
    def test_print_error_text_and_structured_record(self):
        import json

        from TPTBox.logger import loguru_logger

        records = []
        sid = loguru_logger.add(lambda m: records.append(str(m)), serialize=True, level=0)
        try:
            out = _cap(self._raise_and_log)
        finally:
            loguru_logger.remove(sid)
        self.assertIn("ZeroDivisionError", out)
        self.assertTrue(out.startswith(type2bcolors[Log_Type.FAIL][0]))  # FAIL-colored text
        recs = [json.loads(r)["record"] for r in records]
        self.assertTrue(any(r.get("exception") and r["exception"]["type"] == "ZeroDivisionError" for r in recs))

    @staticmethod
    def _raise_and_log():
        try:
            _ = 1 / 0
        except ZeroDivisionError:
            Print_Logger().print_error()


if __name__ == "__main__":
    unittest.main()
