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
from TPTBox.logger.log_constants import type2bcolors
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


_TS_RE = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"


class TestLoggerColorContract(unittest.TestCase):
    def test_each_log_type_renders_like_type2bcolors(self):
        # Format: `<color>[prefix] YYYY-MM-DD HH:MM:SS body<reset>\n`. `[]` comes first,
        # then the timestamp, no level name. Compare against a regex per Log_Type.
        for lt in Log_Type:
            if lt == Log_Type.WARNING_THROW:
                continue  # routed to warnings.warn, asserted separately
            got = _cap(lambda lt=lt: No_Logger().print("Hello World", ltype=lt))
            color = re.escape(type2bcolors[lt][0])
            prefix = re.escape(type2bcolors[lt][1])
            if color and type2bcolors[lt][0] != "\x1b[0m":
                pattern = rf"^{color}{prefix} {_TS_RE} Hello World\x1b\[0m\n$"
            else:
                # TEXT/NEUTRAL currently map to the reset code; the emitter skips
                # coloring so no ANSI is present.
                pattern = rf"^{prefix} {_TS_RE} Hello World\n$"
            self.assertRegex(got, pattern, f"color/format changed for {lt.name}")

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

    def test_empty_print_emits_only_newline(self):
        # `logger.print()` with no args must produce exactly one blank line — no prefix,
        # no timestamp, no color codes.
        self.assertEqual(_cap(lambda: No_Logger().print()), "\n")

    def test_warning_throw_emits_warning_not_stdout(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _cap(lambda: No_Logger().print("danger", ltype=Log_Type.WARNING_THROW))
        self.assertEqual(out, "")
        self.assertEqual(len(w), 1)

    def test_on_helpers_and_prefix_attr(self):
        self.assertRegex(_cap(lambda: No_Logger().on_ok("ok")), rf"\x1b\[92m\[\+\] {_TS_RE} ok")
        self.assertRegex(_cap(lambda: No_Logger().on_fail("bad")), rf"\x1b\[91m\[!\] {_TS_RE} bad")

        def with_prefix():
            lg = No_Logger()
            lg.prefix = "API"
            lg.print("hi", ltype=Log_Type.SAVE)

        self.assertRegex(_cap(with_prefix), rf"\[API\] {_TS_RE} hi")

    def test_multi_arg_and_positional_ltype(self):
        got = _cap(lambda: No_Logger().print("Saved:", "/p/f", 42, Log_Type.SAVE))
        self.assertRegex(got, rf"\[\*\] {_TS_RE} Saved: /p/f 42")
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
        # Each line begins with `[prefix] YYYY-MM-DD HH:MM:SS <body>` — assert the pattern
        # per needle rather than the exact literal.
        for prefix, body in [("#", "Log started at:"), (r"\*", "Saved /p/f.nii.gz"), ("!", "oops"), ("#", "Program duration:")]:
            self.assertRegex(content, rf"\[{prefix}\] {_TS_RE} {re.escape(body)}")

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


class TestLoggerProxyAndSilence(unittest.TestCase):
    def test_logger_proxy_forwards_to_current_default(self):
        from TPTBox.logger import get_default_logger, logger
        from TPTBox.logger.log_file import String_Logger, _set_default_logger

        previous = get_default_logger()
        buf = String_Logger()
        _set_default_logger(buf)
        try:
            logger.on_ok("hello")
        finally:
            _set_default_logger(previous)
        self.assertIn("hello", buf.log_content)

    def test_log_is_alias_of_logger(self):
        from TPTBox.logger import log, logger

        self.assertIs(log, logger)

    def test_vert_constants_log_is_proxy(self):
        from TPTBox.core.vert_constants import log as vc_log
        from TPTBox.logger import logger

        self.assertIs(vc_log, logger)

    def test_configure_logging_silent_kill_switch(self):
        from TPTBox.logger import configure_logging, get_default_logger
        from TPTBox.logger._loguru_backend import _set_verbosity_level
        from TPTBox.logger.log_file import No_Logger, _set_default_logger

        previous = get_default_logger()
        configure_logging(silent=True)
        try:
            for method in ("on_text", "on_ok", "on_fail", "on_warning", "on_neutral"):
                self.assertEqual(_cap(lambda m=method: getattr(No_Logger(), m)("x")), "", f"{method} not silenced")
        finally:
            _set_default_logger(previous)
            _set_verbosity_level(0)

    def test_configure_logging_warnings_capture_round_trip(self):
        import warnings

        from TPTBox.logger import configure_logging, get_default_logger
        from TPTBox.logger.log_file import String_Logger, _set_default_logger

        previous = get_default_logger()
        buf = String_Logger()
        _set_default_logger(buf)
        configure_logging(capture_warnings=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("captured!", UserWarning, stacklevel=1)
        finally:
            configure_logging(capture_warnings=False)
            _set_default_logger(previous)
        self.assertIn("UserWarning: captured!", buf.log_content)

    def test_capture_exceptions_logs_and_reraises(self):
        from TPTBox.logger import get_default_logger
        from TPTBox.logger.log_file import String_Logger, _capture_exceptions, _set_default_logger

        previous = get_default_logger()
        buf = String_Logger()
        _set_default_logger(buf)

        @_capture_exceptions
        def boom():
            raise ValueError("kaboom")

        try:
            with self.assertRaises(ValueError):
                boom()
        finally:
            _set_default_logger(previous)
        self.assertIn("ValueError", buf.log_content)
        self.assertIn("kaboom", buf.log_content)


class TestConfigureLogging(unittest.TestCase):
    def test_configure_logging_end_to_end(self):
        import tempfile
        from pathlib import Path

        from TPTBox.logger import configure_logging, get_default_logger, logger
        from TPTBox.logger._loguru_backend import _set_verbosity_level
        from TPTBox.logger.log_file import _set_default_logger

        previous = get_default_logger()
        d = tempfile.mkdtemp()
        cfg = configure_logging(file=d, filename="pipeline", level="WARNING", verbose=True)
        try:
            self.assertEqual(cfg["level"], 30)
            self.assertTrue(cfg["verbose"])
            self.assertIsNotNone(cfg["file"])

            # Level threshold gates terminal
            self.assertEqual(_cap(lambda: logger.on_ok("hidden")), "")
            self.assertNotEqual(_cap(lambda: logger.on_warning("visible")), "")

            # File captures both regardless
            get_default_logger().close()
            files = list(Path(d).glob("logs/*.log"))
            self.assertEqual(len(files), 1)
            content = files[0].read_text()
            self.assertIn("hidden", content)
            self.assertIn("visible", content)
        finally:
            _set_default_logger(previous)
            _set_verbosity_level(0)

    def test_configure_logging_no_args_returns_current_config(self):
        from TPTBox.logger import configure_logging

        cfg = configure_logging()
        self.assertIn("level", cfg)
        self.assertIn("verbose", cfg)
        self.assertIn("capture_warnings", cfg)
        self.assertIn("capture_exceptions", cfg)

    def test_readme_symbols_all_importable(self):
        """Guard against fabricated names in the README code fences."""
        import importlib

        tp = importlib.import_module("TPTBox")
        for name in (
            "logger",
            "log",
            "configure_logging",
            "get_default_logger",
            "loguru_logger",
            "Logger",
            "Print_Logger",
            "No_Logger",
            "String_Logger",
            "Reflection_Logger",
            "Logger_Interface",
            "Log_Type",
        ):
            self.assertTrue(hasattr(tp, name), f"README cites `TPTBox.{name}` but it isn't exported")


class TestGlobalLevelAndDefaultLogger(unittest.TestCase):
    def test_configure_logging_level_filters_below_threshold(self):
        from TPTBox.logger import configure_logging
        from TPTBox.logger._loguru_backend import _set_verbosity_level

        configure_logging(level="WARNING")  # 30 — drops TEXT (20), OK (25)
        try:
            self.assertEqual(_cap(lambda: No_Logger().on_text("hidden")), "")
            self.assertEqual(_cap(lambda: No_Logger().on_ok("hidden")), "")
            self.assertNotEqual(_cap(lambda: No_Logger().on_warning("visible")), "")
            self.assertNotEqual(_cap(lambda: No_Logger().on_fail("visible")), "")
        finally:
            _set_verbosity_level(0)

    def test_default_logger_roundtrip(self):
        from TPTBox.logger import get_default_logger
        from TPTBox.logger.log_file import String_Logger, _set_default_logger

        previous = get_default_logger()
        buf = String_Logger()
        _set_default_logger(buf)
        try:
            self.assertIs(get_default_logger(), buf)
        finally:
            _set_default_logger(previous)


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
