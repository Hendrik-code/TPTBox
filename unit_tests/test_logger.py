"""Unit tests for the logging infrastructure in TPTBox/logger/log_file.py."""

from __future__ import annotations

import io
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

from TPTBox.logger import Print_Logger
from TPTBox.logger.log_constants import Log_Type
from TPTBox.logger.log_file import (
    Logger,
    No_Logger,
    Reflection_Logger,
    String_Logger,
    _set_indent,
    indentation_level,
    sub_log_call_func,
)

ALL_LOG_TYPES = list(Log_Type)


class Test_No_Logger(unittest.TestCase):
    def setUp(self):
        _set_indent(0)

    def tearDown(self):
        _set_indent(0)

    def test_print_logger_is_no_logger(self):
        # Print_Logger is exported as an alias of No_Logger.
        self.assertIs(Print_Logger, No_Logger)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_print_default_verbose_true(self, mock_stdout):
        No_Logger().print("hello world")
        self.assertIn("hello world", mock_stdout.getvalue())

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_print_all_log_types(self, mock_stdout):
        log = No_Logger()
        for ltype in ALL_LOG_TYPES:
            if ltype == Log_Type.WARNING_THROW:
                continue  # handled separately — it warns instead of printing
            with self.subTest(ltype=ltype):
                log.print("payload", ltype=ltype, verbose=True)
        self.assertIn("payload", mock_stdout.getvalue())

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_on_methods(self, mock_stdout):
        log = No_Logger()
        log.on_fail("f")
        log.on_save("s")
        log.on_ok("o")
        log.on_warning("w")
        log.on_text("t")
        log.on_neutral("n")
        log.on_log("l")
        log.on_bold("b")
        log.on_debug("d")
        out = mock_stdout.getvalue()
        for token in ("f", "s", "o", "w", "t", "n", "l", "b", "d"):
            self.assertIn(token, out)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_drop_in_replacement_methods(self, mock_stdout):
        log = No_Logger()
        log.warning("a-warn")
        log.error("a-error")
        log.info("a-info")
        out = mock_stdout.getvalue()
        self.assertIn("a-warn", out)
        self.assertIn("a-error", out)
        self.assertIn("a-info", out)

    def test_warning_throw_raises_warning(self):
        with self.assertWarns(Warning):
            No_Logger().print("careful", ltype=Log_Type.WARNING_THROW, verbose=True)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_print_empty_is_blank(self, mock_stdout):
        from TPTBox.logger.log_constants import _clean_all_color_from_text

        No_Logger().print(verbose=True)
        # Empty print emits no prefix, just a newline (ignoring ANSI color codes).
        self.assertNotIn("[", _clean_all_color_from_text(mock_stdout.getvalue()))

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_log_type_embedded_in_text(self, mock_stdout):
        # If a Log_Type is among the args (and ltype is default TEXT), it is used.
        No_Logger().print("important", Log_Type.OK, verbose=True)
        out = mock_stdout.getvalue()
        self.assertIn("important", out)
        self.assertIn("[+]", out)  # OK prefix

    def test_preprocess_text_with_prefix(self):
        log = No_Logger()
        out = log._preprocess_text(("msg",), ltype=Log_Type.OK)
        self.assertIn("msg", out)
        self.assertTrue(out.startswith("[+]"))

    def test_preprocess_text_ignore_prefix(self):
        log = No_Logger()
        out = log._preprocess_text(("msg",), ltype=Log_Type.OK, ignore_prefix=True)
        self.assertEqual(out, "msg")

    def test_preprocess_text_custom_prefix(self):
        log = No_Logger(prefix="ABC")
        out = log._preprocess_text(("msg",), ltype=Log_Type.OK)
        self.assertIn("[ABC]", out)

    def test_preprocess_text_with_dict(self):
        log = No_Logger()
        out = log._preprocess_text(({"a": 1, "b": "x"},), ltype=Log_Type.TEXT)
        self.assertIn("'a'", out)
        self.assertIn("'b'", out)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_print_error_logs_traceback(self, mock_stdout):
        log = No_Logger()
        try:
            int("not a number")
        except ValueError:
            log.print_error()
        self.assertIn("ValueError", mock_stdout.getvalue())

    def test_add_sub_logger_returns_self(self):
        log = No_Logger()
        self.assertIs(log.add_sub_logger("x"), log)

    def test_noop_methods(self):
        log = No_Logger()
        # All of these must be safe no-ops.
        self.assertIsNone(log.flush())
        self.assertIsNone(log.close())
        sub = String_Logger(finalize=False)
        self.assertIsNone(log.flush_sub_logger(sub))

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_print_log_started(self, mock_stdout):
        No_Logger(print_log_started=True)
        self.assertIn("Log started at", mock_stdout.getvalue())

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_log_statistic_and_print_statistic(self, mock_stdout):
        log = No_Logger()
        log.log_statistic("dice", 0.5, key2="caseA")
        log.log_statistic("dice", 0.7, key2="caseB")
        log.log_statistic("dice", 0.9)  # key2 defaults to current count
        log.print_statistic()
        out = mock_stdout.getvalue()
        self.assertIn("dice", out)
        self.assertIn("Accumulated Statistics", out)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_print_statistic_without_state(self, mock_stdout):
        log = No_Logger()
        log.print_statistic()
        self.assertIn("No Accumulated Statistics", mock_stdout.getvalue())


class Test_Indentation(unittest.TestCase):
    def setUp(self):
        _set_indent(0)

    def tearDown(self):
        _set_indent(0)

    def test_set_indent_bool(self):
        self.assertEqual(_set_indent(True), 1)
        self.assertEqual(_set_indent(True), 2)
        self.assertEqual(_set_indent(False), 1)
        self.assertEqual(_set_indent(False), 0)
        # cannot go below zero
        self.assertEqual(_set_indent(False), 0)

    def test_set_indent_int(self):
        self.assertEqual(_set_indent(5), 5)
        self.assertEqual(_set_indent(0), 0)

    def test_context_manager_changes_indent(self):
        log = No_Logger()
        import TPTBox.logger.log_file as lf

        self.assertEqual(lf.indentation_level, 0)
        with log:
            self.assertEqual(lf.indentation_level, 1)
        self.assertEqual(lf.indentation_level, 0)

    def test_prefix_indentation_level(self):
        log = No_Logger()
        self.assertEqual(log._prefix_indentation_level(), "")
        _set_indent(2)
        self.assertNotEqual(log._prefix_indentation_level(), "")

    def test_module_level_indent_global_exists(self):
        # smoke test that the module exports the global.
        self.assertEqual(indentation_level, 0)


class Test_String_Logger(unittest.TestCase):
    def setUp(self):
        _set_indent(0)

    def tearDown(self):
        _set_indent(0)

    def test_logs_to_string(self):
        log = String_Logger(finalize=False)
        log.print("hello", ltype=Log_Type.OK)
        self.assertIn("hello", log.log_content)
        self.assertIn("hello", log.log_content_colored)
        # plain content has no ANSI escapes
        self.assertNotIn("\x1b", log.log_content)
        # colored content does
        self.assertIn("\x1b", log.log_content_colored)

    def test_close_returns_tuple(self):
        log = String_Logger(finalize=False)
        log.print("content")
        plain, colored = log.close()
        self.assertIn("content", plain)
        self.assertIn("Sub-process duration", plain)
        self.assertIsInstance(colored, str)

    def test_as_sub_logger_sets_head(self):
        head = No_Logger()
        sub = String_Logger.as_sub_logger(head_logger=head, default_verbose=False)
        self.assertIs(sub.head_logger, head)

    def test_flush_forwards_to_head(self):
        head = Logger_for_temp()
        try:
            sub = String_Logger.as_sub_logger(head_logger=head, default_verbose=False)
            head.sub_loggers.append(sub)
            sub.print("forwarded text", verbose=False)
            sub.flush()
            head.flush()
            content = _read_logfile(head)
            self.assertIn("forwarded text", content)
        finally:
            head.remove()

    def test_close_with_nested_sub_loggers(self):
        log = String_Logger(finalize=False)
        nested = String_Logger(finalize=False)
        nested.print("nested content")
        log.sub_loggers.append(nested)
        plain, _ = log.close()
        self.assertIn("sub logger", plain)

    def test_flush_sub_logger_is_noop(self):
        log = String_Logger(finalize=False)
        other = String_Logger(finalize=False)
        self.assertIsNone(log.flush_sub_logger(other))

    def test_finalize_true_registers_finalizer(self):
        # default finalize=True wires up a weakref finalizer.
        log = String_Logger()
        self.assertTrue(log._finalizer.alive)
        log.print("x")
        log.close()


class Test_Log_Constants(unittest.TestCase):
    def test_get_formatted_time(self):
        from TPTBox.logger.log_constants import _format_time, get_formatted_time, get_time

        self.assertIsInstance(get_formatted_time(), str)
        self.assertIsInstance(_format_time(get_time()), str)


class Test_Reflection_Logger(unittest.TestCase):
    def setUp(self):
        _set_indent(0)

    def tearDown(self):
        _set_indent(0)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_print_to_terminal(self, mock_stdout):
        Reflection_Logger().print("reflected", verbose=True)
        self.assertIn("reflected", mock_stdout.getvalue())

    def test_delegates_to_logger(self):
        target = String_Logger(finalize=False)
        Reflection_Logger().print("delegated", verbose=target)
        self.assertIn("delegated", target.log_content)

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_default_ltype_neutral(self, mock_stdout):
        Reflection_Logger().print("plain", verbose=True)
        # NEUTRAL prefix is "[ ]"
        self.assertIn("[ ]", mock_stdout.getvalue())


def Logger_for_temp(default_verbose: bool = False, **kw) -> Logger:
    """Create a file-backed Logger rooted in a fresh temp dir (caller must remove())."""
    tmp = tempfile.mkdtemp()
    return Logger(tmp, "tmplog", default_verbose=default_verbose, **kw)


def _read_logfile(logger: Logger) -> str:
    if not logger.f.closed:
        logger.flush()
    log_files = list(Path(logger.f.name).parent.glob("*.log"))
    return "\n".join(p.read_text(encoding="utf-8") for p in log_files)


class Test_Logger(unittest.TestCase):
    def setUp(self):
        _set_indent(0)

    def tearDown(self):
        _set_indent(0)

    def test_creates_log_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "mylog")
            try:
                self.assertTrue(Path(tmp, "logs").exists())
                self.assertTrue(Path(log.f.name).exists())
            finally:
                log.remove()

    def test_print_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "mylog")
            try:
                log.print("file message", verbose=False)
                content = _read_logfile(log)
                self.assertIn("file message", content)
            finally:
                log.remove()

    def test_log_filename_as_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, {"sub": "001", "ses": "002"})
            try:
                self.assertIn("sub-001_ses-002", Path(log.f.name).name)
            finally:
                log.remove()

    def test_log_arguments_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "args", log_arguments={"alpha": 1, "beta": 2})
            try:
                content = _read_logfile(log)
                self.assertIn("alpha", content)
                self.assertIn("Run with arguments", content)
            finally:
                log.remove()

    def test_log_arguments_non_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "args", log_arguments=["x", "y"])
            try:
                content = _read_logfile(log)
                self.assertIn("Run with arguments", content)
            finally:
                log.remove()

    def test_sub_logger_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "sublog")
            try:
                sub = log.add_sub_logger("childA")
                self.assertIsInstance(sub, String_Logger)
                sub.print("child message", verbose=False)
                log.flush_sub_logger(sub)
                content = _read_logfile(log)
                self.assertIn("child message", content)
                self.assertIn("Flushed sub logger", content)
                # sub content cleared after flush
                self.assertEqual(sub.log_content, "")
            finally:
                log.remove()

    def test_sub_logger_close_forwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "sublog")
            try:
                sub = log.add_sub_logger("childB")
                sub.print("closing child", verbose=False)
                sub.close()  # flushes to head with closed=True
                content = _read_logfile(log)
                self.assertIn("closing child", content)
            finally:
                log.remove()

    def test_close_writes_duration_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "closetest")
            sub = log.add_sub_logger("leftover")
            sub.print("kept around", verbose=False)
            self.assertFalse(log.removed)
            log.close()
            content = _read_logfile(log)
            self.assertIn("Program duration", content)
            # the unflushed sub-logger content is dumped on close
            self.assertIn("kept around", content)
            log.remove()
            self.assertTrue(log.removed)

    def test_log_statistic_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "stats")
            try:
                log.log_statistic("metric", 1.23456789, verbose=False)
                log.log_statistic("metric", 2.0, verbose=False)
                log.print_statistic()
                content = _read_logfile(log)
                self.assertIn("metric", content)
                self.assertIn("Accumulated Statistics", content)
            finally:
                log.remove()

    def test_create_from_bids(self):
        with tempfile.TemporaryDirectory() as tmp:
            bids_like = types.SimpleNamespace(dataset=Path(tmp))
            log = Logger.create_from_bids(bids_like, "frombids", override_prefix="PFX")
            try:
                self.assertTrue(Path(tmp, "logs").exists())
                self.assertEqual(log.prefix, "PFX")
            finally:
                log.remove()

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_sub_log_call_func(self, _mock_stdout):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "subcall")
            try:

                def worker(name, logger):
                    logger.print("inside worker", verbose=False)
                    return name.upper()

                result = sub_log_call_func("task", log, worker)
                self.assertEqual(result, "TASK")
            finally:
                log.remove()


if __name__ == "__main__":
    unittest.main()
