"""Unit tests for `TPTBox.logger`.

`String_Logger` accumulates into `.log_content`, so almost everything here needs
neither files nor stdout capture. `indentation_level` is a module global shared by
every logger instance, so it is asserted back to zero after each test.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from TPTBox.logger import log_file
from TPTBox.logger.log_constants import (
    Log_Type,
    _clean_all_color_from_text,
    _convert_seconds,
    _dict_to_string,
    _format_time,
    color_log_text,
    datatype_to_string,
    format_time_short,
    get_formatted_time,
    get_time,
)
from TPTBox.logger.log_file import (
    Logger,
    No_Logger,
    Reflection_Logger,
    String_Logger,
    _set_indent,
    print_to_terminal,
    sub_log_call_func,
)


class _IndentGuard(unittest.TestCase):
    def setUp(self):
        _set_indent(0)

    def tearDown(self):
        # `indentation_level` is a module global; a leak would silently indent later tests.
        self.assertEqual(log_file.indentation_level, 0)
        _set_indent(0)


class TestStringLogger(_IndentGuard):
    def test_print_joins_parts_and_adds_a_prefix(self):
        log = String_Logger()
        log.print("a", "b")
        self.assertIn("a b", log.log_content)
        self.assertTrue(log.log_content.endswith("\n"))
        self.assertTrue(log.log_content.startswith("["), "default prefix is a bracketed marker")

    def test_plain_content_is_colour_free_and_colored_content_is_not(self):
        log = String_Logger()
        log.print("hello", ltype=Log_Type.OK)
        self.assertNotIn("\x1b", log.log_content)
        self.assertIn("\x1b", log.log_content_colored)
        self.assertEqual(_clean_all_color_from_text(log.log_content_colored), log.log_content)

    def test_ignore_prefix(self):
        log = String_Logger()
        log.print("naked", ignore_prefix=True)
        self.assertEqual(log.log_content, "naked\n")

    def test_text_already_starting_with_a_bracket_is_not_double_prefixed(self):
        log = String_Logger()
        log.print("[x] already prefixed")
        self.assertEqual(log.log_content, "[x] already prefixed\n")

    def test_empty_print_emits_a_bare_newline(self):
        log = String_Logger()
        log.print()
        self.assertEqual(log.log_content, "\n")

    def test_custom_prefix_is_used_for_every_line(self):
        log = String_Logger(prefix="XYZ")
        log.print("one")
        log.print("two")
        self.assertEqual(len([line for line in log.log_content.splitlines() if "XYZ" in line]), 2)

    def test_log_type_passed_inside_text_is_stripped_but_still_selects_the_type(self):
        log = String_Logger()
        log.print("message", Log_Type.FAIL)
        self.assertNotIn("Log_Type", log.log_content)
        self.assertIn("message", log.log_content)

    def test_end_is_honoured(self):
        log = String_Logger()
        log.print("x", end="")
        self.assertEqual(log.log_content, "[*] x")

    def test_dict_values_are_rendered(self):
        log = String_Logger()
        log.print({"a": 1})
        self.assertIn("'a'", log.log_content)
        self.assertIn("1", log.log_content)

    def test_on_shortcuts_all_write(self):
        for name in ("on_fail", "on_warning", "on_ok", "on_text", "on_neutral", "on_debug", "on_save", "on_log"):
            with self.subTest(method=name):
                log = String_Logger()
                getattr(log, name)("msg")
                self.assertIn("msg", log.log_content)

    def test_print_error_captures_the_traceback(self):
        log = String_Logger()
        try:
            raise RuntimeError("boom")  # noqa: TRY301
        except RuntimeError:
            log.print_error(verbose=False)
        self.assertIn("RuntimeError", log.log_content)
        self.assertIn("boom", log.log_content)

    def test_close_returns_both_buffers_and_reports_the_duration(self):
        log = String_Logger(finalize=False)
        log.print("content")
        plain, colored = log.close()
        self.assertIn("content", plain)
        self.assertIn("Sub-process duration", plain)
        self.assertIn("\x1b", colored)


class TestIndentation(_IndentGuard):
    def test_context_manager_indents_and_restores(self):
        log = String_Logger()
        log.print("flat")
        with log:
            self.assertEqual(log_file.indentation_level, 1)
            log.print("nested")
            with log:
                self.assertEqual(log_file.indentation_level, 2)
                log.print("deeper")
        self.assertEqual(log_file.indentation_level, 0)

        lines = [line for line in log.log_content.splitlines() if line]
        self.assertFalse(lines[0].startswith(" -"))
        self.assertTrue(lines[1].startswith(" - "), lines[1])
        self.assertTrue(lines[2].startswith(" ---- "), lines[2])

    def test_set_indent_bool_steps_and_int_sets(self):
        self.assertEqual(_set_indent(True), 1)
        self.assertEqual(_set_indent(True), 2)
        self.assertEqual(_set_indent(False), 1)
        self.assertEqual(_set_indent(7), 7)
        self.assertEqual(_set_indent(0), 0)

    def test_set_indent_never_goes_negative(self):
        self.assertEqual(_set_indent(False), 0)
        self.assertEqual(_set_indent(False), 0)


class TestStatistics(_IndentGuard):
    def test_log_statistic_accumulates_under_key_and_key2(self):
        log = String_Logger()
        log.log_statistic("dice", 0.5, key2="case1", verbose=False)
        log.log_statistic("dice", 0.7, key2="case2", verbose=False)
        self.assertEqual(log.logging_state, {"dice": {"case1": 0.5, "case2": 0.7}})

    def test_key2_defaults_to_a_running_index(self):
        log = String_Logger()
        log.log_statistic("dice", 1.0, verbose=False)
        log.log_statistic("dice", 2.0, verbose=False)
        self.assertEqual(sorted(log.logging_state["dice"]), [0, 1])

    def test_print_statistic_reports_mean_std_median_and_count(self):
        log = String_Logger()
        for v in (1.0, 2.0, 3.0):
            log.log_statistic("metric", v, verbose=False)
        log.print_statistic()
        text = log.log_content
        self.assertIn("Accumulated Statistics", text)
        self.assertIn("2.0", text)  # mean
        self.assertIn("0.816", text)  # std of 1,2,3
        self.assertIn("±", text)  # the character that forces utf-8 log files
        self.assertIn("metric", text)

    def test_print_statistic_without_data(self):
        log = String_Logger()
        log.print_statistic()
        self.assertIn("No Accumulated Statistics", log.log_content)

    def test_logging_state_is_per_instance(self):
        a, b = String_Logger(), String_Logger()
        a.log_statistic("x", 1.0, verbose=False)
        self.assertIsNone(b.logging_state)


class TestSubLoggers(_IndentGuard):
    def test_flush_appends_into_the_parent_and_clears_the_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Logger(tmp, "test", default_verbose=False)
            sub = parent.add_sub_logger("child")
            sub.print("from the child", verbose=False)
            self.assertIn("from the child", sub.log_content)
            sub.flush()
            self.assertEqual(sub.log_content, "")
            self.assertIn(sub, parent.sub_loggers)
            parent.close()
            self.assertIn("from the child", self._read_log(tmp))

    def test_close_removes_the_child_from_the_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Logger(tmp, "test", default_verbose=False)
            sub = parent.add_sub_logger("child")
            sub.print("bye", verbose=False)
            sub.close()
            self.assertNotIn(sub, parent.sub_loggers)
            parent.close()
            text = self._read_log(tmp)
            self.assertIn("bye", text)
            self.assertIn("Sub-process duration", text)

    def test_unflushed_child_content_is_written_by_the_parents_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Logger(tmp, "test", default_verbose=False)
            sub = parent.add_sub_logger("child")
            sub.print("never flushed", verbose=False)
            parent.close()
            text = self._read_log(tmp)
            self.assertIn("Found 1 sub logger", text)
            self.assertIn("never flushed", text)

    def test_sub_log_call_func_passes_name_and_sub_logger(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Logger(tmp, "test", default_verbose=False)
            seen = {}

            def work(name, logger, extra):
                seen["name"] = name
                seen["is_sub"] = isinstance(logger, String_Logger)
                logger.print("worked on", extra, verbose=False)
                return extra * 2

            result = sub_log_call_func("job", parent, work, extra=21)
            self.assertEqual(result, 42)
            self.assertEqual(seen, {"name": "job", "is_sub": True})
            parent.close()

    @staticmethod
    def _read_log(folder: str) -> str:
        logs = sorted(Path(folder, "logs").glob("*.log"))
        assert len(logs) == 1, logs
        return logs[0].read_text(encoding="utf-8")


class TestFileLogger(_IndentGuard):
    def test_creates_the_logs_folder_and_a_named_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "myrun", default_verbose=False)
            log.print("hello", verbose=False)
            log.flush()
            logs = list(Path(tmp, "logs").glob("*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("myrun", logs[0].name)
            self.assertIn("hello", logs[0].read_text(encoding="utf-8"))
            log.close()

    def test_dict_filename_becomes_bids_like_key_value_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, {"sub": "001", "task": "seg"}, default_verbose=False)
            name = next(iter(Path(tmp, "logs").glob("*.log"))).name
            self.assertIn("sub-001", name)
            self.assertIn("task-seg", name)
            log.close()

    def test_log_arguments_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "r", default_verbose=False, log_arguments={"lr": 0.1})
            log.close()
            text = TestSubLoggers._read_log(tmp)
            self.assertIn("Run with arguments", text)
            self.assertIn("lr", text)
            self.assertIn("0.1", text)

    def test_log_arguments_non_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "r", default_verbose=False, log_arguments="--flag")
            log.close()
            self.assertIn("--flag", TestSubLoggers._read_log(tmp))

    def test_close_is_idempotent_and_writes_the_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "r", default_verbose=False)
            log.close()
            log.close()
            self.assertIn("Program duration", TestSubLoggers._read_log(tmp))

    def test_remove_marks_the_logger_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "r", default_verbose=False)
            self.assertFalse(log.removed)
            log.remove()
            self.assertTrue(log.removed)

    def test_log_file_is_utf8_so_the_statistics_table_can_be_written(self):
        # print_statistic emits "±"; a C/POSIX locale would raise UnicodeEncodeError
        # if the file were opened with the platform default encoding.
        with tempfile.TemporaryDirectory() as tmp:
            log = Logger(tmp, "r", default_verbose=False)
            log.log_statistic("m", 1.0, verbose=False)
            log.print_statistic()
            log.close()
            self.assertIn("±", TestSubLoggers._read_log(tmp))

    def test_create_from_bids_uses_the_dataset_folder(self):
        from TPTBox.core.bids_files import BIDS_FILE  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "dataset-x"
            f = ds / "rawdata" / "sub-001" / "sub-001_ct.nii.gz"
            f.parent.mkdir(parents=True)
            f.touch()
            bids = BIDS_FILE(f, ds, verbose=False)
            log = Logger.create_from_bids(bids, "frombids", default_verbose=False)
            self.assertTrue((ds / "logs").is_dir())
            log.close()


class TestNoLoggerAndReflection(_IndentGuard):
    def test_no_logger_swallows_everything(self):
        log = No_Logger()
        log.print("to terminal only")
        log.flush()
        log.close()
        self.assertIs(log.add_sub_logger("x"), log)

    def test_no_logger_start_banner(self):
        log = No_Logger(print_log_started=True)
        self.assertTrue(hasattr(log, "start_time"))

    def test_reflection_logger_delegates_to_the_logger_given_as_verbose(self):
        target = String_Logger()
        Reflection_Logger().print("delegated", verbose=target)
        self.assertIn("delegated", target.log_content)


class TestLogConstants(unittest.TestCase):
    def test_colour_roundtrip(self):
        for ltype in Log_Type:
            with self.subTest(ltype=ltype):
                self.assertEqual(_clean_all_color_from_text(color_log_text(ltype, "text")), "text")

    def test_convert_seconds(self):
        self.assertEqual(_convert_seconds(0), "0:00:00 h:mm:ss")
        self.assertEqual(_convert_seconds(61), "0:01:01 h:mm:ss")
        self.assertEqual(_convert_seconds(3661), "1:01:01 h:mm:ss")

    def test_format_time_short(self):
        import time as _time  # noqa: PLC0415

        t = _time.struct_time((2024, 1, 15, 9, 30, 0, 0, 15, 0))
        self.assertEqual(format_time_short(t), "date-2024-1-15_time-9-30-0")
        self.assertIn("2024", _format_time(t))

    def test_get_time_and_formatted_time(self):
        self.assertTrue(hasattr(get_time(), "tm_year"))
        self.assertTrue(get_formatted_time().startswith("date-"))

    def test_datatype_to_string(self):
        self.assertEqual(datatype_to_string(5, Log_Type.TEXT), "5")
        self.assertEqual(datatype_to_string("s", Log_Type.TEXT), "s")
        self.assertIn("'k'", datatype_to_string({"k": "v"}, Log_Type.TEXT))

    def test_dict_to_string_quotes_str_keys_and_values(self):
        text = _clean_all_color_from_text(_dict_to_string({"k": "v", 1: 2}))
        self.assertIn("'k': 'v'", text)
        self.assertIn("1: 2", text)

    def test_print_to_terminal_warning_throw_raises_a_warning(self):
        with self.assertWarns(Warning):
            print_to_terminal("careful", end="\n", ltype=Log_Type.WARNING_THROW)


if __name__ == "__main__":
    unittest.main()
