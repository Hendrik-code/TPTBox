# Logger (`TPTBox.logger`)

Consistent, colored, structured logging for TPTBox pipelines. One import for
library code, one function to configure everything.

## Quick start

```python
from TPTBox import configure_logging, logger

configure_logging(file="run.log", verbose=True)
logger.on_ok("segmentation finished")
logger.on_warning("no cord found in sub-007")
logger.on_fail("could not open file")
```

Terminal output:

```
[+] 2026-07-07 08:11:53 segmentation finished
[?] 2026-07-07 08:11:53 no cord found in sub-007
[!] 2026-07-07 08:11:53 could not open file
```

The same lines (ANSI-free) also land in the log file next to `run.log`.

## Everything in one function

```python
configure_logging(
    file=None,                # str | Path — enable file logging
    filename=None,            # str — stem when `file` is a directory
    level=None,               # int | str | Log_Type — terminal severity threshold
    verbose=None,             # bool — default_verbose on the default logger
    silent=False,             # bool — nothing prints (kill-switch)
    capture_warnings=None,    # bool — warnings.warn → logger
    capture_exceptions=None,  # bool — uncaught exceptions → logger
    loguru_sink=None,         # dict — extra Loguru sink (e.g. structured JSON)
)
```

Call with no arguments to inspect the current config:

```python
>>> configure_logging()
{'file': PosixPath('.../logs/....log'), 'level': 30, 'verbose': True,
 'capture_warnings': False, 'capture_exceptions': False}
```

### Recipes

```python
# Terminal + file for everything
configure_logging(file="run.log", verbose=True)

# Only warnings/errors on terminal; file still gets everything
configure_logging(file="run.log", verbose=True, level="WARNING")

# File-only, no terminal
configure_logging(file="run.log", verbose=False)

# Silence everything
configure_logging(silent=True)

# Capture warnings and uncaught exceptions
configure_logging(capture_warnings=True, capture_exceptions=True)

# Attach a structured JSON sink alongside the terminal/file output
configure_logging(loguru_sink={"sink": "run.jsonl", "serialize": True,
                               "level": "TPTBOX_WARNING"})
```

Environment variables (auto-applied on import):

| Variable                        | Effect |
|---------------------------------|--------|
| `TPTBOX_LOG_LEVEL=WARNING`      | Global terminal severity threshold. |
| `TPTBOX_SILENT=1`               | Silence every printout. |
| `TPTBOX_CAPTURE_WARNINGS=1`     | `warnings.warn(...)` → logger. |
| `TPTBOX_CAPTURE_EXCEPTIONS=1`   | Uncaught exceptions → logger. |
| `TPTBOX_LOGGER_TAKEOVER=0`      | Keep Loguru's built-in default stderr handler. |

## The `verbose` kwargs on TPTBox functions

Many TPTBox functions accept a `verbose: bool` argument
(`nii.reorient_(..., verbose=True)`, `BIDS_Global_info(..., verbose=True)`, ...).
They control **only** whether *that specific line* reaches the **terminal**.
The **file** sink always records everything (as long as a `Logger` is the
default).

Combined with `configure_logging`, that gives you two dials:

| You want …                                        | Set …                                                |
|---------------------------------------------------|------------------------------------------------------|
| Every message to terminal AND file                | `configure_logging(file=..., verbose=True)`          |
| Only WARNING+ on terminal; file gets everything   | `configure_logging(file=..., verbose=True, level="WARNING")` |
| Only *specific* noisy calls on terminal           | leave `verbose=False` globally, pass `verbose=True` at the call sites you want to see |
| No terminal at all; file gets everything          | `configure_logging(file=..., verbose=False)`         |
| No terminal, no file — nothing anywhere           | `configure_logging(silent=True)`                     |

## Log types

`Log_Type` — the type marker controls prefix, color, and severity.

| Log_Type          | Prefix       | Color            | Convenience helper |
|-------------------|--------------|------------------|--------------------|
| `TEXT`            | `[*]`        | default          | `logger.on_text` / `logger.info` |
| `NEUTRAL`         | `[ ]`        | default          | `logger.on_neutral`              |
| `SAVE`            | `[*]`        | cyan             | `logger.on_save`                 |
| `LOG`             | `[#]`        | blue             | `logger.on_log`                  |
| `OK`              | `[+]`        | green            | `logger.on_ok`                   |
| `WARNING`         | `[?]`        | yellow           | `logger.on_warning` / `logger.warning` |
| `WARNING_THROW`   | `[?]`        | yellow (via `warnings.warn`) | `logger.print(..., ltype=Log_Type.WARNING_THROW)` |
| `FAIL`            | `[!]`        | red              | `logger.on_fail` / `logger.error` |
| `STRANGE`         | `[-]`        | magenta          | `logger.on_debug`                |
| `BOLD`            | `[*]`        | bold             | `logger.on_bold`                 |
| `UNDERLINE`       | `[_]`        | underline        | `logger.print(..., ltype=Log_Type.UNDERLINE)` |
| `ITALICS`         | `[ ]`        | italics          | `logger.print(..., ltype=Log_Type.ITALICS)`   |
| `Yellow`          | `[*]`        | yellow           | `logger.print(..., ltype=Log_Type.Yellow)`    |
| `DOCKER`          | `[Docker]`   | italics          | `logger.print(..., ltype=Log_Type.DOCKER)`    |
| `TOTALSEG`        | `[TOTALSEG]` | italics          | `logger.print(..., ltype=Log_Type.TOTALSEG)`  |
| `STAGE`           | `[*]`        | blue background  | `logger.print(..., ltype=Log_Type.STAGE)`     |

`level=` in `configure_logging` accepts:
- integer Loguru severities (`10`, `20`, `25`, `30`, `40`, ...),
- TPTBox names — `"TEXT"`, `"OK"`, `"WARNING"`, `"FAIL"`, `"TPTBOX_OK"`, ..., or a `Log_Type` value,
- standard Loguru names — `"TRACE"`, `"DEBUG"`, `"INFO"`, `"SUCCESS"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`.

## For developers adding new code

Import the module-level proxy and call it. Nothing else.

```python
# my_module.py
from TPTBox.logger import logger

def compute_something(x, verbose: bool = False):
    logger.print("starting compute for", x, verbose=verbose)
    if x < 0:
        logger.on_warning("negative input, using absolute value")
        x = abs(x)
    ...
    logger.on_ok("compute done")
```

Rules of thumb:

- Do **not** manually prepend `[!]` / `[?]` / `[*]` to your message. The helpers do it.
- Do **not** import a concrete `Logger` class in library code. Use the proxy.
- Pass `verbose=verbose` on `logger.print(...)` when the caller has explicit
  control over terminal noise. Skip the `verbose=` for warnings and failures —
  they should always reach the terminal if the level threshold permits.

If a user has not called `configure_logging(...)`, the default is a silent
`No_Logger` — so library code stays quiet unless the caller opts in.

## Escape hatches

For fine-grained control beyond `configure_logging`:

- **Custom Loguru sink**: `from TPTBox import loguru_logger` then
  `loguru_logger.add(sink, ...)`. Structured records for exceptions and file
  output flow through Loguru; the terminal is written directly to `sys.stdout`
  and does not go through Loguru sinks.
- **Direct access to the current logger**: `from TPTBox import get_default_logger`.
- **Private helpers** (rarely needed) live in `TPTBox.logger.log_file` and
  `TPTBox.logger._loguru_backend` under underscore-prefixed names
  (`_set_default_logger`, `_set_verbosity_level`, `_set_logger_silence`,
  `_install_warnings_hook`, `_install_excepthook`, `_capture_exceptions`, ...).

## Loguru backend details

- Terminal output is written **directly to `sys.stdout`** with our own ANSI
  coloring — it does NOT go through Loguru's global handler routing. That
  guarantees TPTBox's terminal output is unaffected by whatever handlers a host
  application has active on the Loguru logger.
- File output (from `Logger`) and structured exception records DO go through
  Loguru so your custom sinks receive them. Each `Logger` instance's file has
  its own Loguru sink with a per-instance filter.
- `configure()` runs eagerly at import. It removes Loguru's built-in default
  stderr handler (id 0) so file/exception records don't get printed with the
  `time | LEVEL | module:func:line - msg` default format. Sinks a host
  application registered *before* importing TPTBox are preserved. Set
  `TPTBOX_LOGGER_TAKEOVER=0` before importing TPTBox to keep the default.

Line format:

```
[prefix] YYYY-MM-DD HH:MM:SS body
```

The prefix (`[*]`, `[!]`, `[?]`, `[+]`, `[#]`, `[Docker]`, ...) comes first,
then the timestamp, then the message body. No Loguru level name is emitted.
`logger.print()` with no arguments prints exactly one blank line.

Thread-safety: Loguru serializes file-sink writes; terminal writes acquire a
small lock in `emit_terminal`. `enqueue=True` when constructing `Logger`
additionally makes the sink thread/process-safe.

## Public API

```python
from TPTBox import (
    # the proxy — use everywhere in library code
    logger, log,
    # one-call setup
    configure_logging,
    # the current default (for advanced inspection)
    get_default_logger,
    # concrete logger classes
    Logger, Print_Logger, No_Logger, String_Logger, Reflection_Logger,
    Logger_Interface, Log_Type,
    # Loguru escape hatch (add your own sinks)
    loguru_logger,
)
```
