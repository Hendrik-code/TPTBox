# Logger (`TPTBox.logger`)

Structured, consistent logging for long-running medical image processing pipelines.
Provides a simple interface with configurable verbosity, message categories, and output targets.

## Public API

```python
from TPTBox import Logger, Print_Logger, No_Logger, String_Logger, Log_Type
```

## Key classes

| Class | Description |
|---|---|
| `Logger` | Base logger; prints to stdout with optional file output and timestamps |
| `Print_Logger` | Always-verbose logger — prints every message regardless of `verbose` flag |
| `No_Logger` | Silent logger — discards all messages; useful in batch/library code |
| `String_Logger` | Accumulates messages into an in-memory string; useful for testing |
| `Reflection_Logger` | Wraps another logger and mirrors its messages to a second logger |
| `Logger_Interface` | Abstract base class for custom logger implementations |

## Log_Type enum

| Member | Meaning |
|---|---|
| `Log_Type.BOLD` | Highlighted/important message |
| `Log_Type.OK` | Success confirmation |
| `Log_Type.WARNING` | Non-fatal warning |
| `Log_Type.FAIL` | Error or failure |
| `Log_Type.TEXT` | Plain informational text |

## Example

```python
from TPTBox import Logger, Log_Type

log = Logger(path="run.log", log_filename="pipeline", default_verbose=True)

log.print("Starting segmentation", Log_Type.BOLD)
log.print("Loaded 42 subjects", Log_Type.OK)
log.print("Missing T2w for sub-007", Log_Type.WARNING)

# Suppress all output (e.g. in a library function)
from TPTBox import No_Logger
log = No_Logger()
log.print("This is silently discarded")
```

## Loguru backend

Emission is backed by [Loguru](https://github.com/Delgan/loguru) (`TPTBox/logger/_loguru_backend.py`).
The public API, classes, `verbose` semantics, and the terminal **color coding** are unchanged —
each `Log_Type` maps to a custom `TPTBOX_*` Loguru level whose color reproduces the original ANSI,
so terminal output renders identically. `WARNING_THROW` still raises a Python `warnings.warn`.

On import, TPTBox configures the global Loguru logger (removing Loguru's own default handler).
Set `TPTBOX_LOGGER_TAKEOVER=0` *before importing TPTBox* to opt out.

### What this enables

```python
from TPTBox.logger import loguru_logger, configure, install_excepthook

# 1) Attach your own sink — e.g. structured JSON logs, with level filtering:
loguru_logger.add("run.jsonl", serialize=True, level="TPTBOX_WARNING")  # WARNING and worse

# 2) File rotation / retention on the file-backed Logger:
from TPTBox import Logger
log = Logger("dataset/", "pipeline", rotation="20 MB", retention="10 days", enqueue=True)

# 3) Exception capture: print_error() emits the structured exception to your sinks;
#    optionally route *uncaught* exceptions through Loguru too:
install_excepthook()
```

Thread-safety: Loguru serializes sink writes, so concurrent `print()` calls no longer interleave
mid-line (`enqueue=True` additionally makes a sink thread/process-safe).
