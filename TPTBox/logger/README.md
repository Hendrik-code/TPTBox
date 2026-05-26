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
