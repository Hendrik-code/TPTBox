# Logger (`TPTBox.logger`)

Structured, consistent logging for long-running medical image processing pipelines.
Provides a simple interface with configurable verbosity, message categories, and output targets.

## Public API

```python
from TPTBox import Logger, Print_Logger, No_Logger, String_Logger, Log_Type
```

![Example of logging](logging.png?raw=true "Example of logging messages")

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
| `Log_Type.SAVE` | Saving files to disk |
| `Log_Type.STAGE` | Marking start of different phases |
| `Log_Type.LOG` | Information regarding the logger itself |

## Example

```python
logger.print() # logs/prints empty line

logger.print("Started logging to path: ./logs/test.log", lt.LOG)
logger.print()
logger.print("Phase 1: Data Preprocessing", lt.STAGE)
with logger:
    logger.print("Loading data...")
    logger.print("Data loaded successfully.", lt.OK)
    logger.print("Saving preprocessed data...", lt.SAVE)
    logger.print("Warning: Some data points were missing and have been filled with default values.", lt.WARNING)
logger.print("Phase 2: Measurement", lt.STAGE)
with logger:
    logger.print("Starting measurements...")
    logger.print("Error: Measurement failed due to missing data.", lt.FAIL)
```

The output would be:
![Example of logging](loggingexample.png?raw=true "Example of logging messages")


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
