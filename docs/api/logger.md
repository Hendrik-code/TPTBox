# Logger

Structured, consistent logging for long-running medical image processing pipelines.

All logger implementations conform to the structural
[`Logger_Interface`][TPTBox.logger.log_file.Logger_Interface] protocol, so
client code can type-hint against the interface and stay decoupled from the
concrete backend:

- [`Logger`][TPTBox.logger.log_file.Logger] — writes messages to a timestamped
  file inside a `logs/` folder next to a dataset root; supports sub-loggers
  and accumulated statistics.
- [`No_Logger`][TPTBox.logger.log_file.No_Logger] — verbose-to-terminal fallback
  that persists nothing; safe drop-in when a file log is not wanted.
- [`String_Logger`][TPTBox.logger.log_file.String_Logger] — buffers into an
  in-memory string, optionally forwarding to a parent logger on flush/close.

Log entries are classified with [`Log_Type`][TPTBox.logger.log_constants.Log_Type]
which drives both the terminal color and the file-level prefix.

## Logger

::: TPTBox.logger.log_file.Logger
    options:
      show_source: true

## Logger_Interface

::: TPTBox.logger.log_file.Logger_Interface
    options:
      show_source: true

## No_Logger

::: TPTBox.logger.log_file.No_Logger
    options:
      show_source: true

## String_Logger

::: TPTBox.logger.log_file.String_Logger
    options:
      show_source: true

## Log_Type

::: TPTBox.logger.log_constants.Log_Type
    options:
      show_source: true
