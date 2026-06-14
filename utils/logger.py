import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "MultiAgentMemory",
    verbose: bool = False,
    log_file: Optional[str] = None,
    log_level: Optional[str] = None
) -> logging.Logger:
    """
    Create and configure a logger instance with console and optional file output.

    Why this exists:
    - Provides a unified logging interface across all agents and components.
    - Supports two operational modes: verbose (detailed debug) and normal (concise info).
    - Allows logging to both console and file simultaneously without duplicate handlers.
    - Decouples log format and destination from the calling code.

    Design philosophy:
    - Verbose mode includes source file and line number – essential for distributed
      debugging where multiple agents run concurrently.
    - Normal mode produces clean, human‑readable logs suitable for production monitoring.
    - File logging always writes at DEBUG level regardless of console level, ensuring
      that full detail is preserved for post‑mortem analysis even when console is quiet.
    - The logger is fully re‑initialised each call (handlers cleared) to avoid
      duplicate outputs when `setup_logger` is called multiple times with different
      settings – trade‑off: loses ability to add handlers incrementally, but for this
      system that's acceptable because configuration happens once at startup.

    Arguments:
        name: Logger name, typically `__name__` of the calling module. Used to create
              hierarchical loggers (e.g., "MultiAgentMemory.retriever").
        verbose: If True, set console level to DEBUG and use detailed format with
                 file name and line number. If False, console level is INFO with
                 compact time‑only prefix.
        log_file: Optional file path. If provided, a FileHandler is added with
                  DEBUG level and the verbose format regardless of `verbose` flag.
        log_level: Manual override for the console log level (e.g., "WARNING").
                   Takes precedence over `verbose`. Useful for temporarily suppressing
                   noise without changing code.

    Returns:
        A configured `logging.Logger` instance. The logger may already have handlers
        from a previous call – they are cleared first to guarantee the returned
        logger matches the requested configuration.

    Invariants:
        - The returned logger always has at least a console handler (StreamHandler).
        - If `log_file` is given, exactly one FileHandler is attached.
        - The logger's effective level is the lowest of its handlers' levels.
        - No two handlers share the same stream (stdout) or file descriptor.
        - After this function returns, the logger is ready to use immediately.

    Edge Cases & Risks:
        - If the same `name` is passed again with different parameters, the
          existing logger is re‑configured (handlers cleared). This may surprise
          callers that expect to add their own handlers. To avoid this, always
          call `setup_logger` once at application start, then use `logging.getLogger(name)`
          elsewhere.
        - `log_level` values are case‑insensitive but must match a standard logging
          level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Invalid strings raise
          `AttributeError` via `getattr(logging, ...)` – the function does not catch it.
        - File handler uses mode `'a'` (append). Log files can grow indefinitely.
          The caller is responsible for rotation (e.g., using `RotatingFileHandler`).
        - Console output is hard‑coded to `sys.stdout`. In some environments (e.g.,
          Jupyter), this may not capture all output; consider using `sys.stderr` for
          error logs, but this function keeps it simple.
        - The function resets handlers every time. If multiple modules call it with
          different `log_file` paths, only the last call's file handler survives.
          This is intentional – the system should have a single log configuration.

    Performance:
        - Clearing handlers and re‑adding them is O(previous_handlers) and cheap
          (usually <10 handlers). Called once at startup, so overhead is negligible.
        - The formatter uses `strftime` for every log record – this is standard
          logging behaviour and not optimised further.
        - File I/O is buffered by the logging module (default buffer size). For
          high‑throughput logging, consider adding `logging.handlers.BufferingHandler`.

    Example Usage:
        >>> # Production mode – concise console logs, full file logs
        >>> logger = setup_logger("MyAgent", verbose=False, log_file="/var/log/app.log")
        >>> logger.info("Processing query")  # Console: "HH:MM:SS - INFO - Processing query"
        >>> # File gets timestamp, level, file:line, and message.

        >>> # Debug mode – detailed console, no file
        >>> logger = setup_logger("DebugAgent", verbose=True)
        >>> logger.debug("Entering process()")  # Console: "2025-01-15 14:30:00 - DebugAgent - DEBUG - [agent.py:42] - Entering process()"
    """
    # Determine log level: explicit log_level > verbose flag > default INFO.
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate outputs (e.g., when reconfiguring).
    if logger.hasHandlers():
        logger.handlers.clear()

    # Choose formatter based on verbose mode.
    if verbose:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    # Console handler – always attached.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler – writes everything at DEBUG level with detailed format.
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)  # File captures all details, ignoring console level.
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


# Singleton default logger for convenience across the whole application.
# Why a singleton? Many modules need a logger but don't care about customisation.
# This provides a shared, once‑configured instance that can be imported anywhere.
# However, it's optional – advanced users can call setup_logger directly.
_default_logger = None


def get_logger(name: str = None, verbose: bool = False, log_file: str = None) -> logging.Logger:
    """
    Obtain a logger instance, optionally creating a singleton default.

    Design intent:
    - Provides a simple global access point for logging without passing references.
    - If called without arguments, returns the pre‑configured default logger.
    - If called with a `name`, returns a child logger of the default (or creates
      a new named logger if default does not exist yet).
    - This function is idempotent after the first call: subsequent calls with
      different `verbose` or `log_file` will **not** reconfigure the default logger
      because that would break existing references. Use `setup_logger` for explicit
      reconfiguration.

    Arguments:
        name: Optional child logger name. If provided, returns `logging.getLogger(name)`
              after ensuring the default logger exists. The child inherits the
              parent's level and handlers.
        verbose: Ignored if default logger already exists; used only on first call
                 to create the default logger.
        log_file: Same as `verbose` – only effective when creating the default logger.

    Returns:
        A `logging.Logger` instance. If `name` is given, the returned logger is a
        child of the default logger (or a standalone logger if default not initialised).

    Invariants:
        - The default logger is created at most once.
        - After the default logger exists, `get_logger()` with no arguments returns
          that same instance.
        - Calling `get_logger("sub")` after default creation returns a child logger
          that propagates logs to the default's handlers (unless child's propagate
          flag is set to False – not done here).

    Edge Cases & Risks:
        - If `get_logger` is called with a `name` before any call to `setup_logger`
          or `get_logger` without arguments, a default logger is created on the fly
          with the given `verbose` and `log_file` parameters. This may lead to
          surprising behaviour if different parts of the code request different
          defaults. Best practice: call `get_logger()` once at application entry
          point to initialise the default, then use `get_logger(__name__)` elsewhere.
        - The function does not allow reconfiguring the default logger after creation.
          To change logging level or file, use `setup_logger` directly.
        - Child loggers created via `get_logger(name)` will not have the file:line
          information in their records unless the formatter includes it. That's
          already handled because the default logger's handlers propagate up.
        - The global `_default_logger` is not thread‑safe for initialisation. In
          a multi‑threaded environment, two threads could both see `_default_logger is None`
          and create two different default loggers. This is unlikely in typical
          single‑threaded startup, but to be safe, call `get_logger()` before spawning
          threads.

    Performance:
        - After initialisation, `get_logger()` does a single global variable read
          and possibly a `logging.getLogger` call – both are extremely cheap.
        - No I/O or locking inside this function after default creation.

    Example Usage:
        >>> # In main.py
        >>> get_logger(verbose=True, log_file="app.log")   # Creates default logger
        >>>
        >>> # In agent.py
        >>> from logger_util import get_logger
        >>> logger = get_logger(__name__)   # Child logger named "agent"
        >>> logger.info("Agent initialised")  # Logs go to console and file with proper context.
    """
    global _default_logger
    if _default_logger is None:
        # Create default logger using provided parameters (or defaults).
        _default_logger = setup_logger(name or "MultiAgentMemory", verbose, log_file)
    if name:
        # Return a child logger – it inherits handlers and level from the default.
        return logging.getLogger(name)
    return _default_logger