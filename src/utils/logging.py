"""Logging utilities for VMS_Help_Menu chatbot."""

import json
import logging as std_logging
from logging.handlers import RotatingFileHandler
import sys
import traceback
from datetime import datetime
from pathlib import Path

from src.config import LOG_DIR, LOG_FILE

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure internal logger
_logger = std_logging.getLogger("vaps_help_menu")
_logger.setLevel(std_logging.INFO)
_logger.propagate = False

if not _logger.handlers:
    # Rotating file handler (10MB per file, max 5 backups)
    _file_handler = RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    _file_handler.setLevel(std_logging.INFO)
    _file_format = std_logging.Formatter("%(message)s")
    _file_handler.setFormatter(_file_format)
    _logger.addHandler(_file_handler)

    # Console stream handler for container/Docker stdout
    _console_handler = std_logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(std_logging.INFO)
    _console_format = std_logging.Formatter("%(message)s")
    _console_handler.setFormatter(_console_format)
    _logger.addHandler(_console_handler)


def _format_value(value: object) -> str:
    """Format a value for logging output."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def log_step(step_name: str, **values: object) -> None:
    """
    Log a processing step with optional key-value pairs.

    Writes to both rotating log file and stdout.

    Args:
        step_name: Name of the processing step
        **values: Additional key-value pairs to log
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    details = " ".join(f"{key}={_format_value(value)}" for key, value in values.items())
    line = f"[{timestamp}] {step_name}"
    if details:
        line = f"{line} {details}"

    _logger.info(line)


def log_error(step_name: str, error: Exception, **values: object) -> None:
    """
    Log an error with exception details.

    Args:
        step_name: Name of the processing step
        error: The exception that occurred
        **values: Additional key-value pairs to log
    """
    payload = dict(values)
    payload["error_type"] = type(error).__name__
    payload["error"] = str(error)
    payload["traceback"] = traceback.format_exc(limit=10)
    log_step(step_name, **payload)
