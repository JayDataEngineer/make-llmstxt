"""Logging configuration for make-llmstxt.

Sets up loguru to write to:
- Console (INFO and above)
- File: logs/make_llmstxt.log (DEBUG and above, with rotation)
"""

import sys
import json
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Default log file path - in the project's logs/ directory
DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "make_llmstxt.log"


def json_formatter(record):
    """Format log record as JSON."""
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    if record["exception"]:
        log_entry["exception"] = str(record["exception"])
    return json.dumps(log_entry) + "\n"


def setup_logging(
    level: str = "INFO",
    log_file: Path = None,
    json_format: bool = False,
    log_to_file: bool = True,
) -> Path:
    """Set up logging with console and file handlers.

    Args:
        level: Minimum log level for console (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (default: logs/make_llmstxt.log)
        json_format: Whether to use JSON format for file logging
        log_to_file: Whether to log to file at all

    Returns:
        Path to the log file (or None if not logging to file)
    """
    # Console handler - colored output
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File handler - plain text with rotation
    if log_to_file:
        if log_file is None:
            log_file = DEFAULT_LOG_FILE
            log_dir = DEFAULT_LOG_DIR
        else:
            log_dir = log_file.parent

        log_dir.mkdir(parents=True, exist_ok=True)

        if json_format:
            file_format = json_formatter
        else:
            file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

        logger.add(
            str(log_file),
            level="DEBUG",  # Always capture DEBUG to file
            format=file_format,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="7 days",  # Keep logs for 7 days
            compression="gz",  # Compress rotated logs
            enqueue=True,  # Thread-safe logging
        )

        logger.info(f"Logging to file: {log_file}")
        return log_file

    return None


def get_log_file_path() -> Path:
    """Return the default path to the log file."""
    return DEFAULT_LOG_FILE
