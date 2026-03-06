"""Logging configuration using Loguru.

Provides structured, colored logging with:
- Console output with Rich formatting
- Optional file logging with rotation
- Per-module log levels
- JSON format option for production
"""

import sys
import os
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Log format with colors
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <5}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Simpler format for file
FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

# JSON format for production
JSON_FORMAT = '{{"timestamp": "{time:YYYY-MM-DDTHH:mm:ssZ}", "level": "{level}", "logger": "{name}", "function": "{function}", "line": {line}, "message": "{message}"}}'


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    json_format: bool = False,
    rich_console: bool = True,
) -> None:
    """Configure logging with Loguru.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_format: Use JSON format for structured logging
        rich_console: Use Rich for colored console output
    """
    # Determine console format
    if json_format:
        console_format = JSON_FORMAT
    elif rich_console:
        console_format = CONSOLE_FORMAT
    else:
        console_format = "{time:HH:mm:ss} | {level: <5} | {name} - {message}"

    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=rich_console and not json_format,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_file),
            format=FILE_FORMAT if not json_format else JSON_FORMAT,
            level=level,
            rotation="10 MB",  # Rotate at 10MB
            retention="7 days",  # Keep logs for 7 days
            compression="gz",  # Compress rotated logs
        )

    logger.debug(f"Logging initialized: level={level}, file={log_file}, json={json_format}")


def get_logger(name: str = __name__):
    """Get a logger with module name context.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger with module context bound
    """
    return logger.bind(name=name)


# Convenience function for modules
def configure_from_env():
    """Configure logging from environment variables.

    Environment variables:
        LOG_LEVEL: Log level (default: INFO)
        LOG_FILE: Path to log file (optional)
        LOG_JSON: Use JSON format (default: false)
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE")
    json_format = os.getenv("LOG_JSON", "false").lower() == "true"

    setup_logging(
        level=level,
        log_file=Path(log_file) if log_file else None,
        json_format=json_format,
    )


# Export logger for direct use
__all__ = ["logger", "setup_logging", "get_logger", "configure_from_env"]
