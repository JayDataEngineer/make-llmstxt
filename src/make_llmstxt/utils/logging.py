"""Logging configuration for make-llmstxt.

Sets up loguru to write to:
- Console (INFO and above)
- File: logs/make_llmstxt.log (DEBUG and above, with rotation)

Features:
- Module-specific loggers with bind() for filtering
- Correlation ID support for tracing across components
- Environment-aware defaults (production vs development)
- Duplicate handler prevention
"""

import os
import sys
import json
import threading
import uuid
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any

from loguru import logger

# Remove default handler
logger.remove()

# Default log file path - in the project's logs/ directory
DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "make_llmstxt.log"

# Track handler IDs to prevent duplicates
_handler_ids: Dict[str, int] = {}
_setup_lock = threading.Lock()


def _is_production() -> bool:
    """Check if running in production environment."""
    return os.getenv("ENVIRONMENT", "development").lower() == "production"


def _generate_correlation_id() -> str:
    """Generate a unique correlation ID for tracing."""
    return str(uuid.uuid4())[:8]


# Thread-local storage for correlation ID
_correlation_id = threading.local()


def get_correlation_id() -> str:
    """Get the current correlation ID, or 'none' if not set."""
    return getattr(_correlation_id, 'value', 'none')


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current thread."""
    _correlation_id.value = correlation_id


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager for setting correlation ID.
    
    Args:
        correlation_id: ID to use, or None to generate a new one
        
    Yields:
        The correlation ID being used
    """
    old_id = get_correlation_id()
    new_id = correlation_id or _generate_correlation_id()
    try:
        set_correlation_id(new_id)
        yield new_id
    finally:
        set_correlation_id(old_id)


def json_formatter(record: dict) -> str:
    """Format log record as JSON with full context.

    Includes:
    - timestamp, level, message
    - module, function, line, thread, process
    - correlation_id for tracing
    - exception details if present
    - any extra fields bound to the logger
    """
    log_entry: Dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "thread": record["thread"].id,
        "thread_name": record["thread"].name,
        "process": record["process"].id,
        "correlation_id": record["extra"].get("correlation_id", "none"),
    }

    # Include any extra context bound to the logger
    extra_context = {
        k: v for k, v in record["extra"].items()
        if k not in ("correlation_id",)
    }
    if extra_context:
        log_entry["extra"] = extra_context

    if record["exception"]:
        log_entry["exception"] = {
            "type": type(record["exception"]).__name__,
            "message": str(record["exception"]),
        }

    return json.dumps(log_entry) + "\n"


def _json_sink(message):
    """Custom sink for JSON logging that writes raw JSON without processing."""
    # message is already formatted by json_formatter
    import sys
    sys.stderr.write(message)


def _create_json_file_sink(log_file: Path, rotation: str, retention: str, compression: str):
    """Create a custom sink that writes JSON formatted logs to a file with rotation.
    
    Note: This is a simplified implementation that doesn't support all rotation features.
    For full rotation support, consider using a logging handler from the standard library.
    """
    import os
    from datetime import datetime
    
    def sink(message):
        """Write JSON formatted message to file."""
        # The message object has a 'record' attribute with all the log record data
        record = message.record
        json_str = json_formatter(record)
        
        # Simple rotation check
        if os.path.exists(str(log_file)):
            file_size = os.path.getsize(str(log_file))
            # Parse rotation size (e.g., "10 MB" -> 10 * 1024 * 1024)
            rotation_parts = rotation.split()
            if len(rotation_parts) == 2:
                size_num = int(rotation_parts[0])
                size_unit = rotation_parts[1].upper()
                max_size = size_num * 1024 * 1024 if size_unit == "MB" else size_num * 1024
                if file_size > max_size:
                    # Rotate the file
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    rotated_path = Path(str(log_file) + f".{timestamp}")
                    os.rename(str(log_file), str(rotated_path))
                    if compression == "gz":
                        import gzip
                        with open(str(rotated_path), 'rb') as f_in:
                            with gzip.open(str(rotated_path) + '.gz', 'wb') as f_out:
                                f_out.write(f_in.read())
                        os.remove(str(rotated_path))
        
        with open(str(log_file), "a") as f:
            f.write(json_str)
    
    return sink


def _add_correlation_id_to_record(record: dict) -> None:
    """Add correlation ID to log record."""
    record["extra"]["correlation_id"] = get_correlation_id()


def _json_patcher(record: dict) -> None:
    """Patcher that converts message to JSON format."""
    record["message"] = json_formatter(record).strip()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: Optional[bool] = None,
    log_to_file: bool = True,
) -> Optional[Path]:
    """Set up logging with console and file handlers.

    Args:
        level: Minimum log level for console (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (default: logs/make_llmstxt.log)
        json_format: Whether to use JSON format for file logging.
                     Defaults to True in production, False in development.
        log_to_file: Whether to log to file at all

    Returns:
        Path to the log file (or None if not logging to file)
    """
    with _setup_lock:
        # Remove existing handlers to prevent duplicates
        _remove_all_handlers()
        
        # Add correlation ID processor
        logger.configure(patcher=_add_correlation_id_to_record)
        
        # Auto-detect JSON format for production
        if json_format is None:
            json_format = _is_production()
        
        # Console handler - colored output with structured context
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<magenta>corr:{extra[correlation_id]}</magenta> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stderr,
            level=level,
            format=console_format,
            colorize=True,
            filter=_console_filter,
        )

        # File handler - with rotation
        if log_to_file:
            if log_file is None:
                log_file = DEFAULT_LOG_FILE
                log_dir = DEFAULT_LOG_DIR
            else:
                log_dir = log_file.parent

            log_dir.mkdir(parents=True, exist_ok=True)

            if json_format:
                # For JSON format, use a custom sink with JSON formatting
                json_sink = _create_json_file_sink(log_file, "10 MB", "7 days", "gz")
                logger.add(
                    json_sink,
                    level="DEBUG",
                    enqueue=True,
                )
            else:
                file_format = (
                    "{time:YYYY-MM-DD HH:mm:ss} | "
                    "{level: <8} | "
                    "{name}:{function}:{line} | "
                    "corr:{extra[correlation_id]} | "
                    "{message}"
                )

                logger.add(
                    str(log_file),
                    level="DEBUG",
                    format=file_format,
                    rotation="10 MB",
                    retention="7 days",
                    compression="gz",
                    enqueue=True,
                )

            logger.info(f"Logging to file: {log_file}")
            return log_file

        return None


def _console_filter(record: dict) -> bool:
    """Filter for console output.
    
    Suppresses DEBUG logs from verbose libraries unless LOG_LEVEL=DEBUG.
    """
    # Always show INFO and above
    if record["level"].no >= 20:  # INFO
        return True
    
    # Filter out noisy DEBUG logs from certain libraries
    noisy_modules = [
        "httpx", "httpcore", "urllib3", "asyncio", 
        "langchain", "langgraph", "openai", "anthropic"
    ]
    
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    if log_level == "DEBUG":
        return True
    
    for module in noisy_modules:
        if record["name"].startswith(module):
            return False
    
    return True


def _remove_all_handlers() -> None:
    """Remove all handlers and reset tracking."""
    global _handler_ids
    logger.remove()
    _handler_ids = {}


def get_logger(
    module: Optional[str] = None,
    **extra_context: Any
) -> Any:
    """Get a logger instance with optional module binding.

    Args:
        module: Module name for filtering (e.g., "llm", "scraper", "critic")
        **extra_context: Additional context to bind to log messages

    Returns:
        A loguru logger instance with bound context

    Example:
        >>> logger = get_logger("llm", model="gpt-4")
        >>> logger.info("Processing request")  # Includes module=llm, model=gpt-4
    """
    bound_logger = logger

    if module:
        bound_logger = bound_logger.bind(module=module)

    if extra_context:
        bound_logger = bound_logger.bind(**extra_context)

    return bound_logger


class StructuredLogger:
    """Wrapper for structured logging with consistent formatting.
    
    Provides DRY methods for common logging patterns with automatic
    context binding and professional formatting.
    
    Usage:
        >>> log = StructuredLogger("agent")
        >>> log.start("generation", url="https://example.com")
        >>> log.end("generation", duration=1.5, tokens=100)
        >>> log.error("operation", "Failed to process", url="https://example.com")
    """
    
    def __init__(self, module: str, **default_context: Any):
        self.module = module
        self.default_context = default_context
        self.logger = get_logger(module, **default_context)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context as key=value pairs for logging."""
        if not context:
            return ""
        pairs = []
        for key, value in context.items():
            if isinstance(value, float):
                pairs.append(f"{key}={value:.2f}")
            else:
                pairs.append(f"{key}={value}")
        return " | " + " ".join(pairs)
    
    def _log(self, level: str, event: str, **context: Any) -> None:
        """Internal log method with consistent formatting."""
        merged = {**self.default_context, **context}
        context_str = self._format_context(merged)
        message = f"{event}{context_str}"
        self.logger.log(level, message)
    
    def start(self, operation: str, **context: Any) -> None:
        """Log operation start."""
        self._log("INFO", f"[START] {operation}", **context)
    
    def end(self, operation: str, duration: float, **context: Any) -> None:
        """Log operation end with timing."""
        self._log("INFO", f"[END] {operation}", duration=duration, **context)
    
    def error(self, operation: str, message: str, **context: Any) -> None:
        """Log error with context."""
        self._log("ERROR", f"[ERROR] {operation}", message=message, **context)
    
    def warning(self, operation: str, message: str, **context: Any) -> None:
        """Log warning with context."""
        self._log("WARNING", f"[WARN] {operation}", message=message, **context)
    
    def info(self, message: str, **context: Any) -> None:
        """Log info message."""
        self._log("INFO", message, **context)
    
    def debug(self, message: str, **context: Any) -> None:
        """Log debug message."""
        self._log("DEBUG", message, **context)
    
    def bind(self, **extra_context: Any) -> "StructuredLogger":
        """Return a new logger with additional context bound."""
        return StructuredLogger(
            self.module,
            **{**self.default_context, **extra_context}
        )


@contextmanager
def log_context(**kwargs):
    """Context manager for temporarily adding context to log messages.
    
    Args:
        **kwargs: Key-value pairs to add to log context
        
    Example:
        >>> with log_context(user_id="123", action="generate"):
        ...     logger.info("Starting generation")  # Includes user_id and action
    """
    # Store original logger state
    original_logger = logger
    
    # Bind new context
    bound_logger = logger.bind(**kwargs)
    
    # Temporarily replace the module-level logger reference
    # Note: This only works for modules that import logger from this module
    import make_llmstxt.utils.logging as logging_module
    original = logging_module.logger
    logging_module.logger = bound_logger
    
    try:
        yield bound_logger
    finally:
        logging_module.logger = original


def get_log_file_path() -> Path:
    """Return the default path to the log file."""
    return DEFAULT_LOG_FILE


# Re-export logger for backward compatibility
# Modules should prefer get_logger() for module-specific logging
# Use StructuredLogger for new code with DRY patterns
__all__ = [
    "setup_logging",
    "get_logger",
    "get_log_file_path",
    "get_correlation_id",
    "set_correlation_id",
    "correlation_context",
    "log_context",
    "StructuredLogger",  # Professional DRY logging
    "logger",  # For backward compatibility
]
