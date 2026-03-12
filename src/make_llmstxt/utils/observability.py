"""Langfuse observability integration.

Uses Langfuse SDK v3+ with OpenTelemetry for automatic tracing.
All LLM calls are captured automatically without passing callbacks.

Usage:
    # Just initialize at app start - traces are automatic
    from make_llmstxt.utils.observability import init_langfuse, flush_langfuse

    init_langfuse()  # Call once at app start

    # ... run your LLM code ...

    flush_langfuse()  # Call at end of short-lived scripts

    # Or use @observe decorator for custom spans
    from make_llmstxt.utils.observability import observe

    @observe()
    async def my_function():
        # This function and all nested calls are traced
        ...
"""

import os
import uuid
from typing import Optional, Callable, Any
from functools import wraps
from contextlib import contextmanager

from .logging import StructuredLogger

log = StructuredLogger("observability")

# Singleton instances
_langfuse_client = None
_langfuse_initialized = False
_current_session_id = None


def init_langfuse() -> bool:
    """Initialize Langfuse SDK for automatic tracing.

    Call once at application start. The SDK uses OpenTelemetry to
    automatically capture all LLM calls.

    Returns:
        True if initialized successfully, False otherwise
    """
    global _langfuse_client, _langfuse_initialized

    if _langfuse_initialized:
        return _langfuse_client is not None

    _langfuse_initialized = True

    # Check if credentials are configured
    if not (os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")):
        log.debug("Langfuse not configured (set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY)")
        return False

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )

        if _langfuse_client.auth_check():
            log.info(
                "Langfuse initialized",
                host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
            )
            return True
        else:
            log.warning("Langfuse auth check failed")
            _langfuse_client = None
            return False

    except ImportError:
        log.debug("Langfuse SDK not installed - pip install langfuse")
        return False
    except Exception as e:
        log.warning("Langfuse initialization failed", error=str(e))
        _langfuse_client = None
        return False


def get_langfuse_client():
    """Get the Langfuse client (initializes if needed).

    Returns:
        Langfuse client or None if not configured
    """
    global _langfuse_client

    if _langfuse_client is None and not _langfuse_initialized:
        init_langfuse()

    return _langfuse_client


def flush_langfuse():
    """Flush pending events to Langfuse.

    Call at the end of short-lived scripts to ensure all traces are sent.
    """
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
            log.debug("Langfuse events flushed")
        except Exception as e:
            log.warning("Failed to flush Langfuse", error=str(e))


def is_langfuse_enabled() -> bool:
    """Check if Langfuse tracing is enabled."""
    return get_langfuse_client() is not None


def create_session_id() -> str:
    """Create a new session ID for grouping traces.

    Returns:
        Unique session ID string
    """
    return f"session-{uuid.uuid4().hex[:12]}"


@contextmanager
def session_context(session_id: str):
    """Context manager to propagate session_id to all nested observations.

    Usage:
        with session_context("my-session-123"):
            # All LLM calls here are grouped under this session
            ...

    Args:
        session_id: Session ID to propagate
    """
    global _current_session_id
    old_session_id = _current_session_id
    _current_session_id = session_id

    try:
        client = get_langfuse_client()
        if client is None:
            yield
            return

        # Use propagate_attributes if available
        try:
            from langfuse import propagate_attributes
            with propagate_attributes(session_id=session_id):
                yield
        except ImportError:
            # Fallback: just set the global and yield
            yield
    finally:
        _current_session_id = old_session_id


def get_current_session_id() -> Optional[str]:
    """Get the current session ID if set."""
    return _current_session_id


def observe(name: Optional[str] = None, as_type: str = "span"):
    """Decorator to trace a function with Langfuse.

    Usage:
        @observe()
        async def my_function(arg1, arg2):
            # Function is traced as a span
            ...

        @observe(name="custom-name", as_type="generation")
        async def llm_call(prompt):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = get_langfuse_client()
            if client is None:
                # Langfuse not configured, just run the function
                return await func(*args, **kwargs)

            trace_name = name or func.__name__
            session_id = get_current_session_id()

            try:
                obs_kwargs = {
                    "as_type": as_type,
                    "name": trace_name,
                    "input": {"args": str(args)[:1000], "kwargs": str(kwargs)[:1000]}
                }
                if session_id:
                    obs_kwargs["session_id"] = session_id

                with client.start_as_current_observation(**obs_kwargs) as obs:
                    result = await func(*args, **kwargs)
                    obs.update(output=str(result)[:2000])
                    return result
            except Exception as e:
                log.debug("Langfuse tracing error", error=str(e))
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            client = get_langfuse_client()
            if client is None:
                return func(*args, **kwargs)

            trace_name = name or func.__name__
            session_id = get_current_session_id()

            try:
                obs_kwargs = {
                    "as_type": as_type,
                    "name": trace_name,
                    "input": {"args": str(args)[:1000], "kwargs": str(kwargs)[:1000]}
                }
                if session_id:
                    obs_kwargs["session_id"] = session_id

                with client.start_as_current_observation(**obs_kwargs) as obs:
                    result = func(*args, **kwargs)
                    obs.update(output=str(result)[:2000])
                    return result
            except Exception as e:
                log.debug("Langfuse tracing error", error=str(e))
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Legacy compatibility - get_langfuse_callback for LangChain
def get_langfuse_callback() -> list:
    """Get Langfuse callback handler for LangChain.

    Note: Session ID is propagated via propagate_attributes() context manager,
    not via the handler constructor.

    Returns:
        List containing CallbackHandler if configured, empty list otherwise
    """
    client = get_langfuse_client()
    if client is None:
        return []

    try:
        # Use langfuse.langchain for LangChain integration (v3+)
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()
        log.debug("LangChain CallbackHandler created")
        return [handler]
    except ImportError:
        # Fallback to old import path for older SDK versions
        try:
            from langfuse.callback import CallbackHandler
            handler = CallbackHandler()
            log.debug("LangChain CallbackHandler created (legacy import)")
            return [handler]
        except Exception as e:
            log.warning("Failed to create Langfuse callback", error=str(e))
            return []
    except Exception as e:
        log.warning("Failed to create Langfuse callback", error=str(e))
        return []
