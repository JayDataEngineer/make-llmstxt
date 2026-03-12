# Langfuse Integration Guide: From Zero to Full Observability

This guide documents the complete journey from basic logging to full Langfuse observability with sessions and traces.

## Table of Contents

1. [Starting Point: Basic Logging](#1-starting-point-basic-logging)
2. [Phase 1: Infrastructure Setup](#2-phase-1-infrastructure-setup)
3. [Phase 2: SDK Initialization](#3-phase-2-sdk-initialization)
4. [Phase 3: Adding Traces](#4-phase-3-adding-traces)
5. [Phase 4: Adding Sessions](#5-phase-4-adding-sessions)
6. [Phase 5: Automatic Model Name Detection](#6-phase-5-automatic-model-name-detection)
7. [Complete Integration](#7-complete-integration)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Starting Point: Basic Logging

Before Langfuse, we had simple structured logging:

```python
from make_llmstxt.utils.logging import StructuredLogger

log = StructuredLogger("my_module")

async def generate_content(prompt: str):
    log.info("Starting generation", prompt_length=len(prompt))
    result = await llm.ainvoke(messages)
    log.info("Generation complete", output_length=len(result.content))
    return result
```

**Problems:**
- No visibility into LLM calls
- No token tracking
- No cost analysis
- Hard to debug issues across multiple calls
- No grouping of related operations

---

## 2. Phase 1: Infrastructure Setup

### 2.1 Docker Compose Configuration

Create a `docker-compose.yml` with Langfuse services:

```yaml
services:
  langfuse-web:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@postgres:5432/langfuse
      REDIS_CONNECTION_STRING: redis://redis:6379
      S3_STORAGE_UPLOAD_BUCKET_NAME: langfuse-storage
      # Headless initialization (no login required for API)
      LANGFUSE_INIT_PROJECT_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:-pk-lf-local}
      LANGFUSE_INIT_PROJECT_SECRET_KEY: ${LANGFUSE_SECRET_KEY:-sk-lf-local}
      LANGFUSE_INIT_USER_EMAIL: ${LANGFUSE_INIT_USER_EMAIL:-admin@example.com}
      LANGFUSE_INIT_USER_PASSWORD: ${LANGFUSE_INIT_USER_PASSWORD:-password123}
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse

  redis:
    image: redis:7

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    # For analytics storage
```

### 2.2 Generate Secrets Script

Create `scripts/generate-secrets.sh`:

```bash
#!/bin/bash
set -e

# Generate random 32-character hex keys
PUBLIC_KEY="pk-lf-$(openssl rand -hex 16)"
SECRET_KEY="sk-lf-$(openssl rand -hex 16)"

# Append to .env if not already present
if ! grep -q "LANGFUSE_PUBLIC_KEY" .env 2>/dev/null; then
    echo "LANGFUSE_PUBLIC_KEY=$PUBLIC_KEY" >> .env
    echo "LANGFUSE_SECRET_KEY=$SECRET_KEY" >> .env
    echo "Generated Langfuse API keys"
else
    echo "Keys already exist in .env"
fi
```

### 2.3 Makefile Commands

```makefile
up: setup-secrets
	docker compose --profile all up -d

down:
	docker compose --profile all down

up-observability: setup-secrets
	docker compose --profile observability up -d

reset-observability:
	docker compose --profile observability down -v

setup-secrets:
	@if [ ! -f .env ] || ! grep -q "LANGFUSE_PUBLIC_KEY=." .env; then \
		./scripts/generate-secrets.sh; \
	fi
```

### 2.4 Start Infrastructure

```bash
make up
# Wait for services (~30s on first run)
make status
```

---

## 3. Phase 2: SDK Initialization

### 3.1 Install Dependencies

```bash
uv add langfuse
```

### 3.2 Create Observability Module

Create `src/make_llmstxt/utils/observability.py`:

```python
"""Langfuse observability integration."""

import os
from typing import Optional
from contextlib import contextmanager

from .logging import StructuredLogger

log = StructuredLogger("observability")

# Singleton instances
_langfuse_client = None
_langfuse_initialized = False


def init_langfuse() -> bool:
    """Initialize Langfuse SDK for automatic tracing.

    Call once at application start.

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
            host=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
        )

        if _langfuse_client.auth_check():
            log.info("Langfuse initialized", host=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"))
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
    """Get the Langfuse client (initializes if needed)."""
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
```

### 3.3 Environment Variables

Add to `.env`:

```bash
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxx
```

### 3.4 Initialize at App Start

```python
from make_llmstxt.utils.observability import init_langfuse, flush_langfuse

# At application start
init_langfuse()

# ... run your code ...

# At application end (for short-lived scripts)
flush_langfuse()
```

---

## 4. Phase 3: Adding Traces

### 4.1 LangChain Integration

The key insight: **Callbacks must be attached to the LLM instance**, not passed to `ainvoke()`.

```python
# src/make_llmstxt/utils/observability.py

def get_langfuse_callback() -> list:
    """Get Langfuse callback handler for LangChain.

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
            return [handler]
        except Exception as e:
            log.warning("Failed to create Langfuse callback", error=str(e))
            return []
    except Exception as e:
        log.warning("Failed to create Langfuse callback", error=str(e))
        return []
```

### 4.2 Attach to LLM Instance

**WRONG** (callbacks don't propagate to nested calls):

```python
# This does NOT work - callbacks are lost in nested calls
result = await graph.ainvoke(
    state,
    config={"callbacks": get_langfuse_callback()}
)
```

**CORRECT** (attach to LLM instance):

```python
# src/make_llmstxt/utils/llm.py

def create_llm(config: LLMConfig) -> ChatOpenAI:
    """Create LLM instance from config."""
    from ..utils.observability import get_langfuse_callback
    callbacks = get_langfuse_callback()

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        api_key=config.api_key,
        base_url=config.base_url,
        callbacks=callbacks if callbacks else None,  # Key: attach here!
    )
```

### 4.3 Verify Traces

```bash
# Run your application
uv run make-llmstxt llmstxt https://example.com/docs

# Check Langfuse UI
open http://localhost:3000
```

You should now see traces with:
- Input/output messages
- Token counts
- Latency
- Model information

---

## 5. Phase 4: Adding Sessions

Sessions group all traces from a single run together.

### 5.1 Session ID Generation

```python
# src/make_llmstxt/utils/observability.py

import uuid

_current_session_id = None

def create_session_id() -> str:
    """Create a new session ID for grouping traces."""
    return f"session-{uuid.uuid4().hex[:12]}"


def get_current_session_id() -> Optional[str]:
    """Get the current session ID if set."""
    return _current_session_id
```

### 5.2 Session Propagation (The Key)

**WRONG** (passing via metadata or handler constructor doesn't work):

```python
# These do NOT work for session grouping
handler = CallbackHandler(session_id=session_id)  # Ignored
result = await llm.ainvoke(messages, config={"metadata": {"session_id": session_id}})  # Ignored
```

**CORRECT** (use `propagate_attributes` context manager):

```python
# src/make_llmstxt/utils/observability.py

@contextmanager
def session_context(session_id: str):
    """Context manager to propagate session_id to all nested observations."""
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
```

### 5.3 Use in Agent

```python
# src/make_llmstxt/generators/base_agent.py

from ..utils.observability import (
    init_langfuse,
    flush_langfuse,
    create_session_id,
    session_context,
)

class BaseAgent:
    async def run(self, url: str) -> Dict[str, Any]:
        # Initialize Langfuse
        init_langfuse()

        # Create session ID for this run
        session_id = create_session_id()
        log.info("Starting agent run", session_id=session_id)

        # Wrap entire run with session context
        with session_context(session_id):
            result = await self._run_internal(url)

        # Flush before exit
        flush_langfuse()

        return result
```

### 5.4 Verify Sessions

1. Run multiple times
2. Open Langfuse UI
3. Go to Sessions tab
4. Each run should appear as a separate session with all its traces grouped

---

## 6. Phase 5: Automatic Model Name Detection

When using router/proxy servers (like llama.cpp), the API model name is often just an alias (e.g., `llm`) rather than the actual model name. This makes traces harder to analyze in Langfuse.

### 6.1 The Problem

```python
# Router config uses aliases
[llm]
model = /models/Qwen-2.5-7B-Instruct-Q4_K_M.gguf

# But API only sees "llm" as the model name
# Langfuse shows "llm" instead of "Qwen-2.5-7B"
```

### 6.2 The Solution: Auto-Detection

The system automatically detects the model name using this priority:

1. **Explicit env var**: `LLM_MODEL_DISPLAY_NAME` (manual override)
2. **Model path extraction**: `LLM_MODEL_PATH` filename
3. **Server query**: llama.cpp `/props` endpoint (for local providers)
4. **Fallback**: Use the model name as-is (works for cloud providers)

### 6.3 Implementation

```python
# src/make_llmstxt/config.py

import re
import urllib.request
import json
from pathlib import Path

def _extract_model_display_name(model_path: str) -> str:
    """Extract a clean display name from a model file path.

    Example: /models/Qwen-2.5-7B-Instruct-Q4_K_M.gguf -> "Qwen-2.5-7B"
    """
    # Get filename without extension
    filename = Path(model_path).stem

    # Remove common quantization suffixes
    quant_patterns = [
        r'-[Qq]\d(_[A-Za-z\d]+)?',  # Q4_K_M, q4_k_m, Q4, etc.
        r'-[Ii][Qq]\d+_[A-Za-z]+',   # IQ4_XS, etc.
        r'-[Ff]\d+',                  # F16, F32, etc.
        r'-[Bb][Pp][Ww]\d+(?:-\d+)?', # BPW4, BPW4-8, etc.
        r'-[Ee][Xx][Ll]2',            # EXL2
    ]

    for pattern in quant_patterns:
        filename = re.sub(pattern, '', filename)

    # Remove common suffixes
    suffixes_to_remove = ['-Instruct', '-instruct', '-Chat', '-chat', r'-v\d+']
    for suffix in suffixes_to_remove:
        filename = re.sub(suffix + '$', '', filename)

    return filename


def _fetch_model_name_from_server(base_url: str, model_alias: str) -> Optional[str]:
    """Fetch actual model name from llama.cpp server.

    Queries the server's /props endpoint to get model metadata.
    """
    server_url = base_url.rstrip("/").removesuffix("/v1")

    try:
        props_url = f"{server_url}/props"
        req = urllib.request.Request(props_url, method="GET")
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())

            # llama.cpp returns model path in various fields
            model_path = (
                data.get("model_path") or
                data.get("default_generation_settings", {}).get("model_path") or
                data.get("model") or
                data.get("general_name")
            )

            if model_path:
                return _extract_model_display_name(model_path)

    except Exception:
        pass  # Server not available or doesn't support this endpoint

    return None
```

### 6.4 Config Integration

```python
# In AppConfig.from_env()

@staticmethod
def _get_display_name(provider: str, model: str, base_url: Optional[str]) -> Optional[str]:
    """Get display name with auto-detection priority."""
    # 1. Check explicit env var
    display_name = os.getenv("LLM_MODEL_DISPLAY_NAME")
    if display_name:
        return display_name

    # 2. Extract from model path
    model_path = os.getenv("LLM_MODEL_PATH")
    if model_path:
        return _extract_model_display_name(model_path)

    # 3. For local providers, query server for model info
    if provider in ("local",) and base_url:
        server_name = _fetch_model_name_from_server(base_url, model)
        if server_name:
            return server_name

    # 4. For cloud providers, model name is already descriptive
    return None
```

### 6.5 LLM Integration

```python
# src/make_llmstxt/utils/llm.py

class ChatOpenAIWithDisplayName(ChatOpenAI):
    """ChatOpenAI subclass that reports a display name to observability tools."""

    display_name: str = ""

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Override to return display_name as model_name for observability."""
        params = super()._identifying_params
        if self.display_name:
            params["model_name"] = self.display_name
        return params
```

### 6.6 Example Transformations

| Model Path | Display Name |
|------------|--------------|
| `/models/Qwen-2.5-7B-Instruct-Q4_K_M.gguf` | `Qwen-2.5-7B` |
| `/models/deepseek-r1-671b-BPW4.gguf` | `deepseek-r1-671b` |
| `/models/llama-3.2-3b-chat.F16.gguf` | `llama-3.2-3b` |
| `/models/mistral-7b-v0.3-IQ4_XS.gguf` | `mistral-7b` |

### 6.7 Usage

**Automatic (recommended):**
```bash
# Just set LLM_MODEL_PATH - display name is auto-derived
LLM_MODEL_PATH=/models/Qwen-2.5-7B-Instruct-Q4_K_M.gguf
# Langfuse shows: "Qwen-2.5-7B"
```

**Manual override:**
```bash
# If auto-detection doesn't work, set explicitly
LLM_MODEL_DISPLAY_NAME="Qwen-2.5-7B-Custom"
```

**Cloud providers:**
```bash
# No config needed - model name is already descriptive
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
# Langfuse shows: "gpt-4o-mini"
```

---

## 7. Complete Integration

### 6.1 Full Observability Module

```python
# src/make_llmstxt/utils/observability.py
"""Langfuse observability integration.

Uses Langfuse SDK v3+ with OpenTelemetry for automatic tracing.
All LLM calls are captured automatically without passing callbacks.

Usage:
    from make_llmstxt.utils.observability import (
        init_langfuse,
        flush_langfuse,
        create_session_id,
        session_context,
        get_langfuse_callback,
    )

    # Initialize at app start
    init_langfuse()

    # Create session for grouping
    session_id = create_session_id()

    # Run with session context
    with session_context(session_id):
        result = await my_llm_function()

    # Flush at end
    flush_langfuse()
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
    """Initialize Langfuse SDK for automatic tracing."""
    global _langfuse_client, _langfuse_initialized

    if _langfuse_initialized:
        return _langfuse_client is not None

    _langfuse_initialized = True

    if not (os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")):
        log.debug("Langfuse not configured")
        return False

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
        )

        if _langfuse_client.auth_check():
            log.info("Langfuse initialized")
            return True
        else:
            log.warning("Langfuse auth check failed")
            _langfuse_client = None
            return False

    except ImportError:
        log.debug("Langfuse SDK not installed")
        return False
    except Exception as e:
        log.warning("Langfuse initialization failed", error=str(e))
        _langfuse_client = None
        return False


def get_langfuse_client():
    """Get the Langfuse client (initializes if needed)."""
    global _langfuse_client
    if _langfuse_client is None and not _langfuse_initialized:
        init_langfuse()
    return _langfuse_client


def flush_langfuse():
    """Flush pending events to Langfuse."""
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
    """Create a new session ID for grouping traces."""
    return f"session-{uuid.uuid4().hex[:12]}"


@contextmanager
def session_context(session_id: str):
    """Context manager to propagate session_id to all nested observations."""
    global _current_session_id
    old_session_id = _current_session_id
    _current_session_id = session_id

    try:
        client = get_langfuse_client()
        if client is None:
            yield
            return

        try:
            from langfuse import propagate_attributes
            with propagate_attributes(session_id=session_id):
                yield
        except ImportError:
            yield
    finally:
        _current_session_id = old_session_id


def get_current_session_id() -> Optional[str]:
    """Get the current session ID if set."""
    return _current_session_id


def get_langfuse_callback() -> list:
    """Get Langfuse callback handler for LangChain."""
    client = get_langfuse_client()
    if client is None:
        return []

    try:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()
        log.debug("LangChain CallbackHandler created")
        return [handler]
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler
            handler = CallbackHandler()
            return [handler]
        except Exception as e:
            log.warning("Failed to create Langfuse callback", error=str(e))
            return []
    except Exception as e:
        log.warning("Failed to create Langfuse callback", error=str(e))
        return []


def observe(name: Optional[str] = None, as_type: str = "span"):
    """Decorator to trace a function with Langfuse."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = get_langfuse_client()
            if client is None:
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
```

### 6.2 Usage in Agent

```python
# src/make_llmstxt/generators/base_agent.py

from ..utils.observability import (
    init_langfuse,
    flush_langfuse,
    create_session_id,
    session_context,
    get_langfuse_callback,
)

class MyAgent:
    def _create_llm(self, config: LLMConfig) -> ChatOpenAI:
        """Create LLM with Langfuse callbacks attached."""
        callbacks = get_langfuse_callback()
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
            callbacks=callbacks if callbacks else None,
        )

    async def run(self, url: str) -> Dict[str, Any]:
        """Run agent with full observability."""
        # 1. Initialize Langfuse
        init_langfuse()

        # 2. Create session ID
        session_id = create_session_id()
        log.info("Starting run", session_id=session_id, url=url)

        # 3. Wrap with session context
        with session_context(session_id):
            try:
                result = await self._run_internal(url)
                log.info("Run complete", session_id=session_id)
                return result
            except Exception as e:
                log.error("Run failed", session_id=session_id, error=str(e))
                raise
            finally:
                # 4. Flush traces
                flush_langfuse()
```

### 6.3 Quick Start Checklist

```bash
# 1. Start infrastructure
make up

# 2. Verify environment
grep LANGFUSE .env
# Should show:
# LANGFUSE_BASE_URL=http://localhost:3000
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...

# 3. Run your application
uv run make-llmstxt llmstxt https://example.com/docs

# 4. View traces
open http://localhost:3000
# Login: admin@example.com / password123
```

---

## 8. Troubleshooting

### No Traces Appearing

1. **Check API keys are set:**
   ```bash
   grep LANGFUSE .env
   ```

2. **Check callbacks are attached to LLM:**
   ```python
   # Must attach to ChatOpenAI constructor, NOT ainvoke config
   llm = ChatOpenAI(..., callbacks=get_langfuse_callback())
   ```

3. **Check flush is called:**
   ```python
   flush_langfuse()  # At end of script
   ```

### Sessions Not Grouping

1. **Use `propagate_attributes` context manager:**
   ```python
   from langfuse import propagate_attributes
   with propagate_attributes(session_id=session_id):
       # All LLM calls here are grouped
   ```

2. **Don't use CallbackHandler constructor:**
   ```python
   # WRONG
   handler = CallbackHandler(session_id=session_id)

   # CORRECT
   handler = CallbackHandler()  # Session from propagate_attributes
   ```

### Login Issues

Default credentials (from docker-compose.yml):
- Email: `admin@example.com`
- Password: `password123`

To reset:
```bash
make reset-observability
make up-observability
```

### Import Errors

```bash
# Install Langfuse with LangChain support
uv add langfuse

# Verify
python -c "from langfuse.langchain import CallbackHandler; print('OK')"
```

---

## Summary

| Phase | What | Key Insight |
|-------|------|-------------|
| 1 | Infrastructure | Headless init with `LANGFUSE_INIT_*` env vars |
| 2 | SDK Init | `init_langfuse()` at app start |
| 3 | Tracing | Attach callbacks to LLM instance, not ainvoke |
| 4 | Sessions | Use `propagate_attributes(session_id=...)` context |

**The two critical mistakes to avoid:**

1. **Don't pass callbacks to ainvoke** - They don't propagate to nested calls
2. **Don't pass session_id to CallbackHandler** - Use `propagate_attributes` instead
