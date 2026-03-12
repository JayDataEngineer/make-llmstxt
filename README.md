# Make LLMs.txt

Generate `llms.txt` and skill packages for documentation websites using MCP scraper, LangChain, and AI agents with semantic search.

Supports multiple LLM providers: OpenAI, Anthropic, DeepSeek, OpenRouter, local servers (llama.cpp, vLLM), and more.

## What is llms.txt?

`llms.txt` is a standardized format for making website content more accessible to Large Language Models:

- **llms.txt**: A concise index of all pages with titles and descriptions
- **llms-full.txt**: Complete content of all pages for comprehensive access

## Features

- **Parallel scraping**: Discovers and scrapes multiple pages concurrently
- **Semantic search**: Uses embeddings for intelligent content retrieval
- **Generator-Critic loop**: AI agent iteratively improves output quality
- **Multiple providers**: OpenAI, Anthropic, DeepSeek, OpenRouter, local LLMs
- **Skill generation**: Create skill packages (SKILL.md, scripts/, references/)
- **Observability**: Langfuse integration for LLM tracing and debugging

## Installation

### Using uv (recommended)

```bash
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx

# MCP Scraper (Tailscale)
MCP_HOST=100.85.22.99
MCP_PORT=8000
```

### 2. Run Generation

```bash
# Generate llms.txt for a website
uv run make-llmstxt llmstxt https://example.com/docs

# Generate skill package
uv run make-llmstxt skill https://nextjs.org/docs
```

## Usage

### Generate llms.txt

```bash
# Basic usage
uv run make-llmstxt llmstxt https://example.com/docs

# Limit to 50 URLs
uv run make-llmstxt llmstxt https://example.com/docs --max-urls 50

# Save to specific directory
uv run make-llmstxt llmstxt https://example.com/docs -o ./output

# Use different provider
uv run make-llmstxt llmstxt https://example.com/docs --provider deepseek

# Use local LLM server
uv run make-llmstxt llmstxt https://example.com/docs \
    --provider local \
    --base-url http://localhost:8001/v1 \
    --api-key sk-llama

# Verbose logging
uv run make-llmstxt llmstxt https://example.com/docs --verbose
```

### Generate Skill Packages

```bash
# Basic usage
uv run make-llmstxt skill https://react.dev

# Validate existing skill package
uv run make-llmstxt skill https://react.dev --validate

# Auto-fix issues before generation
uv run make-llmstxt skill https://react.dev --clean

# Custom rounds for critic loop
uv run make-llmstxt skill https://react.dev --max-rounds 5
```

### List Available Providers

```bash
uv run make-llmstxt --list-providers
```

## LLM Providers

| Provider | Env Key | Default Model |
|----------|---------|---------------|
| `openai` | `OPENAI_API_KEY` | gpt-4o-mini |
| `anthropic` | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| `deepseek` | `DEEPSEEK_API_KEY` | deepseek-chat |
| `openrouter` | `OPENROUTER_API_KEY` | multiple available |
| `local` | `LOCAL_API_KEY` | custom |
| `zai`/`glm` | `ZAI_API_KEY` | glm-4 |

## Local LLM Setup (llama.cpp)

Run a local LLM server with lazy-loaded models:

### 1. Configure Models

Edit `.env`:

```bash
LLM_MODEL_PATH=/path/to/your/llm-model.gguf
EMBED_MODEL_PATH=/path/to/your/embedding-model.gguf
```

### 2. Start Server

```bash
make up-llm
# or: make up (starts observability too)
```

Models are **lazy-loaded** - they load on first request, saving GPU memory when idle.

### 3. Verify

```bash
# List available models
curl localhost:8001/v1/models | jq '.data[].id'

# Test chat (loads LLM model on first call)
curl -s localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "llm", "messages": [{"role": "user", "content": "Hi"}]}'

# Test embeddings (loads embed model on first call)
curl -s localhost:8001/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{"model": "embed", "input": "test"}'
```

### 4. Use with make-llmstxt

```bash
uv run make-llmstxt llmstxt https://example.com/docs \
    --provider local \
    --base-url http://localhost:8001/v1 \
    --api-key local
```

## Environment Variables

### LLM Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Provider to use | `openai` |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `LLM_BASE_URL` | API base URL | (provider default) |
| `LLM_API_KEY` | Generic API key | - |

### Provider-Specific Keys

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `OPENROUTER_API_KEY` | OpenRouter |
| `ZAI_API_KEY` | ZAI/GLM |
| `LOCAL_API_KEY` | Local server |

### Embedding Configuration (Semantic Search)

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_BASE_URL` | Embedding API URL | Same as `LLM_BASE_URL` |
| `EMBEDDING_MODEL` | Embedding model name | `embed` |
| `EMBEDDING_DIMS` | Embedding dimensions | `1024` |

### MCP Scraper Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_HOST` | MCP server host | `100.85.22.99` |
| `MCP_PORT` | MCP server port | `8000` |

### Langfuse Observability

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGFUSE_BASE_URL` | Langfuse server URL | `http://localhost:3000` |
| `LANGFUSE_PUBLIC_KEY` | Public API key | - |
| `LANGFUSE_SECRET_KEY` | Secret API key | - |

When both keys are set, LLM tracing is automatically enabled.

### Advanced

| Variable | Description | Default |
|----------|-------------|---------|
| `SKILL_MAX_ROUNDS` | Max critic rounds for skill generation | `3` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Python API

```python
import asyncio
from make_llmstxt import generate_llmstxt

async def main():
    result = await generate_llmstxt(
        "https://example.com/docs",
        max_urls=50,
        output_dir="./output",
    )
    print(f"Processed {result.num_urls_processed} URLs")
    print(result.llmstxt)

asyncio.run(main())
```

### Custom Configuration

```python
from make_llmstxt.config import AppConfig, LLMConfig
from make_llmstxt import generate_llmstxt

config = AppConfig(
    llm=LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        api_key="sk-xxx",
    ),
    max_urls=100,
)

result = await generate_llmstxt(
    "https://example.com/docs",
    config=config,
)
```

## Output Format

### llms.txt

```markdown
# Example
> A sample website demonstrating the llms.txt format.

## Core
- [Getting Started](https://example.com/): Welcome page with main navigation
- [About Us](https://example.com/about): Company history and team
- [Products](https://example.com/products): Product catalog and pricing

## API Reference
- [API Docs](https://example.com/api): REST API documentation

## Optional
- [Blog](https://example.com/blog): Company news and updates
```

### llms-full.txt

```markdown
# Example llms-full.txt

<|page-1|>
## Getting Started
URL: https://example.com/

Full markdown content of the page...

<|page-2|>
## About Us
URL: https://example.com/about

Full markdown content of another page...
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      make-llmstxt Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   MCP Server │    │  LLM Server  │    │ Embed Server │      │
│  │ (Scraper)    │    │ (Chat Model) │    │ (Embeddings) │      │
│  │ Port 8000    │    │ Port 8001    │    │ Port 8001    │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                    │              │
│         ▼                   ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Deep Agent Generator                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │ Discovery   │→ │  Parallel   │→ │   Synthesis     │  │   │
│  │  │ (map_domain)│  │  Scrapers   │  │ (write output)  │  │   │
│  │  └─────────────┘  └──────┬──────┘  └─────────────────┘  │   │
│  │                          │                               │   │
│  │                          ▼                               │   │
│  │              ┌───────────────────────┐                   │   │
│  │              │  InMemory Store       │                   │   │
│  │              │  (Semantic Search)    │                   │   │
│  │              └───────────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Observability

make-llmstxt supports LLM observability via [Langfuse](https://langfuse.com) for tracing, token tracking, and debugging.

### Quick Start

```bash
# 1. Start all infrastructure (LLM + observability)
make up

# 2. Wait for services to start (~30s on first run)
# API keys are auto-generated and added to .env

# 3. Run generation - traces appear in Langfuse automatically
uv run make-llmstxt llmstxt https://docs.python.org/3/

# 4. View traces (optional)
open http://localhost:3000
# Login: admin@example.com / password123
```

**No manual setup needed!** On first run, Langfuse auto-creates:
- Organization and project
- API keys (added to your `.env`)
- Admin user (for optional UI access)

### Makefile Commands

```bash
make up              # Start all services
make down            # Stop all services
make status          # Show service status
make logs            # Follow all logs

# Partial startup
make up-llm          # Only LLM server (port 8001)
make up-observability # Only observability (port 3000)

# Cleanup (WARNING: deletes all data)
make clean
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `llama-server` | 8001 | LLM + embeddings (lazy-loaded) |
| `langfuse-web` | 3000 | Observability UI |
| `postgres` | 5432 | Langfuse database |
| `clickhouse` | 8123 | Analytics database |
| `redis` | 6379 | Queue/cache |
| `minio` | 9090 | S3 storage |

### Features

- **Trace Visibility**: See all LLM calls with inputs/outputs
- **Token Tracking**: Monitor usage across providers
- **Cost Analysis**: Track costs per project/run
- **Debugging**: Inspect failed calls and retry logic
- **Latency Metrics**: Identify slow operations

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"
# or
pip install -e ".[dev]"

# Run tests
pytest

# Run with verbose logging
uv run make-llmstxt llmstxt https://example.com/docs --verbose --log-file debug.log
```

## Troubleshooting

### API Key Errors

Ensure the correct environment variable is set for your provider:

```bash
# For OpenAI
export OPENAI_API_KEY=sk-xxx

# For local server (any non-empty value works)
export LOCAL_API_KEY=sk-llama
```

### MCP Connection Issues

Verify the MCP server is reachable:

```bash
curl http://<MCP_HOST>:<MCP_PORT>/health
```

### Local LLM Errors

Check the Docker container status:

```bash
docker ps | grep make-llmstxt-llama
docker logs make-llmstxt-llama
```

### Out of Memory (GPU)

Reduce context size in `docker/preset.ini` or use CPU for embeddings:

```ini
[embed]
n-gpu-layers = 0  ; Run embeddings on CPU
```

## License

MIT License
