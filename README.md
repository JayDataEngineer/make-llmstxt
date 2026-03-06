# Make LLMs.txt

Generate `llms.txt` and `llms-full.txt` files for any website using Firecrawl and LangChain.

Supports multiple LLM providers: OpenAI, Anthropic, DeepSeek, OpenRouter, local servers, and more.

## What is llms.txt?

`llms.txt` is a standardized format for making website content more accessible to Large Language Models:

- **llms.txt**: A concise index of all pages with titles and descriptions
- **llms-full.txt**: Complete content of all pages for comprehensive access

## Installation

```bash
pip install -e .
```

Or with uv:

```bash
uv pip install -e .
```

## Quick Start

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:

```bash
FIRECRAWL_API_KEY=fc-xxx
OPENAI_API_KEY=sk-xxx
```

3. Run the generator:

```bash
make-llmstxt https://example.com
```

## Usage

### Basic Usage

```bash
# Generate llms.txt for a website
make-llmstxt https://example.com

# Limit to 50 URLs
make-llmstxt https://example.com --max-urls 50

# Save to specific directory
make-llmstxt https://example.com --output-dir ./output

# Only generate llms.txt (skip full text)
make-llmstxt https://example.com --no-full-text
```

### Using Different LLM Providers

```bash
# Use DeepSeek instead of OpenAI
make-llmstxt https://example.com --provider deepseek

# Use local server (llama.cpp, vLLM, etc.)
make-llmstxt https://example.com --provider local --base-url http://localhost:8000/v1

# Use OpenRouter
make-llmstxt https://example.com --provider openrouter

# Use ZAI/GLM
make-llmstxt https://example.com --provider zai
```

### List Available Providers

```bash
make-llmstxt --list-providers
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | Provider to use (openai, anthropic, deepseek, local, etc.) |
| `LLM_MODEL` | Model name to use |
| `LLM_BASE_URL` | Base URL for API (for custom providers) |
| `LLM_API_KEY` | Generic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `ZAI_API_KEY` | ZAI/GLM API key |
| `FIRECRAWL_API_KEY` | Firecrawl API key |

## Python API

```python
import asyncio
from make_llmstxt import generate_llmstxt

async def main():
    result = await generate_llmstxt(
        "https://example.com",
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
    "https://example.com",
    config=config,
)
```

## Output Format

### llms.txt

```
# https://example.com llms.txt

- [Home](https://example.com/): Welcome page with main navigation
- [About Us](https://example.com/about): Company history and team
- [Products](https://example.com/products): Product catalog and pricing
```

### llms-full.txt

```
# https://example.com llms-full.txt

<|page-1|>
## Home
URL: https://example.com/

Full markdown content of the page...

<|page-2|>
## About Us
URL: https://example.com/about

Full markdown content of another page...
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT License
