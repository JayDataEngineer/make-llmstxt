"""Make LLMs.txt - Generate llms.txt files for any website.

Uses Firecrawl or MCP for scraping and LangChain for LLM abstraction,
supporting multiple providers (OpenAI, Anthropic, local servers, etc.)
"""

__version__ = "0.1.0"

from .config import AppConfig, LLMConfig, PROVIDER_PROFILES
from .generator import generate_llmstxt, LLMsTxtGenerator, GenerationResult, PageResult
from .mcp_scraper import MCPWebScraper, MCPConfig
from .firecrawl import FirecrawlClient, FirecrawlConfig

__all__ = [
    "AppConfig",
    "LLMConfig",
    "PROVIDER_PROFILES",
    "generate_llmstxt",
    "LLMsTxtGenerator",
    "GenerationResult",
    "PageResult",
    "MCPWebScraper",
    "MCPConfig",
    "FirecrawlClient",
    "FirecrawlConfig",
]
