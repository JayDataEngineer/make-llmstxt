"""Make LLMs.txt - Generate llms.txt files for any website.

Uses MCP for scraping and LangChain for LLM abstraction,
supporting multiple providers (OpenAI, Anthropic, local servers, etc.)
"""

__version__ = "0.1.0"

from .config import AppConfig, LLMConfig, MCPConfig, PROVIDER_PROFILES
from .generator import generate_llmstxt, LLMsTxtGenerator, GenerationResult, PageResult
from .mcp_scraper import MCPWebScraper
from .critic import Critic, CriticResult
from .deep_draft import (
    DeepDraftConfig,
    DraftState,
    SimpleDraftCritic,
    DeepDraftAgent,
    build_drafter_prompt,
)

__all__ = [
    "AppConfig",
    "LLMConfig",
    "MCPConfig",
    "PROVIDER_PROFILES",
    "generate_llmstxt",
    "LLMsTxtGenerator",
    "GenerationResult",
    "PageResult",
    "MCPWebScraper",
    "Critic",
    "CriticResult",
    "DeepDraftConfig",
    "DraftState",
    "SimpleDraftCritic",
    "DeepDraftAgent",
    "build_drafter_prompt",
]
