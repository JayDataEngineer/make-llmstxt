"""Make LLMs.txt - Generate llms.txt files for any website.

Uses MCP for scraping and LangChain for LLM abstraction,
supporting multiple providers (OpenAI, Anthropic, local servers, etc.)
"""

__version__ = "0.1.0"

from .config import AppConfig, LLMConfig, MCPConfig, PROVIDER_PROFILES
from .generator import generate_llmstxt, LLMsTxtGenerator, GenerationResult, PageResult
from .mcp_tools import (
    create_mcp_client,
    get_mcp_tools,
    filter_tools_by_name,
    MAIN_AGENT_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
    mcp_map_domain,
    mcp_crawl_site,
    mcp_scrape_url,
    mcp_scrape_batch,
    mcp_map_website,
)
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
    # MCP tools
    "create_mcp_client",
    "get_mcp_tools",
    "filter_tools_by_name",
    "MAIN_AGENT_TOOL_NAMES",
    "SUBAGENT_TOOL_NAMES",
    "mcp_map_domain",
    "mcp_crawl_site",
    "mcp_scrape_url",
    "mcp_scrape_batch",
    "mcp_map_website",
    # Other
    "Critic",
    "CriticResult",
    "DeepDraftConfig",
    "DraftState",
    "SimpleDraftCritic",
    "DeepDraftAgent",
    "build_drafter_prompt",
]
