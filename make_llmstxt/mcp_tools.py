"""Shared MCP tools module for both llms.txt and skill generators.

Uses langchain-mcp-adapters to load tools from MCP server.
"""

from typing import List, Dict, Any, Optional, Set
from contextlib import asynccontextmanager

from loguru import logger

from langchain_mcp_adapters.client import MultiServerMCPClient


# ==============================================================================
# Tool name constants for filtering
# ==============================================================================

MAIN_AGENT_TOOL_NAMES: Set[str] = {"map_domain", "crawl_site", "scrape_url"}
SUBAGENT_TOOL_NAMES: Set[str] = {"scrape_url"}
ALL_SCRAPER_TOOLS: Set[str] = MAIN_AGENT_TOOL_NAMES | SUBAGENT_TOOL_NAMES


# ==============================================================================
# Tool filtering utilities
# ==============================================================================

def filter_tools_by_name(tools: List, tool_names: Set[str]) -> List:
    """Filter tools by their names.

    Args:
        tools: List of LangChain tools
        tool_names: Set of tool names to include

    Returns:
        Filtered list of tools
    """
    return [t for t in tools if t.name in tool_names]


# ==============================================================================
# MCP Client Creation
# ==============================================================================

def create_mcp_client(host: str, port: int) -> MultiServerMCPClient:
    """Create MCP client for webtools server.

    Args:
        host: MCP server host
        port: MCP server port

    Returns:
        MultiServerMCPClient instance
    """
    return MultiServerMCPClient(
        {
            "webtools": {
                "transport": "http",
                "url": f"http://{host}:{port}/mcp",
            }
        }
    )


@asynccontextmanager
async def get_mcp_tools(host: str, port: int, max_urls: int = 100):
    """Context manager to connect to MCP server and get LangChain tools.

    Usage:
        async with get_mcp_tools(host, port) as tools:
            main_tools = filter_tools_by_name(tools, MAIN_AGENT_TOOL_NAMES)
            subagent_tools = filter_tools_by_name(tools, SUBAGENT_TOOL_NAMES)
    """
    client = create_mcp_client(host, port)

    # Use session for stateful connection during graph execution
    async with client.session("webtools") as session:
        from langchain_mcp_adapters.tools import load_mcp_tools
        tools = await load_mcp_tools(session)
        scraper_tools = filter_tools_by_name(tools, ALL_SCRAPER_TOOLS)
        logger.info(f"[MCP] Loaded {len(scraper_tools)} tools: {[t.name for t in scraper_tools]}")
        yield scraper_tools


# ==============================================================================
# Direct async functions for programmatic use (llms.txt generator)
# ==============================================================================

def _extract_tool_result(result: Any) -> Any:
    """Extract result from MCP tool response.

    MCP tools return content as a list: [{'type': 'text', 'text': '...'}]
    This extracts the text content and parses JSON if needed.
    """
    import json

    if isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        if isinstance(first_item, dict) and first_item.get("type") == "text":
            text = first_item.get("text", "")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return result
    elif isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


async def _call_mcp_tool(host: str, port: int, tool_name: str, args: dict) -> Any:
    """Helper to call an MCP tool and return extracted result.

    Reduces boilerplate in mcp_* functions.
    """
    client = create_mcp_client(host, port)
    tools = await client.get_tools()
    tool = next((t for t in tools if t.name == tool_name), None)
    if not tool:
        raise RuntimeError(f"MCP tool '{tool_name}' not found")
    result = await tool.ainvoke(args)
    return _extract_tool_result(result)


async def mcp_map_domain(
    host: str,
    port: int,
    domain: str,
    max_urls: int = 100,
    pattern: str = "*",
) -> List[str]:
    """Discover URLs from a domain using MCP map_domain tool."""
    data = await _call_mcp_tool(host, port, "map_domain", {
        "domain": domain,
        "max_urls": max_urls,
        "pattern": pattern,
    })

    raw_urls = data.get("urls", []) if isinstance(data, dict) else []

    # Extract URL strings from [{"url": "..."}, ...] format
    urls = []
    for item in raw_urls:
        if isinstance(item, dict) and "url" in item:
            urls.append(item["url"])
        elif isinstance(item, str):
            urls.append(item)

    logger.info(f"[MCP] map_domain found {len(urls)} URLs for {domain}")
    return urls


async def mcp_crawl_site(
    host: str,
    port: int,
    url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Deep crawl a site following links using MCP crawl_site tool.

    Args:
        host: MCP server host
        port: MCP server port
        url: Starting URL to crawl
        max_depth: Maximum depth to crawl (1-5)
        max_pages: Maximum pages to crawl (1-200)
        include_patterns: URL patterns to include
        exclude_patterns: URL patterns to exclude

    Returns:
        List of crawled pages with content
    """
    args = {
        "url": url,
        "max_depth": max_depth,
        "max_pages": max_pages,
    }
    if include_patterns:
        args["include_patterns"] = include_patterns
    if exclude_patterns:
        args["exclude_patterns"] = exclude_patterns

    data = await _call_mcp_tool(host, port, "crawl_site", args)
    pages = data.get("pages", []) if isinstance(data, dict) else []

    logger.info(f"[MCP] crawl_site found {len(pages)} pages from {url}")
    return pages


async def mcp_scrape_url(
    host: str,
    port: int,
    url: str,
) -> Optional[Dict[str, Any]]:
    """Scrape a single URL using MCP scrape_url tool.

    Args:
        host: MCP server host
        port: MCP server port
        url: URL to scrape

    Returns:
        Dict with url, title, content, metadata or raises on failure
    """
    data = await _call_mcp_tool(host, port, "scrape_url", {"url": url})

    if not isinstance(data, dict):
        raise RuntimeError(f"MCP scrape_url returned unexpected type: {type(data)}")

    if not data.get("success"):
        raise RuntimeError(f"MCP scrape_url failed for {url}: {data.get('error', 'Unknown error')}")

    return {
        "url": data.get("url", url),
        "title": data.get("title", ""),
        "content": data.get("content", ""),
        "markdown": data.get("content", ""),
        "metadata": {
            "title": data.get("title"),
            "word_count": data.get("word_count"),
        },
    }


async def mcp_scrape_batch(
    host: str,
    port: int,
    urls: List[str],
) -> List[Dict[str, Any]]:
    """Scrape multiple URLs concurrently.

    Args:
        host: MCP server host
        port: MCP server port
        urls: URLs to scrape

    Returns:
        List of scraped pages (excludes failures)
    """
    import asyncio

    async def scrape_one(url: str) -> Optional[Dict[str, Any]]:
        return await mcp_scrape_url(host, port, url)

    tasks = [scrape_one(url) for url in urls]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


async def mcp_map_website(
    host: str,
    port: int,
    url: str,
    limit: Optional[int] = None,
) -> List[str]:
    """Map website to get URLs (compatibility function).

    Uses map_domain to discover URLs for the given website.

    Args:
        host: MCP server host
        port: MCP server port
        url: Website URL
        limit: Maximum URLs to return

    Returns:
        List of discovered URLs
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = parsed.netloc

    urls = await mcp_map_domain(
        host,
        port,
        domain,
        max_urls=limit or 100,
    )

    logger.info(f"[MCP] map_website found {len(urls)} URLs for {url}")
    return urls


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Constants
    "MAIN_AGENT_TOOL_NAMES",
    "SUBAGENT_TOOL_NAMES",
    "ALL_SCRAPER_TOOLS",
    # Utilities
    "filter_tools_by_name",
    # Client creation
    "create_mcp_client",
    # Context managers
    "get_mcp_tools",
    # Direct functions
    "mcp_map_domain",
    "mcp_crawl_site",
    "mcp_scrape_url",
    "mcp_scrape_batch",
    "mcp_map_website",
]
