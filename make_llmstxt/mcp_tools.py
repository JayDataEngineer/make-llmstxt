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
        # MCP content format: [{'type': 'text', 'text': '...'}]
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


async def mcp_map_domain(
    host: str,
    port: int,
    domain: str,
    max_urls: int = 100,
    pattern: str = "*",
) -> List[str]:
    """Discover URLs from a domain using MCP map_domain tool.

    Args:
        host: MCP server host
        port: MCP server port
        domain: Domain to map (e.g., 'example.com')
        max_urls: Maximum URLs to return
        pattern: URL pattern filter

    Returns:
        List of discovered URLs
    """
    client = create_mcp_client(host, port)
    tools = await client.get_tools()

    # Find the map_domain tool
    map_domain_tool = next((t for t in tools if t.name == "map_domain"), None)
    if not map_domain_tool:
        logger.error("[MCP] map_domain tool not found")
        return []

    result = await map_domain_tool.ainvoke({
        "domain": domain,
        "max_urls": max_urls,
        "pattern": pattern,
    })

    data = _extract_tool_result(result)
    urls = data.get("urls", []) if isinstance(data, dict) else []

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
    client = create_mcp_client(host, port)
    tools = await client.get_tools()

    # Find the crawl_site tool
    crawl_site_tool = next((t for t in tools if t.name == "crawl_site"), None)
    if not crawl_site_tool:
        logger.error("[MCP] crawl_site tool not found")
        return []

    args = {
        "url": url,
        "max_depth": max_depth,
        "max_pages": max_pages,
    }
    if include_patterns:
        args["include_patterns"] = include_patterns
    if exclude_patterns:
        args["exclude_patterns"] = exclude_patterns

    result = await crawl_site_tool.ainvoke(args)

    data = _extract_tool_result(result)
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
        Dict with url, title, content, metadata or None on failure
    """
    client = create_mcp_client(host, port)
    tools = await client.get_tools()

    # Find the scrape_url tool
    scrape_url_tool = next((t for t in tools if t.name == "scrape_url"), None)
    if not scrape_url_tool:
        logger.error("[MCP] scrape_url tool not found")
        return None

    result = await scrape_url_tool.ainvoke({"url": url})

    data = _extract_tool_result(result)

    if isinstance(data, dict):
        if data.get("success"):
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
        else:
            logger.warning(f"[MCP] Scrape failed for {url}: {data.get('error', 'Unknown error')}")
            return None
    elif isinstance(data, str):
        # Fallback: treat as raw markdown
        return {
            "url": url,
            "title": "",
            "content": data,
            "markdown": data,
            "metadata": {},
        }
    else:
        logger.warning(f"[MCP] Unexpected scrape_url result type: {type(data)}")
        return None


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
