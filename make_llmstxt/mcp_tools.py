"""Shared MCP tools module for both llms.txt and skill generators.

Provides:
- get_mcp_tools(): Context manager for loading MCP tools as LangChain tools
- Direct async functions for programmatic use (map_domain, crawl_site, scrape_url)
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
# MCP Client Context Manager
# ==============================================================================

@asynccontextmanager
async def get_mcp_client(host: str, port: int):
    """Context manager to connect to MCP server.

    Usage:
        async with get_mcp_client(host, port) as client:
            tools = await client.get_tools()
            # or use direct functions below
    """
    mcp_url = f"http://{host}:{port}/sse"

    client = MultiServerMCPClient({
        "webtools": {
            "transport": "sse",
            "url": mcp_url,
        }
    })

    async with client:
        logger.info(f"[MCP] Connected to {mcp_url}")
        yield client


@asynccontextmanager
async def get_mcp_tools(host: str, port: int):
    """Context manager to connect to MCP server and get LangChain tools.

    Usage:
        async with get_mcp_tools(host, port) as tools:
            main_tools = filter_tools_by_name(tools, MAIN_AGENT_TOOL_NAMES)
            subagent_tools = filter_tools_by_name(tools, SUBAGENT_TOOL_NAMES)
    """
    async with get_mcp_client(host, port) as client:
        tools = await client.get_tools()
        # Filter to only scraper tools
        scraper_tools = filter_tools_by_name(tools, ALL_SCRAPER_TOOLS)
        logger.info(f"[MCP] Loaded {len(scraper_tools)} tools: {[t.name for t in scraper_tools]}")
        yield scraper_tools


# ==============================================================================
# Direct async functions for programmatic use
# ==============================================================================

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
    async with get_mcp_client(host, port) as client:
        tools = await client.get_tools()
        map_tool = next((t for t in tools if t.name == "map_domain"), None)

        if not map_tool:
            raise RuntimeError("map_domain tool not found on MCP server")

        result = await map_tool.ainvoke({
            "domain": domain,
            "max_urls": max_urls,
            "pattern": pattern,
        })

        # Parse result - MCP tools return content as list of dicts
        if isinstance(result, list) and len(result) > 0:
            content = result[0]
            if isinstance(content, dict) and "text" in content:
                import json
                data = json.loads(content["text"])
                urls = data.get("urls", [])
                logger.info(f"[MCP] map_domain found {len(urls)} URLs for {domain}")
                return urls

        logger.warning(f"[MCP] Unexpected map_domain result format")
        return []


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
    async with get_mcp_client(host, port) as client:
        tools = await client.get_tools()
        crawl_tool = next((t for t in tools if t.name == "crawl_site"), None)

        if not crawl_tool:
            raise RuntimeError("crawl_site tool not found on MCP server")

        args = {
            "url": url,
            "max_depth": max_depth,
            "max_pages": max_pages,
        }
        if include_patterns:
            args["include_patterns"] = include_patterns
        if exclude_patterns:
            args["exclude_patterns"] = exclude_patterns

        result = await crawl_tool.ainvoke(args)

        # Parse result
        if isinstance(result, list) and len(result) > 0:
            content = result[0]
            if isinstance(content, dict) and "text" in content:
                import json
                data = json.loads(content["text"])
                pages = data.get("pages", [])
                logger.info(f"[MCP] crawl_site found {len(pages)} pages from {url}")
                return pages

        logger.warning(f"[MCP] Unexpected crawl_site result format")
        return []


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
    async with get_mcp_client(host, port) as client:
        tools = await client.get_tools()
        scrape_tool = next((t for t in tools if t.name == "scrape_url"), None)

        if not scrape_tool:
            raise RuntimeError("scrape_url tool not found on MCP server")

        result = await scrape_tool.ainvoke({"url": url})

        # Parse result
        if isinstance(result, list) and len(result) > 0:
            content = result[0]
            if isinstance(content, dict) and "text" in content:
                import json
                data = json.loads(content["text"])
                if data.get("success"):
                    return {
                        "url": data.get("url", url),
                        "title": data.get("title", ""),
                        "content": data.get("content", ""),
                        "markdown": data.get("content", ""),
                        "metadata": {
                            "title": data.get("title"),
                            "word_count": data.get("word_count"),
                            "method": data.get("method_used"),
                        },
                    }
                else:
                    logger.warning(f"[MCP] Scrape failed for {url}: {data.get('error')}")
                    return None

        logger.warning(f"[MCP] Unexpected scrape_url result format")
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

    Uses map_domain under the hood for consistency.

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
        host=host,
        port=port,
        domain=domain,
        max_urls=limit or 100,
    )

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
    # Context managers
    "get_mcp_client",
    "get_mcp_tools",
    # Direct functions
    "mcp_map_domain",
    "mcp_crawl_site",
    "mcp_scrape_url",
    "mcp_scrape_batch",
    "mcp_map_website",
]
