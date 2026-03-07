"""Shared MCP tools module for both llms.txt and skill generators.

Provides:
- Tool name constants for filtering
- Wrapper functions using MCPWebScraper for direct programmatic use
- LangChain tool creation for use with deep agents
"""

from typing import List, Dict, Any, Optional, Set
from contextlib import asynccontextmanager

from loguru import logger
from langchain_core.tools import tool

from .mcp_scraper import MCPWebScraper, MCPConfig


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
# LangChain Tool Creation (for deep agents)
# ==============================================================================

def create_mcp_tools(scraper: MCPWebScraper, max_urls: int = 100) -> List:
    """Create LangChain tools wrapping MCP scraper methods.

    Args:
        scraper: MCPWebScraper instance
        max_urls: Default max URLs for mapping operations

    Returns:
        List of LangChain tools
    """
    import asyncio

    tools = []

    @tool
    def map_domain(domain: str, pattern: str = "*") -> dict:
        """Discover URLs from a domain using sitemaps or Common Crawl.

        Use this to find all available URLs on a website before crawling.

        Args:
            domain: Domain to map (e.g., 'docs.python.org')
            pattern: URL pattern filter (e.g., '*/api/*')

        Returns:
            Dict with 'urls' (list of discovered URLs) and 'total' count
        """
        urls = asyncio.run(scraper.map_domain(
            domain,
            max_urls=max_urls,
            pattern=pattern,
        ))
        return {"urls": urls, "total": len(urls)}

    tools.append(map_domain)

    @tool
    def crawl_site(url: str, max_depth: int = 2, max_pages: int = 50) -> dict:
        """Deep crawl a site following links (BFS strategy).

        Use this to crawl multiple pages from a documentation site.

        Args:
            url: Starting URL to crawl
            max_depth: Maximum depth to crawl (1-5)
            max_pages: Maximum pages to crawl (1-200)

        Returns:
            Dict with 'pages' (list of crawled pages with url, title, content)
        """
        pages = asyncio.run(scraper.crawl_site(
            url,
            max_depth=max_depth,
            max_pages=min(max_pages, max_urls),
        ))
        return {"pages": pages, "total": len(pages)}

    tools.append(crawl_site)

    @tool
    def scrape_url(url: str) -> dict:
        """Scrape a single URL and return markdown content.

        Args:
            url: URL to scrape

        Returns:
            Dict with url, title, content, metadata
        """
        result = asyncio.run(scraper.scrape_url(url))
        if result:
            return {
                "url": url,
                "title": result.get("metadata", {}).get("title", ""),
                "content": result.get("markdown", ""),
                "metadata": result.get("metadata", {}),
            }
        return {"error": f"Failed to scrape {url}"}

    tools.append(scrape_url)

    return tools


@asynccontextmanager
async def get_mcp_tools(host: str, port: int, max_urls: int = 100):
    """Context manager to connect to MCP server and get LangChain tools.

    Usage:
        async with get_mcp_tools(host, port) as tools:
            main_tools = filter_tools_by_name(tools, MAIN_AGENT_TOOL_NAMES)
            subagent_tools = filter_tools_by_name(tools, SUBAGENT_TOOL_NAMES)
    """
    config = MCPConfig(host=host, port=port)
    scraper = MCPWebScraper(config)

    try:
        tools = create_mcp_tools(scraper, max_urls=max_urls)
        scraper_tools = filter_tools_by_name(tools, ALL_SCRAPER_TOOLS)
        logger.info(f"[MCP] Created {len(scraper_tools)} tools: {[t.name for t in scraper_tools]}")
        yield scraper_tools
    finally:
        await scraper.close()


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
    config = MCPConfig(host=host, port=port)
    scraper = MCPWebScraper(config)
    try:
        urls = await scraper.map_domain(
            domain,
            max_urls=max_urls,
            pattern=pattern,
        )
        logger.info(f"[MCP] map_domain found {len(urls)} URLs for {domain}")
        return urls
    finally:
        await scraper.close()


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
    config = MCPConfig(host=host, port=port)
    scraper = MCPWebScraper(config)
    try:
        pages = await scraper.crawl_site(
            url,
            max_depth=max_depth,
            max_pages=max_pages,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        logger.info(f"[MCP] crawl_site found {len(pages)} pages from {url}")
        return pages
    finally:
        await scraper.close()


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
    config = MCPConfig(host=host, port=port)
    scraper = MCPWebScraper(config)
    try:
        result = await scraper.scrape_url(url)
        if result:
            return {
                "url": result.get("url", url),
                "title": result.get("metadata", {}).get("title", ""),
                "content": result.get("markdown", ""),
                "markdown": result.get("markdown", ""),
                "metadata": result.get("metadata", {}),
            }
        return None
    finally:
        await scraper.close()


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
    config = MCPConfig(host=host, port=port)
    scraper = MCPWebScraper(config)
    try:
        pages = await scraper.scrape_batch(urls)
        return [
            {
                "url": p.get("url", ""),
                "title": p.get("metadata", {}).get("title", ""),
                "content": p.get("markdown", ""),
                "markdown": p.get("markdown", ""),
                "metadata": p.get("metadata", {}),
            }
            for p in pages
        ]
    finally:
        await scraper.close()


async def mcp_map_website(
    host: str,
    port: int,
    url: str,
    limit: Optional[int] = None,
) -> List[str]:
    """Map website to get URLs (compatibility function).

    Uses the existing MCPWebScraper.map_website method.

    Args:
        host: MCP server host
        port: MCP server port
        url: Website URL
        limit: Maximum URLs to return

    Returns:
        List of discovered URLs
    """
    config = MCPConfig(host=host, port=port)
    scraper = MCPWebScraper(config)
    try:
        urls = await scraper.map_website(url, limit=limit)
        logger.info(f"[MCP] map_website found {len(urls)} URLs for {url}")
        return urls
    finally:
        await scraper.close()


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
    # Tool creation
    "create_mcp_tools",
    # Context managers
    "get_mcp_tools",
    # Direct functions
    "mcp_map_domain",
    "mcp_crawl_site",
    "mcp_scrape_url",
    "mcp_scrape_batch",
    "mcp_map_website",
]
