"""MCP Web Scraper - Connects to MCP research server for web scraping.

Uses Streamable HTTP transport (MCP spec 2025-03-26) to connect to MCP server.
Tools available:
- search_web: Search the web
- scrape_url: Scrape URL content (returns markdown)
- map_domain: Discover URLs from a domain
- crawl_site: Deep crawl a site following links
- docs_list_sources: List llms.txt sources
- docs_fetch_docs: Fetch docs from llms.txt source

Configuration via environment:
- MCP_HOST: MCP server host (default: 100.85.22.99)
- MCP_PORT: MCP server port (default: 8000)
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin

from loguru import logger
import httpx
from pydantic import BaseModel

# Configuration
MCP_HOST = os.getenv("MCP_HOST", "100.85.22.99")
MCP_PORT = os.getenv("MCP_PORT", "8000")
MCP_BASE_URL = f"http://{MCP_HOST}:{MCP_PORT}"


class MCPConfig(BaseModel):
    """MCP scraper configuration."""

    host: str = MCP_HOST
    port: int = int(MCP_PORT)
    timeout: float = 120.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class MCPClient:
    """Client for MCP server using Streamable HTTP transport.

    Streamable HTTP protocol (MCP spec 2025-03-26):
    1. POST JSON-RPC requests to /mcp endpoint
    2. Server responds with JSON or SSE stream
    3. No persistent connection needed
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.mcp_endpoint = f"{self.base_url}/mcp"

        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None

        # Session state
        self._session_id: Optional[str] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._request_counter = 1

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            # Add session ID header if we have one
            if self._session_id:
                headers["mcp-session-id"] = self._session_id

            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, read=None),
                headers=headers,
            )
        elif self._session_id:
            # Update headers on existing client
            self._http_client.headers["mcp-session-id"] = self._session_id

        return self._http_client

    async def _ensure_initialized(self) -> None:
        """Ensure MCP session is initialized (thread-safe)."""
        if self._initialized:
            return

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            logger.debug("[MCP] Sending initialize request...")
            init_result = await self._call_jsonrpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "make-llmstxt",
                    "version": "0.1.0"
                }
            })
            logger.debug(f"[MCP] Initialize result: {init_result}")

            # Send initialized notification
            await self._send_notification("notifications/initialized")
            logger.debug("[MCP] Sent initialized notification")

            self._initialized = True

    async def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no response expected).

        Note: Notifications in Streamable HTTP may return 202 or 400 depending
        on server implementation. We don't raise on non-2xx for notifications.
        """
        client = await self._get_client()

        payload = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            payload["params"] = params

        logger.debug(f"[MCP] POST notification {method}")
        try:
            response = await client.post(self.mcp_endpoint, json=payload)
            # Don't raise_for_status - notifications may return various codes
            if response.status_code >= 400:
                logger.debug(f"[MCP] Notification {method} returned {response.status_code}")
        except Exception as e:
            # Notifications are fire-and-forget, don't fail on errors
            logger.debug(f"[MCP] Notification {method} error: {e}")

    async def _call_jsonrpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Make JSON-RPC call via Streamable HTTP."""
        client = await self._get_client()

        request_id = self._request_counter
        self._request_counter += 1

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }

        logger.debug(f"[MCP] POST {method} (id={request_id})")

        response = await client.post(self.mcp_endpoint, json=payload)

        # Debug: log response for non-2xx
        if response.status_code >= 400:
            logger.debug(f"[MCP] Error response ({response.status_code}): {response.text[:500]}")

        response.raise_for_status()

        # Capture session ID from response header (Streamable HTTP)
        session_id = response.headers.get("mcp-session-id")
        if session_id and not self._session_id:
            self._session_id = session_id
            logger.debug(f"[MCP] Session ID: {session_id}")

        # Check content type to determine how to parse response
        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            # Parse SSE response
            return await self._parse_sse_response(response)
        else:
            # Parse JSON response
            data = response.json()
            if "error" in data:
                raise RuntimeError(f"MCP error: {data['error']}")
            return data.get("result", {})

    async def _parse_sse_response(self, response: httpx.Response) -> Any:
        """Parse SSE streaming response."""
        text = response.text
        result = None

        # Parse SSE events
        event_type = None
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    data = json.loads(data_str)
                    if "result" in data:
                        result = data["result"]
                    elif "error" in data:
                        raise RuntimeError(f"MCP error: {data['error']}")
                except json.JSONDecodeError:
                    logger.warning(f"[MCP] Failed to parse SSE data: {data_str[:100]}")

        return result

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        await self._ensure_initialized()
        result = await self._call_jsonrpc("tools/list")
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        await self._ensure_initialized()

        params = {"name": tool_name, "arguments": arguments}
        logger.info(f"[MCP] Calling: {tool_name}({list(arguments.keys())})")

        result = await self._call_jsonrpc("tools/call", params)

        # Extract text content from MCP response
        if isinstance(result, dict):
            content = result.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                first = content[0]
                if isinstance(first, dict) and first.get("type") == "text":
                    return first.get("text", "")
            return result

        return result

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._initialized = False


class MCPWebScraper:
    """High-level web scraper using MCP server.

    Provides:
    - map_website: Discover URLs via sitemap or search
    - scrape_url: Scrape single URL to markdown
    - scrape_batch: Scrape multiple URLs concurrently
    - map_domain: Discover URLs from a domain
    - crawl_site: Deep crawl a site following links
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self._client: Optional[MCPClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> MCPClient:
        """Get or create MCP client (thread-safe)."""
        async with self._client_lock:
            if self._client is None:
                self._client = MCPClient(self.config)
        return self._client

    async def close(self):
        """Close client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single URL.

        Args:
            url: URL to scrape

        Returns:
            Dict with 'url', 'markdown', 'metadata' or None on failure
        """
        try:
            client = await self._get_client()
            result = await client.call_tool("scrape_url", {"url": url})

            # MCP server returns JSON string with scrape result
            if isinstance(result, str):
                try:
                    data = json.loads(result)
                    if data.get("success"):
                        return {
                            "url": data.get("url", url),
                            "markdown": data.get("content", ""),
                            "metadata": {
                                "title": data.get("title"),
                                "word_count": data.get("word_count"),
                                "method": data.get("method_used"),
                            },
                        }
                    else:
                        logger.warning(f"[MCP] Scrape failed for {url}: {data.get('error', 'Unknown error')}")
                        return None
                except json.JSONDecodeError:
                    # Fallback: treat as raw markdown
                    return {
                        "url": url,
                        "markdown": result,
                        "metadata": {},
                    }
            elif isinstance(result, dict):
                return {
                    "url": url,
                    "markdown": result.get("content", result.get("markdown", "")),
                    "metadata": result.get("metadata", {}),
                }
            else:
                logger.error(f"[MCP] Unexpected scrape result type: {type(result)}")
                return None

        except Exception as e:
            logger.error(f"[MCP] Scrape failed for {url}: {e}")
            return None

    async def scrape_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently.

        Args:
            urls: URLs to scrape

        Returns:
            List of scraped pages (excludes failures)
        """
        async def scrape_one(url: str) -> Optional[Dict[str, Any]]:
            return await self.scrape_url(url)

        tasks = [scrape_one(url) for url in urls]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]

    async def map_domain(
        self,
        domain: str,
        max_urls: int = 100,
        pattern: str = "*",
    ) -> List[str]:
        """Discover URLs from a domain using MCP map_domain tool.

        Uses sitemaps or Common Crawl to discover URLs.

        Args:
            domain: Domain to map (e.g., 'example.com' or 'https://example.com')
            max_urls: Maximum URLs to return (default: 100)
            pattern: URL pattern filter (e.g., '*/docs/*')

        Returns:
            List of discovered URLs
        """
        try:
            client = await self._get_client()
            result = await client.call_tool("map_domain", {
                "domain": domain,
                "max_urls": max_urls,
                "pattern": pattern,
            })

            # Parse the result
            if isinstance(result, str):
                try:
                    data = json.loads(result)
                    urls = data.get("urls", [])
                    logger.info(f"[MCP] map_domain found {len(urls)} URLs for {domain}")
                    return urls
                except json.JSONDecodeError:
                    logger.warning(f"[MCP] Failed to parse map_domain result")
                    return []
            elif isinstance(result, dict):
                urls = result.get("urls", [])
                logger.info(f"[MCP] map_domain found {len(urls)} URLs for {domain}")
                return urls
            else:
                logger.warning(f"[MCP] Unexpected map_domain result type: {type(result)}")
                return []

        except Exception as e:
            logger.error(f"[MCP] map_domain failed for {domain}: {e}")
            return []

    async def crawl_site(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 50,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Deep crawl a site following links using MCP crawl_site tool.

        Args:
            url: Starting URL to crawl
            max_depth: Maximum depth to crawl (1-5, default: 2)
            max_pages: Maximum pages to crawl (1-200, default: 50)
            include_patterns: URL patterns to include (e.g., ['*api*', '*docs*'])
            exclude_patterns: URL patterns to exclude (e.g., ['*deprecated*'])

        Returns:
            List of crawled pages with content
        """
        try:
            client = await self._get_client()
            args = {
                "url": url,
                "max_depth": max_depth,
                "max_pages": max_pages,
            }
            if include_patterns:
                args["include_patterns"] = include_patterns
            if exclude_patterns:
                args["exclude_patterns"] = exclude_patterns

            result = await client.call_tool("crawl_site", args)

            # Parse the result
            if isinstance(result, str):
                try:
                    data = json.loads(result)
                    pages = data.get("pages", [])
                    logger.info(f"[MCP] crawl_site found {len(pages)} pages from {url}")
                    return pages
                except json.JSONDecodeError:
                    logger.warning(f"[MCP] Failed to parse crawl_site result")
                    return []
            elif isinstance(result, dict):
                pages = result.get("pages", [])
                logger.info(f"[MCP] crawl_site found {len(pages)} pages from {url}")
                return pages
            else:
                logger.warning(f"[MCP] Unexpected crawl_site result type: {type(result)}")
                return []

        except Exception as e:
            logger.error(f"[MCP] crawl_site failed for {url}: {e}")
            return []

    async def map_website(
        self,
        url: str,
        limit: Optional[int] = None,
        use_sitemap: bool = True,
    ) -> List[str]:
        """Discover URLs for a website.

        Strategy:
        1. Try to fetch sitemap.xml
        2. Fall back to search if sitemap not available

        Args:
            url: Website URL
            limit: Maximum URLs to return (None = unlimited)
            use_sitemap: Try sitemap.xml first

        Returns:
            List of discovered URLs
        """
        discovered = set()

        # Parse base URL
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Strategy 1: Try sitemap
        if use_sitemap:
            sitemap_urls = await self._fetch_sitemap(base_url)
            discovered.update(sitemap_urls)
            logger.info(f"[MCP] Found {len(sitemap_urls)} URLs in sitemap")

        # Strategy 2: If sitemap didn't give enough, try common paths
        if limit is None or len(discovered) < limit:
            # Scrape homepage to find links
            homepage = await self.scrape_url(base_url)
            if homepage:
                markdown = homepage.get("markdown", "")
                if markdown:
                    links = self._extract_links(markdown, base_url)
                    discovered.update(links)
                    logger.info(f"[MCP] Found {len(links)} links on homepage")
                else:
                    logger.debug(f"[MCP] No markdown content in homepage response")

        # Filter to same domain and limit
        same_domain = [
            u for u in discovered
            if urlparse(u).netloc == parsed.netloc
        ]

        # Prioritize important pages
        same_domain = self._prioritize_urls(same_domain, base_url)

        # Apply limit if specified
        if limit is not None:
            return same_domain[:limit]
        return same_domain

    async def _fetch_sitemap(self, base_url: str) -> List[str]:
        """Fetch and parse sitemap.xml.

        Uses direct HTTP fetch to get raw XML, since MCP scraper
        may transform XML to markdown.
        """
        sitemap_url = f"{base_url}/sitemap.xml"
        urls = []

        try:
            # Fetch sitemap directly with httpx to get raw XML
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                response = await http_client.get(sitemap_url)
                response.raise_for_status()
                xml_content = response.text

            if xml_content and "<urlset" in xml_content:
                # Parse XML sitemap
                import xml.etree.ElementTree as ET

                # Remove namespace issues
                xml_content = re.sub(r'\sxmlns="[^"]+"', '', xml_content)

                root = ET.fromstring(xml_content)
                for url_elem in root.findall(".//url/loc"):
                    if url_elem.text:
                        urls.append(url_elem.text.strip())

                logger.debug(f"[MCP] Parsed {len(urls)} URLs from sitemap")

        except Exception as e:
            logger.debug(f"[MCP] Sitemap fetch failed: {e}")

        return urls

    def _extract_links(self, markdown: str, base_url: str) -> List[str]:
        """Extract links from markdown content."""
        links = set()

        # Match markdown links: [text](url)
        md_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(md_pattern, markdown):
            url = match.group(2)
            if url.startswith('http'):
                links.add(url)
            elif url.startswith('/'):
                links.add(urljoin(base_url, url))

        # Match HTML links: <a href="url">
        html_pattern = r'href=["\']([^"\']+)["\']'
        for match in re.finditer(html_pattern, markdown, re.IGNORECASE):
            url = match.group(1)
            if url.startswith('http'):
                links.add(url)
            elif url.startswith('/'):
                links.add(urljoin(base_url, url))

        return list(links)

    def _prioritize_urls(self, urls: List[str], base_url: str) -> List[str]:
        """Prioritize important URLs (homepage, about, docs, etc.)."""
        parsed = urlparse(base_url)
        domain = parsed.netloc

        # Priority patterns (higher = more important)
        priority_patterns = [
            (r'^https?://' + re.escape(domain) + '/?$', 100),  # Homepage
            (r'/about', 80),
            (r'/docs', 80),
            (r'/documentation', 80),
            (r'/guide', 80),
            (r'/tutorial', 80),
            (r'/api', 70),
            (r'/reference', 70),
            (r'/getting-started', 70),
            (r'/quickstart', 70),
            (r'/README', 70),
            (r'/index', 60),
            (r'/home', 60),
            (r'/main', 60),
        ]

        def get_priority(url: str) -> int:
            for pattern, priority in priority_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return priority
            return 50  # Default priority

        # Sort by priority (descending)
        return sorted(urls, key=get_priority, reverse=True)


# Convenience functions for drop-in replacement with firecrawl module

async def map_website(url: str, limit: Optional[int] = None, **kwargs) -> List[str]:
    """Map website to get URLs."""
    config = MCPConfig()
    scraper = MCPWebScraper(config)
    try:
        return await scraper.map_website(url, limit=limit, **kwargs)
    finally:
        await scraper.close()


async def scrape_url(url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Scrape a single URL."""
    config = MCPConfig()
    scraper = MCPWebScraper(config)
    try:
        return await scraper.scrape_url(url)
    finally:
        await scraper.close()


async def scrape_batch(urls: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Scrape multiple URLs."""
    config = MCPConfig()
    scraper = MCPWebScraper(config)
    try:
        return await scraper.scrape_batch(urls)
    finally:
        await scraper.close()


async def map_domain(
    domain: str,
    max_urls: int = 100,
    pattern: str = "*",
    **kwargs,
) -> List[str]:
    """Discover URLs from a domain using MCP map_domain tool."""
    config = MCPConfig()
    scraper = MCPWebScraper(config)
    try:
        return await scraper.map_domain(domain, max_urls=max_urls, pattern=pattern)
    finally:
        await scraper.close()


async def crawl_site(
    url: str,
    max_depth: int = 2,
    max_pages: int = 50,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Deep crawl a site following links using MCP crawl_site tool."""
    config = MCPConfig()
    scraper = MCPWebScraper(config)
    try:
        return await scraper.crawl_site(
            url,
            max_depth=max_depth,
            max_pages=max_pages,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    finally:
        await scraper.close()


__all__ = [
    "MCPConfig",
    "MCPClient",
    "MCPWebScraper",
    "map_website",
    "scrape_url",
    "scrape_batch",
    "map_domain",
    "crawl_site",
]
