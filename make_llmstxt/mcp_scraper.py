"""MCP Web Scraper - Connects to MCP research server for web scraping.

Uses the MCP SSE protocol to connect to your MCP research server.
Tools available:
- search_web: Search the web
- scrape_url: Scrape URL content (returns markdown)
- docs_list_sources: List llms.txt sources
- docs_fetch_docs: Fetch docs from llms.txt source

Configuration:
- MCP_HOST: MCP server host (default: 100.85.22.99 - Tailscale)
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
    timeout: float = 60.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class MCPClient:
    """Client for MCP server using SSE transport.

    The MCP SSE protocol:
    1. Open persistent SSE connection to /sse
    2. Receive session ID via "event: endpoint" event
    3. Make POST requests to /messages/?session_id=XXX
    4. Receive tool results via "event: message" on SSE connection
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.session_id: Optional[str] = None
        self.message_endpoint: Optional[str] = None

        # HTTP clients
        self._sse_client: Optional[httpx.AsyncClient] = None
        self._http_client: Optional[httpx.AsyncClient] = None

        # SSE state
        self._sse_cm: Optional[Any] = None
        self._sse_response: Optional[httpx.Response] = None
        self._running = False
        self._sse_task: Optional[asyncio.Task] = None

        # Request tracking
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._request_counter = 1

        # Tools cache
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    async def connect(self) -> bool:
        """Connect to MCP server and establish SSE connection."""
        try:
            sse_url = f"{self.base_url}/sse"
            logger.info(f"[MCP] Connecting to: {sse_url}")

            self._sse_client = httpx.AsyncClient(timeout=None)
            self._sse_cm = self._sse_client.stream("GET", sse_url)
            self._sse_response = await self._sse_cm.__aenter__()
            self._sse_response.raise_for_status()

            # Start background SSE processor (handles both init and events)
            self._running = True
            self._connected = asyncio.Event()
            self._sse_task = asyncio.create_task(self._process_sse_events())

            # Wait for connection to be established (session received)
            try:
                await asyncio.wait_for(self._connected.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("[MCP] Timeout waiting for session")
                self._running = False
                await self._sse_cm.__aexit__(None, None, None)
                return False

            if not self.session_id:
                logger.warning("[MCP] No session ID received")
                self._running = False
                await self._sse_cm.__aexit__(None, None, None)
                return False

            # MCP initialization handshake
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

            logger.info(f"[MCP] Connected: {self.session_id}")
            return True

        except Exception as e:
            logger.error(f"[MCP] Connection failed: {e}")
            return False

    async def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.message_endpoint:
            raise RuntimeError("Not connected. Call connect() first.")

        client = await self._ensure_http_client()

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }

        url = f"{self.base_url}{self.message_endpoint}"
        logger.debug(f"[MCP] POST notification {method}")

        response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()

    async def _process_sse_events(self):
        """Process SSE events - handles both initial connection and subsequent messages."""
        try:
            event_type = None
            async for line in self._sse_response.aiter_lines():
                if not self._running:
                    break

                line = line.strip()
                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()

                    # Handle initial endpoint event for session
                    if event_type == "endpoint":
                        self.message_endpoint = data_str
                        if "session_id=" in data_str:
                            self.session_id = data_str.split("session_id=")[1].split("&")[0]
                        logger.info(f"[MCP] Session: {self.session_id}")
                        self._connected.set()  # Signal that we're connected
                        continue

                    # Handle message events (tool results)
                    if event_type == "message":
                        try:
                            data = json.loads(data_str)
                            request_id = data.get("id")
                            logger.debug(f"[MCP] SSE message id={request_id}: {str(data)[:200]}")

                            if request_id is not None and request_id in self._pending_requests:
                                future = self._pending_requests.pop(request_id)
                                if not future.done():
                                    future.set_result(data)
                        except json.JSONDecodeError:
                            logger.warning(f"[MCP] Failed to parse: {data_str[:100]}")

        except Exception as e:
            if self._running:
                logger.error(f"[MCP] SSE error: {e}")

    async def _ensure_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def _call_jsonrpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Make JSON-RPC call via POST, receive result via SSE."""
        if not self.message_endpoint:
            raise RuntimeError("Not connected. Call connect() first.")

        client = await self._ensure_http_client()

        request_id = self._request_counter
        self._request_counter += 1

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }

        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future

        url = f"{self.base_url}{self.message_endpoint}"
        logger.debug(f"[MCP] POST {method}")

        try:
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise

        try:
            result = await asyncio.wait_for(future, timeout=self.timeout)
            return result.get("result", {})
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise RuntimeError(f"Timeout waiting for MCP response to {method}")

    async def list_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """List available tools."""
        if self._tools_cache and not force_refresh:
            return self._tools_cache

        result = await self._call_jsonrpc("tools/list")
        self._tools_cache = result.get("tools", [])
        return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
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
        """Close connections."""
        self._running = False

        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
            self._sse_task = None

        if self._sse_cm:
            try:
                await self._sse_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._sse_cm = None
            self._sse_response = None

        if self._sse_client:
            await self._sse_client.aclose()
            self._sse_client = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self.session_id = None
        self.message_endpoint = None


class MCPWebScraper:
    """High-level web scraper using MCP server.

    Provides:
    - map_website: Discover URLs via sitemap or search
    - scrape_url: Scrape single URL to markdown
    - scrape_batch: Scrape multiple URLs concurrently
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self._client: Optional[MCPClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> MCPClient:
        """Get or create MCP client (thread-safe)."""
        async with self._client_lock:
            if self._client is None or self._client.session_id is None:
                self._client = MCPClient(self.config)
                connected = await self._client.connect()
                if not connected:
                    raise RuntimeError("Failed to connect to MCP server")
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
        import asyncio

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
