"""Firecrawl API client for website mapping and scraping.

Provides:
- Website mapping (discover all URLs)
- URL scraping (extract markdown content)
"""

from typing import Optional, List, Dict, Any

from loguru import logger
import httpx

from .config import FirecrawlConfig


class FirecrawlClient:
    """Firecrawl API client."""

    def __init__(self, config: FirecrawlConfig):
        """Initialize Firecrawl client.

        Args:
            config: Firecrawl configuration
        """
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    async def map_website(
        self,
        url: str,
        limit: int = 100,
        include_subdomains: bool = False,
    ) -> List[str]:
        """Map a website to get all URLs.

        Args:
            url: Website URL to map
            limit: Maximum URLs to return
            include_subdomains: Include subdomains

        Returns:
            List of URLs
        """
        logger.info(f"Mapping website: {url} (limit: {limit})")

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.config.base_url}/map",
                    headers=self.headers,
                    json={
                        "url": url,
                        "limit": limit,
                        "includeSubdomains": include_subdomains,
                        "ignoreSitemap": False,
                    },
                )
                response.raise_for_status()
                data = response.json()

                if data.get("success") and data.get("links"):
                    urls = data["links"]
                    logger.info(f"Found {len(urls)} URLs")
                    return urls
                else:
                    logger.error(f"Failed to map website: {data}")
                    return []

            except Exception as e:
                logger.error(f"Error mapping website: {e}")
                return []

    async def scrape_url(
        self,
        url: str,
        formats: List[str] = None,
        only_main_content: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Scrape a single URL.

        Args:
            url: URL to scrape
            formats: Output formats (default: ["markdown"])
            only_main_content: Extract only main content

        Returns:
            Dict with 'url', 'markdown', 'metadata' or None on failure
        """
        if formats is None:
            formats = ["markdown"]

        logger.debug(f"Scraping URL: {url}")

        async with httpx.AsyncClient(timeout=self.config.timeout / 1000 + 10) as client:
            try:
                response = await client.post(
                    f"{self.config.base_url}/scrape",
                    headers=self.headers,
                    json={
                        "url": url,
                        "formats": formats,
                        "onlyMainContent": only_main_content,
                        "timeout": self.config.timeout,
                    },
                )
                response.raise_for_status()
                data = response.json()

                if data.get("success") and data.get("data"):
                    return {
                        "url": url,
                        "markdown": data["data"].get("markdown", ""),
                        "metadata": data["data"].get("metadata", {}),
                    }
                else:
                    logger.error(f"Failed to scrape {url}: {data}")
                    return None

            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                return None

    async def scrape_batch(
        self,
        urls: List[str],
        only_main_content: bool = True,
    ) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently.

        Args:
            urls: URLs to scrape
            only_main_content: Extract only main content

        Returns:
            List of scraped page data (excludes failures)
        """
        import asyncio

        async def scrape_one(url: str) -> Optional[Dict[str, Any]]:
            return await self.scrape_url(url, only_main_content=only_main_content)

        # Scrape concurrently
        tasks = [scrape_one(url) for url in urls]
        results = await asyncio.gather(*tasks)

        # Filter out failures
        return [r for r in results if r is not None]


class MockFirecrawlClient:
    """Mock Firecrawl client for testing without API.

    Use FIRECRAWL_MOCK=true to enable.
    """

    def __init__(self, config: FirecrawlConfig):
        self.config = config

    async def map_website(self, url: str, limit: int = 100, **kwargs) -> List[str]:
        """Return mock URLs."""
        logger.info(f"[MOCK] Mapping website: {url}")
        # Return some mock URLs based on the input URL
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return [
            f"https://{domain}/",
            f"https://{domain}/about",
            f"https://{domain}/contact",
            f"https://{domain}/blog",
            f"https://{domain}/docs",
        ][:limit]

    async def scrape_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """Return mock page content."""
        logger.debug(f"[MOCK] Scraping URL: {url}")
        return {
            "url": url,
            "markdown": f"# Mock Content for {url}\n\nThis is mock content for testing.",
            "metadata": {"title": f"Mock Page: {url}"},
        }

    async def scrape_batch(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Return mock pages."""
        results = []
        for url in urls:
            result = await self.scrape_url(url)
            results.append(result)
        return results


def get_firecrawl_client(config: FirecrawlConfig) -> FirecrawlClient:
    """Get Firecrawl client (or mock for testing).

    Args:
        config: Firecrawl configuration

    Returns:
        FirecrawlClient or MockFirecrawlClient
    """
    import os

    if os.getenv("FIRECRAWL_MOCK", "").lower() in ("true", "1", "yes"):
        logger.info("Using mock Firecrawl client")
        return MockFirecrawlClient(config)

    return FirecrawlClient(config)
