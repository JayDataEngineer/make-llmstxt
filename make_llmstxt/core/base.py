"""Base generator class - shared functionality for all generators."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Callable

from loguru import logger

from . import GeneratorConfig, ScrapedPage, GeneratorResult
from ..mcp_tools import mcp_map_website, mcp_scrape_batch


class BaseGenerator(ABC):
    """Abstract base class for all generators.

    Provides:
    - MCP tools for content gathering
    - Configuration management
    - Progress callbacks
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config

    async def gather_content(self, url: str) -> List[ScrapedPage]:
        """Gather content from the target URL.

        Discovers URLs via sitemap, then scrapes each page.

        Args:
            url: Target URL to process

        Returns:
            List of scraped pages with markdown content
        """
        # Discover URLs
        logger.info(f"[BaseGenerator] Mapping website: {url}")
        urls = await mcp_map_website(
            host=self.config.mcp_host,
            port=self.config.mcp_port,
            url=url,
            limit=self.config.max_urls,
        )

        logger.info(f"[BaseGenerator] Found {len(urls)} URLs")

        # Scrape pages
        logger.info(f"[BaseGenerator] Scraping {len(urls)} pages...")
        pages = await mcp_scrape_batch(
            host=self.config.mcp_host,
            port=self.config.mcp_port,
            urls=urls,
        )

        # Convert to ScrapedPage objects
        scraped_pages = []
        for page in pages:
            scraped_pages.append(ScrapedPage(
                url=page.get("url", ""),
                title=page.get("metadata", {}).get("title"),
                markdown=page.get("markdown", ""),
                metadata=page.get("metadata", {}),
            ))

        logger.info(f"[BaseGenerator] Scraped {len(scraped_pages)} pages successfully")
        return scraped_pages

    @abstractmethod
    async def generate(
        self,
        url: str,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
    ) -> GeneratorResult:
        """Generate output for the given URL.

        Args:
            url: Target URL to process
            output_dir: Output directory (defaults to config.output_dir)
            progress_callback: Optional callback for progress updates

        Returns:
            GeneratorResult with output path and stats
        """
        pass

    async def close(self):
        """Clean up resources."""
        pass  # MCP connections managed per-call in mcp_tools
