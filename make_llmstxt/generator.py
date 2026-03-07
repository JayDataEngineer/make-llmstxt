"""LLMs.txt Generator - Main generation logic.

Generates:
- llms.txt: Index of pages with titles and descriptions
- llms-full.txt: Full content of all pages

Scraping via custom MCP server.
Includes Critic + Retry pattern for quality assurance.
"""

import re
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from langchain_openai import ChatOpenAI

from .config import AppConfig
from .mcp_scraper import MCPWebScraper
from .llm import create_llm, generate_summaries_batch
from .critic import Critic, CriticResult


@dataclass
class PageResult:
    """Result of processing a single page."""

    url: str
    title: str
    description: str
    markdown: str
    index: int
    error: Optional[str] = None


@dataclass
class GenerationResult:
    """Result of generating llms.txt files."""

    llmstxt: str
    llms_fulltxt: str
    num_urls_processed: int
    num_urls_total: int
    pages: List[PageResult] = field(default_factory=list)
    # Project metadata
    project_name: str = ""
    project_summary: str = ""
    # Critic metadata
    critic_passed: bool = True
    critic_score: float = 1.0
    critic_issues: List[str] = field(default_factory=list)
    retry_count: int = 0


class LLMsTxtGenerator:
    """Generate llms.txt and llms-full.txt for websites.

    Scraping via custom MCP server.
    """

    def __init__(
        self,
        config: AppConfig,
        llm: Optional[ChatOpenAI] = None,
        scraper: Optional[MCPWebScraper] = None,
        pass_threshold: float = 0.7,
        fail_on_critic_error: bool = False,
    ):
        """Initialize generator.

        Args:
            config: Application configuration
            llm: Optional pre-configured LLM instance
            scraper: Optional pre-configured MCP scraper
            pass_threshold: Minimum critic score to pass (0.0-1.0)
            fail_on_critic_error: Fail generation if critic errors
        """
        self.config = config
        self._external_scraper = scraper is not None
        self.pass_threshold = pass_threshold
        self.fail_on_critic_error = fail_on_critic_error

        # Create LLM if not provided
        self.llm = llm or create_llm(config.llm)

        # Create critic
        self.critic = Critic(
            self.llm,
            pass_threshold=pass_threshold,
            fail_on_error=fail_on_critic_error,
        )

        # Use provided scraper or create from config
        if scraper is not None:
            self.scraper = scraper
        else:
            self.scraper = MCPWebScraper(config.mcp)
            logger.info(f"Using MCP scraper at {config.mcp.base_url}")

    async def map_website(self, url: str) -> List[str]:
        """Map website to get all URLs."""
        if hasattr(self.scraper, 'map_website'):
            return await self.scraper.map_website(url, limit=self.config.max_urls)
        else:
            logger.warning("Scraper does not support mapping, returning single URL")
            return [url]

    async def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single URL."""
        return await self.scraper.scrape_url(url)

    async def process_batch(
        self,
        urls: List[str],
        start_index: int = 0,
        feedback: Optional[List[str]] = None,
    ) -> List[PageResult]:
        """Process a batch of URLs.

        Args:
            urls: URLs to process
            start_index: Starting index for ordering
            feedback: Optional critic feedback to improve summaries

        Returns:
            List of PageResult objects
        """
        # Scrape all URLs
        if hasattr(self.scraper, 'scrape_batch'):
            scraped = await self.scraper.scrape_batch(urls)
        else:
            # Fallback: scrape one by one
            scraped = []
            for url in urls:
                result = await self.scrape_url(url)
                if result:
                    scraped.append(result)

        if not scraped:
            return []

        # Generate summaries using LLM
        pages_for_llm = [
            {"url": page["url"], "markdown": page["markdown"]}
            for page in scraped
        ]
        summaries = await generate_summaries_batch(self.llm, pages_for_llm, feedback=feedback)

        # Combine results
        results = []
        for i, (page, summary) in enumerate(zip(scraped, summaries)):
            results.append(
                PageResult(
                    url=page["url"],
                    title=summary["title"],
                    description=summary["description"],
                    markdown=page["markdown"],
                    index=start_index + i,
                )
            )

        return results

    async def generate(
        self,
        url: str,
        max_urls: Optional[int] = None,
        include_full_text: bool = True,
        progress_callback: Optional[callable] = None,
        enable_critic: bool = True,
        max_retries: int = 2,
    ) -> GenerationResult:
        """Generate llms.txt files for a website.

        Args:
            url: Website URL
            max_urls: Override max URLs from config
            include_full_text: Whether to include full text
            progress_callback: Optional callback for progress updates
            enable_critic: Whether to use critic validation
            max_retries: Maximum retry attempts on critic failure

        Returns:
            GenerationResult with generated content
        """
        max_urls = max_urls or self.config.max_urls
        previous_issues: List[str] = []
        result: Optional[GenerationResult] = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(f"Retry attempt {attempt}/{max_retries} with {len(previous_issues)} issues to fix")

            # Generate draft
            result = await self._generate_draft(
                url=url,
                max_urls=max_urls,
                include_full_text=include_full_text,
                progress_callback=progress_callback,
                feedback=previous_issues if attempt > 0 else None,
            )

            # Skip critic if disabled or first attempt
            if not enable_critic:
                return result

            # Run critic
            if progress_callback:
                progress_callback("critiquing", 0, 1, "Validating output quality...")

            critic_result = await self.critic.evaluate(
                llmstxt=result.llmstxt,
                llms_fulltxt=result.llms_fulltxt if include_full_text else None,
                url=url,
            )

            # Update result with critic metadata
            result.critic_passed = critic_result.passed
            result.critic_score = critic_result.score
            result.critic_issues = critic_result.issues
            result.retry_count = attempt

            if critic_result.passed:
                logger.info(f"Critic passed with score {critic_result.score:.2f}")
                return result

            # Prepare feedback for next iteration
            logger.warning(f"Critic failed (score {critic_result.score:.2f}): {critic_result.issues}")
            previous_issues = critic_result.issues + critic_result.suggestions

        # Max retries reached
        logger.error(f"Max retries ({max_retries}) reached, returning best effort output")
        return result

    async def _generate_draft(
        self,
        url: str,
        max_urls: int,
        include_full_text: bool,
        progress_callback: Optional[callable],
        feedback: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Internal method to generate a draft.

        Args:
            url: Website URL
            max_urls: Maximum URLs to process
            include_full_text: Whether to include full text
            progress_callback: Progress callback
            feedback: Optional critic feedback for retry

        Returns:
            GenerationResult
        """
        # Step 1: Map website
        if progress_callback:
            progress_callback("mapping", 0, 1, f"Mapping {url}...")

        all_urls = await self.map_website(url)

        if not all_urls:
            raise ValueError(f"No URLs found for website: {url}")

        # Limit URLs
        all_urls = all_urls[:max_urls]

        # Step 2: Process in batches
        all_results: List[PageResult] = []
        batch_size = self.config.batch_size
        total_batches = (len(all_urls) + batch_size - 1) // batch_size

        for i in range(0, len(all_urls), batch_size):
            batch_num = i // batch_size + 1
            batch = all_urls[i : i + batch_size]

            if progress_callback:
                progress_callback(
                    "scraping",
                    batch_num,
                    total_batches,
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} URLs)",
                )

            results = await self.process_batch(batch, start_index=i, feedback=feedback)
            all_results.extend(results)

            # Small delay between batches
            if i + batch_size < len(all_urls):
                await asyncio.sleep(1)

        # Sort by index
        all_results.sort(key=lambda x: x.index)

        # Step 3: Generate output
        if progress_callback:
            progress_callback("generating", 0, 1, "Generating output files...")

        llmstxt = self._format_llmstxt(url, all_results)
        llms_fulltxt = (
            self._format_llms_fulltxt(url, all_results) if include_full_text else ""
        )

        return GenerationResult(
            llmstxt=llmstxt,
            llms_fulltxt=llms_fulltxt,
            num_urls_processed=len(all_results),
            num_urls_total=len(all_urls),
            pages=all_results,
        )

    def _format_llmstxt(self, url: str, pages: List[PageResult]) -> str:
        """Format llms.txt content following the official spec.

        Format:
        # Project Name
        > Summary of the project

        ## Section
        - [Title](URL): Description

        ## Optional
        - [Blog](URL): Description
        """
        from urllib.parse import urlparse

        # Extract project name from URL
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        # Convert domain to project name (e.g., "docs.python.org" -> "Python")
        parts = domain.split(".")
        if len(parts) >= 2:
            project_name = parts[-2].capitalize() if parts[-2] not in ("www", "docs", "api") else parts[0].capitalize()
        else:
            project_name = parts[0].capitalize()

        # Generate summary - use first page description or fallback
        summary = f"Documentation and resources for {project_name}."
        if pages and pages[0].description:
            # Use the generated description as summary
            summary = pages[0].description
        elif pages and pages[0].markdown:
            # Try to extract first meaningful sentence (skip if it looks like form data)
            first_para = pages[0].markdown.split("\n\n")[0][:200].strip()
            # Skip if it looks like form fields or raw data
            if first_para and not any(kw in first_para.lower() for kw in ["customer name", "telephone", "email address", "submit", ": \n"]):
                summary = first_para

        # Group pages into sections
        core_pages = []
        api_pages = []
        guide_pages = []
        optional_pages = []

        optional_keywords = ("blog", "news", "changelog", "release", "announcement",
                            "twitter", "facebook", "linkedin", "social", "community",
                            "sponsor", "donate", "careers", "jobs", "press")

        for page in pages:
            url_lower = page.url.lower()
            title_lower = page.title.lower()

            # Check if optional
            is_optional = any(kw in url_lower or kw in title_lower for kw in optional_keywords)

            if is_optional:
                optional_pages.append(page)
            elif any(kw in url_lower for kw in ("api", "reference", "function", "class")):
                api_pages.append(page)
            elif any(kw in url_lower for kw in ("guide", "tutorial", "learn", "getting", "start", "intro")):
                guide_pages.append(page)
            else:
                core_pages.append(page)

        # Build output
        lines = [
            f"# {project_name}",
            f"> {summary}",
            "",
        ]

        # Core section
        if core_pages:
            lines.append("## Core")
            for page in core_pages:
                lines.append(f"- [{page.title}]({page.url}): {page.description}")
            lines.append("")

        # Guide section
        if guide_pages:
            lines.append("## Guides")
            for page in guide_pages:
                lines.append(f"- [{page.title}]({page.url}): {page.description}")
            lines.append("")

        # API section
        if api_pages:
            lines.append("## API Reference")
            for page in api_pages:
                lines.append(f"- [{page.title}]({page.url}): {page.description}")
            lines.append("")

        # Optional section
        if optional_pages:
            lines.append("## Optional")
            for page in optional_pages:
                lines.append(f"- [{page.title}]({page.url}): {page.description}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _format_llms_fulltxt(self, url: str, pages: List[PageResult]) -> str:
        """Format llms-full.txt content."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        parts = domain.split(".")
        project_name = parts[-2].capitalize() if len(parts) >= 2 and parts[-2] not in ("www", "docs", "api") else parts[0].capitalize()

        lines = [f"# {project_name} llms-full.txt", ""]

        for i, page in enumerate(pages, 1):
            lines.append(f"<|page-{i}|>")
            lines.append(f"## {page.title}")
            lines.append(f"URL: {page.url}")
            lines.append("")
            lines.append(page.markdown)
            lines.append("")

        return "\n".join(lines)

    async def close(self) -> None:
        """Close the scraper connection."""
        if hasattr(self.scraper, 'close'):
            await self.scraper.close()


async def generate_llmstxt(
    url: str,
    config: Optional[AppConfig] = None,
    max_urls: Optional[int] = None,
    include_full_text: bool = True,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[callable] = None,
    enable_critic: bool = True,
    max_retries: int = 2,
    pass_threshold: float = 0.7,
    fail_on_critic_error: bool = False,
) -> GenerationResult:
    """Generate llms.txt files for a website.

    Args:
        url: Website URL
        config: Optional app configuration
        max_urls: Override max URLs
        include_full_text: Include full text file
        output_dir: Directory to save files
        progress_callback: Progress callback
        enable_critic: Enable critic validation
        max_retries: Max retry attempts
        pass_threshold: Minimum score to pass (0.0-1.0)
        fail_on_critic_error: Fail if critic errors

    Returns:
        GenerationResult
    """
    from dotenv import load_dotenv

    load_dotenv()

    config = config or AppConfig.from_env()
    generator = LLMsTxtGenerator(
        config,
        pass_threshold=pass_threshold,
        fail_on_critic_error=fail_on_critic_error,
    )

    try:
        result = await generator.generate(
            url,
            max_urls=max_urls,
            include_full_text=include_full_text,
            progress_callback=progress_callback,
            enable_critic=enable_critic,
            max_retries=max_retries,
        )

        # Save files if output_dir provided
        if output_dir:
            from urllib.parse import urlparse

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            domain = urlparse(url).netloc.replace("www.", "")

            # Save llms.txt
            llmstxt_path = output_dir / f"{domain}-llms.txt"
            llmstxt_path.write_text(result.llmstxt, encoding="utf-8")
            logger.info(f"Saved llms.txt to {llmstxt_path}")

            # Save llms-full.txt if included
            if include_full_text and result.llms_fulltxt:
                llms_fulltxt_path = output_dir / f"{domain}-llms-full.txt"
                llms_fulltxt_path.write_text(result.llms_fulltxt, encoding="utf-8")
                logger.info(f"Saved llms-full.txt to {llms_fulltxt_path}")

        return result

    finally:
        await generator.close()
