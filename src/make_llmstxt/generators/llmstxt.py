"""LLMs.txt Generator - Main generation logic.

Generates llms.txt: Index of pages with titles and descriptions.

Architecture:
1. Scrape pages via MCP
2. Generate summaries per page (LLM with structured output)
3. Format llms.txt directly (deterministic Python)
4. Validate with critic (LLM with structured output)

No LLM "draft" step - summaries are already generated per-page.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger
from langchain_openai import ChatOpenAI

from ..config import AppConfig
from ..scrapers import mcp_map_website, mcp_scrape_batch
from ..utils.llm import create_llm, generate_summaries_batch
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
    """Result of generating llms.txt."""

    llmstxt: str
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
    """Generate llms.txt for websites.

    Architecture:
    1. Scrape pages via MCP
    2. Generate summaries per page (LLM with structured output)
    3. Format llms.txt directly (deterministic Python)
    4. Validate with critic (LLM with structured output)
    """

    def __init__(
        self,
        config: AppConfig,
        llm: Optional[ChatOpenAI] = None,
        pass_threshold: float = 0.7,
        max_rounds: int = 10
    ):
        """Initialize generator.

        Args:
            config: Application configuration
            llm: Optional pre-configured LLM instance
            pass_threshold: Minimum critic score to pass (0.0-1.0)
            max_rounds: Maximum revision rounds (for critic feedback)
        """
        self.config = config
        self.pass_threshold = pass_threshold
        self.max_rounds = max_rounds

        # Create LLM if not provided
        self.llm = llm or create_llm(config.llm)

        # Create critic
        self.critic = Critic(
            llm=self.llm,
            pass_threshold=pass_threshold,
        )

        logger.info(f"Using MCP server at {config.mcp.base_url}")

    async def map_website(self, url: str) -> List[str]:
        """Map website to get all URLs."""
        return await mcp_map_website(
            host=self.config.mcp.host,
            port=self.config.mcp.port,
            url=url,
            limit=self.config.max_urls,
        )

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
        # Scrape all URLs using shared MCP function
        scraped = await mcp_scrape_batch(
            host=self.config.mcp.host,
            port=self.config.mcp.port,
            urls=urls,
        )

        if not scraped:
            return []

        # Generate summaries using LLM (with structured output)
        pages_for_llm = [
            {"url": page["url"], "markdown": page["markdown"]}
            for page in scraped
        ]
        summaries = await generate_summaries_batch(
            self.llm, pages_for_llm, feedback=feedback
        )

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
        progress_callback: Optional[Callable] = None,
        enable_critic: bool = True,
        max_retries: int = 2,
    ) -> GenerationResult:
        """Generate llms.txt for a website.

        Architecture:
        1. Scrape website content
        2. Generate summaries per page (LLM with structured output)
        3. Format llms.txt directly (deterministic Python)
        4. Validate with critic (LLM with structured output)
        5. If critic fails, regenerate summaries with feedback

        Args:
            url: Website URL
            max_urls: Override max URLs from config
            progress_callback: Optional callback(stage, current, total, message)
            enable_critic: Whether to use critic validation
            max_retries: Maximum revision rounds

        Returns:
            GenerationResult with generated content and critic metadata
        """
        max_urls = max_urls or self.config.max_urls

        # Step 1: Scrape website content
        if progress_callback:
            progress_callback("scraping", 0, 1, f"Scraping {url}...")

        pages, source_content = await self._scrape_website(
            url=url,
            max_urls=max_urls,
            progress_callback=progress_callback,
        )

        if not pages:
            raise ValueError(f"No content scraped from: {url}")

        # Step 2: Format llms.txt directly (no LLM needed)
        llmstxt = self._format_llmstxt(url, pages)

        # Step 3: Validate with critic (if enabled)
        if enable_critic:
            result = await self._validate_with_critic(
                url=url,
                llmstxt=llmstxt,
                pages=pages,
                source_content=source_content,
                max_retries=max_retries,
                progress_callback=progress_callback,
            )
        else:
            result = GenerationResult(
                llmstxt=llmstxt,
                num_urls_processed=len(pages),
                num_urls_total=max_urls if max_urls is not None else len(pages),
                pages=pages,
                critic_passed=True,
                critic_score=1.0,
                critic_issues=[],
                retry_count=0,
            )

        return result

    async def _scrape_website(
        self,
        url: str,
        max_urls: Optional[int],
        progress_callback: Optional[Callable] = None,
        feedback: Optional[List[str]] = None,
    ) -> tuple[List[PageResult], str]:
        """Scrape website and return page results and source content.

        Args:
            url: Website URL
            max_urls: Maximum URLs to scrape (None = unlimited)
            progress_callback: Progress callback
            feedback: Optional critic feedback to improve summaries

        Returns:
            Tuple of (list_of_page_results, source_content_string)
        """
        # Map website
        if progress_callback:
            progress_callback("mapping", 0, 1, f"Mapping {url}...")

        all_urls = await self.map_website(url)

        if not all_urls:
            raise ValueError(f"No URLs found for website: {url}")

        # Limit URLs (None = unlimited)
        if max_urls is not None:
            all_urls = all_urls[:max_urls]

        # Process in batches
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

        # Build source content string for critic
        content_parts = []
        for page in all_results:
            content_parts.append(f"URL: {page.url}")
            content_parts.append(f"Title: {page.title}")
            content_parts.append(f"Description: {page.description}")
            content_parts.append("---")

        source_content = "\n".join(content_parts)

        return all_results, source_content

    async def _validate_with_critic(
        self,
        url: str,
        llmstxt: str,
        pages: List[PageResult],
        source_content: str,
        max_retries: int,
        progress_callback: Optional[Callable] = None,
    ) -> GenerationResult:
        """Validate llms.txt with critic and optionally revise.

        Args:
            url: Website URL
            llmstxt: Generated llms.txt content
            pages: List of page results
            source_content: Source content for coverage validation
            max_retries: Maximum revision rounds
            progress_callback: Progress callback

        Returns:
            GenerationResult
        """
        current_llmstxt = llmstxt
        current_pages = pages

        for round_num in range(max_retries + 1):
            if progress_callback:
                progress_callback(
                    "critiquing",
                    round_num + 1,
                    max_retries + 1,
                    f"Validating round {round_num + 1}/{max_retries + 1}",
                )

            # Evaluate with critic
            critique = await self.critic.evaluate(
                llmstxt=current_llmstxt,
                url=url,
                source_content=source_content,
            )

            logger.info(
                f"[Critic] Round {round_num + 1}: passed={critique.passed}, "
                f"score={critique.score:.2f}, issues={len(critique.issues)}"
            )

            if critique.passed:
                return GenerationResult(
                    llmstxt=current_llmstxt,
                    num_urls_processed=len(current_pages),
                    num_urls_total=len(current_pages),
                    pages=current_pages,
                    critic_passed=True,
                    critic_score=critique.score,
                    critic_issues=critique.issues,
                    retry_count=round_num,
                )

            # If not passed and have retries left, regenerate summaries with feedback
            if round_num < max_retries and critique.issues:
                logger.info(f"[Critic] Regenerating summaries with feedback...")
                feedback = critique.issues + critique.suggestions
                logger.debug(f"[Critic] Feedback: {feedback}")

                # Re-scrape with feedback (only pages that had issues)
                current_pages, _ = await self._scrape_website(
                    url=url,
                    max_urls=len(pages),
                    progress_callback=None,
                    feedback=feedback,
                )

                current_llmstxt = self._format_llmstxt(url, current_pages)

        # Max rounds reached without approval
        logger.warning(f"[Critic] Max rounds ({max_retries + 1}) reached without approval")

        return GenerationResult(
            llmstxt=current_llmstxt,
            num_urls_processed=len(current_pages),
            num_urls_total=len(current_pages),
            pages=current_pages,
            critic_passed=False,
            critic_score=critique.score,
            critic_issues=critique.issues,
            retry_count=max_retries + 1,
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
        # Extract project name from URL
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        parts = domain.split(".")
        if len(parts) >= 2:
            project_name = (
                parts[-2].capitalize()
                if parts[-2] not in ("www", "docs", "api")
                else parts[0].capitalize()
            )
        else:
            project_name = parts[0].capitalize()

        # Generate summary - use first page description or fallback
        summary = f"Documentation and resources for {project_name}."
        if pages and pages[0].description:
            summary = pages[0].description
        elif pages and pages[0].markdown:
            first_para = pages[0].markdown.split("\n\n")[0][:200].strip()
            if first_para and not any(
                kw in first_para.lower()
                for kw in ["customer name", "telephone", "email address", "submit", ": \n"]
            ):
                summary = first_para

        # Group pages into sections
        core_pages = []
        api_pages = []
        guide_pages = []
        optional_pages = []

        optional_keywords = (
            "blog", "news", "changelog", "release", "announcement",
            "twitter", "facebook", "linkedin", "social", "community",
            "sponsor", "donate", "careers", "jobs", "press"
        )

        for page in pages:
            url_lower = page.url.lower()
            title_lower = page.title.lower()

            is_optional = any(
                kw in url_lower or kw in title_lower for kw in optional_keywords
            )

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

        if core_pages:
            lines.append("## Core")
            for page in core_pages:
                lines.append(f"- [{page.title}]({page.url}): {page.description}")
            lines.append("")

        if guide_pages:
            lines.append("## Guides")
            for page in guide_pages:
                lines.append(f"- [{page.title}]({page.url}): {page.description}")
            lines.append("")

        if api_pages:
            lines.append("## API Reference")
            for page in api_pages:
                lines.append(f"- [{page.title}]({page.url}): {page.description}")
            lines.append("")

        if optional_pages:
            lines.append("## Optional")
            for page in optional_pages:
                lines.append(f"- [{page.title}]({page.url}): {page.description}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    async def close(self) -> None:
        """Close any resources."""
        pass


async def generate_llmstxt(
    url: str,
    config: Optional[AppConfig] = None,
    max_urls: Optional[int] = None,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable] = None,
    enable_critic: bool = True,
    max_retries: int = 2,
    pass_threshold: float = 0.7,
) -> GenerationResult:
    """Generate llms.txt for a website.

    Args:
        url: Website URL
        config: Optional app configuration
        max_urls: Override max URLs
        output_dir: Directory to save files
        progress_callback: Progress callback(stage, current, total, message)
        enable_critic: Enable critic validation
        max_retries: Maximum revision rounds
        pass_threshold: Minimum score to pass (0.0-1.0)

    Returns:
        GenerationResult
    """
    from dotenv import load_dotenv

    load_dotenv()

    config = config or AppConfig.from_env()
    generator = LLMsTxtGenerator(
        config,
        pass_threshold=pass_threshold,
        max_rounds=max_retries + 1,
    )

    try:
        result = await generator.generate(
            url,
            max_urls=max_urls,
            progress_callback=progress_callback,
            enable_critic=enable_critic,
            max_retries=max_retries,
        )

        # Save files if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            domain = urlparse(url).netloc.replace("www.", "")

            llmstxt_path = output_dir / f"{domain}-llms.txt"
            llmstxt_path.write_text(result.llmstxt, encoding="utf-8")
            logger.info(f"Saved llms.txt to {llmstxt_path}")

        return result

    finally:
        await generator.close()
