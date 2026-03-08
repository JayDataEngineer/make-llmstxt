"""LLMs.txt Generator - Main generation logic.

Generates:
- llms.txt: Index of pages with titles and descriptions
- llms-full.txt: Full content of all pages

Scraping via MCP server using langchain-mcp-adapters.
Uses Deep Agents Draft-Critic pattern for quality assurance.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from langchain_openai import ChatOpenAI

from .config import AppConfig
from .mcp_tools import mcp_map_website, mcp_scrape_batch, mcp_scrape_url
from .llm import create_llm, generate_summaries_batch
from .critic import CriticResult
from .deep_draft import (
    DeepDraftConfig,
    SimpleDraftCritic,
    DraftState,
)


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

    Scraping via MCP server using langchain-mcp-adapters.
    Uses Deep Agents Draft-Critic pattern for iterative quality improvement.
    """

    def __init__(
        self,
        config: AppConfig,
        llm: Optional[ChatOpenAI] = None,
        pass_threshold: float = 0.7,
        fail_on_critic_error: bool = False,
        max_rounds: int = 3,
    ):
        """Initialize generator.

        Args:
            config: Application configuration
            llm: Optional pre-configured LLM instance
            pass_threshold: Minimum critic score to pass (0.0-1.0)
            fail_on_critic_error: Fail generation if critic errors
            max_rounds: Maximum draft-critic rounds (default: 3)
        """
        self.config = config
        self.pass_threshold = pass_threshold
        self.fail_on_critic_error = fail_on_critic_error
        self.max_rounds = max_rounds

        # Create LLM if not provided
        self.llm = llm or create_llm(config.llm)

        # Create Deep Draft Critic
        self.draft_config = DeepDraftConfig(
            max_rounds=max_rounds,
            pass_threshold=pass_threshold,
            drafter_model=config.llm.model,
            temperature=config.llm.temperature,
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

    async def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single URL."""
        from .mcp_tools import mcp_scrape_url
        return await mcp_scrape_url(
            host=self.config.mcp.host,
            port=self.config.mcp.port,
            url=url,
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
        progress_callback: Optional[Callable] = None,
        enable_critic: bool = True,
        max_retries: int = 2,
    ) -> GenerationResult:
        """Generate llms.txt files for a website.

        Uses the Deep Agents Draft-Critic pattern:
        1. Scrape website content
        2. Generate initial draft
        3. Critic evaluates draft (structured Pydantic output)
        4. If not approved, revise with feedback
        5. Repeat until approved or max_rounds reached

        Args:
            url: Website URL
            max_urls: Override max URLs from config
            include_full_text: Whether to include full text
            progress_callback: Optional callback(stage, current, total, message)
            enable_critic: Whether to use critic validation
            max_retries: Maximum revision rounds (deprecated: use max_rounds in config)

        Returns:
            GenerationResult with generated content and critic metadata
        """
        max_urls = max_urls or self.config.max_urls

        # Step 1: Scrape website content
        if progress_callback:
            progress_callback("scraping", 0, 1, f"Scraping {url}...")

        scraped_content, pages = await self._scrape_website(
            url=url,
            max_urls=max_urls,
            progress_callback=progress_callback,
        )

        if not pages:
            raise ValueError(f"No content scraped from: {url}")

        # Step 2: Generate using Deep Draft pattern (if critic enabled)
        if enable_critic:
            result = await self._generate_with_deep_draft(
                url=url,
                scraped_content=scraped_content,
                pages=pages,
                include_full_text=include_full_text,
                progress_callback=progress_callback,
            )
        else:
            # Skip critic - just format the output
            llmstxt = self._format_llmstxt(url, pages)
            llms_fulltxt = (
                self._format_llms_fulltxt(url, pages) if include_full_text else ""
            )
            result = GenerationResult(
                llmstxt=llmstxt,
                llms_fulltxt=llms_fulltxt,
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
    ) -> tuple[str, List[PageResult]]:
        """Scrape website and return content string and page results.

        Args:
            url: Website URL
            max_urls: Maximum URLs to scrape (None = unlimited)
            progress_callback: Progress callback

        Returns:
            Tuple of (scraped_content_string, list_of_page_results)
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

            results = await self.process_batch(batch, start_index=i)
            all_results.extend(results)

            # Small delay between batches
            if i + batch_size < len(all_urls):
                await asyncio.sleep(1)

        # Sort by index
        all_results.sort(key=lambda x: x.index)

        # Build content string for LLM
        content_parts = []
        for page in all_results:
            content_parts.append(f"URL: {page.url}")
            content_parts.append(f"Title: {page.title}")
            content_parts.append(f"Description: {page.description}")
            content_parts.append(f"Content:\n{page.markdown[:500]}...")
            content_parts.append("---")

        scraped_content = "\n".join(content_parts)

        return scraped_content, all_results

    async def _generate_with_deep_draft(
        self,
        url: str,
        scraped_content: str,
        pages: List[PageResult],
        include_full_text: bool,
        progress_callback: Optional[Callable] = None,
    ) -> GenerationResult:
        """Generate using Deep Draft-Critic pattern.

        Args:
            url: Website URL
            scraped_content: Formatted scraped content
            pages: List of page results
            include_full_text: Whether to include full text
            progress_callback: Progress callback

        Returns:
            GenerationResult
        """
        logger.info(f"[DeepDraft] Starting generation for {url}")
        logger.info(f"[DeepDraft] Config: max_rounds={self.draft_config.max_rounds}, threshold={self.draft_config.pass_threshold}")

        # Create Deep Draft Critic
        critic = SimpleDraftCritic(
            config=self.draft_config,
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
        )

        # Custom progress callback adapter
        def draft_progress_callback(draft: str, round_num: int, critique: Optional[CriticResult]):
            if progress_callback and critique:
                progress_callback(
                    "critiquing",
                    round_num,
                    self.draft_config.max_rounds,
                    f"Round {round_num}: score={critique.score:.2f}, passed={critique.passed}",
                )

        # Generate with iterative refinement
        final_content, state = await critic.generate(
            url=url,
            content=scraped_content,
            progress_callback=draft_progress_callback,
        )

        # Build result
        llms_fulltxt = (
            self._format_llms_fulltxt(url, pages) if include_full_text else ""
        )

        result = GenerationResult(
            llmstxt=final_content,
            llms_fulltxt=llms_fulltxt,
            num_urls_processed=len(pages),
            num_urls_total=len(pages),
            pages=pages,
            critic_passed=state.agreed,
            critic_score=state.critique.score if state.critique else 1.0,
            critic_issues=state.critique.issues if state.critique else [],
            retry_count=state.round - 1 if state.agreed else state.round,
        )

        if state.agreed:
            logger.info(f"[DeepDraft] Approved at round {state.round} with score {result.critic_score:.2f}")
        else:
            logger.warning(f"[DeepDraft] Max rounds reached without approval, returning best effort")

        return result

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
        """Close any resources."""
        pass  # MCP connections are managed per-call in mcp_tools


async def generate_llmstxt(
    url: str,
    config: Optional[AppConfig] = None,
    max_urls: Optional[int] = None,
    include_full_text: bool = True,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable] = None,
    enable_critic: bool = True,
    max_retries: int = 2,
    pass_threshold: float = 0.7,
    fail_on_critic_error: bool = False,
) -> GenerationResult:
    """Generate llms.txt files for a website.

    Uses the Deep Agents Draft-Critic pattern for iterative quality improvement.

    Args:
        url: Website URL
        config: Optional app configuration
        max_urls: Override max URLs
        include_full_text: Include full text file
        output_dir: Directory to save files
        progress_callback: Progress callback(stage, current, total, message)
        enable_critic: Enable critic validation
        max_retries: Maximum draft-critic rounds (alias for max_rounds)
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
        max_rounds=max_retries + 1,  # Convert retries to rounds (rounds = retries + 1)
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
