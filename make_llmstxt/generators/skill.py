"""Skill generator using Deep Agents.

This generator creates SKILL.md files using the Deep Agents framework.
The agent has access to:
- File tools (read/write)
- Task tool (spawn sub-agents)
- MCP tools (scrape, map)
- Todo planning

"""

from pathlib import Path
from typing import Optional, List, Callable

from loguru import logger
from deepagents import create_deep_agent
from langchain_core.tools import tool

from make_llmstxt.core import BaseGenerator, GeneratorConfig, ScrapedPage, GeneratorResult
from make_llmstxt.templates.skill import SKILL_MAKER_SYSTEM_PROMPT


class SkillGenerator(BaseGenerator):
    """Generate skill packages using Deep Agents.
    This uses the Deep Agent with:
    - Sub-agents for per-page processing
    - File tools for output
    - Todo planning for workflow
    - Plan → Double check → Build → Validate flow
    """

    def _create_tools(self) -> List:
        """Create tools for the Deep Agent."""
        tools = []

        # MCP scrape tool
        @tool
        def scrape_page(url: str) -> dict:
            """Scrape a single URL and return markdown content.

            Args:
                url: URL to scrape

            Returns:
                Dict with url, title, content, metadata
            """
            result = self._scraper.scrape_url(url)
            if result:
                return {
                    "url": url,
                    "title": result.get("metadata", {}).get("title", ""),
                    "content": result.get("markdown", ""),
                    "metadata": result.get("metadata", {}),
                }
            return {"error": f"Failed to scrape {url}"}

        tools.append(scrape_page)

        # MCP map tool
        @tool
        def map_site(url: str) -> dict:
            """Discover URLs from a website using sitemap.

            Args:
                url: Base URL to map

            Returns:
                Dict with urls (list of discovered URLs)
            """
            urls = self._scraper.map_website(url, limit=self.config.max_urls)
            return {"urls": urls}

        tools.append(map_site)

        return tools

    async def generate(
        self,
        url: str,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
    ) -> GeneratorResult:
        """Generate a SKILL.md for the given URL.

        Args:
            url: Target URL to process
            output_dir: Output directory (defaults to config.output_dir)
            progress_callback: Optional callback for progress updates

        Returns:
            GeneratorResult with output path and stats
        """
        output_dir = output_dir or self.config.output_dir
        output_path = output_dir / f"{self._extract_name(url)}-SKILL.md"

        logger.info(f"[SkillGenerator] Starting skill generation for {url}")

        # Gather content first
        if progress_callback:
            progress_callback("Mapping website...", 10)

        pages = await self.gather_content(url)
        logger.info(f"[SkillGenerator] Gathered {len(pages)} pages")
        if progress_callback:
            progress_callback(f"Gathered {len(pages)} pages", 20)

        # Create Deep Agent with tools
        tools = self._create_tools()
        # Create the agent
        agent = create_deep_agent(
            model=self.config.model,
            tools=tools,
            system_prompt=SKILL_MAKER_SYSTEM_PROMPT,
        )
        logger.info(f"[SkillGenerator] Deep Agent created")
        if progress_callback:
            progress_callback("Analyzing documentation and planning skill...", 30)
        # Build initial message with gathered content
        content_summary = self._build_content_summary(pages)
        initial_message = f"""Create a skill package for the following library:

URL: {url}

Gathered Documentation:
---
{content_summary}
---

Instructions:
1. Plan the skill creation using `write_todos`
2. Analyze the documentation structure
3. Create a comprehensive SKILL.md file
4. Write the output to {output_path}

Start by planning the skill creation."""
        logger.info(f"[SkillGenerator] Invoking Deep Agent...")
        # Invoke the agent
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": initial_message}]
        })
        # Extract the generated content
        final_message = result.get("messages", [])[-1] if result.get("messages") else None
        skill_content = ""
        if final_message:
            if hasattr(final_message, 'content'):
                skill_content = final_message.content
            elif isinstance(final_message, dict):
                skill_content = final_message.get("content", "")
        # Write the output
        with open(output_path, "w") as f:
            f.write(skill_content)
        logger.info(f"[SkillGenerator] Wrote skill to {output_path}")
        if progress_callback:
            progress_callback(f"Skill package created at {output_path}", 100)
        return GeneratorResult(
            output_path=output_path,
            stats={
                "pages_processed": len(pages),
                "output_file": str(output_path),
            },
        )

    def _extract_name(self, url: str) -> str:
        """Extract library name from URL.

        Args:
            url: URL to extract name from

        Returns:
            Library name (e.g., "nextra" from "nextra.site")
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # Remove www. or docs. prefix
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        if domain.startswith("docs."):
            domain = domain[5:]
        return domain.split(".")[0]

    def _build_content_summary(self, pages: List[ScrapedPage]) -> str:
        """Build a summary of gathered content for the agent.

        Args:
            pages: List of scraped pages

        Returns:
            Summary string for the agent
        """
        summaries = []
        for page in pages[:10]:  # Limit to first 10 for summary
            title = page.title or page.url
            content = page.markdown[:500]  # First 500 chars
            summaries.append(f"### {title}\n{content}...\n")
        return "\n\n".join(summaries)
