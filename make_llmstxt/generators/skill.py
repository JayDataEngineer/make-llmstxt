"""Skill generator using hierarchical LangGraph pattern.

Architecture:
- Supervisor: Plans the skill package structure
- Generator: Deep Agent that creates folder structure (SKILL.md, scripts/, references/)
- Critic: Validates folder structure and YAML frontmatter

The generator uses Deep Agents' built-in filesystem tools to create:
- SKILL.md (with YAML frontmatter)
- references/ (scraped documentation)
- scripts/ (extracted code examples)
"""

from pathlib import Path
from typing import Optional, List, Callable, Dict, Any, Literal
from urllib.parse import urlparse

from loguru import logger
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import create_react_agent
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel

from make_llmstxt.core import GeneratorConfig, ScrapedPage, GeneratorResult
from make_llmstxt.mcp_scraper import MCPWebScraper, MCPConfig


# ==============================================================================
# Prompts
# ==============================================================================

SUPERVISOR_PROMPT = """You are a skill package supervisor. Your job is to plan the creation of an AI Skill Package.

When given a library name and documentation, you will:
1. Analyze the documentation structure
2. Plan the folder layout:
   - SKILL.md (with YAML frontmatter: name, description, version)
   - references/ (key documentation files)
   - scripts/ (code examples extracted from docs)
3. Delegate to the generator to create the actual files

Output a clear plan for what files to create."""

GENERATOR_PROMPT = """You are a skill package generator. You create the actual skill package files.

You have access to filesystem tools (write_file, read_file, etc.) and scraping tools.

Your task is to create a complete skill package folder structure:

1. Create the base directory: {output_dir}/{library_name}/
2. Write SKILL.md with proper YAML frontmatter:
   ---
   name: {library_name}
   description: Brief description
   version: 1.0.0
   ---

3. Create references/ folder with key documentation
4. Create scripts/ folder with extracted code examples

IMPORTANT: You MUST create a folder structure, not just a single file.
Use write_file to create each file in the correct path."""

CRITIC_PROMPT = """You are a skill package critic. You validate the generated skill package.

Check for:
1. SKILL.md exists with valid YAML frontmatter (name, description, version)
2. references/ folder exists with at least one .md file
3. scripts/ folder exists with at least one .py file
4. Content quality: clear sections, working code examples

Respond with:
- "APPROVE" if the package is complete and valid
- Specific issues to fix if something is missing or invalid

Be strict - a partial skill package is useless."""


# ==============================================================================
# Tools
# ==============================================================================

def create_tools(scraper: MCPWebScraper, config: GeneratorConfig) -> List:
    """Create tools for the agents."""

    tools = []

    @tool
    def scrape_page(url: str) -> dict:
        """Scrape a single URL and return markdown content.

        Args:
            url: URL to scrape

        Returns:
            Dict with url, title, content, metadata
        """
        import asyncio
        result = asyncio.run(scraper.scrape_url(url))
        if result:
            return {
                "url": url,
                "title": result.get("metadata", {}).get("title", ""),
                "content": result.get("markdown", ""),
                "metadata": result.get("metadata", {}),
            }
        return {"error": f"Failed to scrape {url}"}

    tools.append(scrape_page)

    @tool
    def map_site(url: str) -> dict:
        """Discover URLs from a website using sitemap.

        Args:
            url: Base URL to map

        Returns:
            Dict with urls (list of discovered URLs)
        """
        import asyncio
        urls = asyncio.run(scraper.map_website(url, limit=config.max_urls))
        return {"urls": urls}

    tools.append(map_site)

    return tools


# ==============================================================================
# Graph Builder
# ==============================================================================

class SkillGenerator:
    """Generate skill packages using hierarchical LangGraph pattern.

    Architecture:
    - Supervisor: Plans the skill package structure
    - Generator: Deep Agent that creates folder structure
    - Critic: Validates folder structure and content

    Output:
    - {library_name}/SKILL.md
    - {library_name}/references/*.md
    - {library_name}/scripts/*.py
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._scraper: Optional[MCPWebScraper] = None

    async def _get_scraper(self) -> MCPWebScraper:
        """Get or create MCP scraper instance."""
        if self._scraper is None:
            mcp_config = MCPConfig(
                host=self.config.mcp_host,
                port=self.config.mcp_port,
            )
            self._scraper = MCPWebScraper(mcp_config)
        return self._scraper

    async def gather_content(self, url: str) -> List[ScrapedPage]:
        """Gather content from the target URL.

        Args:
            url: Target URL to process

        Returns:
            List of scraped pages with markdown content
        """
        scraper = await self._get_scraper()

        # Discover URLs
        logger.info(f"[SkillGenerator] Mapping website: {url}")
        urls = await scraper.map_website(url, limit=self.config.max_urls)
        logger.info(f"[SkillGenerator] Found {len(urls)} URLs")

        # Scrape pages
        logger.info(f"[SkillGenerator] Scraping {len(urls)} pages...")
        pages = await scraper.scrape_batch(urls)

        # Convert to ScrapedPage objects
        scraped_pages = []
        for page in pages:
            scraped_pages.append(ScrapedPage(
                url=page.get("url", ""),
                title=page.get("metadata", {}).get("title"),
                markdown=page.get("markdown", ""),
                metadata=page.get("metadata", {}),
            ))

        logger.info(f"[SkillGenerator] Scraped {len(scraped_pages)} pages successfully")
        return scraped_pages

    def _extract_name(self, url: str) -> str:
        """Extract library name from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        if domain.startswith("docs."):
            domain = domain[5:]
        return domain.split(".")[0]

    def _build_content_summary(self, pages: List[ScrapedPage]) -> str:
        """Build a summary of gathered content for the agent."""
        summaries = []
        for page in pages[:20]:  # Include more pages for skill generation
            title = page.title or page.url
            content = page.markdown[:1000]  # More content for better skill generation
            summaries.append(f"### {title}\nURL: {page.url}\n{content}...\n")
        return "\n\n".join(summaries)

    async def generate(
        self,
        url: str,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
    ) -> GeneratorResult:
        """Generate a skill package for the given URL using hierarchical LangGraph.

        Args:
            url: Target URL to process
            output_dir: Output directory (defaults to config.output_dir)
            progress_callback: Optional callback for progress updates

        Returns:
            GeneratorResult with output path and stats
        """
        output_dir = output_dir or self.config.output_dir
        library_name = self._extract_name(url)
        skill_dir = output_dir / library_name

        logger.info(f"[SkillGenerator] Starting skill generation for {url}")
        logger.info(f"[SkillGenerator] Output directory: {skill_dir}")

        if progress_callback:
            progress_callback("Gathering documentation...", 10)

        # Step 1: Gather content
        pages = await self.gather_content(url)
        logger.info(f"[SkillGenerator] Gathered {len(pages)} pages")

        if progress_callback:
            progress_callback(f"Gathered {len(pages)} pages", 20)

        # Step 2: Create tools
        scraper = await self._get_scraper()
        tools = create_tools(scraper, self.config)

        if progress_callback:
            progress_callback("Building LangGraph...", 30)

        # Step 3: Build the hierarchical graph
        llm = ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

        # Supervisor agent
        supervisor = create_react_agent(
            llm,
            tools=[],  # Supervisor just plans, doesn't scrape
            state_modifier=SUPERVISOR_PROMPT,
        )

        # Generator agent (Deep Agent with filesystem tools)
        generator_agent = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=GENERATOR_PROMPT.format(
                output_dir=str(output_dir),
                library_name=library_name,
            ),
        )

        # Critic agent
        critic = create_react_agent(
            llm,
            tools=[],  # Critic just reviews
            state_modifier=CRITIC_PROMPT,
        )

        if progress_callback:
            progress_callback("Running skill generation graph...", 40)

        # Step 4: Build the graph with max iterations guard
        MAX_ITERATIONS = 3  # Prevent infinite loops

        def should_continue(state: MessagesState) -> str:
            """Router: Check if critic approved or needs fixes."""
            # Check iteration count to prevent infinite loops
            message_count = len(state.get("messages", []))
            iteration = message_count // 3  # Each iteration = supervisor + generator + critic

            if iteration >= MAX_ITERATIONS:
                logger.warning(f"[SkillGenerator] Max iterations ({MAX_ITERATIONS}) reached, ending")
                return END

            last_message = state["messages"][-1].content
            if "APPROVE" in last_message.upper():
                return END
            return "generator"  # Loop back to fix issues

        graph = StateGraph(MessagesState)
        graph.add_node("supervisor", supervisor)
        graph.add_node("generator", generator_agent)
        graph.add_node("critic", critic)

        graph.set_entry_point("supervisor")
        graph.add_edge("supervisor", "generator")
        graph.add_edge("generator", "critic")
        graph.add_conditional_edges("critic", should_continue)

        app = graph.compile()

        # Step 5: Run the graph
        content_summary = self._build_content_summary(pages)
        initial_message = f"""Create a skill package for {library_name}.

URL: {url}
Output directory: {output_dir}
Library name: {library_name}

Gathered Documentation:
---
{content_summary}
---

Create the folder structure:
1. {skill_dir}/SKILL.md (with YAML frontmatter)
2. {skill_dir}/references/ (documentation files)
3. {skill_dir}/scripts/ (code examples)
"""

        logger.info(f"[SkillGenerator] Running LangGraph...")
        result = await app.ainvoke({
            "messages": [{"role": "user", "content": initial_message}]
        })

        if progress_callback:
            progress_callback("Skill package generated", 90)

        # Step 6: Check output
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists():
            logger.info(f"[SkillGenerator] Skill package created at {skill_dir}")
            if progress_callback:
                progress_callback(f"Skill package created at {skill_dir}", 100)
            return GeneratorResult(
                output_path=skill_file,
                stats={
                    "pages_processed": len(pages),
                    "output_dir": str(skill_dir),
                    "files_created": len(list(skill_dir.rglob("*"))) if skill_dir.exists() else 0,
                },
            )
        else:
            logger.warning(f"[SkillGenerator] SKILL.md not created, checking graph output")
            # Fallback: extract content from last message
            final_message = result.get("messages", [])[-1] if result.get("messages") else None
            if final_message:
                content = final_message.content if hasattr(final_message, 'content') else str(final_message)
                # Write the content anyway
                skill_dir.mkdir(parents=True, exist_ok=True)
                skill_file.write_text(content)
                logger.info(f"[SkillGenerator] Wrote fallback SKILL.md")

            return GeneratorResult(
                output_path=skill_file,
                stats={
                    "pages_processed": len(pages),
                    "output_dir": str(skill_dir),
                    "fallback": True,
                },
            )

    async def close(self):
        """Clean up resources."""
        if self._scraper:
            await self._scraper.close()
            self._scraper = None
