"""LLMs.txt Generator using Deep Agent pattern.

Architecture:
1. Generator Deep Agent orchestrates URL discovery and llms.txt creation
2. Page Scraper Subagent scrapes individual URLs for title/description
3. Critic validates the output (structured output)
4. Loop until approved or max rounds reached

Same pattern as skill.py but with different prompts and output.
"""

import asyncio
from typing import List, Optional, Dict, Any, TypedDict
from pathlib import Path
from urllib.parse import urlparse
import time

from loguru import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.subagents import SubAgent

from ..core import GeneratorConfig, GeneratorResult
from ..config import AppConfig
from ..scrapers import (
    get_mcp_tools,
    filter_tools_by_name,
    LLMSTXT_AGENT_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
)
from .critic import Critic, CriticResult


# ==============================================================================
# Prompts
# ==============================================================================

GENERATOR_SYSTEM_PROMPT = """You are an expert llms.txt generator. You create standardized llms.txt files for AI model context ingestion.

## Your Task

Generate a complete llms.txt file for the website.

## llms.txt Format

```markdown
# Project Name
> One-line summary of the project.

## Section Name
- [Page Title](https://example.com/page): Concise description of the page.

## Optional
- [Blog](https://example.com/blog): Non-essential content.
```

## Critical Rules

1. **H1 Header**: Project NAME (not URL) at the top
2. **Blockquote**: Project summary right after H1
3. **H2 Sections**: Group URLs logically (Core, API Reference, Guides, etc.)
4. **Link Format**: `- [Title](URL): Description.` (colon then space then description)
5. **Include ALL URLs**: Every discovered URL must appear in the file
6. **Titles**: 2-5 words, specific (NOT "Home", "Page", "Documentation")
7. **Descriptions**: 5-12 words, informative, end with period
8. **Optional Section**: Put blogs, changelogs, social links under `## Optional`
9. **No Placeholders**: Never use "No description" or "Documentation for..."

## Process

1. Use `map_domain` to discover all URLs on the website
2. Use the `page_scraper` subagent to get title/description for each URL
3. Write the complete llms.txt file using `write_file`
4. If the critic provides feedback, revise and rewrite

## Output

Write the llms.txt to: {output_file}
Target URL: {url}
Max URLs to include: {max_urls}
"""

SUBAGENT_SYSTEM_PROMPT = """You are a page summarizer. Extract title and description from scraped web pages.

## Your Task

For the given URL, scrape it and return a JSON object with:
- title: 2-5 words, specific and descriptive
- description: 5-12 words, informative summary ending with a period

## Rules

- Title: Be specific (NOT "Home", "Page", "About")
- Description: Summarize what the page IS, not what it "contains"
- Never use placeholders like "No description available"
- Extract meaning from the actual content

Return ONLY valid JSON:
{"title": "Specific Title", "description": "Informative description of page purpose."}
"""

CRITIC_FEEDBACK_PROMPT = """The critic found issues with your llms.txt:

{critic_feedback}

Please fix these issues and rewrite the complete llms.txt file.

Remember:
- Include ALL URLs from the original discovery
- Fix the specific issues mentioned
- Maintain proper format
- End all descriptions with periods
"""


# ==============================================================================
# State Definition
# ==============================================================================

class LLMsTxtState(TypedDict):
    """State for llms.txt generation graph."""
    messages: List[Any]
    url: str
    output_file: str
    max_urls: int
    current_round: int
    max_rounds: int
    critic_passed: bool
    critic_score: float
    critic_feedback: List[str]


# ==============================================================================
# Logging Callback Handler
# ==============================================================================

class LLMsTxtLoggingHandler(BaseCallbackHandler):
    """Custom callback handler for detailed llms.txt generation logging."""

    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.chain_depth: int = 0

    def _indent(self) -> str:
        return "  " * self.chain_depth

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times[run_id] = time.time()
        name = serialized.get("name", kwargs.get("name", "unnamed")) if serialized else kwargs.get("name", "unnamed")
        self.chain_depth += 1
        logger.info(f"{self._indent()}┌─ CHAIN START: {name}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        duration = time.time() - self.start_times.pop(run_id, time.time())
        logger.info(f"{self._indent()}└─ CHAIN END ({duration:.2f}s)")
        self.chain_depth = max(0, self.chain_depth - 1)

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times.pop(run_id, None)
        logger.error(f"{self._indent()}💥 CHAIN ERROR: {error}")
        self.chain_depth = max(0, self.chain_depth - 1)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times[run_id] = time.time()
        model = kwargs.get("invocation_params", {}).get("model", "unknown")
        logger.info(f"{self._indent()}🤖 LLM START: {model}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        duration = time.time() - self.start_times.pop(run_id, time.time())
        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        total_tokens = token_usage.get("total_tokens", "N/A")
        logger.info(f"{self._indent()}✅ LLM END ({duration:.2f}s, {total_tokens} tokens)")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times.pop(run_id, None)
        logger.error(f"{self._indent()}💥 LLM ERROR: {error}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times[run_id] = time.time()
        tool_name = serialized.get("name", kwargs.get("name", "unknown")) if serialized else kwargs.get("name", "unknown")
        logger.info(f"{self._indent()}🔧 TOOL START: {tool_name}")

    def on_tool_end(self, output: Any, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        duration = time.time() - self.start_times.pop(run_id, time.time())
        output_str = str(output)[:150] + "..." if len(str(output)) > 150 else str(output)
        logger.info(f"{self._indent()}✅ TOOL END ({duration:.2f}s): {output_str}")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times.pop(run_id, None)
        logger.error(f"{self._indent()}💥 TOOL ERROR: {error}")


# ==============================================================================
# Agent Generator
# ==============================================================================

class LLMsTxtAgentGenerator:
    """Generate llms.txt using Deep Agent pattern.

    Flow:
    1. Generator Deep Agent creates llms.txt using MCP tools
    2. Critic validates with structured output
    3. If not approved, generator revises with feedback

    Output:
    - llms.txt file at specified output path
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config

    def _extract_name(self, url: str) -> str:
        """Extract project name from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        if domain.startswith("docs."):
            domain = domain[5:]
        return domain.split(".")[0]

    async def generate(
        self,
        url: str,
        output_file: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> GeneratorResult:
        """Generate llms.txt for the given URL.

        Args:
            url: Target URL to process
            output_file: Output file path (defaults to output_dir/llms.txt)
            progress_callback: Optional callback for progress updates

        Returns:
            GeneratorResult with output path and stats
        """
        output_file = Path(output_file or self.config.output_dir / "llms.txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        project_name = self._extract_name(url)

        logger.info(f"[LLMsTxtAgent] Starting generation for {url}")
        logger.info(f"[LLMsTxtAgent] Output file: {output_file}")
        logger.info(f"[LLMsTxtAgent] Project name: {project_name}")

        if progress_callback:
            progress_callback("Connecting to MCP server...", 5)

        # Connect to MCP and build agents
        async with get_mcp_tools(
            self.config.mcp_host,
            self.config.mcp_port,
            max_urls=self.config.max_urls,
        ) as all_tools:
            main_tools = filter_tools_by_name(all_tools, LLMSTXT_AGENT_TOOL_NAMES)
            subagent_tools = filter_tools_by_name(all_tools, SUBAGENT_TOOL_NAMES)

            logger.info(f"[LLMsTxtAgent] Main agent tools: {[t.name for t in main_tools]}")
            logger.info(f"[LLMsTxtAgent] Subagent tools: {[t.name for t in subagent_tools]}")

            if progress_callback:
                progress_callback("Building agents...", 10)

            # Create base LLM
            llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            logger.info(f"[LLMsTxtAgent] Using LLM: {self.config.model}")

            # Create critic
            critic = Critic(llm, pass_threshold=self.config.pass_threshold)

            # Create filesystem backend
            fs_backend = FilesystemBackend(root_dir=str(output_file.parent))

            # Create page scraper subagent
            scraper_subagent = SubAgent(
                name="page_scraper",
                description="Scrapes a URL and returns title/description JSON.",
                system_prompt=SUBAGENT_SYSTEM_PROMPT,
                tools=subagent_tools,
            )

            # Create generator agent
            generator_agent = create_deep_agent(
                model=llm,
                tools=main_tools,
                subagents=[scraper_subagent],
                system_prompt=GENERATOR_SYSTEM_PROMPT.format(
                    url=url,
                    output_file=str(output_file),
                    max_urls=self.config.max_urls or "unlimited",
                ),
                backend=fs_backend,
            )

            if progress_callback:
                progress_callback("Running generator → critic loop...", 20)

            # Build the graph
            def should_continue(state: dict) -> str:
                current_round = state.get("current_round", 0)
                max_rounds = state.get("max_rounds", 3)

                if current_round >= max_rounds:
                    logger.warning(f"[LLMsTxtAgent] Max rounds ({max_rounds}) reached")
                    return END

                critic_passed = state.get("critic_passed", False)
                if critic_passed:
                    logger.info("[LLMsTxtAgent] Critic APPROVED the llms.txt")
                    return END

                return "generator"

            async def critic_node(state: dict) -> dict:
                """Run critic on the generated llms.txt."""
                current_round = state.get("current_round", 0) + 1

                if progress_callback:
                    progress_callback(f"Critic round {current_round}...", 20 + current_round * 20)

                # Read the generated llms.txt
                try:
                    llmstxt_content = output_file.read_text()
                except FileNotFoundError:
                    logger.error(f"[LLMsTxtAgent] llms.txt not found at {output_file}")
                    return {
                        **state,
                        "current_round": current_round,
                        "critic_passed": False,
                        "critic_score": 0.0,
                        "critic_feedback": ["llms.txt file was not created"],
                    }

                logger.info(f"[LLMsTxtAgent] Critic round {current_round}: evaluating ({len(llmstxt_content)} chars)")

                # Run critic (we're already in async context)
                critique = await critic.evaluate(
                    llmstxt=llmstxt_content,
                    url=url,
                    source_content=None,  # TODO: pass scraped content for coverage check
                )

                logger.info(
                    f"[LLMsTxtAgent] Critic result: passed={critique.passed}, "
                    f"score={critique.score:.2f}, issues={len(critique.issues)}"
                )

                feedback = critique.issues + critique.suggestions

                if not critique.passed:
                    # Add feedback message for generator
                    feedback_msg = CRITIC_FEEDBACK_PROMPT.format(
                        critic_feedback="\n".join(f"- {f}" for f in feedback)
                    )
                    messages = state.get("messages", [])
                    messages = list(messages) + [{"role": "user", "content": feedback_msg}]

                    return {
                        **state,
                        "messages": messages,
                        "current_round": current_round,
                        "critic_passed": False,
                        "critic_score": critique.score,
                        "critic_feedback": feedback,
                    }

                return {
                    **state,
                    "current_round": current_round,
                    "critic_passed": True,
                    "critic_score": critique.score,
                    "critic_feedback": [],
                }

            # Build graph
            graph = StateGraph(LLMsTxtState)
            graph.add_node("generator", generator_agent)
            graph.add_node("critic", critic_node)

            graph.set_entry_point("generator")
            graph.add_edge("generator", "critic")
            graph.add_conditional_edges("critic", should_continue)

            app = graph.compile()

            # Run the graph
            initial_message = f"""Generate llms.txt for {url}.

Steps:
1. Use map_domain to discover all URLs on {url}
2. Use page_scraper subagent to get title/description for each URL
3. Write the complete llms.txt to {output_file}

Remember to:
- Include ALL discovered URLs
- Use specific titles (2-5 words)
- Write informative descriptions (5-12 words)
- End descriptions with periods
- Group URLs into logical sections
"""

            logger.info("[LLMsTxtAgent] Running generator → critic loop...")

            logging_handler = LLMsTxtLoggingHandler()

            import asyncio
            result = await app.ainvoke(
                {
                    "messages": [{"role": "user", "content": initial_message}],
                    "url": url,
                    "output_file": str(output_file),
                    "max_urls": self.config.max_urls or 100,
                    "current_round": 0,
                    "max_rounds": self.config.max_rounds,
                    "critic_passed": False,
                    "critic_score": 0.0,
                    "critic_feedback": [],
                },
                config={"callbacks": [logging_handler]}
            )

            if progress_callback:
                progress_callback("Generation complete", 100)

            # Check output
            if output_file.exists():
                content = output_file.read_text()
                logger.info(f"[LLMsTxtAgent] Generated llms.txt ({len(content)} chars)")

                return GeneratorResult(
                    output_path=output_file,
                    stats={
                        "file_size": len(content),
                        "critic_passed": result.get("critic_passed", False),
                        "critic_score": result.get("critic_score", 0.0),
                        "rounds": result.get("current_round", 0),
                    },
                )
            else:
                raise RuntimeError(f"llms.txt generation failed: file not created at {output_file}")

    async def close(self):
        """Clean up resources."""
        pass


