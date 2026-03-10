"""Shared infrastructure for Deep Agent pattern.

Both llms.txt and skill generators use this same infrastructure,
with different prompts injected via configuration.
"""

import time
import asyncio
import hashlib
import json
import re
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Optional, Callable, Annotated
import operator
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.store.base import BaseStore
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.subagents import SubAgent

from ..core import GeneratorConfig, GeneratorResult, AgentPrompts
from ..scrapers import (
    get_mcp_tools,
    filter_tools_by_name,
    MAIN_AGENT_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
)
from ..store import get_store_from_config
from .schemas import PageSummary


# ==============================================================================
# Utility Functions
# ==============================================================================

def extract_name_from_url(url: str) -> str:
    """Extract library/project name from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    if domain.startswith("docs."):
        domain = domain[5:]
    return domain.split(".")[0]


# ==============================================================================
# Message Cleaning (Qwen/llama.cpp compatibility)
# ==============================================================================

def clean_messages(messages: List[Any]) -> List[Any]:
    """Merge consecutive assistant messages into one.

    Some LLM APIs (like Qwen/llama.cpp) don't allow consecutive assistant messages.
    This function merges them by concatenating their content.
    """
    if not messages:
        return messages

    cleaned = []
    for msg in messages:
        # Handle both dict format and LangChain message objects
        if isinstance(msg, dict):
            msg_type = msg.get("role", "unknown")
            msg_content = msg.get("content", "")
        else:
            msg_type = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
            msg_content = getattr(msg, "content", "")

        # Check if we need to merge with previous message
        if cleaned:
            prev_msg = cleaned[-1]
            if isinstance(prev_msg, dict):
                prev_type = prev_msg.get("role", "unknown")
            else:
                prev_type = getattr(prev_msg, "type", None) or getattr(prev_msg, "role", "unknown")

            # Merge consecutive assistant messages
            if prev_type == "assistant" and msg_type == "assistant":
                if isinstance(prev_msg, dict):
                    prev_msg["content"] = f"{prev_msg.get('content', '')}\n{msg_content}"
                else:
                    prev_msg.content = f"{prev_msg.content}\n{msg_content}"
                continue

        cleaned.append(msg)

    return cleaned


# ==============================================================================
# Logging Callback Handler
# ==============================================================================

class DeepAgentLoggingHandler(BaseCallbackHandler):
    """Custom callback handler for detailed Deep Agent logging.

    Tracks:
    - Chain/agent start/end with timing
    - LLM calls and token usage
    - Tool invocations
    - Errors
    """

    def __init__(self, prefix: str = ""):
        self.start_times: Dict[str, float] = {}
        self.chain_depth: int = 0
        self.prefix = prefix

    def _indent(self) -> str:
        return "  " * self.chain_depth

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times[run_id] = time.time()

        name = "unnamed"
        if serialized and isinstance(serialized, dict):
            name = serialized.get("name", kwargs.get("name", "unnamed"))
        elif kwargs.get("name"):
            name = kwargs["name"]
        self.chain_depth += 1
        logger.info(f"{self.prefix}{self._indent()}┌─ CHAIN START: {name}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        duration = time.time() - self.start_times.pop(run_id, time.time())
        logger.info(f"{self.prefix}{self._indent()}└─ CHAIN END ({duration:.2f}s)")
        self.chain_depth = max(0, self.chain_depth - 1)

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times.pop(run_id, None)
        logger.error(f"{self.prefix}{self._indent()}💥 CHAIN ERROR: {error}")
        self.chain_depth = max(0, self.chain_depth - 1)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times[run_id] = time.time()
        model = kwargs.get("invocation_params", {}).get("model", "unknown")
        logger.info(f"{self.prefix}{self._indent()}🤖 LLM START: {model}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        duration = time.time() - self.start_times.pop(run_id, time.time())
        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        total_tokens = token_usage.get("total_tokens", "N/A")
        logger.info(f"{self.prefix}{self._indent()}✅ LLM END ({duration:.2f}s, {total_tokens} tokens)")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times.pop(run_id, None)
        logger.error(f"{self.prefix}{self._indent()}💥 LLM ERROR: {error}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times[run_id] = time.time()
        tool_name = "unknown"
        if serialized and isinstance(serialized, dict):
            tool_name = serialized.get("name", kwargs.get("name", "unknown"))
        elif kwargs.get("name"):
            tool_name = kwargs["name"]
        logger.info(f"{self.prefix}{self._indent()}🔧 TOOL START: {tool_name}")

    def on_tool_end(self, output: Any, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        duration = time.time() - self.start_times.pop(run_id, time.time())
        output_str = str(output)[:150] + "..." if len(str(output)) > 150 else str(output)
        logger.info(f"{self.prefix}{self._indent()}✅ TOOL END ({duration:.2f}s): {output_str}")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times.pop(run_id, None)
        logger.error(f"{self.prefix}{self._indent()}💥 TOOL ERROR: {error}")


# ==============================================================================
# State definition
# ==============================================================================

class DeepAgentState(TypedDict):
    """Base state for Deep Agent graph.

    Generic state used by all generators. Keep this clean - no domain-specific fields.
    """
    messages: List[Any]
    url: str
    output_path: str
    current_round: int
    max_rounds: int
    critic_passed: bool
    critic_score: float
    critic_feedback: List[str]
    # Generic context for task-specific data
    context: Dict[str, Any]


class WebScrapingState(DeepAgentState):
    """Specialized state for web scraping agents (Map-Reduce pattern).

    Extends DeepAgentState with parallel scraping fields. Use this
    only in generators that do web scraping (e.g., LLMsTxtGenerator).
    """
    # Parallel scraping state
    discovered_urls: List[str]  # URLs from map_domain
    scraped_docs: Annotated[List[Dict], operator.add]  # Parallel-safe accumulator
    scraping_errors: Annotated[List[str], operator.add]  # Failed URLs
    scraping_complete: bool  # Flag to indicate all scrapers finished


# ==============================================================================
# Deep Agent Generator Base Class
# ==============================================================================

class DeepAgentGenerator:
    """Unified Deep Agent generator for both llms.txt and skill generation.

    Architecture:
    1. Generator Deep Agent creates output using MCP tools
    2. Critic validates (either structured output or LLM-based)
    3. Loop until approved or max rounds reached

    Subclasses provide:
    - prompts: AgentPrompts instance
    - _run_critic(): Custom critic implementation
    - _get_output_path(): Determine output path from config
    """

    def __init__(
        self,
        config: GeneratorConfig,
        log_prefix: str = "[DeepAgent]",
        default_prompts: Optional[AgentPrompts] = None,
    ):
        # Set prompts if not already set
        if config.prompts is None and default_prompts is not None:
            config = config.model_copy(update={"prompts": default_prompts})

        self.config = config
        self.log_prefix = log_prefix

        # Ensure prompts are set
        if config.prompts is None:
            raise ValueError(f"{log_prefix} config.prompts must be set")

    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance."""
        return ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def _create_subagent(self, tools: List) -> Optional[SubAgent]:
        """Create subagent if subagent_system is provided."""
        prompts = self.config.prompts
        if not prompts or not prompts.subagent_system:
            return None

        return SubAgent(
            name=prompts.subagent_name,
            description=prompts.subagent_description,
            system_prompt=prompts.subagent_system,
            tools=tools,
        )

    async def generate(
        self,
        url: str,
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
    ) -> GeneratorResult:
        """Run the Deep Agent generation.

        Args:
            url: Target URL to process
            output_path: Output file/directory path (defaults to config.output_dir)
            progress_callback: Optional callback(message, percent)

        Returns:
            GeneratorResult with output path and stats
        """
        output_path = Path(output_path or self.config.output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"{self.log_prefix} Starting generation for {url}")
        logger.info(f"{self.log_prefix} Output: {output_path}")

        if progress_callback:
            progress_callback("Connecting to MCP server...", 5)

        # Connect to MCP and get tools
        async with get_mcp_tools(
            self.config.mcp_host,
            self.config.mcp_port,
            max_urls=self.config.max_urls,
        ) as all_tools:
            main_tools = filter_tools_by_name(all_tools, MAIN_AGENT_TOOL_NAMES)
            subagent_tools = filter_tools_by_name(all_tools, SUBAGENT_TOOL_NAMES)

            logger.info(f"{self.log_prefix} Main tools: {[t.name for t in main_tools]}")
            logger.info(f"{self.log_prefix} Subagent tools: {[t.name for t in subagent_tools]}")

            if progress_callback:
                progress_callback("Building agents...", 10)

            # Create LLM
            llm = self._create_llm()
            logger.info(f"{self.log_prefix} Using LLM: {self.config.model}")

            # Create subagent
            subagent = self._create_subagent(subagent_tools)
            subagents = [subagent] if subagent else []

            # Format system prompt with context
            system_prompt = self._format_system_prompt(url, output_path)

            # Create filesystem backend
            fs_backend = FilesystemBackend(root_dir=str(output_path.parent))

            # Create generator agent
            generator_agent = create_deep_agent(
                model=llm,
                tools=main_tools,
                subagents=subagents,
                system_prompt=system_prompt,
                backend=fs_backend,
            )

            if progress_callback:
                progress_callback("Running generator → critic loop...", 20)

            # Build the appropriate graph based on config
            if self.config.enable_parallel:
                logger.info(f"{self.log_prefix} Using PARALLEL scraping (max_concurrent={self.config.max_concurrent})")
                # Get scrape tool for parallel workers
                scrape_tool = filter_tools_by_name(all_tools, ["scrape_url"])[0] if filter_tools_by_name(all_tools, ["scrape_url"]) else None
                if scrape_tool:
                    graph = self._build_parallel_graph(generator_agent, llm, output_path, url, scrape_tool)
                else:
                    logger.warning(f"{self.log_prefix} scrape_url tool not found, falling back to serial mode")
                    graph = self._build_graph(generator_agent, llm, output_path, url)
            else:
                logger.info(f"{self.log_prefix} Using SERIAL scraping (conversational)")
                graph = self._build_graph(generator_agent, llm, output_path, url)

            # Create store for full content persistence (optional)
            store = get_store_from_config(self.config)

            # Run the graph
            initial_message = self._format_initial_message(url, output_path)
            logger.info(f"{self.log_prefix} Running generator → critic loop...")

            logging_handler = DeepAgentLoggingHandler(self.log_prefix)

            # Build config with store (if available)
            invoke_config = {
                "callbacks": [logging_handler],
                "configurable": {},
            }
            if store:
                invoke_config["configurable"]["store"] = store
                logger.info(f"{self.log_prefix} Store enabled for content persistence")

            result = await graph.ainvoke(
                {
                    "messages": [{"role": "user", "content": initial_message}],
                    "url": url,
                    "output_path": str(output_path),
                    "current_round": 0,
                    "max_rounds": self.config.max_rounds,
                    "critic_passed": False,
                    "critic_score": 0.0,
                    "critic_feedback": [],
                    "context": self.config.context,
                    # Parallel scraping state
                    "discovered_urls": [],
                    "scraped_docs": [],
                    "scraping_errors": [],
                    "scraping_complete": False,
                },
                config=invoke_config
            )

            if progress_callback:
                progress_callback("Generation complete", 100)

            # Check output
            return self._create_result(output_path, result)

    def _format_system_prompt(self, url: str, output_path: Path) -> str:
        """Format system prompt with context variables."""
        prompts = self.config.prompts
        context = {
            "url": url,
            "output_path": str(output_path),
            "max_urls": self.config.max_urls or "unlimited",
            **self.config.context,
        }
        try:
            return prompts.generator_system.format(**context)
        except KeyError as e:
            logger.warning(f"{self.log_prefix} Missing context key in prompt: {e}")
            return prompts.generator_system

    def _format_initial_message(self, url: str, output_path: Path) -> str:
        """Format initial message with context variables."""
        prompts = self.config.prompts
        context = {
            "url": url,
            "output_path": str(output_path),
            **self.config.context,
        }
        try:
            return prompts.generator_initial.format(**context)
        except KeyError as e:
            logger.warning(f"{self.log_prefix} Missing context key in initial message: {e}")
            return prompts.generator_initial

    def _format_feedback_message(self, feedback: List[str]) -> str:
        """Format critic feedback for next generator round."""
        return self.config.prompts.generator_feedback.format(
            critic_feedback="\n".join(f"- {f}" for f in feedback)
        )

    def _build_graph(self, generator_agent, llm, output_path: Path, url: str) -> StateGraph:
        """Build the StateGraph with generator → critic → loop pattern."""

        def should_continue(state: dict) -> str:
            current_round = state.get("current_round", 0)
            max_rounds = state.get("max_rounds", 3)

            if current_round >= max_rounds:
                logger.warning(f"{self.log_prefix} Max rounds ({max_rounds}) reached")
                return END

            critic_passed = state.get("critic_passed", False)
            if critic_passed:
                logger.info(f"{self.log_prefix} Critic APPROVED")
                return END

            return "generator"

        async def generator_node(state: dict) -> dict:
            """Run generator with message cleaning for Qwen compatibility."""
            messages = state.get("messages", [])
            cleaned = clean_messages(messages)
            if len(cleaned) != len(messages):
                logger.debug(f"{self.log_prefix} Cleaned {len(messages)} -> {len(cleaned)} messages")
            result = await generator_agent.ainvoke({**state, "messages": cleaned})
            return result

        async def critic_node(state: dict) -> dict:
            """Run critic on the generated output."""
            current_round = state.get("current_round", 0) + 1

            # Check if output exists and validate
            passed, score, feedback = await self._run_critic(
                state, output_path, url, current_round
            )

            logger.info(
                f"{self.log_prefix} Critic round {current_round}: "
                f"passed={passed}, score={score:.2f}, issues={len(feedback)}"
            )

            if not passed:
                # Add feedback message for generator
                feedback_msg = self._format_feedback_message(feedback)
                messages = state.get("messages", [])
                messages = list(messages) + [{"role": "user", "content": feedback_msg}]
                messages = clean_messages(messages)

                return {
                    **state,
                    "messages": messages,
                    "current_round": current_round,
                    "critic_passed": False,
                    "critic_score": score,
                    "critic_feedback": feedback,
                }

            return {
                **state,
                "current_round": current_round,
                "critic_passed": True,
                "critic_score": score,
                "critic_feedback": [],
            }

        # Build graph
        graph = StateGraph(DeepAgentState)
        graph.add_node("generator", generator_node)
        graph.add_node("critic", critic_node)
        graph.set_entry_point("generator")
        graph.add_edge("generator", "critic")
        graph.add_conditional_edges("critic", should_continue)
        return graph.compile()

    def _build_parallel_graph(
        self,
        generator_agent,
        llm: ChatOpenAI,
        output_path: Path,
        url: str,
        scrape_tool,
    ) -> StateGraph:
        """Build graph with parallel scraping via Send (Map-Reduce pattern).

        Flow:
        1. Discovery - agent runs map_domain to discover URLs
        2. Router - fans out to parallel scraper workers
        3. Scraper workers - each scrapes one URL (runs in parallel)
           - Stores full content (Phase 3: will use BaseStore)
           - Generates structured summary for state
        4. Synthesis - LLM generates output from scraped summaries
        5. Critic - validates output
        """
        max_concurrent = self.config.max_concurrent or 5
        max_content = self.config.max_content_per_doc
        log_prefix = self.log_prefix

        # Semaphore for explicit concurrency limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        # Fast LLM for structured summarization (cheaper model)
        fast_llm = ChatOpenAI(
            model=self.config.fast_model,
            temperature=self.config.fast_model_temperature,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        structured_llm = fast_llm.with_structured_output(PageSummary)

        # Bind static methods for use in nested functions
        extract_urls = self._extract_urls_from_messages
        parse_result = self._parse_scrape_result
        format_docs = self._format_scraped_docs_for_synthesis

        def should_continue(state: dict) -> str:
            current_round = state.get("current_round", 0)
            max_rounds = state.get("max_rounds", 3)

            if current_round >= max_rounds:
                logger.warning(f"{log_prefix} Max rounds ({max_rounds}) reached")
                return END

            critic_passed = state.get("critic_passed", False)
            if critic_passed:
                logger.info(f"{log_prefix} Critic APPROVED")
                return END

            return "discovery"

        async def discovery_node(state: dict) -> dict:
            """Run generator to discover URLs via map_domain."""
            messages = state.get("messages", [])
            cleaned = clean_messages(messages)
            result = await generator_agent.ainvoke({**state, "messages": cleaned})

            # Extract URLs from tool responses
            urls = extract_urls(result.get("messages", []))

            # Apply max_urls limit
            max_urls = self.config.max_urls
            if max_urls and len(urls) > max_urls:
                logger.info(f"{log_prefix} Limiting URLs from {len(urls)} to {max_urls}")
                urls = urls[:max_urls]

            logger.info(f"{log_prefix} Discovered {len(urls)} URLs for parallel scraping")

            return {
                **result,
                "discovered_urls": urls,
                "scraped_docs": [],
                "scraping_errors": [],
            }

        def router_node(state: dict) -> List[Send]:
            """Fan out to parallel scraper workers."""
            urls = state.get("discovered_urls", [])
            if not urls:
                logger.warning(f"{log_prefix} No URLs discovered, skipping scraping")
                return []

            logger.info(f"{log_prefix} Fanning out to {len(urls)} scraper workers")
            return [
                Send("scraper", {
                    "url": url,
                    "output_path": state.get("output_path", ""),
                    "max_content": max_content,
                })
                for url in urls
            ]

        async def scraper_node(state: dict, config: RunnableConfig) -> dict:
            """Scrape a single URL, store full content, return structured summary.

            Runs in parallel via Send with semaphore limiting.
            Stage 1 of two-stage pipeline: small summaries in state, full content persists.
            """
            url = state["url"]
            max_c = state.get("max_content")

            # Get store from config (if available)
            store: Optional[BaseStore] = config.get("configurable", {}).get("store")

            # Use semaphore to limit concurrent scrapers
            async with semaphore:
                try:
                    logger.debug(f"{log_prefix} Scraping: {url}")
                    result = await scrape_tool.ainvoke({"url": url})

                    # Parse result and extract content
                    full_content = parse_result(result)

                    # ACTION 1: Store FULL content in BaseStore (persists across runs)
                    if store:
                        # Use URL + content prefix for deduplication (handles content changes)
                        content_hash = hashlib.md5(f"{url}:{full_content[:1000]}".encode()).hexdigest()

                        await store.aput(
                            namespace=("memories", "raw_pages"),
                            key=content_hash,
                            value={
                                "url": url,
                                "content": full_content,
                                "scraped_at": datetime.now().isoformat(),
                            }
                        )
                        logger.debug(f"{log_prefix} Stored {url} in BaseStore (key={content_hash[:8]}...)")

                    # Truncate for LLM if configured (for summary generation)
                    content_for_llm = full_content
                    if max_c and len(full_content) > max_c:
                        content_for_llm = full_content[:max_c]
                        logger.debug(f"{log_prefix} Truncated {url} to {max_c} chars for summary")

                    # ACTION 2: Generate structured summary (tiny footprint for state)
                    summary: PageSummary = await structured_llm.ainvoke(
                        f"Analyze this documentation page.\n\n"
                        f"URL: {url}\n\n"
                        f"Content:\n{content_for_llm[:15000]}"
                    )

                    # Ensure URL is set correctly (in case LLM makes mistakes)
                    summary.url = url

                    logger.info(f"{log_prefix} Scraped {url} → {summary.title} ({len(full_content)} chars)")
                    return {"scraped_docs": [summary.model_dump()]}

                except Exception as e:
                    logger.error(f"{log_prefix} Failed to scrape {url}: {e}")
                    return {"scraping_errors": [f"{url}: {str(e)}"]}

        async def synthesis_node(state: dict) -> dict:
            """Generate final output from scraped docs."""
            docs = state.get("scraped_docs", [])
            errors = state.get("scraping_errors", [])

            logger.info(f"{log_prefix} Synthesis: {len(docs)} docs, {len(errors)} errors")

            # Format scraped content for LLM
            context = format_docs(docs, errors)

            # Create synthesis prompt
            synthesis_message = self._format_synthesis_message(context, output_path)

            # Run generator with synthesis context
            messages = state.get("messages", [])
            messages = list(messages) + [{"role": "user", "content": synthesis_message}]
            messages = clean_messages(messages)

            result = await generator_agent.ainvoke({**state, "messages": messages})
            return result

        async def critic_node(state: dict) -> dict:
            """Run critic on the generated output."""
            current_round = state.get("current_round", 0) + 1

            passed, score, feedback = await self._run_critic(
                state, output_path, url, current_round
            )

            logger.info(
                f"{log_prefix} Critic round {current_round}: "
                f"passed={passed}, score={score:.2f}, issues={len(feedback)}"
            )

            if not passed:
                feedback_msg = self._format_feedback_message(feedback)
                messages = state.get("messages", [])
                messages = list(messages) + [{"role": "user", "content": feedback_msg}]
                messages = clean_messages(messages)

                return {
                    **state,
                    "messages": messages,
                    "current_round": current_round,
                    "critic_passed": False,
                    "critic_score": score,
                    "critic_feedback": feedback,
                    # Reset for next round
                    "discovered_urls": [],
                    "scraped_docs": [],
                    "scraping_errors": [],
                }

            return {
                **state,
                "current_round": current_round,
                "critic_passed": True,
                "critic_score": score,
                "critic_feedback": [],
            }

        # Build graph with specialized state for web scraping
        graph = StateGraph(WebScrapingState)
        graph.add_node("discovery", discovery_node)
        graph.add_node("router", router_node)
        graph.add_node("scraper", scraper_node)
        graph.add_node("synthesis", synthesis_node)
        graph.add_node("critic", critic_node)

        # Wire up the flow
        graph.set_entry_point("discovery")
        # If URLs discovered, go to router (which fans out), else skip to synthesis
        graph.add_conditional_edges(
            "discovery",
            lambda s: "router" if s.get("discovered_urls") else "synthesis"
        )
        # router fans out to scrapers via Send (handled automatically by returning List[Send])
        graph.add_edge("scraper", "synthesis")  # All scrapers converge to synthesis
        graph.add_edge("synthesis", "critic")
        graph.add_conditional_edges("critic", should_continue)

        return graph.compile()

    def _format_synthesis_message(self, context: str, output_path: Path) -> str:
        """Format the synthesis prompt. Override in subclasses for customization."""
        return f"""Generate the output file based on the scraped page summaries below.

Output file: {output_path}

Scraped page summaries:
---
{context}
---

Use write_file to create the output file with the appropriate content."""

    # ==============================================================================
    # Helper functions for parallel scraping
    # ==============================================================================

    @staticmethod
    def _extract_urls_from_messages(messages: List[Any]) -> List[str]:
        """Extract URLs from tool responses (map_domain results)."""
        urls = []
        for msg in messages:
            # Handle dict format
            if isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = getattr(msg, "content", "")

            if not content:
                continue

            # Try to parse as JSON (map_domain returns JSON)
            try:
                if isinstance(content, str) and content.startswith("["):
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "url" in item:
                                urls.append(item["url"])
                            elif isinstance(item, str) and item.startswith("http"):
                                urls.append(item)
                elif isinstance(content, str) and '"urls"' in content:
                    # Handle nested format: {"urls": [...]}
                    data = json.loads(content)
                    if "urls" in data:
                        for item in data["urls"]:
                            if isinstance(item, dict) and "url" in item:
                                urls.append(item["url"])
            except (json.JSONDecodeError, TypeError):
                # Fallback: regex for URLs
                found = re.findall(r'https?://[^\s"<>\]]+', content)
                urls.extend(found)

        # Dedupe while preserving order
        seen = set()
        unique = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique

    @staticmethod
    def _parse_scrape_result(result: Any) -> str:
        """Parse scrape tool result to extract content."""
        if isinstance(result, str):
            # Try to parse JSON
            try:
                data = json.loads(result)
                if isinstance(data, dict):
                    return data.get("content", "") or data.get("markdown", "") or result
            except json.JSONDecodeError:
                return result
        elif isinstance(result, dict):
            return result.get("content", "") or result.get("markdown", "") or str(result)
        elif isinstance(result, list) and result:
            # Handle list format from MCP tools
            first = result[0]
            if isinstance(first, dict) and "text" in first:
                text = first["text"]
                try:
                    data = json.loads(text)
                    return data.get("content", "") or data.get("markdown", "") or text
                except json.JSONDecodeError:
                    return text
        return str(result)

    @staticmethod
    def _format_scraped_docs_for_synthesis(docs: List[Dict], errors: List[str]) -> str:
        """Format scraped docs for the synthesis LLM.

        Works with PageSummary format (url, title, description, key_topics).
        This creates a compact representation for synthesis without context explosion.
        """
        parts = []

        if docs:
            parts.append(f"=== {len(docs)} Scraped Pages (Summaries) ===\n")
            for i, doc in enumerate(docs, 1):
                url = doc.get("url", "unknown")
                title = doc.get("title", "Untitled")
                description = doc.get("description", "No description")
                topics = doc.get("key_topics", [])
                topics_str = ", ".join(topics) if topics else "N/A"
                parts.append(
                    f"--- Page {i}: {title} ---\n"
                    f"URL: {url}\n"
                    f"Description: {description}\n"
                    f"Topics: {topics_str}\n"
                )

        if errors:
            parts.append(f"\n=== {len(errors)} Failed Pages ===\n")
            for err in errors:
                parts.append(f"- {err}\n")

        return "\n".join(parts)

    async def _run_critic(
        self,
        state: dict,
        output_path: Path,
        url: str,
        round_num: int
    ) -> tuple[bool, float, List[str]]:
        """Run critic validation. Override in subclasses for different critic types.

        Returns:
            Tuple of (passed, score, feedback_list)
        """
        # Default implementation: check if file exists
        if output_path.exists():
            return True, 1.0, []
        else:
            return False, 0.0, [f"Output file not created at {output_path}"]

    def _create_result(self, output_path: Path, result: dict) -> GeneratorResult:
        """Create GeneratorResult from graph output."""
        if output_path.exists():
            content = output_path.read_text() if output_path.is_file() else ""
            logger.info(f"{self.log_prefix} Generated output ({len(content)} chars)")
            return GeneratorResult(
                output_path=output_path,
                stats={
                    "file_size": len(content),
                    "critic_passed": result.get("critic_passed", False),
                    "critic_score": result.get("critic_score", 0.0),
                    "rounds": result.get("current_round", 0),
                },
            )
        else:
            raise RuntimeError(f"Generation failed: file not created at {output_path}")
