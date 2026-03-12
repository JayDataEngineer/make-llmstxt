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

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.store.base import BaseStore
from deepagents.graph import create_agent, FilesystemMiddleware
from deepagents.backends import FilesystemBackend, CompositeBackend
from ..backends.docs_backend import DocsBackend
from deepagents.middleware.subagents import SubAgent
from ..utils.logging import StructuredLogger
from ..utils.observability import init_langfuse, flush_langfuse, create_session_id, session_context

log = StructuredLogger("agent")

from ..core import GeneratorConfig, GeneratorResult, AgentPrompts
from ..scrapers import (
    get_mcp_tools,
    filter_tools_by_name,
    MAIN_AGENT_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
)
from ..store import create_store
from ..tools import search_docs, get_doc_by_url
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
    """Merge consecutive assistant messages and ensure valid message order.

    Some LLM APIs (like Qwen/llama.cpp) don't allow:
    - Consecutive assistant messages
    - Messages ending with multiple assistant messages

    This function merges consecutive assistant messages by concatenating content.
    Also handles LangChain AIMessage, HumanMessage, etc. objects.
    """
    if not messages:
        return messages

    def get_role(msg) -> str:
        """Extract role from dict or LangChain message object."""
        if isinstance(msg, dict):
            return msg.get("role", "unknown")
        # LangChain message types
        msg_type = getattr(msg, "type", None)
        if msg_type:
            # Map LangChain types to roles
            type_to_role = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
                "tool": "tool",
            }
            return type_to_role.get(msg_type, msg_type)
        return getattr(msg, "role", "unknown")

    def get_content(msg) -> str:
        """Extract content from dict or LangChain message object."""
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
        return getattr(msg, "content", "") or ""

    def set_content(msg, content: str):
        """Set content on dict or LangChain message object."""
        if isinstance(msg, dict):
            msg["content"] = content
        else:
            msg.content = content

    cleaned = []
    for msg in messages:
        msg_role = get_role(msg)
        msg_content = get_content(msg)

        # Check if we need to merge with previous message
        if cleaned:
            prev_msg = cleaned[-1]
            prev_role = get_role(prev_msg)

            # Merge consecutive assistant messages
            if prev_role == "assistant" and msg_role == "assistant":
                prev_content = get_content(prev_msg)
                set_content(prev_msg, f"{prev_content}\n{msg_content}")
                log.debug("Merged consecutive assistant messages")
                continue

            # Also merge consecutive user messages (some APIs don't like this either)
            if prev_role == "user" and msg_role == "user":
                prev_content = get_content(prev_msg)
                set_content(prev_msg, f"{prev_content}\n\n{msg_content}")
                log.debug("Merged consecutive user messages")
                continue

        cleaned.append(msg)

    # Final pass: ensure we don't end with multiple assistant messages
    # (should already be handled by merge above, but double-check)
    while len(cleaned) >= 2:
        if get_role(cleaned[-1]) == "assistant" and get_role(cleaned[-2]) == "assistant":
            # Merge last two messages
            last = cleaned.pop()
            second_last = cleaned[-1]
            set_content(second_last, f"{get_content(second_last)}\n{get_content(last)}")
            log.debug("Merged trailing assistant messages")
        else:
            break

    return cleaned


def estimate_tokens(messages: List[Any]) -> int:
    """Estimate token count for messages (rough: 4 chars per token)."""
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "")
        total += len(str(content)) // 4 + 10  # +10 for message metadata overhead
    return total


def trim_messages(messages: List[Any], max_tokens: int = 24000) -> List[Any]:
    """Trim old messages to keep within token budget.

    Always keeps:
    - First user message (the task)
    - Most recent messages (the active context)

    Removes middle messages if needed.
    """
    if not messages:
        return messages

    current_tokens = estimate_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    log.debug("Trimming messages", before=current_tokens, after=max_tokens)

    # Always keep first user message (the task)
    first_msg = messages[0]
    remaining = messages[1:]

    # Keep most recent messages that fit
    trimmed = [first_msg]
    recent = []

    for msg in reversed(remaining):
        recent.insert(0, msg)
        test_messages = trimmed + recent
        if estimate_tokens(test_messages) > max_tokens:
            recent.pop(0)  # Remove the one that pushed us over
            break

    result = trimmed + recent
    log.debug("Messages trimmed", before=len(messages), after=len(result), tokens=estimate_tokens(result))
    return result


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
        default_prompts: Optional[AgentPrompts] = None,
    ):
        # Set prompts if not already set
        if config.prompts is None and default_prompts is not None:
            config = config.model_copy(update={"prompts": default_prompts})

        self.config = config
        self.log = log.bind(url=url if (url := config.url) else "unknown")

        # Ensure prompts are set
        if config.prompts is None:
            raise ValueError("config.prompts must be set")

    def _create_llm(self, enable_thinking: Optional[bool] = None, model: Optional[str] = None, temperature: Optional[float] = None) -> ChatOpenAI:
        """Create LLM instance with optional reasoning/thinking mode.

        Different providers/models use different parameters to control thinking:
        - llama.cpp (Qwen): extra_body.chat_template_kwargs.enable_thinking
        - llama.cpp (DeepSeek-R1): reasoning_effort
        - OpenAI o1/o3: reasoning_effort

        Args:
            enable_thinking: Enable thinking/reasoning mode.
                             Defaults to config.enable_thinking if not specified.
            model: Model name to use. Defaults to config.model.
            temperature: Temperature to use. Defaults to config.temperature.

        Returns:
            ChatOpenAI instance configured with reasoning mode
        """
        thinking = enable_thinking if enable_thinking is not None else self.config.enable_thinking
        model_name = model or self.config.model
        temp = temperature if temperature is not None else self.config.temperature

        # Build provider-specific reasoning parameters
        extra_kwargs: Dict[str, Any] = {}
        model_lower = model_name.lower()

        # Qwen models via llama.cpp use enable_thinking in chat-template-kwargs
        if "qwen" in model_lower:
            extra_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": thinking
                }
            }
        # DeepSeek-R1, o1, o3 use reasoning_effort
        elif any(x in model_lower for x in ["deepseek-r1", "deepseek-reasoner", "o1-", "o3-"]):
            extra_kwargs["reasoning_effort"] = "high" if thinking else "low"
        # Other models: no thinking parameter needed

        # Get Langfuse callback handler for automatic tracing
        from ..utils.observability import get_langfuse_callback
        callbacks = get_langfuse_callback()

        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            callbacks=callbacks if callbacks else None,
            **extra_kwargs,
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
        output_is_dir: bool = False,
    ) -> GeneratorResult:
        """Run the Deep Agent generation.

        Args:
            url: Target URL to process
            output_path: Output file/directory path (defaults to config.output_dir)
            progress_callback: Optional callback(message, percent)
            output_is_dir: If True, output_path is a directory (for skill packages).
                          If False (default), output_path is a file (for llms.txt).

        Returns:
            GeneratorResult with output path and stats
        """
        # Initialize Langfuse for automatic LLM tracing
        init_langfuse()

        # Create session ID to group all traces for this run
        session_id = create_session_id()
        self.log.info("Session created", session_id=session_id)

        output_path = Path(output_path or self.config.output_dir)
        if output_is_dir:
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log.bind(url=url)

        self.log.start("generation", output_path=str(output_path), session_id=session_id)

        if progress_callback:
            progress_callback("Connecting to MCP server...", 5)

        # Create store FIRST so we can add search tools
        store = create_store(
            embedding_base_url=self.config.embedding_base_url,
            embedding_model=self.config.embedding_model,
            embedding_dims=self.config.embedding_dims,
        )
        if store:
            self.log.info("Store created", purpose="content_persistence")

        # Connect to MCP and get tools
        async with get_mcp_tools(
            self.config.mcp_host,
            self.config.mcp_port,
            url=self.config.mcp_url,
            max_urls=self.config.max_urls,
        ) as all_tools:
            main_tools = filter_tools_by_name(all_tools, MAIN_AGENT_TOOL_NAMES)
            subagent_tools = filter_tools_by_name(all_tools, SUBAGENT_TOOL_NAMES)

            # Add store search tools if store is configured
            if store:
                main_tools = list(main_tools) + [search_docs, get_doc_by_url]
                self.log.info("Store search tools added")

            self.log.debug("Tools configured", main=[t.name for t in main_tools], subagent=[t.name for t in subagent_tools])

            if progress_callback:
                progress_callback("Building agents...", 10)

            # Create LLM
            llm = self._create_llm()
            self.log.info("LLM initialized", model=self.config.model)

            # Create subagent
            subagent = self._create_subagent(subagent_tools)
            subagents = [subagent] if subagent else []

            # Format system prompt with context
            system_prompt = self._format_system_prompt(url, output_path)

            # Create backends:
            # - FilesystemBackend for writing output files
            # - DocsBackend for reading scraped content via semantic search
            # Use virtual_mode=True so LLM paths like "/llms.txt" resolve to root_dir
            # For directory output (skills), root_dir is output_path itself
            # For file output (llms.txt), root_dir is output_path.parent
            root_dir = output_path if output_is_dir else output_path.parent
            fs_backend = FilesystemBackend(
                root_dir=str(root_dir),
                virtual_mode=True
            )
            docs_backend = DocsBackend(store=store)
            backend = CompositeBackend(
                default=fs_backend,
                routes={"/docs/": docs_backend}
            )
            self.log.info("CompositeBackend created", routes=["/docs/ -> semantic_search"])

            # Create generator agent with filesystem middleware for backend access
            generator_agent = create_agent(
                model=llm,
                tools=main_tools,
                system_prompt=system_prompt,
                middleware=[
                    FilesystemMiddleware(backend=backend),
                ],
            )

            # Create synthesis agent with ONLY filesystem tools (no scraping)
            # This prevents synthesis from calling map_domain/scrape_url
            # Use virtual path format since fs_backend uses virtual_mode=True
            # For directory output (skills), use "/" since root_dir is already the target directory
            # For file output (llms.txt), use the filename
            virtual_output_path = "/" if output_is_dir else f"/{output_path.name}"
            synthesis_agent = create_agent(
                model=llm,
                tools=[],  # No scraping tools - only filesystem middleware
                system_prompt=f"""You are an output file generator. Your ONLY job is to write the output file.

You will receive page summaries and must write a properly formatted output file.
Use the write_file tool to create the output file.

Output path: {virtual_output_path}
Target URL: {url}

DO NOT use any scraping tools. Just write the output file based on the summaries provided.""",
                middleware=[
                    FilesystemMiddleware(backend=fs_backend),  # Only filesystem, no docs backend
                ],
            )

            if progress_callback:
                progress_callback("Running generator → critic loop...", 20)

            # Build the appropriate graph based on config
            if self.config.enable_parallel:
                self.log.info("Using PARALLEL scraping", max_concurrent=self.config.max_concurrent)
                # Get tools for parallel workers
                scrape_tool = filter_tools_by_name(all_tools, ["scrape_url"])[0] if filter_tools_by_name(all_tools, ["scrape_url"]) else None
                map_domain_tool = filter_tools_by_name(all_tools, ["map_domain"])[0] if filter_tools_by_name(all_tools, ["map_domain"]) else None
                if scrape_tool and map_domain_tool:
                    graph = self._build_parallel_graph(generator_agent, synthesis_agent, llm, output_path, url, scrape_tool, map_domain_tool)
                else:
                    self.log.warning("Required tools not found, falling back to serial mode")
                    graph = self._build_graph(generator_agent, llm, output_path, url)
            else:
                self.log.info("Using SERIAL scraping", mode="conversational")
                graph = self._build_graph(generator_agent, llm, output_path, url)

            # Run the graph
            initial_message = self._format_initial_message(url, output_path)
            self.log.start("generator_critic_loop")

            # Build config with store (if available)
            invoke_config = {
                "configurable": {},
            }
            if store:
                invoke_config["configurable"]["store"] = store
                self.log.info("Store enabled for content persistence")

            # Wrap invocation with session propagation for Langfuse
            try:
                from langfuse import propagate_attributes
                with propagate_attributes(session_id=session_id):
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
            except ImportError:
                # Fallback if propagate_attributes not available
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
            except Exception as e:
                # If generation failed but file exists, use it (best effort)
                if output_path.exists() and (output_path.is_file() or output_path.is_dir()):
                    self.log.warning(
                        "best_effort_output",
                        f"Generation failed but output exists - using best effort (error: {e})",
                        output_path=str(output_path)
                    )
                    # Create a result indicating partial success
                    return GeneratorResult(
                        output_path=output_path,
                        stats={
                            "file_size": output_path.read_text().__len__() if output_path.is_file() else 0,
                            "critic_passed": False,
                            "critic_score": 0.0,
                            "rounds": self.config.max_rounds,
                            "best_effort": True,
                            "error": str(e),
                        },
                    )
                # No output file - re-raise the exception
                raise

            if progress_callback:
                progress_callback("Generation complete", 100)

            # Flush observability events before returning
            flush_langfuse()
            self.log.info("Traces flushed to Langfuse", session_id=session_id)

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
            self.log.warning("Missing context key in prompt", key=str(e))
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
            self.log.warning("Missing context key in initial message", key=str(e))
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
                self.log.warning("Max rounds reached", max_rounds=max_rounds)
                return END

            critic_passed = state.get("critic_passed", False)
            if critic_passed:
                self.log.info("Critic approved")
                return END

            return "generator"

        async def generator_node(state: dict) -> dict:
            """Run generator with message cleaning and trimming for context management."""
            messages = state.get("messages", [])
            # First clean (merge consecutive messages)
            cleaned = clean_messages(messages)
            # Then trim to token budget to prevent context overflow
            trimmed = trim_messages(cleaned, max_tokens=24000)
            if len(trimmed) != len(messages):
                self.log.debug("Messages managed", original=len(messages), cleaned=len(cleaned), trimmed=len(trimmed))
            result = await generator_agent.ainvoke({**state, "messages": trimmed})
            return result

        async def critic_node(state: dict) -> dict:
            """Run critic on the generated output."""
            current_round = state.get("current_round", 0) + 1

            # Check if output exists and validate
            passed, score, feedback = await self._run_critic(
                state, output_path, url, current_round
            )

            self.log.info(
                "Critic evaluation",
                round=current_round,
                passed=passed,
                score=f"{score:.2f}",
                issues=len(feedback)
            )

            if not passed:
                # Add feedback message for generator
                feedback_msg = self._format_feedback_message(feedback)
                messages = state.get("messages", [])
                messages = list(messages) + [{"role": "user", "content": feedback_msg}]
                messages = clean_messages(messages)
                # Trim to prevent context overflow across rounds
                messages = trim_messages(messages, max_tokens=24000)

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
        synthesis_agent,
        llm: ChatOpenAI,
        output_path: Path,
        url: str,
        scrape_tool,
        map_domain_tool=None,
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
        scoped_log = self.log

        # Semaphore for explicit concurrency limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        # Fast LLM for structured summarization (use main model if fast_model not set)
        # Thinking DISABLED for subagents - they do simple summarization
        fast_model_name = self.config.fast_model or self.config.model
        fast_llm = self._create_llm(
            enable_thinking=False,
            model=fast_model_name,
            temperature=self.config.fast_model_temperature
        )
        structured_llm = fast_llm.with_structured_output(PageSummary)

        # Create discovery agent with ONLY map_domain tool
        # This prevents the agent from scraping during discovery
        discovery_tools = [map_domain_tool] if map_domain_tool else []
        discovery_agent = create_agent(
            model=llm,
            tools=discovery_tools,
            system_prompt=f"""You are a URL discovery agent. Your ONLY job is to discover URLs on a website.

Use the map_domain tool to discover all URLs on the target website.

Return the list of discovered URLs. Do NOT scrape any pages - just discover them.

Target website: {url}""",
        )

        # Bind static methods for use in nested functions
        extract_urls = self._extract_urls_from_messages
        parse_result = self._parse_scrape_result
        format_docs = self._format_scraped_docs_for_synthesis

        def should_continue(state: dict) -> str:
            current_round = state.get("current_round", 0)
            max_rounds = state.get("max_rounds", 3)

            if current_round >= max_rounds:
                scoped_log.warning("Max rounds reached", max_rounds=max_rounds)
                return END

            critic_passed = state.get("critic_passed", False)
            if critic_passed:
                scoped_log.info("Critic approved")
                return END

            return "discovery"

        async def discovery_node(state: dict) -> dict:
            """Run discovery agent to discover URLs via map_domain ONLY."""
            messages = state.get("messages", [])

            # Simple message for discovery
            discovery_message = f"Discover all URLs on {url} using map_domain."

            result = await discovery_agent.ainvoke({
                "messages": [{"role": "user", "content": discovery_message}],
            })

            # Extract URLs from tool responses
            urls = extract_urls(result.get("messages", []))

            # Apply max_urls limit
            max_urls = self.config.max_urls
            if max_urls and len(urls) > max_urls:
                scoped_log.info("URLs limited", before=len(urls), after=max_urls)
                urls = urls[:max_urls]

            scoped_log.info("URLs discovered", count=len(urls), parallel_scraping=True)

            return {
                **result,
                "discovered_urls": urls,
                "scraped_docs": [],
                "scraping_errors": [],
            }

        async def scraper_node(state: dict, config: RunnableConfig) -> dict:
            """Scrape a single URL, store full content, return structured summary.

            Runs in parallel via Send with semaphore limiting.
            Full content is stored with auto-generated embeddings for semantic search.
            """
            url = state["url"]
            max_c = state.get("max_content")

            # Get store from config (if available)
            store = config.get("configurable", {}).get("store")

            # Use semaphore to limit concurrent scrapers
            async with semaphore:
                try:
                    scoped_log.debug("Scraping URL", url=url)
                    result = await scrape_tool.ainvoke({"url": url})

                    # Parse result and extract content
                    full_content = parse_result(result)

                    # Store full content with auto-generated embeddings
                    if store:
                        # Normalize content for consistent hashing (whitespace-insensitive)
                        normalized = " ".join(full_content.split()).lower()
                        content_hash = hashlib.md5(f"{url}:{normalized}".encode()).hexdigest()

                        await store.aput(
                            namespace=("memories", "raw_pages"),
                            key=content_hash,
                            value={
                                "url": url,
                                "content": full_content,
                                "scraped_at": datetime.now().isoformat(),
                            },
                            index=None,  # Auto-embed using store's embedding config
                        )
                        scoped_log.debug("Content stored", url=url, hash=content_hash[:8], embedded=True)

                    # Truncate for LLM if configured (for summary generation)
                    content_for_llm = full_content
                    if max_c and len(full_content) > max_c:
                        content_for_llm = full_content[:max_c]
                        scoped_log.debug("Content truncated", url=url, max_chars=max_c)

                    # ACTION 2: Generate structured summary (tiny footprint for state)
                    summary: PageSummary = await structured_llm.ainvoke(
                        f"Analyze this documentation page.\n\n"
                        f"URL: {url}\n\n"
                        f"Content:\n{content_for_llm[:15000]}"
                    )

                    # Ensure URL is set correctly (in case LLM makes mistakes)
                    summary.url = url

                    scoped_log.info("Page scraped", url=url, title=summary.title, content_chars=len(full_content))
                    return {"scraped_docs": [summary.model_dump()]}

                except Exception as e:
                    scoped_log.error("Scrape failed", url=url, error=str(e))
                    return {"scraping_errors": [f"{url}: {str(e)}"]}

        async def synthesis_node(state: dict) -> dict:
            """Generate final output from scraped docs."""
            docs = state.get("scraped_docs", [])
            errors = state.get("scraping_errors", [])

            scoped_log.info("Synthesis starting", docs_count=len(docs), errors_count=len(errors))

            # Format scraped content for LLM (compact format)
            context = format_docs(docs, errors)

            # Create synthesis prompt (no URL list - summaries already have URLs)
            synthesis_message = self._format_synthesis_message(context, output_path, url)

            # Run synthesis agent (only has write_file tool, no scraping)
            # Start fresh - only keep synthesis message to avoid context overflow
            messages = [{"role": "user", "content": synthesis_message}]
            messages = clean_messages(messages)

            result = await synthesis_agent.ainvoke({**state, "messages": messages})
            return result

        async def critic_node(state: dict) -> dict:
            """Run critic on the generated output."""
            current_round = state.get("current_round", 0) + 1

            passed, score, feedback = await self._run_critic(
                state, output_path, url, current_round
            )

            scoped_log.info(
                "Critic evaluation",
                round=current_round,
                passed=passed,
                score=f"{score:.2f}",
                issues=len(feedback)
            )

            if not passed:
                feedback_msg = self._format_feedback_message(feedback)
                messages = state.get("messages", [])
                messages = list(messages) + [{"role": "user", "content": feedback_msg}]
                messages = clean_messages(messages)
                # Trim to prevent context overflow across rounds
                messages = trim_messages(messages, max_tokens=24000)

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

        def route_to_scrapers(state: dict):
            """Conditional edge that fans out to parallel scrapers via Send."""
            urls = state.get("discovered_urls", [])
            if not urls:
                scoped_log.warning("No URLs discovered, going to synthesis")
                return "synthesis"

            scoped_log.info("Fanning out to scrapers", worker_count=len(urls))
            return [
                Send("scraper", {
                    "url": url,
                    "output_path": state.get("output_path", ""),
                    "max_content": max_content,
                })
                for url in urls
            ]

        # Build graph with specialized state for web scraping
        graph = StateGraph(WebScrapingState)
        graph.add_node("discovery", discovery_node)
        graph.add_node("scraper", scraper_node)
        graph.add_node("synthesis", synthesis_node)
        graph.add_node("critic", critic_node)

        # Wire up the flow
        graph.set_entry_point("discovery")
        # Conditional edge: fan out to scrapers or go directly to synthesis
        graph.add_conditional_edges("discovery", route_to_scrapers, ["synthesis"])
        # All scrapers converge to synthesis
        graph.add_edge("scraper", "synthesis")
        graph.add_edge("synthesis", "critic")
        graph.add_conditional_edges("critic", should_continue)

        return graph.compile()

    def _format_synthesis_message(self, context: str, output_path: Path, url: str = None) -> str:
        """Format the synthesis prompt. Override in subclasses for customization."""
        # FilesystemBackend uses virtual_mode=True with root_dir=output_path.parent
        # Use virtual path format: /filename resolves to root_dir/filename
        virtual_path = f"/{output_path.name}"
        return f"""Generate the llms.txt file for {url}.

Write to: {virtual_path}

## llms.txt Format

Use this exact format:

```
# Project Name

> One-line summary of the project.

## Section Name

- [Specific Title](URL): Informative description ending with a period.
- [Another Title](URL): Another clear description here.

## Optional

- [Non-essential Link](URL): Optional resource description.
```

## Format Rules

1. **H1 Header**: Project NAME (not URL) at top
2. **Blockquote Summary**: One-line summary after H1 using `>`
3. **Links**: Use `- [Title](URL): Description.` format
4. **Titles**: 2-5 words, specific (NOT "Home", "Page", "About")
5. **Descriptions**: 5-12 words, informative, end with period
6. **No Placeholders**: Never use "No description" or "N/A"
7. **Sections**: Group URLs under H2 headers (`##`)
8. **Optional Section**: Put blogs, changelogs under `## Optional`

Page summaries to include:
{context}

Use write_file to create the output file with ALL pages included."""

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

            # Handle MCP tool response format: [{'type': 'text', 'text': '...'}]
            if isinstance(content, list) and content and isinstance(content[0], dict):
                if "text" in content[0]:
                    content = content[0]["text"]

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

        Compact format to avoid context explosion.
        """
        lines = []

        for doc in docs:
            url = doc.get("url", "unknown")
            title = doc.get("title", "Untitled")
            description = doc.get("description", "No description")
            # One line per doc: URL | Title | Description
            lines.append(f"{url} | {title} | {description}")

        if errors:
            lines.append(f"\nFailed: {len(errors)} pages")

        return "\n".join(lines)

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
        """Create GeneratorResult from graph output.

        Returns a result even if generation failed (critic didn't pass).
        Check result.stats['critic_passed'] to see if generation succeeded.
        """
        critic_passed = result.get("critic_passed", False)
        critic_score = result.get("critic_score", 0.0)
        rounds = result.get("current_round", 0)

        if output_path.exists():
            content = output_path.read_text() if output_path.is_file() else ""
            self.log.info("Output generated", chars=len(content), critic_passed=critic_passed)
            return GeneratorResult(
                output_path=output_path,
                stats={
                    "file_size": len(content),
                    "critic_passed": critic_passed,
                    "critic_score": critic_score,
                    "rounds": rounds,
                },
            )
        else:
            # File not created - generation failed
            self.log.warning(
                "Output file not created",
                path=str(output_path),
                critic_passed=critic_passed,
                rounds=rounds
            )
            # Return result with failure stats instead of raising
            return GeneratorResult(
                output_path=output_path,
                stats={
                    "file_size": 0,
                    "critic_passed": False,
                    "critic_score": critic_score,
                    "rounds": rounds,
                    "error": "Output file not created",
                },
            )
