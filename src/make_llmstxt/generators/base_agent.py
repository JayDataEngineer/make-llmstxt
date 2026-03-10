"""Shared infrastructure for Deep Agent pattern.

Both llms.txt and skill generators use this same infrastructure,
with different prompts injected via configuration.
"""

import time
from typing import List, Dict, Any, TypedDict, Optional, Callable
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
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
    """Shared state for Deep Agent graph.

    Used by both llms.txt and skill generators.
    """
    messages: List[Any]
    url: str
    output_path: str
    current_round: int
    max_rounds: int
    critic_passed: bool
    critic_score: float
    critic_feedback: List[str]
    # Additional context (llms.txt content for skill generation)
    context: Dict[str, Any]


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

            # Build the graph
            graph = self._build_graph(generator_agent, llm, output_path, url)

            # Run the graph
            initial_message = self._format_initial_message(url, output_path)
            logger.info(f"{self.log_prefix} Running generator → critic loop...")

            logging_handler = DeepAgentLoggingHandler(self.log_prefix)

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
                },
                config={"callbacks": [logging_handler]}
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
