"""Skill generator using llms.txt + Deep Agent pattern.

Architecture:
1. Generate llms.txt from the URL (condensed documentation summary)
2. Generator (Deep Agent): Uses llms.txt as reference to create skill package
3. Critic (Deep Agent): Uses llms.txt as ground truth to validate coverage

The generator creates:
- SKILL.md (with YAML frontmatter)
- references/ (key documentation files)
- scripts/ (code examples)

MCP tools are loaded via langchain-mcp-adapters.
"""

import time
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from urllib.parse import urlparse

from loguru import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langgraph.graph import StateGraph, MessagesState, END
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI

from ..core import GeneratorConfig, GeneratorResult
from ..config import AppConfig
from .llmstxt import LLMsTxtGenerator
from ..scrapers import (
    get_mcp_tools,
    filter_tools_by_name,
    MAIN_AGENT_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
)
from deepagents.middleware.subagents import SubAgent


# ==============================================================================
# Custom Logging Callback Handler
# ==============================================================================

class SkillGeneratorLoggingHandler(BaseCallbackHandler):
    """Custom callback handler for detailed skill generation logging.

    Tracks:
    - Chain/agent start/end with timing
    - LLM calls and token usage
    - Tool invocations
    - Errors
    """

    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.chain_depth: int = 0

    def _indent(self) -> str:
        return "  " * self.chain_depth

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times[run_id] = time.time()

        name = "unnamed"
        if serialized and isinstance(serialized, dict):
            name = serialized.get("name", kwargs.get("name", "unnamed"))
        elif kwargs.get("name"):
            name = kwargs["name"]

        self.chain_depth += 1
        input_preview = self._preview_inputs(inputs)
        logger.info(f"{self._indent()}┌─ CHAIN START: {name}")
        logger.info(f"{self._indent()}│  inputs: {input_preview}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        duration = time.time() - self.start_times.pop(run_id, time.time())
        output_preview = self._preview_outputs(outputs)
        logger.info(f"{self._indent()}└─ CHAIN END ({duration:.2f}s): {output_preview}")
        self.chain_depth = max(0, self.chain_depth - 1)

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times.pop(run_id, None)
        logger.error(f"{self._indent()}💥 CHAIN ERROR: {error}")
        self.chain_depth = max(0, self.chain_depth - 1)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ) -> None:
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

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times[run_id] = time.time()
        tool_name = "unknown"
        if serialized and isinstance(serialized, dict):
            tool_name = serialized.get("name", kwargs.get("name", "unknown"))
        elif kwargs.get("name"):
            tool_name = kwargs["name"]
        logger.info(f"{self._indent()}🔧 TOOL START: {tool_name}")

    def on_tool_end(self, output: Any, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        duration = time.time() - self.start_times.pop(run_id, time.time())
        output_str = self._preview_tool_output(output)
        logger.info(f"{self._indent()}✅ TOOL END ({duration:.2f}s): {output_str}")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", "unknown"))
        self.start_times.pop(run_id, None)
        logger.error(f"{self._indent()}💥 TOOL ERROR: {error}")

    def _preview_inputs(self, inputs: Any) -> str:
        if not inputs:
            return "empty"
        if isinstance(inputs, dict) and "messages" in inputs:
            msgs = inputs["messages"]
            if isinstance(msgs, list) and len(msgs) > 0:
                last_msg = msgs[-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                else:
                    content = getattr(last_msg, "content", str(last_msg))
                return f"[{len(msgs)} messages, last: {str(content)[:100]}...]"
        return str(inputs)[:150]

    def _preview_outputs(self, outputs: Any) -> str:
        if not outputs:
            return "empty"
        if isinstance(outputs, dict) and "messages" in outputs:
            msgs = outputs["messages"]
            if isinstance(msgs, list) and len(msgs) > 0:
                last_msg = msgs[-1]
                if hasattr(last_msg, "content"):
                    return f"[{len(msgs)} messages, last: {str(last_msg.content)[:100]}...]"
                return f"[{len(msgs)} messages]"
        return str(outputs)[:150]

    def _preview_tool_output(self, output: Any) -> str:
        if output is None:
            return "None"
        if hasattr(output, "__class__") and output.__class__.__name__ == "Command":
            return f"Command({str(output)[:100]}...)"
        if hasattr(output, "content"):
            return str(output.content)[:150] + "..."
        return str(output)[:150] + "..."


# ==============================================================================
# Prompts
# ==============================================================================

GENERATOR_PROMPT = """You are a skill package generator. You create comprehensive skill packages for AI assistants.

You have access to:
- Filesystem tools (write_file, read_file, ls, etc.)
- Web scraping tools (map_domain, crawl_site, scrape_url)

Your task is to create a complete skill package based on the llms.txt reference provided.

## llms.txt Reference
This is a condensed summary of the library documentation:

{llmstxt_content}

## Output Structure

Create in {output_dir}/{library_name}/:

1. **SKILL.md** - Main skill file with YAML frontmatter:
   ---
   name: {library_name}
   description: Clear, specific description
   version: 1.0.0
   ---

   Then comprehensive sections covering all important topics from the llms.txt.

2. **references/** - Documentation files (one per major topic)
   - Each file should cover a specific topic in depth
   - Include code examples, API references, usage patterns

3. **scripts/** - Practical code examples
   - Working, runnable examples
   - Cover common use cases
   - Include comments explaining the code

## Requirements

- Cover ALL important topics from llms.txt
- Include working code examples (not placeholder comments)
- Make it genuinely useful for an AI assistant
- Be comprehensive, not sparse

Use write_file to create each file. Scrape additional pages if you need more detail."""

CRITIC_PROMPT = """You are a skill package critic. You validate skill packages against the llms.txt reference.

You have access to filesystem tools (read_file, ls, grep) to inspect the generated files.

## llms.txt Reference (Ground Truth)

{llmstxt_content}

## Your Task

1. Read the generated SKILL.md and all files in references/ and scripts/
2. Compare coverage against the llms.txt reference
3. Check for:
   - All major topics from llms.txt are covered
   - Code examples are complete (not placeholders)
   - Content is substantive, not sparse
   - YAML frontmatter is correct

## Response Format

If APPROVED:
- Say "APPROVE" and briefly explain why it's good

If NEEDS WORK:
- List specific missing topics
- List specific improvements needed
- Be constructive and specific

Do NOT approve sparse or incomplete packages. Quality matters."""

# ==============================================================================
# Skill Generator
# ==============================================================================

class SkillGenerator:
    """Generate skill packages using llms.txt + Deep Agent pattern.

    Flow:
    1. Generate llms.txt from URL (condensed summary)
    2. Generator Deep Agent creates skill package using llms.txt as reference
    3. Critic Deep Agent validates against llms.txt

    Output:
    - {library_name}/SKILL.md
    - {library_name}/references/*.md
    - {library_name}/scripts/*.*
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config

    def _extract_name(self, url: str) -> str:
        """Extract library name from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        if domain.startswith("docs."):
            domain = domain[5:]
        return domain.split(".")[0]

    async def _generate_llmstxt(
        self,
        url: str,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """Generate llms.txt for the URL.

        Returns the llms.txt content (not full text).
        """
        if progress_callback:
            progress_callback("Generating llms.txt reference...", 10)

        # Create AppConfig from GeneratorConfig
        app_config = AppConfig(
            llm={
                "model": self.config.model,
                "api_key": self.config.api_key,
                "base_url": self.config.base_url,
                "temperature": self.config.temperature,
                "provider": self.config.provider,
            },
            mcp={
                "host": self.config.mcp_host,
                "port": self.config.mcp_port,
            },
            max_urls=self.config.max_urls,
        )

        generator = LLMsTxtGenerator(
            config=app_config,
            pass_threshold=self.config.pass_threshold,
            max_rounds=10,  # Allow enough rounds for comprehensive coverage
        )

        result = await generator.generate(
            url=url,
            max_urls=self.config.max_urls,
            enable_critic=True,
            progress_callback=progress_callback,
        )

        logger.info(f"[SkillGenerator] Generated llms.txt ({len(result.llmstxt)} chars)")
        return result.llmstxt

    async def generate(
        self,
        url: str,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
    ) -> GeneratorResult:
        """Generate a skill package for the given URL.

        Args:
            url: Target URL to process
            output_dir: Output directory (defaults to config.output_dir)
            progress_callback: Optional callback for progress updates

        Returns:
            GeneratorResult with output path and stats
        """
        output_dir = Path(output_dir or self.config.output_dir).resolve()
        library_name = self._extract_name(url)
        skill_dir = output_dir / library_name

        logger.info(f"[SkillGenerator] Starting skill generation for {url}")
        logger.info(f"[SkillGenerator] Output directory: {skill_dir}")

        # Step 1: Generate llms.txt as reference
        llmstxt_content = await self._generate_llmstxt(url, progress_callback)

        if progress_callback:
            progress_callback("Connecting to MCP server...", 25)

        # Step 2: Connect to MCP and build agents
        async with get_mcp_tools(
            self.config.mcp_host,
            self.config.mcp_port,
            max_urls=self.config.max_urls,
        ) as all_tools:
            main_tools = filter_tools_by_name(all_tools, MAIN_AGENT_TOOL_NAMES)
            subagent_tools = filter_tools_by_name(all_tools, SUBAGENT_TOOL_NAMES)

            logger.info(f"[SkillGenerator] Main agent tools: {[t.name for t in main_tools]}")

            if progress_callback:
                progress_callback("Building agents...", 30)

            # Create LLM
            llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

            # Generator Deep Agent
            subagent_spec = SubAgent(
                name="scraper",
                description="Scrapes individual URLs to fetch documentation content.",
                system_prompt="You are a documentation scraper. Use scrape_url to fetch content.",
                tools=subagent_tools,
            )

            generator_fs_backend = FilesystemBackend(root_dir=str(output_dir))

            generator_agent = create_deep_agent(
                model=llm,
                tools=main_tools,
                subagents=[subagent_spec],
                system_prompt=GENERATOR_PROMPT.format(
                    llmstxt_content=llmstxt_content,
                    output_dir=str(output_dir),
                    library_name=library_name,
                ),
                backend=generator_fs_backend,
            )

            # Critic Deep Agent (with filesystem tools AND web tools to verify content)
            critic_fs_backend = FilesystemBackend(root_dir=str(output_dir))

            critic_agent = create_deep_agent(
                model=llm,
                tools=main_tools,  # Web tools to verify against source
                system_prompt=CRITIC_PROMPT.format(
                    llmstxt_content=llmstxt_content,
                ),
                backend=critic_fs_backend,
            )

            if progress_callback:
                progress_callback("Running skill generation...", 40)

            # Build simple 2-node graph: generator → critic → (end or loop)
            def should_continue(state: MessagesState) -> str:
                message_count = len(state.get("messages", []))
                iteration = message_count // 2  # Each iteration = generator + critic

                if iteration >= self.config.max_rounds:
                    logger.warning(f"[SkillGenerator] Max iterations ({self.config.max_rounds}) reached")
                    return END

                last_message = state["messages"][-1]
                content = getattr(last_message, "content", "")
                if "APPROVE" in content.upper():
                    logger.info("[SkillGenerator] Critic APPROVED the skill package")
                    return END
                return "generator"

            graph = StateGraph(MessagesState)
            graph.add_node("generator", generator_agent)
            graph.add_node("critic", critic_agent)

            graph.set_entry_point("generator")
            graph.add_edge("generator", "critic")
            graph.add_conditional_edges("critic", should_continue)

            app = graph.compile()

            # Run the graph
            initial_message = f"""Create a comprehensive skill package for {library_name}.

Use the llms.txt reference provided in your system prompt to guide what topics to cover.

Output directory: {output_dir}
Library name: {library_name}
Skill directory: {skill_dir}/

Create:
1. {skill_dir}/SKILL.md (with YAML frontmatter)
2. {skill_dir}/references/ (detailed documentation files)
3. {skill_dir}/scripts/ (working code examples)

Be comprehensive - cover all important topics from the llms.txt reference.
"""

            logger.info("[SkillGenerator] Running generator → critic loop...")

            logging_handler = SkillGeneratorLoggingHandler()

            result = await app.ainvoke(
                {"messages": [{"role": "user", "content": initial_message}]},
                config={"callbacks": [logging_handler]}
            )

            if progress_callback:
                progress_callback("Skill package generated", 90)

            # Check output
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                files_created = len(list(skill_dir.rglob("*"))) if skill_dir.exists() else 0
                logger.info(f"[SkillGenerator] Skill package created at {skill_dir} ({files_created} files)")
                if progress_callback:
                    progress_callback(f"Skill package created: {files_created} files", 100)
                return GeneratorResult(
                    output_path=skill_file,
                    stats={
                        "output_dir": str(skill_dir),
                        "files_created": files_created,
                        "llmstxt_length": len(llmstxt_content),
                    },
                )
            else:
                raise RuntimeError(f"Skill generation failed: SKILL.md not created at {skill_dir}")

    async def close(self):
        """Clean up resources."""
        pass
