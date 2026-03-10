"""Skill generator using Deep Agent pattern.

This extends DeepAgentGenerator with:
- Prompts from prompts/skill.py
- LLM-based critic checking for "APPROVE" keyword
- Pre-generation of llms.txt as reference
"""

from pathlib import Path
from typing import List, Optional, Callable

from loguru import logger

from ..core import GeneratorConfig, GeneratorResult
from ..config import AppConfig
from .base_agent import DeepAgentGenerator, extract_name_from_url
from .prompts.skill import SKILL_PROMPTS
from .llmstxt import LLMsTxtGenerator


class SkillGenerator(DeepAgentGenerator):
    """Generate skill packages using Deep Agent pattern.

    Flow:
    1. Generate llms.txt from URL (condensed documentation summary)
    2. Generator Deep Agent creates skill package using llms.txt as reference
    3. Critic validates against llms.txt (checks for "APPROVE")

    Output:
    - {library_name}/SKILL.md
    - {library_name}/references/*.md
    - {library_name}/scripts/*.*
    """

    def __init__(self, config: GeneratorConfig):
        super().__init__(config, log_prefix="[SkillGenerator]", default_prompts=SKILL_PROMPTS)
        self._llmstxt_content: Optional[str] = None

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
        library_name = extract_name_from_url(url)
        skill_dir = output_dir / library_name

        logger.info(f"{self.log_prefix} Starting skill generation for {url}")
        logger.info(f"{self.log_prefix} Output directory: {skill_dir}")

        # Step 1: Generate llms.txt as reference
        self._llmstxt_content = await self._generate_llmstxt(url, progress_callback)

        if progress_callback:
            progress_callback("Creating skill package...", 50)

        # Add context for prompt formatting
        self.config.context["library_name"] = library_name
        self.config.context["output_dir"] = str(output_dir)
        self.config.context["llmstxt_content"] = self._llmstxt_content

        # Run the Deep Agent generation
        result = await super().generate(url, skill_dir, progress_callback)

        # Update stats with llms.txt info
        result.stats["llmstxt_length"] = len(self._llmstxt_content)

        return result

    async def _generate_llmstxt(
        self,
        url: str,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """Generate llms.txt for the URL.

        Returns the llms.txt content.
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

        logger.info(f"{self.log_prefix} Generated llms.txt ({len(result.llmstxt)} chars)")
        return result.llmstxt

    async def _run_critic(
        self,
        state: dict,
        output_path: Path,
        url: str,
        round_num: int
    ) -> tuple[bool, float, List[str]]:
        """Run LLM-based critic on the skill package.

        Checks the critic's response for "APPROVE" keyword.
        """
        # Check if SKILL.md exists
        skill_file = output_path / "SKILL.md"
        if not skill_file.exists():
            logger.error(f"{self.log_prefix} SKILL.md not found at {skill_file}")
            return False, 0.0, ["SKILL.md file was not created"]

        # Check the last message for APPROVE
        messages = state.get("messages", [])
        if not messages:
            return False, 0.0, ["No critic response found"]

        last_message = messages[-1]
        content = ""
        if isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = getattr(last_message, "content", "")

        # Check for APPROVE keyword
        if "APPROVE" in content.upper():
            logger.info(f"{self.log_prefix} Critic APPROVED the skill package")
            return True, 1.0, []

        # Extract feedback from critic response
        feedback = [
            line.strip()
            for line in content.split("\n")
            if line.strip() and not line.strip().upper().startswith("APPROVE")
        ]

        logger.info(f"{self.log_prefix} Critic round {round_num}: needs improvement, {len(feedback)} issues")

        return False, 0.5, feedback[:5]  # Limit feedback to top 5 issues
