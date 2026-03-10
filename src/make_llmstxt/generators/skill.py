"""Skill generator using Deep Agent pattern.

This extends DeepAgentGenerator with:
- Prompts from prompts/skill.py
- Structured critic using Critic class
- Pre-generation of llms.txt as reference
"""

from pathlib import Path
from typing import List, Optional, Callable

from loguru import logger

from ..core import GeneratorConfig, GeneratorResult
from .base_agent import DeepAgentGenerator, extract_name_from_url
from .prompts.skill import SKILL_PROMPTS
from .llmstxt import LLMsTxtGenerator
from .critic import Critic


class SkillGenerator(DeepAgentGenerator):
    """Generate skill packages using Deep Agent pattern.

    Flow:
    1. Generate llms.txt from URL (condensed documentation summary)
    2. Generator Deep Agent creates skill package using llms.txt as reference
    3. Structured Critic validates against llms.txt

    Output:
    - {library_name}/SKILL.md
    - {library_name}/references/*.md
    - {library_name}/scripts/*.*
    """

    def __init__(self, config: GeneratorConfig):
        super().__init__(config, log_prefix="[SkillGenerator]", default_prompts=SKILL_PROMPTS)
        self._llmstxt_content: Optional[str] = None
        self._critic = None

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

        # Create a config for the llms.txt generator (reuse our settings)
        llmstxt_config = GeneratorConfig(
            url=url,
            output_dir=self.config.output_dir,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            provider=self.config.provider,
            mcp_host=self.config.mcp_host,
            mcp_port=self.config.mcp_port,
            max_urls=self.config.max_urls,
            max_rounds=10,  # Allow enough rounds for comprehensive coverage
            pass_threshold=self.config.pass_threshold,
        )

        generator = LLMsTxtGenerator(llmstxt_config)

        # Generate to a temp file, we just need the content
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = await generator.generate(
                url=url,
                output_file=temp_path,
                progress_callback=progress_callback,
            )
            content = result.llmstxt
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

        logger.info(f"{self.log_prefix} Generated llms.txt ({len(content)} chars)")
        return content

    async def _run_critic(
        self,
        state: dict,
        output_path: Path,
        url: str,
        round_num: int
    ) -> tuple[bool, float, List[str]]:
        """Run structured critic on the skill package."""
        # Read all generated files
        content_parts = []

        # Check SKILL.md
        skill_file = output_path / "SKILL.md"
        if skill_file.exists():
            content_parts.append(f"=== SKILL.md ===\n{skill_file.read_text()}")
        else:
            logger.error(f"{self.log_prefix} SKILL.md not found at {skill_file}")
            return False, 0.0, ["SKILL.md file was not created"]

        # Check references/
        refs_dir = output_path / "references"
        if refs_dir.exists():
            for ref_file in sorted(refs_dir.glob("*.md")):
                content_parts.append(f"=== references/{ref_file.name} ===\n{ref_file.read_text()}")

        # Check scripts/
        scripts_dir = output_path / "scripts"
        if scripts_dir.exists():
            for script_file in sorted(scripts_dir.iterdir()):
                if script_file.is_file():
                    content_parts.append(f"=== scripts/{script_file.name} ===\n{script_file.read_text()}")

        generated_content = "\n\n".join(content_parts)
        logger.info(f"{self.log_prefix} Critic round {round_num}: evaluating ({len(generated_content)} chars)")

        # Create critic if needed (with config prompts)
        if self._critic is None:
            llm = self._create_llm()
            llm.temperature = 0.0  # Critic should be deterministic

            # Build prompt function from template
            def build_skill_prompt(content, url, source_content, **kwargs):
                llmstxt = kwargs.get("llmstxt_content", "")
                return self.config.prompts.critic_prompt_template.format(
                    content=content,
                    llmstxt_content=llmstxt,
                )

            self._critic = Critic(
                llm,
                pass_threshold=self.config.pass_threshold,
                system_prompt=self.config.prompts.critic_system,
                build_prompt=build_skill_prompt,
            )

        # Run critic with llms.txt reference
        critique = await self._critic.evaluate(
            content=generated_content,
            url=url,
            source_content=None,
            llmstxt_content=self._llmstxt_content or "",
        )

        feedback = critique.issues + critique.suggestions

        return critique.passed, critique.score, feedback
