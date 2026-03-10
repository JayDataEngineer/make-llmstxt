"""LLMs.txt Generator using Deep Agent pattern.

This extends DeepAgentGenerator with:
- Prompts from prompts/llmstxt.py
- Structured critic using Critic class
"""

from pathlib import Path
from typing import List

from loguru import logger

from ..core import GeneratorConfig, GeneratorResult
from .base_agent import DeepAgentGenerator, extract_name_from_url
from .prompts.llmstxt import LLMSTXT_PROMPTS
from .critic import Critic


class LLMsTxtGenerator(DeepAgentGenerator):
    """Generate llms.txt using Deep Agent pattern with structured critic.

    Flow:
    1. Generator Deep Agent creates llms.txt using MCP tools
    2. Structured Critic validates output
    3. If not approved, generator revises with feedback

    Output:
    - llms.txt file at specified output path
    """

    def __init__(self, config: GeneratorConfig):
        super().__init__(config, log_prefix="[LLMsTxt]", default_prompts=LLMSTXT_PROMPTS)
        self._critic = None

    async def generate(
        self,
        url: str,
        output_file: Path,
        progress_callback=None,
    ) -> GeneratorResult:
        """Generate llms.txt for the given URL.

        Args:
            url: Target URL to process
            output_file: Output file path
            progress_callback: Optional callback for progress updates

        Returns:
            GeneratorResult with output path and stats
        """
        # Add context for prompt formatting
        project_name = extract_name_from_url(url)
        self.config.context["project_name"] = project_name
        self.config.context["output_file"] = str(output_file)

        result = await super().generate(url, output_file, progress_callback)

        # Add llms.txt content to result for downstream use (skill generator)
        result.llmstxt = output_file.read_text() if output_file.exists() else ""

        return result

    async def _run_critic(
        self,
        state: dict,
        output_path: Path,
        url: str,
        round_num: int
    ) -> tuple[bool, float, List[str]]:
        """Run structured critic on the generated llms.txt."""
        # Read the generated file
        try:
            llmstxt_content = output_path.read_text()
        except FileNotFoundError:
            logger.error(f"{self.log_prefix} llms.txt not found at {output_path}")
            return False, 0.0, ["llms.txt file was not created"]

        logger.info(f"{self.log_prefix} Critic round {round_num}: evaluating ({len(llmstxt_content)} chars)")

        # Create critic if needed (with config prompts)
        if self._critic is None:
            llm = self._create_llm()
            llm.temperature = 0.0  # Critic should be deterministic

            # Build prompt function from template
            def build_llmstxt_prompt(content, url, source_content, **kwargs):
                context = f"Source URL: {url}" if url else ""
                return self.config.prompts.critic_prompt_template.format(
                    content=content,
                    context=context,
                )

            self._critic = Critic(
                llm,
                pass_threshold=self.config.pass_threshold,
                system_prompt=self.config.prompts.critic_system,
                build_prompt=build_llmstxt_prompt,
            )

        # Run critic
        critique = await self._critic.evaluate(
            content=llmstxt_content,
            url=url,
            source_content=None,
        )

        feedback = critique.issues + critique.suggestions

        return critique.passed, critique.score, feedback
