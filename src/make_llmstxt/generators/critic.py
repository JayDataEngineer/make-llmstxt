"""Critic module for llms.txt validation.

Uses .with_structured_output() for single-call validation with guaranteed Pydantic schema.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.prompts import CRITIC_SYSTEM_PROMPT, build_critic_prompt


class CriticResult(BaseModel):
    """Structured result from critic evaluation."""

    passed: bool = Field(
        description="True ONLY if output meets ALL validation rules. Be strict."
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality score: 0.0=fail, 0.5=acceptable, 0.7+=good, 0.9+=excellent"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific problems found. Empty if passed."
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable fixes for each issue. Must match issues count."
    )


class Critic:
    """LLM-based critic for llms.txt validation.

    Uses .with_structured_output() for single-call validation.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        pass_threshold: float = 0.7,
    ):
        """Initialize critic.

        Args:
            llm: LangChain ChatOpenAI instance (or ChatZAI for ZAI/GLM models)
            pass_threshold: Minimum score to pass (default 0.7)
        """
        self.pass_threshold = pass_threshold

        # Create LLM with structured output - single call, guaranteed schema
        self.structured_llm = llm.with_structured_output(CriticResult)

    async def evaluate(
        self,
        llmstxt: str,
        url: Optional[str] = None,
        source_content: Optional[str] = None,
    ) -> CriticResult:
        """Evaluate llms.txt output quality.

        Args:
            llmstxt: The generated llms.txt content
            url: Original URL being processed (for context)
            source_content: The scraped source content (for coverage validation)

        Returns:
            CriticResult with pass/fail and feedback
        """
        prompt = build_critic_prompt(llmstxt, url, source_content)

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            # Single call with structured output - returns CriticResult directly
            logger.debug("[Critic] Evaluating with structured output...")
            logger.debug(f"[Critic] Evaluating llms.txt ({len(llmstxt)} chars)")
            logger.debug(f"[Critic] llms.txt content:\n{llmstxt[:2000]}{'...' if len(llmstxt) > 2000 else ''}")
            result = await self.structured_llm.ainvoke(messages)

            # Apply pass threshold override
            if result.score >= self.pass_threshold and not result.passed:
                logger.info(
                    f"[Critic] Overriding passed=True (score {result.score:.2f} >= threshold {self.pass_threshold})"
                )
                result = CriticResult(
                    passed=True,
                    score=result.score,
                    issues=result.issues,
                    suggestions=result.suggestions,
                )

            logger.info(
                f"[Critic] passed={result.passed}, score={result.score:.2f}, "
                f"issues={len(result.issues)}"
            )

            # Log full critic output
            logger.info(f"[Critic] Full result: {result.model_dump_json(indent=2)}")

            if not result.passed:
                for i, issue in enumerate(result.issues):
                    logger.warning(f"  Issue {i+1}: {issue}")
                for i, suggestion in enumerate(result.suggestions):
                    logger.info(f"  Suggestion {i+1}: {suggestion}")

            return result

        except Exception as e:
            logger.error(f"[Critic] Evaluation failed: {e}")
            raise RuntimeError(f"Critic evaluation failed: {e}") from e


async def critique_generation(
    llm: ChatOpenAI,
    llmstxt: str,
    url: Optional[str] = None,
    pass_threshold: float = 0.7,
    source_content: Optional[str] = None,
) -> CriticResult:
    """Convenience function to critique generation output.

    Args:
        llm: LangChain ChatOpenAI instance
        llmstxt: Generated llms.txt content
        url: Source URL for context
        pass_threshold: Minimum score to pass
        source_content: The scraped source content (for coverage validation)

    Returns:
        CriticResult
    """
    critic = Critic(llm, pass_threshold=pass_threshold)
    return await critic.evaluate(llmstxt, url, source_content)


__all__ = ["Critic", "CriticResult", "critique_generation"]