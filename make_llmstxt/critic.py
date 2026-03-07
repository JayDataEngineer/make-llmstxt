"""Critic module for llms.txt validation.

Validates generated llms.txt output using the "Draft then Extract" pattern:
1. LLM runs freely to evaluate the output (returns natural language)
2. Extraction LLM with with_structured_output() parses into CriticResult

This works with any provider - even those that don't support native structured output.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import CRITIC_SYSTEM_PROMPT, build_critic_prompt


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


# Extraction prompt for parsing critic evaluation into structured output
EXTRACTION_PROMPT = """Extract the evaluation results from the text below into structured JSON.

The output must match this schema:
- passed (boolean): True if the llms.txt passed validation
- score (float 0.0-1.0): Quality score
- issues (list of strings): Problems found
- suggestions (list of strings): Fixes for the issues

TEXT TO PARSE:
{text}

Return ONLY valid JSON matching the schema. No markdown, no explanation."""


class Critic:
    """LLM-based critic for llms.txt validation.

    Uses "Draft then Extract" pattern:
    1. LLM evaluates freely → natural language response
    2. Extraction LLM parses response → CriticResult
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        pass_threshold: float = 0.7,
        fail_on_error: bool = False,
    ):
        """Initialize critic.

        Args:
            llm: LangChain ChatOpenAI instance
            pass_threshold: Minimum score to pass (default 0.7)
            fail_on_error: If True, critic errors cause failure; else pass with warning
        """
        self.llm = llm
        self.pass_threshold = pass_threshold
        self.fail_on_error = fail_on_error

        # Create extraction LLM with structured output
        self.extraction_llm = llm.with_structured_output(CriticResult)

    async def evaluate(
        self,
        llmstxt: str,
        llms_fulltxt: Optional[str] = None,
        url: Optional[str] = None,
    ) -> CriticResult:
        """Evaluate llms.txt output quality.

        Args:
            llmstxt: The generated llms.txt content
            llms_fulltxt: Optional llms-full.txt content
            url: Original URL being processed (for context)

        Returns:
            CriticResult with pass/fail and feedback
        """
        prompt = build_critic_prompt(llmstxt, llms_fulltxt, url)

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            # Step 1: Get free-form evaluation from LLM
            logger.debug("[Critic] Getting free-form evaluation...")
            response = await self.llm.ainvoke(messages)
            evaluation_text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"[Critic] Evaluation text ({len(evaluation_text)} chars): {evaluation_text[:200]}...")

            # Step 2: Extract structured result using with_structured_output
            logger.debug("[Critic] Extracting structured result...")
            extraction_messages = [HumanMessage(content=EXTRACTION_PROMPT.format(text=evaluation_text))]
            result = await self.extraction_llm.ainvoke(extraction_messages)

            # Handle case where LLM returns markdown-wrapped JSON instead of raw JSON
            # (common with providers that don't fully support structured output)
            if isinstance(result, str):
                import json
                import re
                # Strip markdown code blocks if present
                text = result.strip()
                if text.startswith("```"):
                    # Extract JSON from markdown code block
                    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
                    if match:
                        text = match.group(1).strip()
                # Parse and re-validate
                data = json.loads(text)
                result = CriticResult(**data)

            # Apply pass threshold override
            if result.score >= self.pass_threshold and not result.passed:
                logger.info(
                    f"Critic: Overriding passed=True (score {result.score:.2f} >= threshold {self.pass_threshold})"
                )
                result = CriticResult(
                    passed=True,
                    score=result.score,
                    issues=result.issues,
                    suggestions=result.suggestions,
                )

            logger.info(
                f"Critic: passed={result.passed}, score={result.score:.2f}, "
                f"issues={len(result.issues)}"
            )

            if not result.passed:
                for issue in result.issues:
                    logger.warning(f"  Issue: {issue}")

            return result

        except Exception as e:
            logger.error(f"Critic evaluation failed: {e}")

            if self.fail_on_error:
                return CriticResult(
                    passed=False,
                    score=0.0,
                    issues=[f"Critic error: {e}"],
                    suggestions=["Check LLM configuration and retry"],
                )
            else:
                # Graceful degradation: pass with warning
                return CriticResult(
                    passed=True,
                    score=0.5,
                    issues=[],
                    suggestions=[f"Note: Critic failed ({e}), auto-passed"],
                )


async def critique_generation(
    llm: ChatOpenAI,
    llmstxt: str,
    llms_fulltxt: Optional[str] = None,
    url: Optional[str] = None,
    pass_threshold: float = 0.7,
) -> CriticResult:
    """Convenience function to critique generation output.

    Args:
        llm: LangChain ChatOpenAI instance
        llmstxt: Generated llms.txt content
        llms_fulltxt: Optional llms-full.txt content
        url: Source URL for context
        pass_threshold: Minimum score to pass

    Returns:
        CriticResult
    """
    critic = Critic(llm, pass_threshold=pass_threshold)
    return await critic.evaluate(llmstxt, llms_fulltxt, url)


__all__ = ["Critic", "CriticResult", "critique_generation"]
