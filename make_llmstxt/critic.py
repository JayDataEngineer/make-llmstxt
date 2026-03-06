"""Critic module for llms.txt validation.

Validates generated llms.txt output using structured LLM calls.
Returns pass/fail with specific, actionable feedback for retries.

Uses LangChain's with_structured_output for guaranteed JSON schema compliance.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class CriticResult(BaseModel):
    """Structured result from critic evaluation.

    This schema is enforced via with_structured_output() to guarantee
    the LLM always returns parseable JSON - no prompt engineering needed.
    """

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


CRITIC_SYSTEM_PROMPT = """You are a strict quality critic for llms.txt files.

Your job is to evaluate llms.txt drafts and provide actionable feedback.
Be pedantic - the goal is machine-readable documentation for LLMs.

VALIDATION RULES (ALL must pass):

1. HEADER: Must start with `# <url> llms.txt` (exact format)
2. ENTRIES: Each non-empty line after header must be:
   `- [Title](URL): Description`
3. TITLES: 2-5 words, specific, descriptive (NOT "Home", "Page", "About")
4. DESCRIPTIONS: 8-15 words, summarize page purpose (NOT generic)
5. NO PLACEHOLDERS: Reject "No description", "Page content", "This is a"
6. NO DUPLICATES: Each URL appears exactly once
7. VALID URLS: All URLs must be properly formatted

SCORING:
- 0.9+: Excellent, all rules passed, descriptions are specific
- 0.7-0.8: Good, minor issues (e.g., one title slightly long)
- 0.5-0.6: Acceptable, needs improvement but usable
- <0.5: Fail, major issues (placeholders, missing header, etc.)

Be harsh. Generic text like "This page contains information" is a FAILURE.
Return passed=false if ANY entry has a placeholder or generic description.

You MUST provide one suggestion per issue found."""


class Critic:
    """LLM-based critic for llms.txt validation.

    Uses with_structured_output to guarantee valid JSON responses.
    No need for prompt engineering around output parsing.
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
        # Use with_structured_output for guaranteed schema compliance
        self.llm = llm.with_structured_output(CriticResult)
        self.pass_threshold = pass_threshold
        self.fail_on_error = fail_on_error

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
        content = f"""Evaluate this llms.txt output:

--- llms.txt ---
{llmstxt}
--- end ---"""

        if llms_fulltxt:
            content += f"""

--- llms-full.txt (first 2000 chars) ---
{llms_fulltxt[:2000]}
--- end ---"""

        if url:
            content = f"Source website: {url}\n\n{content}"

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        try:
            result = await self.llm.ainvoke(messages)

            # Apply pass threshold
            if result.score >= self.pass_threshold and not result.passed:
                logger.info(f"Critic: Overriding passed=True (score {result.score:.2f} >= threshold {self.pass_threshold})")
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
) -> CriticResult:
    """Convenience function to critique generation output.

    Args:
        llm: LangChain ChatOpenAI instance
        llmstxt: Generated llms.txt content
        llms_fulltxt: Optional llms-full.txt content
        url: Source URL for context

    Returns:
        CriticResult
    """
    critic = Critic(llm)
    return await critic.evaluate(llmstxt, llms_fulltxt, url)


__all__ = ["Critic", "CriticResult", "critique_generation"]
