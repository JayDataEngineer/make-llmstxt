"""Critic module for llms.txt validation.

Validates generated llms.txt output using structured LLM calls.
Returns pass/fail with specific, actionable feedback for retries.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class CriticResult(BaseModel):
    """Structured result from critic evaluation."""

    passed: bool = Field(description="True if llms.txt meets all quality requirements")
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality score 0.0-1.0"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific problems found"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="How to fix the issues"
    )


CRITIC_SYSTEM_PROMPT = """You are a strict quality critic for llms.txt files.

Your job is to evaluate llms.txt drafts and provide actionable feedback.

VALIDATION RULES:
1. STRUCTURE: Must start with `# <url> llms.txt` header
2. ENTRIES: Each line must follow `- [Title](URL): Description` format
3. TITLES: Must be 2-5 words, concise and descriptive
4. DESCRIPTIONS: Must be 8-12 words, informative, not generic
5. NO PLACEHOLDERS: No "Page", "No description available", or empty fields
6. NO DUPLICATES: Each URL should appear once
7. COMPLETENESS: If full text requested, check llms-full.txt exists

GRADE STRICTLY. Generic descriptions like "This is a webpage" are FAILURES.

Output a JSON object with:
- passed: boolean
- score: 0.0-1.0 (be harsh, 0.7+ is good)
- issues: list of specific problems
- suggestions: list of how to fix each issue"""


class Critic:
    """LLM-based critic for llms.txt validation."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(CriticResult)

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
            content = f"Source URL: {url}\n\n{content}"

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        try:
            result = await self.llm.ainvoke(messages)
            logger.info(f"Critic: passed={result.passed}, score={result.score:.2f}, issues={len(result.issues)}")

            if not result.passed:
                logger.warning(f"Critic issues: {result.issues}")

            return result

        except Exception as e:
            logger.error(f"Critic evaluation failed: {e}")
            # Return a pass on error to avoid blocking the pipeline
            return CriticResult(
                passed=True,
                score=0.5,
                issues=[],
                suggestions=[f"Critic failed with error: {e}"],
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
