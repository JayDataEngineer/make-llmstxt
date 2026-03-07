"""Deep Agents Draft Pattern for llms.txt generation.

Implements the Drafter-Critic pattern using LangChain Deep Agents:
1. Drafter subagent creates/revies the llms.txt content
2. Critic subagent evaluates and provides structured feedback
3. Iterative loop until critic approves or max rounds reached

This uses the existing CriticResult Pydantic model for validation.
"""

from typing import Optional, List, Callable, Any
from pydantic import BaseModel, Field
from loguru import logger

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from .critic import CriticResult
from .prompts import GENERATOR_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT, build_critic_prompt


class DraftState(BaseModel):
    """State for the draft-critic loop."""

    draft: str = Field(default="", description="Current draft content")
    critique: Optional[CriticResult] = Field(default=None, description="Latest critique result")
    round: int = Field(default=0, description="Current round number")
    agreed: bool = Field(default=False, description="Whether critic approved")
    history: List[dict] = Field(default_factory=list, description="Round history")


class DeepDraftConfig(BaseModel):
    """Configuration for the Deep Draft pattern."""

    max_rounds: int = Field(default=3, ge=1, le=10, description="Maximum revision rounds")
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum score to pass")
    drafter_model: str = Field(default="claude-sonnet-4-6", description="Model for drafter")
    critic_model: str = Field(default="claude-sonnet-4-6", description="Model for critic")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Generation temperature")


def build_drafter_prompt(
    url: str,
    content: str,
    feedback: Optional[List[str]] = None,
) -> str:
    """Build the drafter prompt with optional critic feedback.

    Args:
        url: The website URL being processed
        content: Scraped content (URLs and summaries)
        feedback: Critic feedback from previous attempt (if retrying)

    Returns:
        Formatted prompt string
    """
    base = f"""Generate an llms.txt file for this website.

Source URL: {url}

Scraped content (URLs and page data):
---
{content}
---

Remember:
1. H1 with project NAME (extract from content/URL), not the URL
2. Blockquote summary of what this project/site is
3. Group related URLs under H2 sections
4. Use `- [Title](URL): Description` format exactly
5. Put low-priority links under `## Optional`"""

    if feedback:
        feedback_block = "\n".join(f"- {f}" for f in feedback)
        return f"""{base}

PREVIOUS ATTEMPT FAILED CRITIC VALIDATION. Fix these specific issues:
{feedback_block}

Only fix the reported issues. Do not change valid content."""

    return base


def create_drafter_tool(url: str, scraped_content: str) -> Callable:
    """Create a tool for the drafter to generate/revise llms.txt content.

    Args:
        url: The website URL
        scraped_content: The scraped content from the website

    Returns:
        A tool function for drafting
    """

    @tool
    def draft_llmstxt(feedback: str = "") -> str:
        """Generate or revise llms.txt content based on feedback.

        Args:
            feedback: Optional feedback from critic to address

        Returns:
            The generated/revised llms.txt content
        """
        feedback_list = [feedback] if feedback else None
        prompt = build_drafter_prompt(url, scraped_content, feedback_list)
        return prompt

    return draft_llmstxt


def create_critic_tool() -> Callable:
    """Create a tool for the critic to evaluate drafts.

    Returns:
        A tool function for critiquing
    """

    @tool
    def critique_draft(llmstxt: str) -> dict:
        """Evaluate an llms.txt draft and return structured feedback.

        Args:
            llmstxt: The llms.txt content to evaluate

        Returns:
            Structured critique with passed, score, issues, and suggestions
        """
        # This tool just passes the content - actual evaluation happens
        # via structured output on the subagent
        return {"llmstxt": llmstxt, "status": "needs_evaluation"}

    return critique_draft


class DeepDraftAgent:
    """Deep Agent implementation of the Draft-Critic pattern.

    Uses Deep Agents' subagent capabilities for context isolation
    between the drafter and critic roles.
    """

    def __init__(
        self,
        config: Optional[DeepDraftConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the Deep Draft Agent.

        Args:
            config: Configuration for the draft pattern
            api_key: API key for the LLM provider
            base_url: Base URL for the LLM API (for custom providers)
        """
        self.config = config or DeepDraftConfig()

        # Initialize models
        model_kwargs = {"temperature": self.config.temperature}
        if api_key:
            model_kwargs["api_key"] = api_key
        if base_url:
            model_kwargs["base_url"] = base_url

        self.drafter_model = init_chat_model(
            model=self.config.drafter_model,
            **model_kwargs,
        )
        self.critic_model = init_chat_model(
            model=self.config.critic_model,
            **model_kwargs,
        )

    def create_agent(
        self,
        url: str,
        scraped_content: str,
    ):
        """Create a Deep Agent with drafter and critic subagents.

        Args:
            url: The website URL to process
            scraped_content: The scraped content from the website

        Returns:
            A compiled Deep Agent graph
        """
        # Define the drafter subagent
        drafter_subagent = {
            "name": "drafter",
            "description": (
                "Creates and revises llms.txt content. "
                "Use this agent to generate the initial draft or revise based on critic feedback."
            ),
            "system_prompt": GENERATOR_SYSTEM_PROMPT,
            "tools": [],  # Uses built-in file tools for context management
            "model": self.config.drafter_model,
        }

        # Define the critic subagent with structured output
        critic_subagent = {
            "name": "critic",
            "description": (
                "Evaluates llms.txt drafts against the specification. "
                "Returns structured feedback with pass/fail decision, score, and specific issues."
            ),
            "system_prompt": CRITIC_SYSTEM_PROMPT,
            "tools": [],
            "model": self.config.critic_model,
        }

        # Create the main orchestrator agent
        orchestrator_prompt = f"""You are orchestrating the generation of an llms.txt file.

Website URL: {url}

Your workflow:
1. First, delegate to the 'drafter' subagent to create an initial llms.txt draft
2. Then, delegate to the 'critic' subagent to evaluate the draft
3. If the critic returns passed=false or score < {self.config.pass_threshold}:
   - Extract the issues and suggestions from the critic's feedback
   - Delegate back to the 'drafter' to revise, providing the specific issues to fix
4. Repeat until the critic approves (passed=true) or you've tried {self.config.max_rounds} times

IMPORTANT:
- Each round, provide the drafter with the specific issues from the critic
- Track the round number - stop after {self.config.max_rounds} rounds even if not perfect
- When the critic approves, return the final llms.txt content
- Use the file system tools to store intermediate drafts if needed

The final output should be the complete, validated llms.txt content."""

        agent = create_deep_agent(
            model=self.config.drafter_model,
            system_prompt=orchestrator_prompt,
            subagents=[drafter_subagent, critic_subagent],
            name="llmstxt-orchestrator",
        )

        return agent

    async def generate(
        self,
        url: str,
        scraped_content: str,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> tuple[str, DraftState]:
        """Generate llms.txt using the Draft-Critic pattern.

        Args:
            url: The website URL to process
            scraped_content: The scraped content from the website
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (final_llmstxt_content, draft_state)
        """
        state = DraftState()

        # Create the agent
        agent = self.create_agent(url, scraped_content)

        # Build initial message with scraped content
        initial_message = f"""Generate an llms.txt file for this website.

URL: {url}

Scraped Content:
---
{scraped_content}
---

Start by delegating to the drafter to create an initial draft."""

        logger.info(f"[DeepDraft] Starting generation for {url}")
        logger.info(f"[DeepDraft] Max rounds: {self.config.max_rounds}, Pass threshold: {self.config.pass_threshold}")

        # Track progress
        if progress_callback:
            progress_callback("Starting Deep Draft generation...", 0)

        try:
            # Invoke the agent
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": initial_message}]
            })

            # Extract the final content from the last message
            final_message = result.get("messages", [])[-1] if result.get("messages") else None
            final_content = ""

            if final_message:
                if hasattr(final_message, 'content'):
                    final_content = final_message.content
                elif isinstance(final_message, dict):
                    final_content = final_message.get("content", "")

            state.draft = final_content
            state.round = self.config.max_rounds  # Assume completed
            state.agreed = True

            if progress_callback:
                progress_callback("Deep Draft generation completed", 100)

            logger.info(f"[DeepDraft] Generation completed, output length: {len(final_content)}")

            return final_content, state

        except Exception as e:
            logger.error(f"[DeepDraft] Generation failed: {e}")
            state.agreed = False
            raise


class SimpleDraftCritic:
    """Simpler implementation using direct subagent calls.

    This version doesn't use Deep Agents' orchestration but implements
    the draft-critic pattern directly with more control.
    """

    def __init__(
        self,
        config: Optional[DeepDraftConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the Simple Draft Critic.

        Args:
            config: Configuration for the draft pattern
            api_key: API key for the LLM provider
            base_url: Base URL for the LLM API
        """
        self.config = config or DeepDraftConfig()

        model_kwargs = {"temperature": self.config.temperature}
        if api_key:
            model_kwargs["api_key"] = api_key
        if base_url:
            model_kwargs["base_url"] = base_url

        self.llm = init_chat_model(
            model=self.config.drafter_model,
            **model_kwargs,
        )

        # Critic with structured output
        self.critic_llm = self.llm.with_structured_output(CriticResult)

    async def draft(
        self,
        url: str,
        content: str,
        feedback: Optional[List[str]] = None,
    ) -> str:
        """Generate or revise a draft.

        Args:
            url: The website URL
            content: Scraped content
            feedback: Optional critic feedback to address

        Returns:
            Generated/revised llms.txt content
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = build_drafter_prompt(url, content, feedback)
        messages = [
            SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    async def critique(self, llmstxt: str, url: Optional[str] = None) -> CriticResult:
        """Evaluate a draft using structured output.

        Args:
            llmstxt: The llms.txt content to evaluate
            url: Optional source URL for context

        Returns:
            Structured CriticResult
        """
        from langchain_core.messages import HumanMessage
        from .prompts import build_critic_prompt

        prompt = build_critic_prompt(llmstxt, None, url)
        messages = [HumanMessage(content=prompt)]

        try:
            result = await self.critic_llm.ainvoke(messages)

            # Handle string response (some providers wrap JSON)
            if isinstance(result, str):
                import json
                import re
                text = result.strip()
                if text.startswith("```"):
                    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
                    if match:
                        text = match.group(1).strip()
                data = json.loads(text)
                result = CriticResult(**data)

            # Apply threshold override
            if result.score >= self.config.pass_threshold and not result.passed:
                logger.info(
                    f"[SimpleDraftCritic] Overriding passed=True "
                    f"(score {result.score:.2f} >= threshold {self.config.pass_threshold})"
                )
                result = CriticResult(
                    passed=True,
                    score=result.score,
                    issues=result.issues,
                    suggestions=result.suggestions,
                )

            return result

        except Exception as e:
            logger.error(f"[SimpleDraftCritic] Critique failed: {e}")
            return CriticResult(
                passed=False,
                score=0.0,
                issues=[f"Critique error: {e}"],
                suggestions=["Check LLM configuration and retry"],
            )

    async def generate(
        self,
        url: str,
        content: str,
        progress_callback: Optional[Callable[[str, int, Optional[CriticResult]], None]] = None,
    ) -> tuple[str, DraftState]:
        """Generate llms.txt using iterative draft-critic loop.

        Args:
            url: The website URL
            content: Scraped content
            progress_callback: Optional callback(draft, round, critique_result)

        Returns:
            Tuple of (final_content, state)
        """
        state = DraftState()
        current_draft = ""
        feedback: List[str] = []

        logger.info(f"[SimpleDraftCritic] Starting generation for {url}")
        logger.info(f"[SimpleDraftCritic] Max rounds: {self.config.max_rounds}")

        for round_num in range(1, self.config.max_rounds + 1):
            state.round = round_num
            logger.info(f"[SimpleDraftCritic] Round {round_num}/{self.config.max_rounds}")

            # Draft
            if round_num > 1:
                logger.info(f"[SimpleDraftCritic] Revising with {len(feedback)} issues to fix")
            current_draft = await self.draft(url, content, feedback if round_num > 1 else None)
            state.draft = current_draft

            # Critique
            critique_result = await self.critique(current_draft, url)
            state.critique = critique_result

            # Record history
            state.history.append({
                "round": round_num,
                "score": critique_result.score,
                "passed": critique_result.passed,
                "issues_count": len(critique_result.issues),
            })

            logger.info(
                f"[SimpleDraftCritic] Critique: passed={critique_result.passed}, "
                f"score={critique_result.score:.2f}, issues={len(critique_result.issues)}"
            )

            # Progress callback
            if progress_callback:
                progress_callback(current_draft, round_num, critique_result)

            # Check if approved
            if critique_result.passed:
                state.agreed = True
                logger.info(f"[SimpleDraftCritic] Approved at round {round_num}")
                break

            # Prepare feedback for next round
            feedback = critique_result.issues

            # Add suggestions as additional feedback
            if critique_result.suggestions:
                feedback = [
                    f"{issue} → Fix: {suggestion}"
                    for issue, suggestion in zip(critique_result.issues, critique_result.suggestions)
                ]
        else:
            # Max rounds reached without approval
            logger.warning(
                f"[SimpleDraftCritic] Max rounds ({self.config.max_rounds}) reached without approval"
            )
            state.agreed = False

        return current_draft, state


__all__ = [
    "DraftState",
    "DeepDraftConfig",
    "DeepDraftAgent",
    "SimpleDraftCritic",
    "build_drafter_prompt",
]
