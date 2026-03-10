"""Prompts for Deep Agent generators."""

from .llmstxt import LLMSTXT_PROMPTS
from .skill import SKILL_PROMPTS
from ...core import AgentPrompts

__all__ = ["AgentPrompts", "LLMSTXT_PROMPTS", "SKILL_PROMPTS"]
