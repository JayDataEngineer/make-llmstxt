"""Generator modules for make-llmstxt.

This package contains:
- LLMsTxtGenerator: Simple Draft-Critic (single file output)
- SkillGenerator: Deep Agent (multi-file output with sub-agents)
"""

from .skill import SkillGenerator

# LLMsTxtGenerator is still in the original location (generator.py)
# We'll import it from the when needed

__all__ = ["LLMsTxtGenerator", "SkillGenerator"]
