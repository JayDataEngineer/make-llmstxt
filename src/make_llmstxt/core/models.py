"""Core data models for make-llmstxt."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AgentPrompts:
    """Prompts configuration for Deep Agent.

    Different modes (llms.txt vs skill) provide different prompts.
    """
    # Generator agent prompts
    generator_system: str
    generator_initial: str
    generator_feedback: str

    # Subagent prompts (for page scraping)
    subagent_name: str = "page_scraper"
    subagent_description: str = "Scrapes a URL and returns title/description JSON."
    subagent_system: str = ""

    # Critic prompts (optional - if using LLM-based critic)
    critic_system: str = ""
    critic_approval_keyword: str = "APPROVE"


class GeneratorConfig(BaseModel):
    """Configuration for all generators.

    Used by both llms.txt and skill generators, with prompts injected
    to customize behavior.
    """
    # Target
    url: str = Field(..., description="Target URL to process")
    output_dir: Path = Field(default=Path("."), description="Output directory")

    # Scraping config
    max_urls: Optional[int] = Field(default=None, description="Max URLs to process (None = unlimited)")
    mcp_host: str = Field(default="100.85.22.99", description="MCP server host")
    mcp_port: int = Field(default=8000, description="MCP server port")

    # LLM config
    model: str = Field(default="glm-4.5-air", description="Model identifier")
    provider: str = Field(default="zai", description="LLM provider name")
    api_key: Optional[str] = Field(default=None, description="API key for LLM provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for LLM API")
    temperature: float = Field(default=0.3, description="Generation temperature")

    # Draft-Critic config
    pass_threshold: float = Field(default=0.7, description="Score threshold for critic approval")
    max_rounds: int = Field(default=3, description="Maximum revision rounds")

    # Deep Agent config (optional - set by subclasses)
    prompts: Optional[AgentPrompts] = Field(default=None, description="Prompts for Deep Agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for prompts")

    class Config:
        arbitrary_types_allowed = True  # Allow AgentPrompts dataclass


class GeneratorResult(BaseModel):
    """Result from a generator.

    Attributes:
        output_path: Path to the output file(s)
        stats: Statistics about the generation
    """
    output_path: Path
    stats: dict = Field(default_factory=dict)
