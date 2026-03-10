"""Core data models for make-llmstxt."""

from typing import Optional
from pydantic import BaseModel, Field
from pathlib import Path


class GeneratorConfig(BaseModel):
    """Shared configuration for all generators.

    Attributes:
        url: Target URL to process
        output_dir: Directory to write output files
        max_urls: Maximum URLs to process (None = unlimited)
        model: Model identifier for LLM
        api_key: API key for LLM provider
        base_url: Base URL for LLM API
        temperature: Temperature for generation
        pass_threshold: Score threshold for critic approval
        max_rounds: Maximum revision rounds
        mcp_host: MCP server host
        mcp_port: MCP server port
    """

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


class GeneratorResult(BaseModel):
    """Result from a generator.

    Attributes:
        output_path: Path to the output file(s)
        stats: Statistics about the generation
    """

    output_path: Path
    stats: dict = Field(default_factory=dict)