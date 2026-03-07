"""Core module - shared types, base classes, and configuration."""

from typing import Optional, List
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
    api_key: Optional[str] = Field(default=None, description="API key for LLM provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for LLM API")
    temperature: float = Field(default=0.3, description="Generation temperature")

    # Draft-Critic config
    pass_threshold: float = Field(default=0.7, description="Score threshold for critic approval")
    max_rounds: int = Field(default=3, description="Maximum revision rounds")


class ScrapedPage(BaseModel):
    """Represents a scraped page.

    Attributes:
        url: URL of the page
        title: Page title
        markdown: Page content in markdown format
        metadata: Additional metadata
    """

    url: str
    title: Optional[str] = None
    markdown: str
    metadata: dict = Field(default_factory=dict)


class GeneratorResult(BaseModel):
    """Result from a generator.

    Attributes:
        output_path: Path to the output file(s)
        stats: Statistics about the generation
    """

    output_path: Path
    stats: dict = Field(default_factory=dict)


class BaseGenerator:
    """Abstract base class for generators.

    All generators share:
    - MCP scraper for content gathering
    - Configuration management
    - Progress callbacks
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config

    async def gather_content(self, url: str) -> List[ScrapedPage]:
        """Gather content from the target URL.

        Args:
            url: Target URL to process

        Returns:
            List of scraped pages
        """
        raise NotImplementedError

    async def generate(
        self,
        url: str,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> GeneratorResult:
        """Generate output for the given URL.

        Args:
            url: Target URL to process
            output_dir: Output directory (defaults to config.output_dir)
            progress_callback: Optional callback for progress updates

        Returns:
            GeneratorResult with output path and stats
        """
        raise NotImplementedError
