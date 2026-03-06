"""LLM Configuration - Provider management and settings.

Supports multiple LLM providers via configuration:
- OpenAI (default)
- Anthropic (via OpenAI-compatible proxy or native)
- Local servers (llama.cpp, vLLM, etc.)
- Custom OpenAI-compatible APIs

Configuration priority:
1. Command line arguments
2. Environment variables
3. .env file
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
import yaml


class LLMConfig(BaseModel):
    """LLM configuration for any provider."""

    model: str = Field(default="gpt-4o-mini", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=150, ge=1)

    # Provider type (openai, anthropic, local, custom)
    provider: str = Field(default="openai", description="Provider type")

    def get_langchain_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for LangChain ChatOpenAI."""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url

        return kwargs


class FirecrawlConfig(BaseModel):
    """Firecrawl API configuration."""

    api_key: Optional[str] = Field(default=None, description="Firecrawl API key")
    base_url: str = Field(
        default="https://api.firecrawl.dev/v1",
        description="Firecrawl API base URL"
    )
    timeout: int = Field(default=30000, description="Request timeout in ms")


class ProviderProfile(BaseModel):
    """Predefined provider profile."""

    name: str
    base_url: Optional[str] = None
    env_key: str  # Environment variable name for API key
    default_model: str
    description: str = ""


# Predefined provider profiles
PROVIDER_PROFILES: Dict[str, ProviderProfile] = {
    "openai": ProviderProfile(
        name="openai",
        env_key="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        description="OpenAI GPT models",
    ),
    "anthropic": ProviderProfile(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        env_key="ANTHROPIC_API_KEY",
        default_model="claude-3-haiku-20240307",
        description="Anthropic Claude models (requires OpenAI-compatible proxy)",
    ),
    "deepseek": ProviderProfile(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        description="DeepSeek models",
    ),
    "openrouter": ProviderProfile(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        default_model="anthropic/claude-3-haiku",
        description="OpenRouter multi-model gateway",
    ),
    "local": ProviderProfile(
        name="local",
        base_url="http://localhost:8000/v1",
        env_key="LOCAL_API_KEY",
        default_model="local-model",
        description="Local server (llama.cpp, vLLM, etc.)",
    ),
    "zai": ProviderProfile(
        name="zai",
        base_url="https://api.z.ai/api/coding/paas/v4",
        env_key="ZAI_API_KEY",
        default_model="glm-4.5-air",
        description="ZAI GLM models",
    ),
    "glm": ProviderProfile(
        name="glm",
        base_url="https://api.z.ai/api/coding/paas/v4",
        env_key="ZAI_API_KEY",
        default_model="glm-4.5-air",
        description="GLM models via ZAI API",
    ),
}


class MCPConfig(BaseModel):
    """MCP scraper configuration."""

    host: str = Field(default="100.85.22.99", description="MCP server host (Tailscale IP)")
    port: int = Field(default=8000, description="MCP server port")
    timeout: float = Field(default=60.0, description="Request timeout")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class ScraperConfig(BaseModel):
    """Scraper backend configuration."""

    backend: str = Field(
        default="mcp",
        description="Scraper backend: 'mcp' or 'firecrawl'"
    )
    firecrawl: FirecrawlConfig = Field(default_factory=FirecrawlConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


class AppConfig(BaseModel):
    """Application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)

    # Legacy support
    firecrawl: FirecrawlConfig = Field(default_factory=FirecrawlConfig)

    # Generation settings
    max_urls: int = Field(default=20, description="Maximum URLs to process")
    batch_size: int = Field(default=10, description="URLs per batch")
    max_workers: int = Field(default=5, description="Concurrent workers")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Load LLM config from env
        llm_provider = os.getenv("LLM_PROVIDER", "openai")

        if llm_provider in PROVIDER_PROFILES:
            profile = PROVIDER_PROFILES[llm_provider]
            config.llm.provider = llm_provider
            config.llm.model = os.getenv("LLM_MODEL", profile.default_model)
            config.llm.api_key = os.getenv(profile.env_key)
            if profile.base_url:
                config.llm.base_url = os.getenv("LLM_BASE_URL", profile.base_url)
        else:
            # Custom provider
            config.llm.provider = llm_provider
            config.llm.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            config.llm.api_key = os.getenv("LLM_API_KEY")
            config.llm.base_url = os.getenv("LLM_BASE_URL")

        # Load Firecrawl config from env
        config.firecrawl.api_key = os.getenv("FIRECRAWL_API_KEY")

        # Load scraper backend selection
        scraper_backend = os.getenv("SCRAPER_BACKEND", "mcp").lower()
        config.scraper.backend = scraper_backend

        # Load MCP config from env
        config.scraper.mcp.host = os.getenv("MCP_HOST", "100.85.22.99")
        config.scraper.mcp.port = int(os.getenv("MCP_PORT", "8000"))

        # Copy firecrawl to scraper for legacy support
        config.scraper.firecrawl = config.firecrawl

        return config

    @classmethod
    def load_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls.from_env()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls.from_env()

        # Override with YAML values
        if "llm" in data:
            config.llm = LLMConfig(**{**config.llm.model_dump(), **data["llm"]})
        if "firecrawl" in data:
            config.firecrawl = FirecrawlConfig(
                **{**config.firecrawl.model_dump(), **data["firecrawl"]}
            )
        if "max_urls" in data:
            config.max_urls = data["max_urls"]
        if "batch_size" in data:
            config.batch_size = data["batch_size"]

        return config
