"""Configuration for make-llmstxt.

Provider-agnostic via LangChain:
- OpenAI, Anthropic, DeepSeek, OpenRouter
- Local servers (llama.cpp, vLLM, Ollama)
- Any OpenAI-compatible API

Scraping via your custom MCP server.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path


class LLMConfig(BaseModel):
    """LLM configuration - works with any OpenAI-compatible provider."""

    model: str = Field(default="gpt-4o-mini", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=150, ge=1)
    provider: str = Field(default="openai", description="Provider name for logging")

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


class MCPConfig(BaseModel):
    """MCP scraper configuration."""

    host: str = Field(default="100.85.22.99", description="MCP server host")
    port: int = Field(default=8000, description="MCP server port")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class AppConfig(BaseModel):
    """Application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    # Generation settings
    max_urls: Optional[int] = Field(default=None, description="Maximum URLs to process (None = unlimited)")
    batch_size: int = Field(default=10, description="URLs per batch")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        config = cls()

        # LLM provider detection
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

        # Map providers to their env keys and default base URLs
        provider_settings = {
            "openai": ("OPENAI_API_KEY", None),
            "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1"),
            "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
            "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
            "zai": ("ZAI_API_KEY", "https://api.z.ai/api/coding/paas/v4"),
            "glm": ("ZAI_API_KEY", "https://api.z.ai/api/coding/paas/v4"),
            "local": ("LOCAL_API_KEY", "http://localhost:8000/v1"),
        }

        if llm_provider in provider_settings:
            env_key, default_base_url = provider_settings[llm_provider]
            config.llm.provider = llm_provider
            config.llm.model = os.getenv("LLM_MODEL", cls._get_default_model(llm_provider))
            config.llm.api_key = os.getenv(env_key)
            if default_base_url:
                config.llm.base_url = os.getenv("LLM_BASE_URL", default_base_url)
        else:
            # Custom provider
            config.llm.provider = llm_provider
            config.llm.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            config.llm.api_key = os.getenv("LLM_API_KEY")
            config.llm.base_url = os.getenv("LLM_BASE_URL")

        # MCP config
        config.mcp.host = os.getenv("MCP_HOST", "100.85.22.99")
        config.mcp.port = int(os.getenv("MCP_PORT", "8000"))

        return config

    @staticmethod
    def _get_default_model(provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "deepseek": "deepseek-chat",
            "openrouter": "anthropic/claude-3-haiku",
            "zai": "glm-4.5-air",
            "glm": "glm-4.5-air",
            "local": "local-model",
        }
        return defaults.get(provider, "gpt-4o-mini")


# Convenience: provider profiles for CLI help
PROVIDER_PROFILES = {
    "openai": {"env_key": "OPENAI_API_KEY", "default_model": "gpt-4o-mini", "description": "OpenAI GPT models"},
    "anthropic": {"env_key": "ANTHROPIC_API_KEY", "default_model": "claude-3-haiku-20240307", "description": "Anthropic Claude"},
    "deepseek": {"env_key": "DEEPSEEK_API_KEY", "default_model": "deepseek-chat", "description": "DeepSeek models"},
    "openrouter": {"env_key": "OPENROUTER_API_KEY", "default_model": "anthropic/claude-3-haiku", "description": "OpenRouter gateway"},
    "zai": {"env_key": "ZAI_API_KEY", "default_model": "glm-4.5-air", "description": "ZAI GLM models"},
    "glm": {"env_key": "ZAI_API_KEY", "default_model": "glm-4.5-air", "description": "GLM models"},
    "local": {"env_key": "LOCAL_API_KEY", "default_model": "local-model", "description": "Local server"},
}
