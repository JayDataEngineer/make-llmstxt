"""Configuration for make-llmstxt.

Provider-agnostic via LangChain:
- OpenAI, Anthropic, DeepSeek, OpenRouter
- Local servers (llama.cpp, vLLM, Ollama)
- Any OpenAI-compatible API

Scraping via your custom MCP server.
"""

import os
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path


def _extract_model_display_name(model_path: str) -> str:
    """Extract a clean display name from a model file path.

    Args:
        model_path: Path to model file (e.g., "/models/Qwen-2.5-7B-Instruct-Q4_K_M.gguf")

    Returns:
        Clean display name (e.g., "Qwen-2.5-7B")
    """
    # Get filename without extension
    filename = Path(model_path).stem

    # Remove common quantization suffixes
    quant_patterns = [
        r'-[Qq]\d(_[A-Za-z\d]+)?',  # Q4_K_M, q4_k_m, Q4, etc.
        r'-[Ii][Qq]\d+_[A-Za-z]+',   # IQ4_XS, etc.
        r'-[Ff]\d+',                  # F16, F32, etc.
        r'-[Bb][Pp][Ww]\d+(?:-\d+)?', # BPW4, BPW4-8, etc.
        r'-[Ee][Xx][Ll]2',            # EXL2
    ]

    for pattern in quant_patterns:
        filename = re.sub(pattern, '', filename)

    # Remove common suffixes
    suffixes_to_remove = [
        '-Instruct',
        '-instruct',
        '-Chat',
        '-chat',
        '-v\d+',  # Version numbers like -v1, -v2
    ]

    for suffix in suffixes_to_remove:
        filename = re.sub(suffix + '$', '', filename)

    return filename


class LLMConfig(BaseModel):
    """LLM configuration - works with any OpenAI-compatible provider."""

    model: str = Field(default="gpt-4o-mini", description="Model name (API identifier)")
    model_display_name: Optional[str] = Field(
        default=None,
        description="Display name for observability (e.g., 'Qwen-2.5-7B' when model='llm')"
    )
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=65536, ge=1)
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
    # Full URL override (e.g., for Tailscale Funnel)
    url: Optional[str] = Field(default=None, description="Full MCP URL (overrides host/port)")

    @property
    def base_url(self) -> str:
        if self.url:
            return self.url
        return f"http://{self.host}:{self.port}"

    @property
    def mcp_endpoint(self) -> str:
        """Get the MCP endpoint URL."""
        if self.url:
            return self.url if self.url.endswith('/mcp') else f"{self.url}/mcp"
        return f"http://{self.host}:{self.port}/mcp"


class LangfuseConfig(BaseModel):
    """Langfuse observability configuration.

    Uses standard Langfuse env vars:
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_BASE_URL

    Tracing is automatically enabled when both keys are present.
    """

    enabled: bool = Field(default=False, description="Auto-enabled when keys are set")
    public_key: Optional[str] = Field(default=None, description="LANGFUSE_PUBLIC_KEY")
    secret_key: Optional[str] = Field(default=None, description="LANGFUSE_SECRET_KEY")
    base_url: str = Field(default="http://localhost:3000", description="LANGFUSE_BASE_URL")

    @classmethod
    def from_env(cls) -> "LangfuseConfig":
        """Load Langfuse config from environment variables."""
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        enabled = bool(secret_key and public_key)

        return cls(
            enabled=enabled,
            public_key=public_key,
            secret_key=secret_key,
            base_url=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
        )


class AppConfig(BaseModel):
    """Application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)

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

            # Auto-derive display name from model path if not explicitly set
            display_name = os.getenv("LLM_MODEL_DISPLAY_NAME")
            if not display_name:
                model_path = os.getenv("LLM_MODEL_PATH")
                if model_path:
                    display_name = _extract_model_display_name(model_path)
            config.llm.model_display_name = display_name

            config.llm.api_key = os.getenv(env_key)
            if default_base_url:
                config.llm.base_url = os.getenv("LLM_BASE_URL", default_base_url)
        else:
            # Custom provider
            config.llm.provider = llm_provider
            config.llm.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

            # Auto-derive display name from model path if not explicitly set
            display_name = os.getenv("LLM_MODEL_DISPLAY_NAME")
            if not display_name:
                model_path = os.getenv("LLM_MODEL_PATH")
                if model_path:
                    display_name = _extract_model_display_name(model_path)
            config.llm.model_display_name = display_name

            config.llm.api_key = os.getenv("LLM_API_KEY")
            config.llm.base_url = os.getenv("LLM_BASE_URL")

        # MCP config
        config.mcp.host = os.getenv("MCP_HOST", "100.85.22.99")
        config.mcp.port = int(os.getenv("MCP_PORT", "8000"))
        config.mcp.url = os.getenv("MCP_URL")  # Full URL override (e.g., Tailscale Funnel)

        # Langfuse observability config
        config.langfuse = LangfuseConfig.from_env()

        return config

    @staticmethod
    def _get_default_model(provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "deepseek": "deepseek-chat",
            "openrouter": "anthropic/claude-3-haiku",
            "zai": "glm-4.7",
            "glm": "glm-4.7",
            "local": "local-model",
        }
        return defaults.get(provider, "gpt-4o-mini")


# Convenience: provider profiles for CLI help
PROVIDER_PROFILES = {
    "openai": {"env_key": "OPENAI_API_KEY", "default_model": "gpt-4o-mini", "description": "OpenAI GPT models"},
    "anthropic": {"env_key": "ANTHROPIC_API_KEY", "default_model": "claude-3-haiku-20240307", "description": "Anthropic Claude"},
    "deepseek": {"env_key": "DEEPSEEK_API_KEY", "default_model": "deepseek-chat", "description": "DeepSeek models"},
    "openrouter": {"env_key": "OPENROUTER_API_KEY", "default_model": "anthropic/claude-3-haiku", "description": "OpenRouter gateway"},
    "zai": {"env_key": "ZAI_API_KEY", "default_model": "glm-4.7", "description": "ZAI GLM models"},
    "glm": {"env_key": "ZAI_API_KEY", "default_model": "glm-4.7", "description": "GLM models"},
    "local": {"env_key": "LOCAL_API_KEY", "default_model": "local-model", "description": "Local server"},
}
