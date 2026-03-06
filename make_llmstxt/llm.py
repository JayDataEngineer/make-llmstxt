"""LangChain LLM Factory - Creates LLM instances for any provider.

Supports:
- OpenAI (default)
- Any OpenAI-compatible API (Anthropic via proxy, DeepSeek, etc.)
- Local servers (llama.cpp, vLLM, etc.)
- Custom providers

Uses LangChain's ChatOpenAI which works with any OpenAI-compatible API.
"""

from typing import Optional, Dict, Any, List

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

from .config import LLMConfig, PROVIDER_PROFILES


# Rate limit configurations per provider
PROVIDER_RATE_LIMITS: Dict[str, tuple[float, int]] = {
    "openai": (0.5, 10),      # 1 req/2s, burst 10
    "anthropic": (0.3, 8),    # 1 req/3.3s, burst 8
    "deepseek": (0.5, 10),    # 1 req/2s, burst 10
    "openrouter": (0.5, 10),  # 1 req/2s, burst 10
    "zai": (0.2, 3),          # 1 req/5s, burst 3
    "glm": (0.2, 3),          # Same as zai
    "local": (100, 1000),     # No limiting for local
}


def get_rate_limiter(
    provider: str,
    enabled: bool = True
) -> Optional[InMemoryRateLimiter]:
    """Get rate limiter for provider.

    Args:
        provider: Provider name
        enabled: If False, returns None

    Returns:
        InMemoryRateLimiter or None
    """
    if not enabled:
        return None

    # Local servers don't need rate limiting
    if provider == "local":
        return None

    rps, bucket = PROVIDER_RATE_LIMITS.get(provider, (0.3, 8))

    return InMemoryRateLimiter(
        requests_per_second=rps,
        check_every_n_seconds=0.1,
        max_bucket_size=bucket,
    )


def create_llm(
    config: LLMConfig,
    rate_limit_enabled: bool = True,
) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance.

    Args:
        config: LLM configuration
        rate_limit_enabled: Enable rate limiting

    Returns:
        ChatOpenAI instance configured for the provider
    """
    kwargs = config.get_langchain_kwargs()

    # Get rate limiter
    rate_limiter = get_rate_limiter(config.provider, enabled=rate_limit_enabled)
    if rate_limiter:
        kwargs["rate_limiter"] = rate_limiter

    # Create LLM
    llm = ChatOpenAI(**kwargs)

    logger.info(
        f"[LLM] Created {config.provider} LLM: model={config.model}, "
        f"base_url={config.base_url or 'default'}, rate_limit={rate_limiter is not None}"
    )

    return llm


async def generate_summary(
    llm: ChatOpenAI,
    url: str,
    markdown: str,
    max_content_chars: int = 4000,
    feedback: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Generate title and description for a page.

    Args:
        llm: LangChain ChatOpenAI instance
        url: Page URL
        markdown: Page content in markdown
        max_content_chars: Max characters to send to LLM
        feedback: Optional critic feedback to improve output

    Returns:
        Dict with 'title' and 'description' keys
    """
    base_prompt = f"""Generate a 9-10 word description and a 3-4 word title for this webpage.
The title and description should help users find this page for its intended purpose.

URL: {url}

Return ONLY valid JSON in this exact format:
{{"title": "3-4 word title", "description": "9-10 word description"}}"""

    # Add feedback if provided (for retry scenarios)
    if feedback:
        feedback_text = "\n".join(f"- {f}" for f in feedback)
        prompt = f"""{base_prompt}

CRITICAL - Previous attempt had these issues that MUST be fixed:
{feedback_text}

Ensure your output addresses these issues."""
    else:
        prompt = base_prompt

    messages = [
        SystemMessage(content="You are a helpful assistant that generates concise titles and descriptions for web pages. Always respond with valid JSON only."),
        HumanMessage(content=f"{prompt}\n\nPage content:\n{markdown[:max_content_chars]}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()

        # Parse JSON response
        import json

        # Try to extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        return {
            "title": result.get("title", "Page"),
            "description": result.get("description", "No description available"),
        }

    except Exception as e:
        logger.error(f"Error generating description for {url}: {e}")
        return {
            "title": "Page",
            "description": "No description available",
        }


async def generate_summaries_batch(
    llm: ChatOpenAI,
    pages: List[Dict[str, str]],
    max_content_chars: int = 4000,
    feedback: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Generate summaries for multiple pages.

    Args:
        llm: LangChain ChatOpenAI instance
        pages: List of dicts with 'url' and 'markdown' keys
        max_content_chars: Max chars per page
        feedback: Optional critic feedback for retry

    Returns:
        List of dicts with 'title' and 'description' keys
    """
    import asyncio

    async def process_page(page: Dict[str, str]) -> Dict[str, str]:
        result = await generate_summary(
            llm, page["url"], page["markdown"], max_content_chars, feedback=feedback
        )
        return {
            "title": result["title"],
            "description": result["description"],
        }

    # Process concurrently
    tasks = [process_page(page) for page in pages]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {pages[i]['url']}: {result}")
            final_results.append({
                "url": pages[i]["url"],
                "title": "Page",
                "description": "No description available",
            })
        else:
            final_results.append(result)

    return final_results
