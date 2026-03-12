"""LangChain LLM Factory - Creates LLM instances for any provider.

Supports:
- OpenAI (default)
- Any OpenAI-compatible API (Anthropic via proxy, DeepSeek, etc.)
- Local servers (llama.cpp, vLLM, Ollama)

Reasoning/Thinking Mode (provider-specific):
- llama.cpp (Qwen): extra_body.chat_template_kwargs.enable_thinking
- llama.cpp (DeepSeek-R1): reasoning_effort
- OpenAI o1/o3: reasoning_effort
"""

import asyncio
import json
from typing import Optional, Dict, Any, List, Type
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ValidationError

from ..config import LLMConfig, PROVIDER_PROFILES
from ..generators.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_RETRY_PROMPT_TEMPLATE,
)
from ..utils.logging import StructuredLogger
from ..utils.observability import get_langfuse_callback

log = StructuredLogger("llm")


# =============================================================================
# JSON Extraction Utility
# =============================================================================

def extract_json_from_text(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks.

    Handles:
    - ```json...``` blocks
    - ```...``` blocks
    - Raw JSON objects

    Args:
        text: Text that may contain JSON

    Returns:
        Extracted JSON string

    Raises:
        ValueError: If no JSON can be found
    """
    # Try ```json block first
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    # Try ``` block
    if "```" in text:
        start = text.find("```") + 3
        # Skip language identifier if present
        if text[start:start+4].lower() in ("json", "java"):
            start = text.find("\n", start) + 1
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    # Try to find raw JSON object
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start != -1 and json_end > json_start:
        return text[json_start:json_end]

    # No JSON found
    raise ValueError(f"No JSON found in response: {text[:200]}...")


# =============================================================================
# ChatZAI - Custom ChatOpenAI subclass for ZAI/GLM models
# =============================================================================

class ChatZAI(ChatOpenAI):
    """ChatOpenAI subclass for ZAI/GLM models with custom structured output.

    ZAI models don't natively support .with_structured_output() - they return
    markdown-wrapped JSON instead of raw JSON. This class overrides the method
    to extract JSON from markdown responses.

    Usage:
        llm = ChatZAI(model="glm-4.5-air", api_key="...", base_url="...")
        structured = llm.with_structured_output(MyPydanticModel)
        result = await structured.ainvoke(messages)
    """

    def with_structured_output(self, schema: Type[BaseModel]) -> RunnableLambda:
        """Override to extract JSON from markdown responses.

        Args:
            schema: Pydantic model schema for structured output

        Returns:
            RunnableLambda that extracts JSON and validates against schema
        """
        log.debug("Structured output configured", schema=schema.__name__)

        def extract_and_validate(response_text: str) -> BaseModel:
            """Extract JSON from response and validate against schema."""
            try:
                json_str = extract_json_from_text(response_text)
                data = json.loads(json_str)
                return schema.model_validate(data)
            except (ValueError, json.JSONDecodeError) as e:
                log.error("JSON parse error", error=str(e), response_preview=response_text[:200])
                raise
            except ValidationError as e:
                log.error(
                    "Validation failed",
                    schema=schema.__name__,
                    error=str(e),
                    response_preview=response_text[:200]
                )
                raise

        return RunnableLambda(extract_and_validate)


# =============================================================================
# Page Summary Generation (for parallel scraping)
# =============================================================================

async def generate_page_summary(
    content: str,
    url: str,
    llm: ChatOpenAI,
    max_content: int = 15000,
) -> Dict[str, Any]:
    """Generate a structured summary of a documentation page.

    Used in parallel scraping to create compact summaries for state.

    Args:
        content: Page content to summarize
        url: Source URL
        llm: LLM instance for summarization
        max_content: Max content chars to send to LLM

    Returns:
        Dict with title, description, and URL
    """
    # Truncate content for LLM
    truncated = content[:max_content] if len(content) > max_content else content

    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=SUMMARY_PROMPT_TEMPLATE.format(
            url=url,
            content=truncated
        )),
    ]

    try:
        response = await llm.ainvoke(
            messages,
            config={"callbacks": get_langfuse_callback()}
        )
        content_text = response.content if hasattr(response, "content") else str(response)

        # Extract JSON from response
        json_str = extract_json_from_text(content_text)
        summary = json.loads(json_str)
        summary["url"] = url

        log.info(
            "Page summary generated",
            url=url,
            title=summary.get("title", "Untitled"),
            content_chars=len(content)
        )
        return summary

    except Exception as e:
        log.error("Summary generation failed", url=url, error=str(e))
        # Return fallback summary
        return {
            "url": url,
            "title": "Untitled",
            "description": f"Failed to summarize: {str(e)}",
        }


async def generate_page_summaries_parallel(
    pages: List[Dict[str, Any]],
    llm: ChatOpenAI,
    max_concurrent: int = 5,
    max_content: int = 15000,
) -> List[Dict[str, Any]]:
    """Generate summaries for multiple pages in parallel.

    Args:
        pages: List of page dicts with 'url' and 'content' keys
        llm: LLM instance for summarization
        max_concurrent: Max concurrent LLM calls
        max_content: Max content chars per page

    Returns:
        List of summary dicts
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def summarize_with_limit(page: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await generate_page_summary(
                page["content"],
                page["url"],
                llm,
                max_content,
            )

    tasks = [summarize_with_limit(page) for page in pages]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries = []
    errors = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(f"{pages[i]['url']}: {str(result)}")
            log.error("Summary failed", url=pages[i]["url"], error=str(result))
        else:
            summaries.append(result)

    log.info(
        "Batch summary complete",
        total=len(pages),
        success=len(summaries),
        errors=len(errors)
    )

    return summaries


# =============================================================================
# Backward compatibility aliases
# =============================================================================

def create_llm(config: LLMConfig) -> ChatOpenAI:
    """Create LLM instance from config (backward compatibility).

    Args:
        config: LLMConfig instance

    Returns:
        ChatOpenAI instance
    """
    # Get Langfuse callback for automatic tracing
    from ..utils.observability import get_langfuse_callback
    callbacks = get_langfuse_callback()

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        api_key=config.api_key,
        base_url=config.base_url,
        callbacks=callbacks if callbacks else None,
    )


generate_summaries_batch = generate_page_summaries_parallel
