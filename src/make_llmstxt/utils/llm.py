"""LangChain LLM Factory - Creates LLM instances for any provider.

Supports:
- OpenAI (default)
- Any OpenAI-compatible API (Anthropic via proxy, DeepSeek, etc.)
- ZAI/GLM models (with custom structured output handling)
- Local servers (llama.cpp, vLLM, etc.)
- Custom providers
"""

import json
from typing import Optional, Dict, Any, List, Type
from urllib.parse import urlparse

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ValidationError

from ..config import LLMConfig, PROVIDER_PROFILES
from ..core.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_RETRY_PROMPT_TEMPLATE,
)


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
            schema: Pydantic model class to validate against

        Returns:
            RunnableLambda that extracts and validates JSON
        """

        async def extract_and_validate(inputs: Any, config: Any = None) -> BaseModel:
            """Extract JSON from LLM response and validate against schema."""
            # Get messages from various input formats
            messages = self._normalize_messages(inputs)

            # Add schema instruction
            schema_instruction = self._build_schema_instruction(schema)
            messages[-1] = HumanMessage(
                content=messages[-1].content + "\n\n" + schema_instruction
            )

            # Call LLM
            response = await self.ainvoke(messages)
            response_text = response.content.strip()

            # Extract JSON from markdown
            json_str = self._extract_json(response_text)

            # Parse and validate
            try:
                parsed = json.loads(json_str)
                validated = schema(**parsed)
                logger.debug(f"[ChatZAI] Structured output validated: {schema.__name__}")
                return validated
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"[ChatZAI] Validation failed for {schema.__name__}: {e}")
                logger.error(f"[ChatZAI] Response was: {response_text[:500]}")
                raise RuntimeError(
                    f"Failed to extract valid {schema.__name__} from response: {e}"
                ) from e

        return RunnableLambda(extract_and_validate, name="zai_structured_output")

    def _normalize_messages(self, inputs: Any) -> List:
        """Convert various input formats to list of messages."""
        if isinstance(inputs, str):
            return [HumanMessage(content=inputs)]
        elif isinstance(inputs, list):
            if all(isinstance(m, dict) for m in inputs):
                # Convert dicts to messages
                messages = []
                for m in inputs:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "system":
                        messages.append(SystemMessage(content=content))
                    else:
                        messages.append(HumanMessage(content=content))
                return messages
            return inputs
        else:
            return [HumanMessage(content=str(inputs))]

    def _build_schema_instruction(self, schema: Type[BaseModel]) -> str:
        """Build instruction string from Pydantic schema."""
        fields_desc = []
        for field_name, field_info in schema.model_fields.items():
            field_type = str(field_info.annotation).replace("typing.", "")
            desc = field_info.description or ""
            required = field_info.is_required()
            req_str = "REQUIRED" if required else "OPTIONAL"
            fields_desc.append(f"  - {field_name} ({field_type}, {req_str}): {desc}")

        return (
            f"[SCHEMA] Output MUST be a JSON object with these fields:\n"
            + "\n".join(fields_desc)
            + "\nOutput ONLY valid JSON. No markdown code blocks."
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text. Delegates to module-level function."""
        return extract_json_from_text(text)


# =============================================================================
# Rate limit configurations
# =============================================================================
PROVIDER_RATE_LIMITS: Dict[str, tuple[float, int]] = {
    "openai": (0.5, 10),      # 1 req/2s, burst 10
    "anthropic": (0.3, 8),    # 1 req/3.3s, burst 8
    "deepseek": (0.5, 10),    # 1 req/2s, burst 10
    "openrouter": (0.5, 10),  # 1 req/2s, burst 10
    "zai": (0.2, 1),          # 1 req/5s, burst 1 (no bursting allowed)
    "glm": (0.2, 1),          # Same as zai
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
        Returns ChatZAI for zai/glm providers (handles markdown-wrapped JSON)
    """
    kwargs = config.get_langchain_kwargs()

    # Get rate limiter
    rate_limiter = get_rate_limiter(config.provider, enabled=rate_limit_enabled)
    if rate_limiter:
        kwargs["rate_limiter"] = rate_limiter

    # Use ChatZAI for ZAI/GLM providers (they return markdown-wrapped JSON)
    if config.provider in ("zai", "glm"):
        llm = ChatZAI(**kwargs)
        logger.info(
            f"[LLM] Created ChatZAI for {config.provider}: model={config.model}, "
            f"base_url={config.base_url}, rate_limit={rate_limiter is not None}"
        )
    else:
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
    import json

    content = markdown[:max_content_chars]

    # Use retry prompt if feedback provided, otherwise use base prompt
    if feedback:
        feedback_text = "\n".join(f"- {f}" for f in feedback)
        prompt = SUMMARY_RETRY_PROMPT_TEMPLATE.format(
            url=url, content=content, feedback=feedback_text
        )
    else:
        prompt = SUMMARY_PROMPT_TEMPLATE.format(url=url, content=content)

    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        response_content = response.content.strip()

        # Try to extract JSON from response
        response_content = extract_json_from_text(response_content)

        result = json.loads(response_content)

        # Validate output quality
        title = result.get("title", "").strip()
        description = result.get("description", "").strip()

        # Reject generic/placeholder titles
        generic_titles = {"page", "home", "about", "about us", "welcome", "untitled"}
        if not title or title.lower() in generic_titles:
            # Try to extract from URL as fallback
            from urllib.parse import urlparse
            path = urlparse(url).path.strip("/")
            if path and "/" in path:
                title = path.split("/")[-1].replace("-", " ").title()
            else:
                title = urlparse(url).netloc.split(".")[0].title()

        # Reject generic/placeholder descriptions
        generic_starters = ("this is", "this page", "page contains", "information about", "no description")
        if not description or any(description.lower().startswith(s) for s in generic_starters):
            description = f"Content from {urlparse(url).netloc}"

        return {
            "title": title,
            "description": description,
        }

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {url}: {e}")
        return {
            "title": urlparse(url).path.split("/")[-1].replace("-", " ").title() or "Page",
            "description": f"Content from {urlparse(url).netloc}",
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
