"""Tools for searching scraped content in LangGraph BaseStore.

Stage 2 of the two-stage pipeline uses these tools to retrieve
full content from the store based on semantic search.
"""

from typing import Optional

from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from loguru import logger


@tool
async def search_docs(query: str, store: BaseStore, limit: int = 5) -> str:
    """Search scraped documentation semantically.

    Use this to find detailed content about specific topics from previously
    scraped pages. Returns the most relevant document sections with source URLs.

    Args:
        query: The search query describing what you're looking for
        store: The LangGraph BaseStore instance (injected by the runtime)
        limit: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with source URLs and content snippets
    """
    if not store:
        return "Error: Store not available. Content search requires a configured store."

    try:
        # Perform semantic search on the store
        results = await store.asearch(
            namespace=("memories", "raw_pages"),
            query=query,
            limit=limit
        )

        if not results:
            return f"No relevant documents found for query: {query}"

        formatted = []
        for r in results:
            url = r.value.get("url", "unknown")
            content = r.value.get("content", "")
            score = getattr(r, "score", 0.0)

            # Truncate content for readability
            content_preview = content[:3000] + "..." if len(content) > 3000 else content

            formatted.append(
                f"**Source: {url}** (relevance: {score:.2f})\n"
                f"{content_preview}"
            )

        logger.info(f"[search_docs] Found {len(results)} results for: {query[:50]}...")
        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error(f"[search_docs] Search failed: {e}")
        return f"Search error: {str(e)}"


@tool
async def get_doc_by_url(url: str, store: BaseStore) -> str:
    """Retrieve the full content of a specific scraped page by URL.

    Use this when you know the exact URL and want the complete content.

    Args:
        url: The exact URL of the page to retrieve
        store: The LangGraph BaseStore instance (injected by the runtime)

    Returns:
        The full content of the page, or an error message if not found
    """
    if not store:
        return "Error: Store not available."

    try:
        # Search for the specific URL
        # Note: This relies on the content being indexed by URL field
        results = await store.asearch(
            namespace=("memories", "raw_pages"),
            query=f"url:{url}",
            limit=1
        )

        if not results:
            return f"No document found for URL: {url}"

        doc = results[0]
        actual_url = doc.value.get("url", "unknown")
        content = doc.value.get("content", "")
        scraped_at = doc.value.get("scraped_at", "unknown")

        logger.info(f"[get_doc_by_url] Retrieved: {url}")
        return (
            f"**URL: {actual_url}**\n"
            f"**Scraped: {scraped_at}**\n\n"
            f"{content}"
        )

    except Exception as e:
        logger.error(f"[get_doc_by_url] Retrieval failed: {e}")
        return f"Retrieval error: {str(e)}"
