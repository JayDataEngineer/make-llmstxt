"""Store factory for creating indexed stores with real-time embeddings.

Uses LangGraph's InMemoryStore with embeddings configured.
Embeddings are generated on put() automatically.
"""

from pathlib import Path
from typing import Optional

from loguru import logger
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore


def create_store(
    embedding_base_url: Optional[str] = None,
    embedding_model: str = "embed",
    embedding_dims: int = 1024,
) -> Optional[InMemoryStore]:
    """Create an InMemoryStore with real-time embeddings.

    When content is stored via put(), embeddings are generated automatically.
    Requires an embedding server (e.g., llama.cpp in router mode with embed model).

    Args:
        embedding_base_url: URL of embedding API (e.g., http://localhost:8001/v1)
        embedding_model: Model name for embeddings (router mode: use preset model name)
        embedding_dims: Embedding dimensions (1024 for jina-v5-small)

    Returns:
        InMemoryStore with embeddings, or None if no embedding_base_url configured
    """
    if not embedding_base_url:
        logger.info("[Store] No embedding_base_url configured, store disabled")
        return None

    try:
        # Create embeddings client pointing to local embedding server
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=embedding_base_url,
            openai_api_key="dummy",  # Required but unused by llama.cpp
        )

        # Create indexed store - embeddings generated on put()
        store = InMemoryStore(
            index={
                "embed": embeddings,
                "dims": embedding_dims,
                "fields": ["content", "url"],
            }
        )

        logger.info(
            f"[Store] Created indexed store: {embedding_base_url}, "
            f"model={embedding_model}, dims={embedding_dims}"
        )
        return store

    except Exception as e:
        logger.error(f"[Store] Failed to create store: {e}")
        return None
