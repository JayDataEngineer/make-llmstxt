"""Store factory for creating indexed stores with local embeddings."""

from typing import Optional
from loguru import logger

from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore


def create_store_with_embeddings(
    embedding_base_url: Optional[str] = None,
    embedding_model: str = "embed",
    embedding_dims: int = 768,
) -> Optional[InMemoryStore]:
    """Create an indexed store with local embeddings.

    Args:
        embedding_base_url: Base URL for embedding API (e.g., http://localhost:8080/v1)
        embedding_model: Model name for embeddings
        embedding_dims: Embedding dimensions

    Returns:
        InMemoryStore with embeddings configured, or None if no base_url provided
    """
    if not embedding_base_url:
        logger.info("[Store] No embedding_base_url configured, store disabled")
        return None

    try:
        # Create embeddings client pointing to local llama.cpp server
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=embedding_base_url,
            openai_api_key="dummy",  # Required but unused by llama.cpp
        )

        # Create indexed in-memory store
        store = InMemoryStore(
            index={
                "embeddings": embeddings,
                "dims": embedding_dims,
                "fields": ["content", "url"],
            }
        )

        logger.info(
            f"[Store] Created indexed store: base_url={embedding_base_url}, "
            f"model={embedding_model}, dims={embedding_dims}"
        )
        return store

    except Exception as e:
        logger.error(f"[Store] Failed to create store: {e}")
        return None


def get_store_from_config(config) -> Optional[InMemoryStore]:
    """Create store from GeneratorConfig.

    Args:
        config: GeneratorConfig instance

    Returns:
        InMemoryStore or None
    """
    return create_store_with_embeddings(
        embedding_base_url=config.embedding_base_url,
        embedding_model=config.embedding_model,
        embedding_dims=config.embedding_dims,
    )
