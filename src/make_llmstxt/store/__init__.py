"""Store factory for creating indexed stores with local embeddings.

Supports two modes:
1. Real-time indexing: Embeddings generated on put()
2. Embed-later mode: Store content first, batch embed later

The embed-later mode enables using large embedding models (e.g., Qwen-8B)
without competing for VRAM with the chat model.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from loguru import logger

from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore


# Default store directory for persistence
DEFAULT_STORE_DIR = Path(".make-llmstxt/store")


class EmbedLaterStore:
    """Store that supports embed-later mode for batch indexing.

    During Stage 1 (scraping), content is stored without embeddings.
    During Stage 1.5 (batch embed), embeddings are generated and added.
    During Stage 2 (generation), semantic search works on embedded content.

    This enables using large embedding models without VRAM competition.
    """

    def __init__(
        self,
        store_dir: Path = DEFAULT_STORE_DIR,
        embedding_base_url: Optional[str] = None,
        embedding_model: str = "embed",
        embedding_dims: int = 768,
    ):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims

        # Internal storage (simple JSON files for now)
        self._content_dir = self.store_dir / "content"
        self._content_dir.mkdir(exist_ok=True)

        # Index file tracking what's embedded
        self._index_file = self.store_dir / "index.json"
        self._index: Dict[str, Dict] = self._load_index()

        # Lazy-loaded embedding store (only when needed)
        self._indexed_store: Optional[InMemoryStore] = None

    def _load_index(self) -> Dict[str, Dict]:
        """Load the index file."""
        if self._index_file.exists():
            return json.loads(self._index_file.read_text())
        return {}

    def _save_index(self):
        """Save the index file."""
        self._index_file.write_text(json.dumps(self._index, indent=2))

    def _content_path(self, key: str) -> Path:
        """Get path for content file."""
        return self._content_dir / f"{key}.json"

    def put(
        self,
        namespace: tuple,
        key: str,
        value: Dict[str, Any],
        index: bool = False,
    ) -> str:
        """Store content with optional indexing.

        Args:
            namespace: Tuple for namespacing (e.g., ("memories", "raw_pages"))
            key: Unique key for this content
            value: Dict with content (must have 'url' and 'content' keys)
            index: If True, generate embeddings immediately.
                   If False, store for later batch embedding.

        Returns:
            The key used for storage
        """
        ns_key = f"{'/'.join(namespace)}/{key}"

        # Store content to disk
        content_path = self._content_path(key)
        content_path.write_text(json.dumps({
            "namespace": namespace,
            "key": key,
            "value": value,
            "stored_at": datetime.now().isoformat(),
            "embedded": False,
        }, indent=2))

        # Update index
        self._index[ns_key] = {
            "key": key,
            "namespace": namespace,
            "embedded": False,
            "stored_at": datetime.now().isoformat(),
        }
        self._save_index()

        logger.debug(f"[Store] Stored {key} (indexed={index})")

        # If indexing requested and embedder available, embed now
        if index and self.embedding_base_url:
            self._embed_single(key, value)

        return key

    async def aput(
        self,
        namespace: tuple,
        key: str,
        value: Dict[str, Any],
        index: bool = False,
    ) -> str:
        """Async version of put()."""
        return self.put(namespace, key, value, index)

    def _embed_single(self, key: str, value: Dict[str, Any]):
        """Embed a single document."""
        if not self.embedding_base_url:
            logger.warning(f"[Store] No embedder configured, skipping {key}")
            return

        try:
            embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_base=self.embedding_base_url,
                openai_api_key="dummy",
            )

            content = value.get("content", "")
            vector = embeddings.embed_query(content)

            # Store vector alongside content
            content_path = self._content_path(key)
            data = json.loads(content_path.read_text())
            data["embedding"] = vector
            data["embedded"] = True
            data["embedded_at"] = datetime.now().isoformat()
            content_path.write_text(json.dumps(data))

            # Update index
            for ns_key, info in self._index.items():
                if info["key"] == key:
                    info["embedded"] = True
            self._save_index()

            logger.debug(f"[Store] Embedded {key}")

        except Exception as e:
            logger.error(f"[Store] Failed to embed {key}: {e}")

    def list_unembedded(self, namespace: tuple = None) -> List[Dict[str, Any]]:
        """List all documents that need embedding.

        Args:
            namespace: Optional namespace filter

        Returns:
            List of dicts with key, namespace, value
        """
        unembedded = []
        for ns_key, info in self._index.items():
            if not info.get("embedded", False):
                if namespace and tuple(info["namespace"]) != namespace:
                    continue

                content_path = self._content_path(info["key"])
                if content_path.exists():
                    data = json.loads(content_path.read_text())
                    unembedded.append({
                        "key": info["key"],
                        "namespace": tuple(info["namespace"]),
                        "value": data["value"],
                    })

        return unembedded

    def batch_embed(
        self,
        embedding_base_url: str,
        embedding_model: str = None,
        embedding_dims: int = None,
        namespace: tuple = None,
        batch_size: int = 10,
    ) -> int:
        """Batch embed all unembedded documents.

        This is the Stage 1.5 step - run with embedding server only.

        Args:
            embedding_base_url: URL of embedding server (e.g., http://localhost:8080/v1)
            embedding_model: Model name (overrides init setting)
            embedding_dims: Embedding dims (overrides init setting)
            namespace: Only embed documents in this namespace
            batch_size: Number of documents to embed at once

        Returns:
            Number of documents embedded
        """
        embeddings = OpenAIEmbeddings(
            model=embedding_model or self.embedding_model,
            openai_api_base=embedding_base_url,
            openai_api_key="dummy",
        )

        unembedded = self.list_unembedded(namespace)
        if not unembedded:
            logger.info("[Store] No documents need embedding")
            return 0

        logger.info(f"[Store] Batch embedding {len(unembedded)} documents...")

        embedded_count = 0
        for i in range(0, len(unembedded), batch_size):
            batch = unembedded[i:i + batch_size]

            # Get content for batch
            texts = [doc["value"].get("content", "") for doc in batch]

            try:
                # Generate embeddings
                vectors = embeddings.embed_documents(texts)

                # Store vectors
                for doc, vector in zip(batch, vectors):
                    content_path = self._content_path(doc["key"])
                    data = json.loads(content_path.read_text())
                    data["embedding"] = vector
                    data["embedded"] = True
                    data["embedded_at"] = datetime.now().isoformat()
                    content_path.write_text(json.dumps(data))

                    # Update index
                    ns_key = f"{'/'.join(doc['namespace'])}/{doc['key']}"
                    self._index[ns_key]["embedded"] = True

                self._save_index()
                embedded_count += len(batch)
                logger.info(f"[Store] Embedded {embedded_count}/{len(unembedded)} documents")

            except Exception as e:
                logger.error(f"[Store] Batch embedding failed at {i}: {e}")
                # Try one by one as fallback
                for doc in batch:
                    try:
                        vector = embeddings.embed_query(doc["value"].get("content", ""))
                        content_path = self._content_path(doc["key"])
                        data = json.loads(content_path.read_text())
                        data["embedding"] = vector
                        data["embedded"] = True
                        content_path.write_text(json.dumps(data))
                        embedded_count += 1
                    except Exception as e2:
                        logger.error(f"[Store] Failed to embed {doc['key']}: {e2}")

        return embedded_count

    def search(self, query: str, namespace: tuple = None, limit: int = 5) -> List[Dict]:
        """Semantic search on embedded content.

        Requires embeddings to have been generated (via batch_embed or put with index=True).
        """
        if not self.embedding_base_url:
            logger.warning("[Store] No embedder configured for search")
            return []

        embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_base=self.embedding_base_url,
            openai_api_key="dummy",
        )

        # Get query embedding
        query_vector = embeddings.embed_query(query)

        # Find all embedded documents
        results = []
        for ns_key, info in self._index.items():
            if not info.get("embedded", False):
                continue

            if namespace and tuple(info["namespace"]) != namespace:
                continue

            content_path = self._content_path(info["key"])
            if not content_path.exists():
                continue

            data = json.loads(content_path.read_text())
            doc_vector = data.get("embedding")
            if not doc_vector:
                continue

            # Cosine similarity
            import math
            dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
            norm_a = math.sqrt(sum(a * a for a in query_vector))
            norm_b = math.sqrt(sum(b * b for b in doc_vector))
            similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

            results.append({
                "key": info["key"],
                "namespace": tuple(info["namespace"]),
                "value": data["value"],
                "score": similarity,
            })

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def asearch(self, query: str, namespace: tuple = None, limit: int = 5) -> List[Dict]:
        """Async version of search()."""
        return self.search(query, namespace, limit)

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        total = len(self._index)
        embedded = sum(1 for info in self._index.values() if info.get("embedded", False))

        return {
            "total_documents": total,
            "embedded": embedded,
            "unembedded": total - embedded,
            "store_dir": str(self.store_dir),
        }


# =============================================================================
# Factory functions
# =============================================================================

def create_store(
    store_dir: Path = DEFAULT_STORE_DIR,
    embedding_base_url: Optional[str] = None,
    embedding_model: str = "embed",
    embedding_dims: int = 768,
) -> EmbedLaterStore:
    """Create an EmbedLaterStore.

    Args:
        store_dir: Directory for persistent storage
        embedding_base_url: URL for embedding API (optional, can set later for batch embed)
        embedding_model: Model name for embeddings
        embedding_dims: Embedding dimensions

    Returns:
        EmbedLaterStore instance
    """
    return EmbedLaterStore(
        store_dir=store_dir,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
    )


def create_store_with_embeddings(
    embedding_base_url: Optional[str] = None,
    embedding_model: str = "embed",
    embedding_dims: int = 768,
) -> Optional[EmbedLaterStore]:
    """Create store with embeddings configured (backward compat).

    Deprecated: Use create_store() instead.
    """
    if not embedding_base_url:
        logger.info("[Store] No embedding_base_url configured, store disabled")
        return None

    return create_store(
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
    )


def get_store_from_config(config) -> Optional[EmbedLaterStore]:
    """Create store from GeneratorConfig.

    Args:
        config: GeneratorConfig instance

    Returns:
        EmbedLaterStore or None
    """
    return create_store(
        store_dir=Path(config.output_dir) / ".store" if config.output_dir else DEFAULT_STORE_DIR,
        embedding_base_url=config.embedding_base_url,
        embedding_model=config.embedding_model,
        embedding_dims=config.embedding_dims,
    )
