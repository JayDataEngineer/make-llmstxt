"""CLI command for batch embedding stored content.

Stage 1.5 of the pipeline - run with embedding server only.

Usage:
    make-llmstxt embed --base-url http://localhost:8080/v1 --model embed
    make-llmstxt embed --base-url http://localhost:8080/v1 --model qwen-embed --dims 4096
    make-llmstxt embed --stats  # Show store statistics
"""

import argparse
from pathlib import Path
from loguru import logger

from ..store import EmbedLaterStore, DEFAULT_STORE_DIR


def add_embed_parser(subparsers):
    """Add embed command to CLI."""
    embed_parser = subparsers.add_parser(
        "embed",
        help="Batch embed stored content (Stage 1.5)",
        description="""
Batch embed all stored content using a dedicated embedding server.

This is Stage 1.5 of the pipeline:
1. Run scraping with chat model (content stored without embeddings)
2. SPIN DOWN chat server
3. SPIN UP embedding server (e.g., Qwen-Embedding-8B)
4. Run this command to generate embeddings
5. SPIN DOWN embedding server
6. SPIN UP chat server for Stage 2 generation

Example:
    # Start embedding server
    llama-server --model qwen-embed-8b.gguf --embedding --port 8081

    # Run batch embedding
    make-llmstxt embed --base-url http://localhost:8081/v1 --model qwen-embed

    # Stop embedding server, start chat server
    # Now semantic search works in Stage 2
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    embed_parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for embedding API (e.g., http://localhost:8081/v1)",
    )
    embed_parser.add_argument(
        "--model",
        type=str,
        default="embed",
        help="Model name for embeddings (default: embed)",
    )
    embed_parser.add_argument(
        "--dims",
        type=int,
        default=768,
        help="Embedding dimensions (768 for nomic, 1024 for bge-large, 4096 for Qwen-8B)",
    )
    embed_parser.add_argument(
        "--store-dir",
        type=str,
        default=str(DEFAULT_STORE_DIR),
        help=f"Store directory (default: {DEFAULT_STORE_DIR})",
    )
    embed_parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Only embed documents in this namespace (e.g., 'memories/raw_pages')",
    )
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of documents to embed per batch (default: 10)",
    )
    embed_parser.add_argument(
        "--stats",
        action="store_true",
        help="Show store statistics instead of embedding",
    )

    embed_parser.set_defaults(func=run_embed)


def run_embed(args):
    """Run the embed command."""
    store = EmbedLaterStore(store_dir=Path(args.store_dir))

    # Show stats and exit
    if args.stats:
        stats = store.stats()
        print("\n📊 Store Statistics:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Embedded:       {stats['embedded']}")
        print(f"   Unembedded:     {stats['unembedded']}")
        print(f"   Store dir:      {stats['store_dir']}")

        if stats['unembedded'] > 0:
            print(f"\n   Run: make-llmstxt embed --base-url <url> --model <model>")
        return 0

    # Require base-url for embedding
    if not args.base_url:
        logger.error("--base-url is required for embedding")
        print("\nError: --base-url is required")
        print("Example: make-llmstxt embed --base-url http://localhost:8081/v1 --model embed")
        return 1

    # Parse namespace if provided
    namespace = None
    if args.namespace:
        namespace = tuple(args.namespace.split("/"))

    # Run batch embedding
    logger.info(f"Starting batch embedding with {args.base_url}")
    logger.info(f"Model: {args.model}, Dims: {args.dims}")

    count = store.batch_embed(
        embedding_base_url=args.base_url,
        embedding_model=args.model,
        embedding_dims=args.dims,
        namespace=namespace,
        batch_size=args.batch_size,
    )

    print(f"\n✅ Embedded {count} documents")

    # Show final stats
    stats = store.stats()
    print(f"   Total: {stats['total_documents']}, Embedded: {stats['embedded']}, Remaining: {stats['unembedded']}")

    return 0
