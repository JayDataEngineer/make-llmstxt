"""Command-line interface for make-llmstxt."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import AppConfig, LLMConfig, PROVIDER_PROFILES
from .generator import generate_llmstxt
from .__init__ import __version__
from .logging import setup_logging

console = Console()


def list_providers():
    """Print available providers."""
    console.print("\n[bold]Available LLM Providers:[/bold]\n")

    for name, profile in PROVIDER_PROFILES.items():
        console.print(f"  [cyan]{name}[/cyan]")
        console.print(f"    Default model: {profile['default_model']}")
        console.print(f"    Env key: {profile['env_key']}")
        console.print(f"    {profile['description']}")
        console.print()

    console.print("[dim]Set provider via LLM_PROVIDER env var or --provider flag[/dim]")
    console.print("[dim]Set API key via provider's env key (e.g., OPENAI_API_KEY)[/dim]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate llms.txt and llms-full.txt files for any website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with OpenAI (default)
  make-llmstxt https://example.com

  # Use a different provider
  make-llmstxt https://example.com --provider deepseek

  # Use local server
  make-llmstxt https://example.com --provider local --base-url http://localhost:8000/v1

  # Limit URLs and output to specific directory
  make-llmstxt https://example.com --max-urls 50 --output-dir ./output

Environment Variables:
  LLM_PROVIDER       Provider to use (openai, anthropic, deepseek, local, etc.)
  LLM_MODEL          Model name to use
  LLM_API_KEY        API key (if not using provider-specific env var)
  LLM_BASE_URL       Base URL for API

  OPENAI_API_KEY     OpenAI API key
  ANTHROPIC_API_KEY  Anthropic API key
  DEEPSEEK_API_KEY   DeepSeek API key
  OPENROUTER_API_KEY OpenRouter API key
  ZAI_API_KEY        ZAI API key

  MCP_HOST           MCP server host (default: 100.85.22.99)
  MCP_PORT           MCP server port (default: 8000)

  LOG_LEVEL          Log level (DEBUG, INFO, WARNING, ERROR)
  LOG_FILE           Path to log file (optional)
  LOG_JSON           Use JSON log format (true/false)
        """,
    )

    parser.add_argument("url", nargs="?", help="Website URL to process")
    parser.add_argument(
        "--version", "-v", action="version", version=f"make-llmstxt {__version__}"
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Maximum number of URLs to process (default: unlimited, process all URLs in sitemap)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory to save output files (default: current directory)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=list(PROVIDER_PROFILES.keys()),
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model name to use",
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for LLM API (for custom providers)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for LLM provider",
    )
    parser.add_argument(
        "--mcp-host",
        default="100.85.22.99",
        help="MCP server host (default: 100.85.22.99 - Tailscale)",
    )
    parser.add_argument(
        "--mcp-port",
        default="8000",
        help="MCP server port (default: 8000)",
    )
    parser.add_argument(
        "--no-full-text",
        action="store_true",
        help="Don't generate llms-full.txt file",
    )
    # Deep Draft-Critic options
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Disable critic validation (faster but lower quality)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum draft-critic rounds (default: 3)",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.7,
        help="Minimum score to pass critic, 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--critic-strict",
        action="store_true",
        help="Fail generation if critic errors occur",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Use JSON log format",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available LLM providers",
    )

    args = parser.parse_args()

    # Load .env file
    load_dotenv()

    # Setup logging with loguru
    log_level = "DEBUG" if args.verbose else os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = Path(args.log_file) if args.log_file else (Path(os.getenv("LOG_FILE")) if os.getenv("LOG_FILE") else None)
    log_json = args.log_json or os.getenv("LOG_JSON", "false").lower() == "true"

    setup_logging(
        level=log_level,
        log_file=log_file,
        json_format=log_json,
    )

    logger.info(f"make-llmstxt v{__version__} starting up")

    # List providers if requested
    if args.list_providers:
        list_providers()
        return

    # Validate URL
    if not args.url:
        parser.error("URL is required (or use --list-providers)")

    # Build configuration
    config = AppConfig.from_env()
    logger.debug(f"Loaded config from env: provider={config.llm.provider}")

    # Configure MCP scraper
    config.mcp.host = args.mcp_host
    config.mcp.port = int(args.mcp_port)
    logger.info(f"Using MCP scraper at {args.mcp_host}:{args.mcp_port}")

    # Override LLM settings with command line args
    if args.provider:
        config.llm.provider = args.provider
        if args.provider in PROVIDER_PROFILES:
            profile = PROVIDER_PROFILES[args.provider]
            config.llm.model = args.model or profile["default_model"]

    if args.model:
        config.llm.model = args.model
    if args.base_url:
        config.llm.base_url = args.base_url
    if args.api_key:
        config.llm.api_key = args.api_key

    logger.info(f"LLM: provider={config.llm.provider}, model={config.llm.model}")

    if not config.llm.api_key and config.llm.provider != "local":
        console.print(f"[red]Error: LLM API key not provided for provider '{config.llm.provider}'[/red]")
        console.print(f"Set the appropriate environment variable or use --api-key")
        logger.error(f"LLM API key not provided for provider '{config.llm.provider}'")
        sys.exit(1)

    # Run generation
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            task_id = None

            def progress_callback(stage, current, total, message):
                nonlocal task_id
                logger.debug(f"Progress: {stage} - {current}/{total} - {message}")
                if stage == "mapping":
                    if task_id is None:
                        task_id = progress.add_task(message, total=total)
                    progress.update(task_id, description=message, completed=current)
                elif stage == "scraping":
                    if task_id is None:
                        task_id = progress.add_task(message, total=total)
                    progress.update(task_id, description=message, completed=current)

            import asyncio

            logger.info(f"Starting generation for {args.url}")
            result = asyncio.run(
                generate_llmstxt(
                    args.url,
                    config=config,
                    max_urls=args.max_urls,
                    include_full_text=not args.no_full_text,
                    output_dir=Path(args.output_dir),
                    progress_callback=progress_callback,
                    enable_critic=not args.no_critic,
                    max_retries=args.max_rounds,
                    pass_threshold=args.pass_threshold,
                    fail_on_critic_error=args.critic_strict,
                )
            )

        # Print summary
        console.print()
        console.print(f"[green]Success![/green] Processed {result.num_urls_processed} out of {result.num_urls_total} URLs")
        console.print(f"Files saved to {args.output_dir}/")
        logger.info(f"Generation complete: {result.num_urls_processed}/{result.num_urls_total} URLs processed")

    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
