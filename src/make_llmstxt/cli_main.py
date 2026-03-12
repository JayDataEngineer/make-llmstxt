"""Command-line interface for make-llmstxt.

Supports two modes:
- llmstxt: Generate llms.txt (index of pages with titles/descriptions)
- skill: Generate skill package (SKILL.md, scripts/, references/)

Both modes use the Deep Agent pattern (generator → critic → loop).
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from .config import AppConfig, PROVIDER_PROFILES
from .generators.llmstxt import LLMsTxtGenerator
from .generators.skill import SkillGenerator
from .generators.prompts.llmstxt import LLMSTXT_PROMPTS
from .generators.prompts.skill import SKILL_PROMPTS
from .__init__ import __version__
from .utils.logging import setup_logging, StructuredLogger
from .core import GeneratorConfig
from .validators import validate_skill_generation

console = Console()
log = StructuredLogger("cli")


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


def handle_llmstxt(args, env_config: AppConfig):
    """Handle llms.txt generation using Deep Agent pattern."""
    # Build unified config
    config = GeneratorConfig(
        url=args.url,
        output_dir=Path(args.output_dir),
        mcp_host=env_config.mcp.host,
        mcp_port=env_config.mcp.port,
        mcp_url=env_config.mcp.url,
        max_rounds=args.max_rounds,
        pass_threshold=args.pass_threshold,
        api_key=env_config.llm.api_key,
        base_url=env_config.llm.base_url,
        model=env_config.llm.model,
        model_display_name=env_config.llm.model_display_name,
        provider=env_config.llm.provider,
        max_urls=args.max_urls,
        prompts=LLMSTXT_PROMPTS,
    )

    # Override with CLI args
    if args.provider:
        config.provider = args.provider
        if args.provider in PROVIDER_PROFILES:
            profile = PROVIDER_PROFILES[args.provider]
            config.model = args.model or profile["default_model"]
            # Also get base_url from provider settings
            provider_base_urls = {
                "openai": None,
                "anthropic": "https://api.anthropic.com/v1",
                "deepseek": "https://api.deepseek.com/v1",
                "openrouter": "https://openrouter.ai/api/v1",
                "zai": "https://api.z.ai/api/coding/paas/v4",
                "glm": "https://api.z.ai/api/coding/paas/v4",
            }
            if args.provider in provider_base_urls:
                config.base_url = args.base_url or provider_base_urls[args.provider]
            # Get API key from correct env var
            env_key = profile["env_key"]
            config.api_key = args.api_key or os.getenv(env_key)
    if args.model:
        config.model = args.model
    if args.model_display_name:
        config.model_display_name = args.model_display_name
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    # Embedding config for semantic search store
    # Priority: CLI args > env vars > default (same as base_url for router mode)
    embedding_base_url = (
        args.embedding_base_url or
        os.getenv("EMBEDDING_BASE_URL") or
        config.base_url  # Default: use same server (router mode)
    )
    config.embedding_base_url = embedding_base_url
    config.embedding_model = args.embedding_model or os.getenv("EMBEDDING_MODEL", "embed")
    embedding_dims_str = os.getenv("EMBEDDING_DIMS", "1024")
    config.embedding_dims = args.embedding_dims or int(embedding_dims_str)
    log.debug(
        "Embedding config",
        base_url=config.embedding_base_url,
        model=config.embedding_model,
        dims=config.embedding_dims
    )

    # Create generator
    generator = LLMsTxtGenerator(config)

    mcp_display = config.mcp_url or f"{config.mcp_host}:{config.mcp_port}"
    console.print(Panel.fit(
        f"[bold cyan]LLMs.txt Generation[/bold cyan]\n"
        f"URL: {args.url}\n"
        f"Provider: {config.provider}\n"
        f"Model: {config.model}\n"
        f"MCP: {mcp_display}",
        title="make-llmstxt",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Starting...", total=None)

        def progress_callback(message, percent):
            progress.update(task_id, description=message)

        try:
            result = asyncio.run(
                generator.generate(
                    url=args.url,
                    output_file=Path(args.output_dir) / "llms.txt",
                    progress_callback=progress_callback,
                )
            )

            console.print()
            console.print(f"[green]✓ Success![/green] Generated llms.txt")
            console.print(f"  Output: {result.output_path}")
            console.print(f"  Critic passed: {result.stats.get('critic_passed', False)}")
            console.print(f"  Rounds: {result.stats.get('rounds', 0)}")
            log.info("Generation complete", output_path=str(result.output_path))

        except Exception as e:
            log.error("Generation", f"Failed: {e}")
            console.print(f"[red]Error: {e}[/red]")
            if args.verbose:
                console.print_exception()
            sys.exit(1)


def handle_skill(args, env_config: AppConfig):
    """Handle skill generation using Deep Agent pattern."""
    library_name = args.url.split("//")[-1].split("/")[0].replace("www.", "").replace("docs.", "").split(".")[0]
    output_subdir = Path(args.output_dir) / library_name

    # Run validation if requested
    if args.validate or args.clean:
        validation_result = validate_skill_generation(
            output_dir=output_subdir,
            mcp_host=env_config.mcp.host,
            mcp_port=env_config.mcp.port,
            llm_base_url=env_config.llm.base_url,
            auto_fix=args.clean,
        )

        if validation_result.fixes_applied:
            for fix in validation_result.fixes_applied:
                console.print(f"[green]✓ {fix}[/green]")

        if validation_result.warnings:
            for warning in validation_result.warnings:
                console.print(f"[yellow]⚠ {warning}[/yellow]")

        if validation_result.errors:
            for error in validation_result.errors:
                console.print(f"[red]✗ {error}[/red]")
            if not args.clean:
                console.print("[dim]Run with --clean to automatically fix issues[/dim]")
            sys.exit(1)

        if args.validate and validation_result.valid:
            console.print("[green]✓ All validation checks passed[/green]")

    # Read max_rounds from env or use CLI default
    max_rounds = args.max_rounds or int(os.getenv("SKILL_MAX_ROUNDS", "3"))

    # Build unified config
    config = GeneratorConfig(
        url=args.url,
        output_dir=Path(args.output_dir),
        mcp_host=env_config.mcp.host,
        mcp_port=env_config.mcp.port,
        max_rounds=max_rounds,
        api_key=env_config.llm.api_key,
        base_url=env_config.llm.base_url,
        model=env_config.llm.model,
        model_display_name=env_config.llm.model_display_name,
        provider=env_config.llm.provider,
        max_urls=args.max_urls,
        prompts=SKILL_PROMPTS,
    )

    # Override with CLI args
    if args.provider:
        config.provider = args.provider
        if args.provider in PROVIDER_PROFILES:
            profile = PROVIDER_PROFILES[args.provider]
            config.model = args.model or profile["default_model"]
            # Also get base_url from provider settings
            provider_base_urls = {
                "openai": None,
                "anthropic": "https://api.anthropic.com/v1",
                "deepseek": "https://api.deepseek.com/v1",
                "openrouter": "https://openrouter.ai/api/v1",
                "zai": "https://api.z.ai/api/coding/paas/v4",
                "glm": "https://api.z.ai/api/coding/paas/v4",
            }
            if args.provider in provider_base_urls:
                config.base_url = args.base_url or provider_base_urls[args.provider]
            # Get API key from correct env var
            env_key = profile["env_key"]
            config.api_key = args.api_key or os.getenv(env_key)
    if args.model:
        config.model = args.model
    if args.model_display_name:
        config.model_display_name = args.model_display_name
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    # Embedding config for semantic search store (same as llmstxt mode)
    embedding_base_url = (
        args.embedding_base_url or
        os.getenv("EMBEDDING_BASE_URL") or
        config.base_url  # Default: use same server (router mode)
    )
    log.debug(f"DEBUG: args.embedding_base_url={args.embedding_base_url}")
    log.debug(f"DEBUG: os.getenv('EMBEDDING_BASE_URL')={os.getenv('EMBEDDING_BASE_URL')}")
    log.debug(f"DEBUG: config.base_url={config.base_url}")
    log.debug(f"DEBUG: final embedding_base_url={embedding_base_url}")
    config.embedding_base_url = embedding_base_url
    config.embedding_model = args.embedding_model or os.getenv("EMBEDDING_MODEL", "embed")
    config.embedding_dims = args.embedding_dims or int(os.getenv("EMBEDDING_DIMS", "1024"))
    log.debug(f"DEBUG: config.embedding_base_url after set={config.embedding_base_url}")

    # Create generator
    generator = SkillGenerator(config)

    console.print(Panel.fit(
        f"[bold cyan]Skill Package Generation[/bold cyan]\n"
        f"URL: {args.url}\n"
        f"Library: {library_name}\n"
        f"Provider: {config.provider}\n"
        f"Model: {config.model}\n"
        f"MCP: {config.mcp_url or f'{config.mcp_host}:{config.mcp_port}'}",
        title="make-llmstxt",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Starting...", total=None)

        def progress_callback(message, percent):
            progress.update(task_id, description=message)

        try:
            result = asyncio.run(
                generator.generate(
                    url=args.url,
                    output_dir=Path(args.output_dir),
                    progress_callback=progress_callback,
                )
            )

            console.print()
            console.print(f"[green]✓ Success![/green] Skill package created")
            console.print(f"  Output: {result.output_path}")
            console.print(f"  Files created: {result.stats.get('files_created', 0)}")
            log.info("Skill package created", output_path=str(result.output_path))

        except Exception as e:
            log.error("Skill generation", f"Failed: {e}")
            console.print(f"[red]Error: {e}[/red]")
            if args.verbose:
                console.print_exception()
            sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate llms.txt or skill packages for documentation websites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--version", "-v", action="version", version=f"make-llmstxt {__version__}"
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available LLM providers",
    )

    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Generation mode")

    # ========== llmstxt mode ==========
    llmstxt_parser = subparsers.add_parser(
        "llmstxt",
        help="Generate llms.txt (index of pages with titles/descriptions)",
    )
    llmstxt_parser.add_argument("url", help="Website URL to process")
    llmstxt_parser.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Maximum number of URLs to process (default: unlimited)",
    )
    llmstxt_parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory to save output files (default: current directory)",
    )
    llmstxt_parser.add_argument(
        "--provider", "-p",
        choices=list(PROVIDER_PROFILES.keys()),
        help="LLM provider to use",
    )
    llmstxt_parser.add_argument("--model", "-m", help="Model name to use (API identifier)")
    llmstxt_parser.add_argument("--model-display-name", help="Display name for observability (e.g., 'Qwen-2.5-7B' when model='llm')")
    llmstxt_parser.add_argument("--base-url", help="Base URL for LLM API")
    llmstxt_parser.add_argument("--api-key", help="API key for LLM provider")
    llmstxt_parser.add_argument("--max-rounds", type=int, default=3, help="Maximum draft-critic rounds (default: 3)")
    llmstxt_parser.add_argument("--pass-threshold", type=float, default=0.7, help="Minimum score to pass critic (default: 0.7)")
    llmstxt_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    llmstxt_parser.add_argument("--log-file", help="Path to log file")
    llmstxt_parser.add_argument("--log-json", action="store_true", help="Use JSON log format")
    # Embedding config (for semantic search store)
    llmstxt_parser.add_argument("--embedding-base-url", help="Base URL for embedding API (default: same as base-url)")
    llmstxt_parser.add_argument("--embedding-model", default="embed", help="Embedding model name (default: embed)")
    llmstxt_parser.add_argument("--embedding-dims", type=int, default=1024, help="Embedding dimensions (default: 1024)")

    # ========== skill mode ==========
    skill_parser = subparsers.add_parser(
        "skill",
        help="Generate a skill package (SKILL.md, scripts/, references/)",
    )
    skill_parser.add_argument("url", help="Library/framework documentation URL")
    skill_parser.add_argument("--output-dir", "-o", default=".", help="Output directory for skill package")
    skill_parser.add_argument("--provider", "-p", choices=list(PROVIDER_PROFILES.keys()), help="LLM provider")
    skill_parser.add_argument("--model", "-m", help="Model name to use (API identifier)")
    skill_parser.add_argument("--model-display-name", help="Display name for observability (e.g., 'Qwen-2.5-7B' when model='llm')")
    skill_parser.add_argument("--base-url", help="Base URL for LLM API")
    skill_parser.add_argument("--api-key", help="API key for LLM provider")
    skill_parser.add_argument("--max-urls", type=int, default=None, help="Maximum URLs to scrape")
    skill_parser.add_argument("--max-rounds", type=int, default=3, help="Maximum critic revision rounds (default: 3)")
    skill_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    skill_parser.add_argument("--log-file", help="Path to log file")
    skill_parser.add_argument("--log-json", action="store_true", help="Use JSON log format")
    skill_parser.add_argument("--clean", action="store_true", help="Clean output directory before generation")
    skill_parser.add_argument("--validate", action="store_true", help="Run validation checks before generation")
    # Embedding config (for semantic search store)
    skill_parser.add_argument("--embedding-base-url", help="Base URL for embedding API (default: same as base-url)")
    skill_parser.add_argument("--embedding-model", default="embed", help="Embedding model name (default: embed)")
    skill_parser.add_argument("--embedding-dims", type=int, default=1024, help="Embedding dimensions (default: 1024)")

    # Parse arguments
    args = parser.parse_args()

    # Load .env file
    load_dotenv()

    # Setup logging
    log_level = "DEBUG" if getattr(args, 'verbose', False) else os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = Path(args.log_file) if hasattr(args, 'log_file') and args.log_file else None
    log_json = hasattr(args, 'log_json') and args.log_json

    actual_log_file = setup_logging(level=log_level, log_file=log_file, json_format=log_json)
    log.start("make_llmstxt", version=__version__, log_file=str(actual_log_file))

    # List providers if requested
    if args.list_providers:
        list_providers()
        return

    # Load environment config
    env_config = AppConfig.from_env()
    log.debug("Config loaded", provider=env_config.llm.provider)
    log.info("MCP scraper configured", host=env_config.mcp.host, port=env_config.mcp.port)

    # Check API key
    if not env_config.llm.api_key and env_config.llm.provider != "local":
        console.print(f"[red]Error: LLM API key not provided for provider '{env_config.llm.provider}'[/red]")
        console.print(f"Set the appropriate environment variable or use --api-key")
        log.error("API key", f"Missing for provider: {env_config.llm.provider}")
        sys.exit(1)

    # Handle modes
    if args.mode == "skill":
        handle_skill(args, env_config)
    elif args.mode == "llmstxt":
        handle_llmstxt(args, env_config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
