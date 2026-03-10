"""Command-line interface for make-llmstxt.

Supports two modes:
- llms.txt: Index of pages with titles and descriptions (default)
- skill: Skill package with folder structure (SKILL.md, scripts/, references/)

"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import asyncio

from .config import AppConfig, LLMConfig, PROVIDER_PROFILES
from .generators.llmstxt import generate_llmstxt
from .generators.skill import SkillGenerator
from .__init__ import __version__
from .utils.logging import setup_logging
from .core import GeneratorConfig
from .validators import validate_skill_generation

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


def handle_skill(args):
    """Handle skill generation with hierarchical LangGraph."""
    # Load env config first
    env_config = AppConfig.from_env()

    # Run validation if requested or if clean is specified
    if args.validate or args.clean:
        library_name = args.url.split("//")[-1].split("/")[0].replace("www.", "").replace("docs.", "").split(".")[0]
        output_subdir = Path(args.output_dir) / library_name

        validation_result = validate_skill_generation(
            output_dir=output_subdir,
            mcp_host=env_config.mcp.host,
            mcp_port=env_config.mcp.port,
            llm_base_url=env_config.llm.base_url,
            auto_fix=args.clean,
        )

        # Print validation results
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

    skill_config = GeneratorConfig(
        url=args.url,
        output_dir=Path(args.output_dir),
        mcp_host=env_config.mcp.host,
        mcp_port=env_config.mcp.port,
        max_rounds=max_rounds,
    )

    # Override LLM settings from env
    skill_config.api_key = env_config.llm.api_key
    skill_config.base_url = env_config.llm.base_url
    skill_config.model = env_config.llm.model
    skill_config.provider = env_config.llm.provider

    # Override with CLI args
    if args.provider:
        skill_config.provider = args.provider
        if args.provider in PROVIDER_PROFILES:
            profile = PROVIDER_PROFILES[args.provider]
            skill_config.model = args.model or profile["default_model"]
    if args.model:
        skill_config.model = args.model
    if args.base_url:
        skill_config.base_url = args.base_url
    if args.api_key:
        skill_config.api_key = args.api_key

    # Create skill generator
    generator = SkillGenerator(skill_config)

    console.print(f"\n[bold cyan]Starting skill generation for {args.url}...[/bold cyan]")
    console.print(f"Provider: {skill_config.provider}")
    console.print(f"Model: {skill_config.model}")
    console.print(f"MCP: {skill_config.mcp_host}:{skill_config.mcp_port}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Initializing...", total=None)

        def progress_callback(*args):
            """Handle progress updates with flexible signature.

            LLMsTxtGenerator calls with 4 args: (stage, current, total, message)
            SkillGenerator calls with 2 args: (message, percent)
            """
            nonlocal task_id
            if len(args) == 4:
                # LLMsTxtGenerator format: (stage, current, total, message)
                message = args[3]
            elif len(args) == 2:
                # SkillGenerator format: (message, percent)
                message = args[0]
            else:
                message = str(args[0]) if args else ""
            progress.update(task_id, description=message)

        try:
            result = asyncio.run(
                generator.generate(
                    args.url,
                    output_dir=Path(args.output_dir),
                    progress_callback=progress_callback,
                )
            )

            console.print()
            console.print(f"[green]Success![/green] Skill package created")
            console.print(f"Output: {result.output_path}")
            console.print(f"Pages processed: {result.stats.get('pages_processed', 0)}")
            console.print(f"Files created: {result.stats.get('files_created', 0)}")
            logger.info(f"Skill package created at {result.output_path}")

        except Exception as e:
            logger.exception(f"Skill generation failed: {e}")
            console.print(f"[red]Error: {e}[/red]")
            if args.verbose:
                console.print_exception()
            sys.exit(1)

        finally:
            asyncio.run(generator.close())


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

    # ========== llms.txt mode (default) ==========
    llmstxt_parser = subparsers.add_parser(
        "llmstxt",
        help="Generate llms.txt and llms-full.txt (default mode)",
    )
    llmstxt_parser.add_argument("url", nargs="?", help="Website URL to process")
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
    llmstxt_parser.add_argument("--model", "-m", help="Model name to use")
    llmstxt_parser.add_argument("--base-url", help="Base URL for LLM API")
    llmstxt_parser.add_argument("--api-key", help="API key for LLM provider")
    llmstxt_parser.add_argument("--no-critic", action="store_true", help="Disable critic validation")
    llmstxt_parser.add_argument("--max-rounds", type=int, default=3, help="Maximum draft-critic rounds")
    llmstxt_parser.add_argument("--pass-threshold", type=float, default=0.7, help="Minimum score to pass critic")
    llmstxt_parser.add_argument("--critic-strict", action="store_true", help="Fail if critic errors")
    llmstxt_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    llmstxt_parser.add_argument("--log-file", help="Path to log file")
    llmstxt_parser.add_argument("--log-json", action="store_true", help="Use JSON log format")

    # ========== skill mode ==========
    skill_parser = subparsers.add_parser(
        "skill",
        help="Generate a skill package (SKILL.md, scripts/, references/) using hierarchical LangGraph",
    )
    skill_parser.add_argument("url", help="Library/framework documentation URL")
    skill_parser.add_argument("--output-dir", "-o", default=".", help="Output directory for skill package")
    skill_parser.add_argument("--provider", "-p", choices=list(PROVIDER_PROFILES.keys()), help="LLM provider")
    skill_parser.add_argument("--model", "-m", help="Model name to use")
    skill_parser.add_argument("--base-url", help="Base URL for LLM API")
    skill_parser.add_argument("--api-key", help="API key for LLM provider")
    skill_parser.add_argument("--max-rounds", type=int, default=3, help="Maximum critic revision rounds (default: 3)")
    skill_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    skill_parser.add_argument("--log-file", help="Path to log file")
    skill_parser.add_argument("--log-json", action="store_true", help="Use JSON log format")
    skill_parser.add_argument("--clean", action="store_true", help="Clean output directory before generation")
    skill_parser.add_argument("--validate", action="store_true", help="Run validation checks before generation")

    # Parse arguments
    args = parser.parse_args()

    # Load .env file
    load_dotenv()

    # Setup logging
    log_level = "DEBUG" if getattr(args, 'verbose', False) else os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = Path(args.log_file) if hasattr(args, 'log_file') and args.log_file else None
    log_json = hasattr(args, 'log_json') and args.log_json

    actual_log_file = setup_logging(level=log_level, log_file=log_file, json_format=log_json)
    logger.info(f"make-llmstxt v{__version__} starting up")
    logger.info(f"Logs: {actual_log_file}")

    # List providers if requested
    if args.list_providers:
        list_providers()
        return

    # Handle skill mode
    if args.mode == "skill":
        handle_skill(args)
        return

    # Handle llms.txt mode (default)
    if not hasattr(args, 'url') or not args.url:
        parser.error("URL is required")

    # Build configuration
    config = AppConfig.from_env()
    logger.debug(f"Loaded config from env: provider={config.llm.provider}")
    logger.info(f"Using MCP scraper at {config.mcp.host}:{config.mcp.port}")

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

    # Run llms.txt generation
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

            logger.info(f"Starting generation for {args.url}")
            result = asyncio.run(
                generate_llmstxt(
                    args.url,
                    config=config,
                    max_urls=args.max_urls,
                    output_dir=Path(args.output_dir),
                    progress_callback=progress_callback,
                    enable_critic=not args.no_critic,
                    max_retries=args.max_rounds,
                    pass_threshold=args.pass_threshold,
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
