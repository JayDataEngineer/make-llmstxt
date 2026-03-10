"""Validators for make-llmstxt skill generation.

Pre-flight checks to ensure skill generation will succeed.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file at import time
load_dotenv()


@dataclass
class ValidationResult:
    """Result of validation checks."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)


class SkillGenerationValidator:
    """Validator for skill generation preconditions.

    Checks:
    - Output directory exists and is empty (or can be cleaned)
    - MCP server is reachable
    - LLM server is reachable
    - Environment variables are set
    """

    def __init__(
        self,
        output_dir: Path,
        mcp_host: str = "localhost",
        mcp_port: int = 8000,
        llm_base_url: str = "http://localhost:8001/v1",
        auto_fix: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        self.llm_base_url = llm_base_url
        self.auto_fix = auto_fix

    def validate(self) -> ValidationResult:
        """Run all validation checks."""
        result = ValidationResult(valid=True)

        # Check output directory
        self._check_output_dir(result)

        # Check environment
        self._check_environment(result)

        # Check network services (optional - can warn but not fail)
        self._check_services(result)

        return result

    def _check_output_dir(self, result: ValidationResult) -> None:
        """Check if output directory is ready for writing."""
        if not self.output_dir.exists():
            if self.auto_fix:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                result.fixes_applied.append(f"Created output directory: {self.output_dir}")
            else:
                result.errors.append(f"Output directory does not exist: {self.output_dir}")
                result.valid = False
            return

        # Check if directory has existing files
        existing_files = list(self.output_dir.rglob("*"))
        existing_files = [f for f in existing_files if f.is_file()]

        if existing_files:
            if self.auto_fix:
                # Clean existing files
                import shutil
                for f in existing_files:
                    f.unlink()
                # Remove empty subdirectories
                for d in sorted(self.output_dir.rglob("*"), reverse=True):
                    if d.is_dir() and not any(d.iterdir()):
                        d.rmdir()
                result.fixes_applied.append(
                    f"Cleaned {len(existing_files)} existing file(s) from {self.output_dir}"
                )
            else:
                result.errors.append(
                    f"Output directory has {len(existing_files)} existing file(s). "
                    f"Use --clean or set auto_fix=True to remove them. "
                    f"Files: {[str(f.relative_to(self.output_dir)) for f in existing_files[:5]]}"
                )
                result.valid = False

    def _check_environment(self, result: ValidationResult) -> None:
        """Check required environment variables."""
        # Check for API key (either OPENAI_API_KEY or LOCAL_API_KEY)
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_API_KEY")
        if not api_key:
            result.errors.append(
                "No API key found. Set OPENAI_API_KEY or LOCAL_API_KEY environment variable."
            )
            result.valid = False

        # Check for provider config
        provider = os.environ.get("LLM_PROVIDER")
        if not provider:
            result.warnings.append(
                "LLM_PROVIDER not set. Defaulting to 'local' or 'openai'."
            )

    def _check_services(self, result: ValidationResult) -> None:
        """Check if required services are reachable."""
        import socket

        # Check MCP server
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            mcp_result = sock.connect_ex((self.mcp_host, self.mcp_port))
            sock.close()
            if mcp_result != 0:
                result.warnings.append(
                    f"MCP server not reachable at {self.mcp_host}:{self.mcp_port}. "
                    "Skill generation may fail."
                )
        except Exception as e:
            result.warnings.append(f"Could not check MCP server: {e}")

        # Check LLM server (parse host:port from base_url)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.llm_base_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 80

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            llm_result = sock.connect_ex((host, port))
            sock.close()
            if llm_result != 0:
                result.warnings.append(
                    f"LLM server not reachable at {host}:{port}. "
                    "Skill generation will fail."
                )
        except Exception as e:
            result.warnings.append(f"Could not check LLM server: {e}")


def validate_skill_generation(
    output_dir: Path,
    mcp_host: str = "localhost",
    mcp_port: int = 8000,
    llm_base_url: str = "http://localhost:8001/v1",
    auto_fix: bool = False,
) -> ValidationResult:
    """Convenience function to validate skill generation preconditions."""
    validator = SkillGenerationValidator(
        output_dir=output_dir,
        mcp_host=mcp_host,
        mcp_port=mcp_port,
        llm_base_url=llm_base_url,
        auto_fix=auto_fix,
    )
    return validator.validate()


def main():
    """CLI entry point for validator."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate skill generation preconditions")
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory for generated skill packages"
    )
    parser.add_argument(
        "--mcp-host",
        default=os.environ.get("MCP_HOST", "localhost"),
        help="MCP server host"
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=int(os.environ.get("MCP_PORT", "8000")),
        help="MCP server port"
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1"),
        help="LLM server base URL"
    )
    parser.add_argument(
        "--auto-fix", "-f",
        action="store_true",
        help="Automatically fix issues (create dirs, clean files)"
    )

    args = parser.parse_args()

    result = validate_skill_generation(
        output_dir=args.output_dir,
        mcp_host=args.mcp_host,
        mcp_port=args.mcp_port,
        llm_base_url=args.llm_base_url,
        auto_fix=args.auto_fix,
    )

    # Print results
    if result.fixes_applied:
        print("\n=== FIXES APPLIED ===")
        for fix in result.fixes_applied:
            print(f"  ✓ {fix}")

    if result.warnings:
        print("\n=== WARNINGS ===")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")

    if result.errors:
        print("\n=== ERRORS ===")
        for error in result.errors:
            print(f"  ✗ {error}")

    if result.valid:
        print("\n✓ All validation checks passed!")
        return 0
    else:
        print("\n✗ Validation failed!")
        if not args.auto_fix:
            print("  Run with --auto-fix to automatically fix issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
