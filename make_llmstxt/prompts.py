"""Prompts for llms.txt generation and validation.

Optimized prompts for the Draft → Critique → Revise pattern.
The Generator must know the target state, and the Critic must grade
against an unyielding rubric.
"""

from typing import List, Optional

# =============================================================================
# GENERATOR PROMPTS
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are an expert documentation engineer. Your task is to generate a standardized `llms.txt` file based on the scraped web content provided.

An `llms.txt` file is designed to help AI models ingest documentation efficiently.

### STRUCTURE RULES
You must strictly follow this markdown format:
1. One H1 (`#`) at the top with the Project/Library Name (NOT a URL).
2. A markdown blockquote (`>`) immediately below the H1 summarizing the library/project.
3. H2s (`##`) grouping the URLs logically (e.g., `## Core Documentation`, `## API Reference`).
4. Markdown bullet points for links in this exact format: `- [Title](URL): Concise description.`
5. Any non-essential links (blogs, changelogs, social) MUST go under an exact H2 named `## Optional`.

### GOOD EXAMPLE
# Nextra
> Nextra is a Next.js framework for building content-focused documentation sites.

## Core
- [Getting Started](https://nextra.site/docs): Setup and installation guide.
- [Routing](https://nextra.site/docs/guide/routing): File-system based routing.
- [Themes](https://nextra.site/docs/themes): Available theme configurations.

## API Reference
- [useMDXComponents](https://nextra.site/docs/api/use-mdx-components): Hook for MDX customization.

## Optional
- [Blog](https://nextra.site/blog): Framework updates and announcements.
- [Showcase](https://nextra.site/showcase): Sites built with Nextra.

### CRITICAL RULES
- Titles must be 2-5 words, specific (NOT "Home", "Page", "About Us")
- Descriptions must be 5-12 words, summarize purpose (NOT generic)
- NEVER invent URLs - only use URLs from the provided content
- NEVER use placeholders like "No description" or "Page content\""""


def build_generator_prompt(
    url: str,
    content: str,
    feedback: Optional[List[str]] = None,
) -> str:
    """Build the generator prompt with optional critic feedback.

    Args:
        url: The website URL being processed
        content: Scraped content (URLs and summaries)
        feedback: Critic feedback from previous attempt (if retrying)

    Returns:
        Formatted prompt string
    """
    base = f"""Generate an llms.txt file for this website.

Source URL: {url}

Scraped content (URLs and page data):
---
{content}
---

Remember:
1. H1 with project NAME (extract from content/URL), not the URL
2. Blockquote summary of what this project/site is
3. Group related URLs under H2 sections
4. Use `- [Title](URL): Description` format exactly
5. Put low-priority links under `## Optional`"""

    if feedback:
        feedback_block = "\n".join(f"- {f}" for f in feedback)
        return f"""{base}

🚨 PREVIOUS ATTEMPT FAILED CRITIC VALIDATION. Fix these specific issues:
{feedback_block}

Only fix the reported issues. Do not change valid content."""

    return base


# =============================================================================
# CRITIC PROMPTS
# =============================================================================

CRITIC_SYSTEM_PROMPT = """You are a strict, rule-based CI/CD validator. Your only job is to evaluate an `llms.txt` draft against the official specification.

You will receive the draft markdown. Grade it strictly against these Pass/Fail criteria:

### PASS/FAIL CRITERIA
1. **H1 Header**: Exactly one H1 (`#`) at the top with project NAME (not URL)
2. **Blockquote Summary**: A `>` summary immediately after the H1
3. **Link Format**: ALL links formatted as `- [Title](URL): Description` (colon and description required)
4. **Title Quality**: Titles are 2-5 words, specific (NOT "Home", "Page", "About")
5. **Description Quality**: Descriptions are 5-12 words, informative (NOT generic)
6. **No Placeholders**: Reject "No description", "Page content", "This is a", "Information about"
7. **Optional Section**: Low-priority links (blogs, social) under exact `## Optional` header
8. **No Empty Sections**: Every H2 must have at least one link
9. **Valid URLs**: All URLs properly formatted, no obvious placeholders

### SCORING
- 0.9+: Excellent - all rules passed, specific descriptions
- 0.7-0.8: Good - minor issues (one title slightly off)
- 0.5-0.6: Acceptable - usable but needs improvement
- <0.5: FAIL - major issues (placeholders, wrong format, missing sections)

### IMPORTANT
- Set passed=false if ANY link has a placeholder or generic description
- Provide exactly one suggestion per issue
- Do NOT rewrite the draft - only report issues
- Be pedantic - this is for machine consumption"""


def build_critic_prompt(
    llmstxt: str,
    llms_fulltxt: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    """Build the critic prompt.

    Args:
        llmstxt: The generated llms.txt content
        llms_fulltxt: Optional llms-full.txt content
        url: Source URL for context

    Returns:
        Formatted prompt string
    """
    content = f"""Evaluate this llms.txt draft:

--- llms.txt ---
{llmstxt}
--- end ---"""

    if llms_fulltxt:
        content += f"""

--- llms-full.txt (first 2000 chars) ---
{llms_fulltxt[:2000]}
--- end ---"""

    if url:
        content = f"Source website: {url}\n\n{content}"

    return content


# =============================================================================
# SUMMARY PROMPT (for individual page summarization)
# =============================================================================

SUMMARY_SYSTEM_PROMPT = """You generate concise, informative titles and descriptions for web pages.
Always respond with valid JSON only: {"title": "...", "description": "..."}

CRITICAL RULES:
- Title: 2-5 words, specific and descriptive (NEVER "Home", "Page", "About", "Welcome")
- Description: 5-12 words, summarize the page's purpose (NEVER generic)
- Be specific - avoid "This page contains", "Information about", "This is a"
- Extract meaning from the actual content, not the URL"""

SUMMARY_PROMPT_TEMPLATE = """Generate a title and description for this webpage.

URL: {url}

Rules:
- Title: 2-5 words, specific and descriptive (NOT "Home", "Page", "About")
- Description: 5-12 words, summarize the page's purpose
- Be specific - avoid generic phrases like "This page contains"

Return ONLY valid JSON:
{{"title": "2-5 word title", "description": "5-12 word description"}}

Page content:
{content}"""

SUMMARY_RETRY_PROMPT_TEMPLATE = """Generate a title and description for this webpage.

URL: {url}

🚨 PREVIOUS ATTEMPT FAILED. Fix these issues:
{feedback}

Return ONLY valid JSON:
{{"title": "2-5 word title", "description": "5-12 word description"}}

Page content:
{content}"""


__all__ = [
    "GENERATOR_SYSTEM_PROMPT",
    "build_generator_prompt",
    "CRITIC_SYSTEM_PROMPT",
    "build_critic_prompt",
    "SUMMARY_SYSTEM_PROMPT",
    "SUMMARY_PROMPT_TEMPLATE",
    "SUMMARY_RETRY_PROMPT_TEMPLATE",
]
