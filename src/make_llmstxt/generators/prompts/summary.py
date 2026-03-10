"""Prompts for individual page summarization.

Used by the page_scraper subagent to extract titles and descriptions.
"""

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
    "SUMMARY_SYSTEM_PROMPT",
    "SUMMARY_PROMPT_TEMPLATE",
    "SUMMARY_RETRY_PROMPT_TEMPLATE",
]
