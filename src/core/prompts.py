

























































































    return base


# =============================================================================
# CRITIC PROMPTS
# =============================================================================

CRITIC_SYSTEM_PROMPT = """You are a strict, rule-based CI/CD validator. Your only job is to evaluate an `llms.txt` draft against the official specification.

You will receive the draft markdown AND the source scraped content. Grade it strictly against these Pass/Fail criteria:

### PASS/FAIL CRITERIA
1. **COVERAGE (CRITICAL)**: The llms.txt MUST include ALL URLs from the source content.
   - Count URLs in source content, count links in llms.txt
   - If source has 10 pages and llms.txt has 3 links = FAIL
   - Missing URLs = automatic FAIL, no exceptions
2. **H1 Header**: Exactly one H1 (`#`) at the top with project NAME (not URL)
3. **Blockquote Summary**: A `>` summary immediately after the H1
4. **Link Format**: ALL links formatted as `- [Title](URL): Description` (colon and description required)
5. **Title Quality**: Titles are 2-5 words, specific (NOT "Home", "Page", "About")
6. **Description Quality**: Descriptions are 5-12 words, informative (NOT generic)
7. **No Placeholders**: Reject "No description", "Page content", "This is a", "Information about"
8. **Optional Section**: Low-priority links (blogs, social) under exact `## Optional` header
9. **No Empty Sections**: Every H2 must have at least one link
10. **Valid URLs**: All URLs properly formatted, no obvious placeholders

### SCORING
- 0.9+: Excellent - all rules passed, COMPLETE coverage, specific descriptions
- 0.7-0.8: Good - minor issues (one title slightly off), complete coverage
- 0.5-0.6: Acceptable - usable but needs improvement
- <0.5: FAIL - INCOMPLETE COVERAGE, placeholders, wrong format, missing sections

### IMPORTANT
- Set passed=false if COVERAGE is incomplete (missing URLs from source)
- Set passed=false if ANY link has a placeholder or generic description
- Provide exactly one suggestion per issue
- Do NOT rewrite the draft - only report issues
- Be pedantic - this is for machine consumption"""


def build_critic_prompt(
    llmstxt: str,
    url: Optional[str] = None,
    source_content: Optional[str] = None,
) -> str:
    """Build the critic prompt.

    Args:
        llmstxt: The generated llms.txt content















































































