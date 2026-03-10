"""Structured output schemas for generators."""

from pydantic import BaseModel, Field


class PageSummary(BaseModel):
    """Structured summary of a scraped page - tiny footprint for state.

    This schema is used by Stage 1 (scraping) to generate compact summaries
    that live in agent state, while full content is stored in BaseStore.

    The small footprint allows handling 70+ pages without context explosion.
    """

    url: str = Field(description="The source URL")
    title: str = Field(description="Page title (2-5 words)")
    description: str = Field(description="What this page covers (1-2 sentences, max 50 words)")
    key_topics: list[str] = Field(
        default_factory=list,
        description="3-5 technical topics covered on this page"
    )
