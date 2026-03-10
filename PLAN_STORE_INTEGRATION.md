# Plan: LangGraph Store Integration for Two-Stage Pipeline

## Overview

Implement a two-stage agentic pipeline that solves context explosion while preserving full content fidelity:

- **Stage 1 (LLMs.txt)**: Parallel scraping → structured summaries in state + full content in Store
- **Stage 2 (Skills)**: Read llms.txt catalog → semantic search for full content → generate skills

---

## Stage 1: Parallel Scraping with Dual-Action Workers

### 1.1 Add PageSummary Structured Output

**File:** `src/make_llmstxt/generators/schemas.py` (NEW)

```python
from pydantic import BaseModel, Field

class PageSummary(BaseModel):
    """Structured summary of a scraped page - tiny footprint for state."""
    url: str = Field(description="The source URL")
    title: str = Field(description="Page title (2-5 words)")
    description: str = Field(description="What this page covers (1-2 sentences, max 50 words)")
    key_topics: list[str] = Field(description="3-5 technical topics covered")
```

### 1.2 Add Fast LLM Configuration

**File:** `src/make_llmstxt/core/models.py`

```python
class GeneratorConfig(BaseModel):
    # ... existing fields ...

    # Fast LLM for parallel summarization (cheaper)
    fast_model: str = Field(default="gpt-4o-mini", description="Fast model for summarization")
    fast_model_temperature: float = Field(default=0.1, description="Temperature for fast model")
```

### 1.3 Update WebScrapingState

**File:** `src/make_llmstxt/generators/base_agent.py`

```python
class WebScrapingState(DeepAgentState):
    """Specialized state for web scraping agents."""
    discovered_urls: List[str]
    scraped_docs: Annotated[List[Dict], operator.add]  # List of PageSummary dicts
    scraping_errors: Annotated[List[str], operator.add]
    scraping_complete: bool
```

### 1.4 Dual-Action Scraper Node

**File:** `src/make_llmstxt/generators/base_agent.py`

```python
import hashlib
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

async def scraper_node(
    state: WebScrapingState,
    config: RunnableConfig,
    store: BaseStore
) -> dict:
    """Scrape URL, store full content, return structured summary."""
    url = state["url"]
    max_content = state.get("max_content")

    async with semaphore:
        try:
            # 1. Scrape full content
            raw_result = await scrape_tool.ainvoke({"url": url})
            full_content = parse_result(raw_result)

            # 2. ACTION 1: Store FULL content in BaseStore (persists across runs)
            # Use URL + content prefix for deduplication (handles content changes)
            content_hash = hashlib.md5(f"{url}:{full_content[:1000]}".encode()).hexdigest()

            await store.aput(
                namespace=("memories", "raw_pages"),
                key=content_hash,
                value={
                    "url": url,
                    "content": full_content,
                    "scraped_at": datetime.now().isoformat(),
                }
            )

            # 3. ACTION 2: Generate structured summary for state (tiny footprint)
            structured_llm = fast_llm.with_structured_output(PageSummary)

            summary: PageSummary = await structured_llm.ainvoke(
                f"Analyze this documentation page.\n\n"
                f"URL: {url}\n\n"
                f"Content:\n{full_content[:15000]}"
            )

            logger.info(f"{log_prefix} Scraped {url} → stored + summarized")

            return {"scraped_docs": [summary.model_dump()]}

        except Exception as e:
            logger.error(f"{log_prefix} Failed {url}: {e}")
            return {"scraping_errors": [f"{url}: {str(e)}"]}
```

---

## Stage 2: Store Configuration for Semantic Search

### 2.1 Add langgraph.json Configuration

**File:** `langgraph.json` (NEW)

```json
{
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536,
      "fields": ["content", "url"]
    }
  }
}
```

### 2.2 Composite Backend Setup

**File:** `src/make_llmstxt/generators/base_agent.py`

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

def _create_backend(self, runtime) -> CompositeBackend:
    """Create backend with /memories/ route to persistent Store."""
    return CompositeBackend(
        default=StateBackend(runtime),  # /tmp/ stays ephemeral
        routes={
            "/memories/": StoreBackend(runtime)  # Persistent + indexed
        }
    )
```

### 2.3 Search Tool for Stage 2

**File:** `src/make_llmstxt/tools/store_tools.py` (NEW)

```python
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

@tool
async def search_docs(query: str, runtime) -> str:
    """Search scraped documentation semantically.

    Use this to find detailed content about specific topics.
    Returns the most relevant document sections.
    """
    store: BaseStore = runtime.store

    results = await store.asearch(
        namespace=("memories", "raw_pages"),
        query=query,
        limit=5
    )

    if not results:
        return "No relevant documents found."

    formatted = []
    for r in results:
        formatted.append(
            f"**Source: {r.value['url']}** (relevance: {r.score:.2f})\n"
            f"{r.value['content'][:3000]}..."
        )

    return "\n\n---\n\n".join(formatted)
```

---

## Implementation Order

### Phase 1: Foundation (Current Commit ✓)
- [x] Parallel graph structure with Send
- [x] WebScrapingState with reducers
- [x] Semaphore for concurrency limiting
- [x] Static helper methods

### Phase 2: Structured Output
- [x] Create `PageSummary` schema
- [x] Add `fast_model` config
- [x] Update scraper_node for structured output
- [x] Test summarization works

### Phase 3: Store Integration
- [x] Add `langgraph.json` with index config
- [x] Implement store access via config (using LangGraph native store)
- [x] Update scraper_node with `store.aput()`
- [ ] Test persistence across runs

### Phase 4: Search Tool
- [x] Create `search_docs` tool
- [x] Create `get_doc_by_url` tool
- [ ] Add to Stage 2 agent tools
- [ ] Test semantic retrieval
- [ ] Verify Stage 1 → Stage 2 handoff

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Content hash key | `md5(url:content[:1000])` | Handles content changes, dedupes |
| Store namespace | `("memories", "raw_pages")` | Organized, queryable |
| Fast model | `gpt-4o-mini` | Cheap, fast, good for summaries |
| Summary max words | 50 words description | Tiny footprint for 70+ pages |
| Search limit | 5 results | Balance breadth vs context |
| Embedding model | `text-embedding-3-small` | Good quality, low cost |

---

## Testing Checklist

- [ ] Parallel scraping runs without context explosion
- [ ] Full content persists in Store across restarts
- [ ] Semantic search retrieves relevant chunks
- [ ] Stage 2 agent can find specific code examples
- [ ] Deduplication works (same URL doesn't duplicate)
- [ ] Concurrency limiting prevents rate limiting
