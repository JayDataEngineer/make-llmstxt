#!/usr/bin/env python3
"""Test different domains using langchain-mcp-adapters."""
import pytest

from make_llmstxt.mcp_tools import mcp_scrape_url


@pytest.mark.asyncio
async def test_different_domains():
    """Test MCP scraper with different domains."""
    host = "100.102.244.97"
    port = 8000

    # Try a different domain
    result = await mcp_scrape_url(host, port, "https://httpbin.org/html")
    assert result is not None

    result = await mcp_scrape_url(host, port, "https://www.iana.org/domains/reserved")
    assert result is not None
