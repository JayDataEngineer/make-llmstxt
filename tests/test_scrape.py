#!/usr/bin/env python3
"""Test scrape_url directly using langchain-mcp-adapters."""
import pytest

from make_llmstxt.mcp_tools import create_mcp_client, mcp_scrape_url


@pytest.mark.asyncio
async def test_scrape_url():
    """Test scrape_url tool directly."""
    host = "100.102.244.97"
    port = 8000

    result = await mcp_scrape_url(host, port, "https://example.com")
    assert result is not None
    assert "content" in result or "markdown" in result


@pytest.mark.asyncio
async def test_get_tools():
    """Test loading tools from MCP server."""
    client = create_mcp_client("100.102.244.97", 8000)
    tools = await client.get_tools()

    tool_names = [t.name for t in tools]
    assert "scrape_url" in tool_names
    assert "map_domain" in tool_names
    assert "crawl_site" in tool_names
