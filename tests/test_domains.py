#!/usr/bin/env python3
"""Test different domains."""
import pytest

from make_llmstxt.mcp_scraper import MCPClient, MCPConfig


@pytest.mark.asyncio
async def test_different_domains():
    """Test MCP scraper with different domains."""
    config = MCPConfig(host="100.102.244.97", port=8000)
    client = MCPClient(config)
    try:
        await client.connect()

        # Try a different domain
        result = await client.call_tool("scrape_url", {"url": "https://httpbin.org/html"})
        assert result is not None

        result = await client.call_tool("scrape_url", {"url": "https://www.iana.org/domains/reserved"})
        assert result is not None
    finally:
        await client.close()
