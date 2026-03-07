#!/usr/bin/env python3
"""Test scrape_url directly."""
import pytest

from make_llmstxt.mcp_scraper import MCPClient, MCPConfig


@pytest.mark.asyncio
async def test_scrape_url():
    """Test scrape_url tool directly."""
    config = MCPConfig(host="100.102.244.97", port=8000)
    client = MCPClient(config)
    try:
        connected = await client.connect()
        assert connected

        # Test scrape_url directly
        result = await client.call_tool("scrape_url", {"url": "https://example.com"})
        assert result is not None
    finally:
        await client.close()
