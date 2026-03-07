#!/usr/bin/env python3
"""Test different domains."""
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from make_llmstxt.mcp_scraper import MCPClient, MCPConfig


async def test():
    config = MCPConfig(host="100.102.244.97", port=8000)
    client = MCPClient(config)
    await client.connect()

    # Try a different domain
    result = await client.call_tool("scrape_url", {"url": "https://httpbin.org/html"})
    print(f"httpbin.org result: {result[:500]}")

    result = await client.call_tool("scrape_url", {"url": "https://www.iana.org/domains/reserved"})
    print(f"iana.org result: {result[:500]}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test())
