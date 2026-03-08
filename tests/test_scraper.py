#!/usr/bin/env python3
"""Test what scrape_url returns using langchain-mcp-adapters."""
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from make_llmstxt.mcp_tools import create_mcp_client, mcp_scrape_url


async def test_scrape():
    host = "100.102.244.97"
    port = 8000

    print("Testing scrape_url via langchain-mcp-adapters...")

    # Test scrape_url
    print("\n--- Testing scrape_url ---")
    result = await mcp_scrape_url(host, port, "https://docs.nextra.site")
    print(f"Result type: {type(result)}")

    if result:
        print(f"URL: {result.get('url')}")
        print(f"Title: {result.get('title')}")
        content = result.get('content', result.get('markdown', ''))
        print(f"Content length: {len(content)}")
        print(f"Content preview (first 500): {content[:500]}")
    else:
        print("Result is None - scrape failed")


async def test_get_tools():
    host = "100.102.244.97"
    port = 8000

    print("\n--- Testing get_tools ---")
    client = create_mcp_client(host, port)
    tools = await client.get_tools()

    print(f"Found {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:100] if tool.description else 'No description'}...")


if __name__ == "__main__":
    asyncio.run(test_scrape())
    asyncio.run(test_get_tools())
