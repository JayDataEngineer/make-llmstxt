#!/usr/bin/env python3
"""Debug MCP server responses."""
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from make_llmstxt.mcp_scraper import MCPClient, MCPConfig


async def debug_mcp():
    config = MCPConfig(host="100.102.244.97", port=8000)
    client = MCPClient(config)

    print("Connecting...")
    connected = await client.connect()
    print(f"Connected: {connected}")
    print(f"Session: {client.session_id}")

    # List tools
    print("\nListing tools...")
    tools = await client.list_tools()
    print(f"Available tools: {[t['name'] for t in tools]}")

    # Test scrape_url
    print("\n--- Testing scrape_url ---")
    result = await client.call_tool("scrape_url", {"url": "https://docs.python.org/3/"})
    print(f"Result type: {type(result)}")
    print(f"Result (first 500 chars): {str(result)[:500]}")

    # Test search_web
    print("\n--- Testing search_web ---")
    result = await client.call_tool("search_web", {"query": "python documentation site:docs.python.org"})
    print(f"Result type: {type(result)}")
    print(f"Result (first 500 chars): {str(result)[:500]}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(debug_mcp())
