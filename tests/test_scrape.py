#!/usr/bin/env python3
"""Test scrape_url directly."""
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from make_llmstxt.mcp_scraper import MCPClient, MCPConfig


async def test_scrape():
    config = MCPConfig(host="100.102.244.97", port=8000)
    client = MCPClient(config)

    print("Connecting...")
    connected = await client.connect()
    print(f"Connected: {connected}")
    print(f"Session: {client.session_id}")

    # Test scrape_url directly
    print("\n--- Testing scrape_url ---")
    result = await client.call_tool("scrape_url", {"url": "https://example.com"})
    print(f"Result type: {type(result)}")
    result_str = str(result)
    print(f"Result length: {len(result_str)}")
    print(f"Result (first 1000 chars): {result_str[:1000]}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_scrape())
