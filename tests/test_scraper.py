#!/usr/bin/env python3
"""Test what scrape_url returns."""
import asyncio
import json
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

    # Test scrape_url
    print("\n--- Testing scrape_url ---")
    result = await client.call_tool("scrape_url", {"url": "https://docs.nextra.site"})
    print(f"Result type: {type(result)}")

    # Parse the JSON if it's a JSON string
    if isinstance(result, str):
        print(f"Result is a string (length: {len(result)})")
        try:
            data = json.loads(result)
            print(f"Parsed JSON: {json.dumps(data, indent=2)[:500]}")
        except json.JSONDecodeError:
            print(f"Not JSON, raw string (first 500): {result[:500]}")
    else:
        print(f"Result (first 500): {str(result)[:500]}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_scrape())
