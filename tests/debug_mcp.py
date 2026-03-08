#!/usr/bin/env python3
"""Debug MCP server responses using langchain-mcp-adapters."""
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from make_llmstxt.mcp_tools import create_mcp_client, mcp_scrape_url


async def debug_mcp():
    host = "100.102.244.97"
    port = 8000

    print("Connecting via langchain-mcp-adapters...")

    # Create client and list tools
    client = create_mcp_client(host, port)
    tools = await client.get_tools()

    print(f"Available tools: {[t.name for t in tools]}")

    # Test scrape_url
    print("\n--- Testing scrape_url ---")
    result = await mcp_scrape_url(host, port, "https://docs.python.org/3/")
    print(f"Result type: {type(result)}")
    if result:
        print(f"URL: {result.get('url')}")
        print(f"Title: {result.get('title')}")
        content = result.get('content', result.get('markdown', ''))
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:500]}")

    # Test search_web via direct tool call
    print("\n--- Testing search_web ---")
    search_tool = next((t for t in tools if t.name == "search_web"), None)
    if search_tool:
        result = await search_tool.ainvoke({"query": "python documentation site:docs.python.org"})
        print(f"Result type: {type(result)}")
        print(f"Result preview: {str(result)[:500]}")
    else:
        print("search_web tool not found")


if __name__ == "__main__":
    asyncio.run(debug_mcp())
