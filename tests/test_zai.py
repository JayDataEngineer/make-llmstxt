#!/usr/bin/env python3
"""Test ZAI LLM integration."""
import asyncio
import os
from pathlib import Path

# Load .env from script directory
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from make_llmstxt.config import AppConfig
from make_llmstxt.llm import create_llm, generate_summary
from langchain_core.messages import HumanMessage, SystemMessage


async def test_zai():
    config = AppConfig.from_env()
    print(f"Provider: {config.llm.provider}")
    print(f"Model: {config.llm.model}")
    print(f"Base URL: {config.llm.base_url}")
    print(f"API Key: {config.llm.api_key[:20] if config.llm.api_key else 'None'}...")

    llm = create_llm(config.llm)

    # Simple test
    messages = [
        SystemMessage(content="You are a helpful assistant. Respond briefly."),
        HumanMessage(content="Say 'Hello, ZAI works!' and nothing else."),
    ]

    print("\nSending request to ZAI...")
    response = await llm.ainvoke(messages)
    print(f"Response: {response.content}")

    # Test summary generation
    print("\nTesting summary generation...")
    summary = await generate_summary(
        llm,
        "https://example.com",
        "# Example Domain\n\nThis domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.",
    )
    print(f"Title: {summary['title']}")
    print(f"Description: {summary['description']}")


if __name__ == "__main__":
    asyncio.run(test_zai())
