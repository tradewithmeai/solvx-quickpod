"""WebSearch tool using Tavily API."""
import os

import httpx


def _get_tavily_key() -> str:
    """Get Tavily API key, loading from .env if needed."""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        # Try loading .env file (may not have been loaded yet)
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv())
        key = os.getenv("TAVILY_API_KEY")
    return key


MAX_OUTPUT_LENGTH = 2000  # Prevent prompt bloat


def websearch(query: str, k: int = 5) -> str:
    """
    Perform a web search using Tavily API.

    Args:
        query: Search query string (required)
        k: Number of results to return (default 5)

    Returns:
        Formatted search results string
    """
    # Mandatory logging (console)
    print(f'[WEBSEARCH EXECUTED] query="{query}"')

    api_key = _get_tavily_key()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    # Call Tavily API
    response = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": k,
            "include_answer": False,
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Tavily API error {response.status_code}: {response.text}")

    data = response.json()
    results = data.get("results", [])

    # Format output - include marker in output so model sees proof too
    lines = ["[WEBSEARCH EXECUTED]", "WEB SEARCH RESULTS:"]
    sources = []

    for i, r in enumerate(results[:k], 1):
        title = r.get("title", "No title")
        snippet = r.get("content", "")[:200]
        url = r.get("url", "")
        lines.append(f"{i}) {title} – {snippet}")
        if url:
            sources.append(url)

    if sources:
        lines.append("\nSources:")
        for url in sources:
            lines.append(f"- {url}")

    output = "\n".join(lines)

    # Cap output length to prevent prompt bloat
    if len(output) > MAX_OUTPUT_LENGTH:
        output = output[:MAX_OUTPUT_LENGTH] + "\n[TRUNCATED]"

    return output
