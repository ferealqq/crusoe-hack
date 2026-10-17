"""Web search for researchers via the You.com Search API."""

from typing import Any

import httpx

from app.config import get_settings

URL = "https://ydc-index.io/v1/search"


async def web_search(query: str, count: int = 5) -> list[dict[str, Any]]:
    """Up to `count` results: title, url, description."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            URL,
            headers={"X-API-Key": get_settings().ydc_api_key},
            json={"query": query, "count": count},
        )
        r.raise_for_status()
    web = r.json().get("results", {}).get("web", [])
    return [
        {"title": w.get("title"), "url": w.get("url"), "description": w.get("description")}
        for w in web[:count]
    ]
