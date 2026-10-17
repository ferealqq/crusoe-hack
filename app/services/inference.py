"""Crusoe Intelligence Foundry inference. The API is OpenAI-compatible."""

import time
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas import Reply, ToolCall, Usage

BASE_URL = "https://api.inference.crusoecloud.com/v1"


@lru_cache
def get_client() -> AsyncOpenAI:
    # The SDK retries 429/5xx with exponential backoff (honours Retry-After)
    return AsyncOpenAI(api_key=get_settings().crusoe_api_key, base_url=BASE_URL, max_retries=6)


async def chat(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    **kwargs,
) -> tuple[str, int]:
    """Returns (content, latency_ms).

    Reasoning models spend `max_tokens` on thinking first; empty content with
    finish_reason == "length" means the budget was too small.
    """
    started = time.perf_counter()
    resp = await get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    return resp.choices[0].message.content or "", latency_ms


async def complete(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]] | None = None,
    **params: Any,
) -> Reply:
    """One chat completion with optional tools. Non-OpenAI params go in `extra_body`."""
    if tools:
        params["tools"] = tools
    resp = await get_client().chat.completions.create(model=model, messages=messages, **params)
    choice = resp.choices[0]
    u = resp.usage
    details = getattr(u, "prompt_tokens_details", None) if u else None
    usage = Usage(
        prompt=u.prompt_tokens if u else 0,
        cached=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
        output=u.completion_tokens if u else 0,
        calls=1,
    )
    return Reply(
        content=choice.message.content or "",
        tool_calls=[
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
            for c in choice.message.tool_calls or []
        ],
        usage=usage,
        finish_reason=choice.finish_reason,
    )


async def list_models() -> list[str]:
    models = await get_client().models.list()
    return [m.id for m in models.data]
