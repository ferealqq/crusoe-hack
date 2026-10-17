"""Crusoe Intelligence Foundry inference. The API is OpenAI-compatible."""

import asyncio
import logging
import random
import time
from functools import lru_cache
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from app.config import get_settings
from app.schemas import Reply, ToolCall, Usage

BASE_URL = "https://api.inference.crusoecloud.com/v1"
# Rate limits (30 requests/min per model without a payment method) and busy endpoints are
# transient: keep retrying for up to 15 minutes instead of failing a 2-hour research run.
RETRY_ON = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)
MAX_RETRY_WAIT_S = 900

log = logging.getLogger(__name__)


@lru_cache
def get_client() -> AsyncOpenAI:
    # The SDK retries 429/5xx a few times quickly (honours Retry-After); complete() then waits
    # longer.
    return AsyncOpenAI(api_key=get_settings().crusoe_api_key, base_url=BASE_URL, max_retries=3)


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
    waited, delay = 0.0, 10.0
    while True:
        try:
            resp = await get_client().chat.completions.create(
                model=model, messages=messages, **params
            )
            break
        except RETRY_ON as e:
            if waited >= MAX_RETRY_WAIT_S:
                raise
            pause = min(delay, 60.0) * random.uniform(0.5, 1.0)
            log.warning("%s from %s; retrying in %.0f s", type(e).__name__, model, pause)
            await asyncio.sleep(pause)
            waited, delay = waited + pause, delay * 2
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
