"""Crusoe Intelligence Foundry inference. The API is OpenAI-compatible."""

import asyncio
import logging
import random
import time
from functools import lru_cache
from typing import Any

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from app import events
from app.config import get_settings

log = logging.getLogger(__name__)

# 429s seen per model during agent turns; the orchestrator reports them.
RATE_LIMITED: dict[str, int] = {}
TRANSIENT = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@lru_cache
def get_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.crusoe_api_key, base_url=s.crusoe_base_url, max_retries=s.crusoe_max_retries
    )


async def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
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
        model=model or get_settings().crusoe_default_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    return resp.choices[0].message.content or "", latency_ms


async def list_models() -> list[str]:
    models = await get_client().models.list()
    return [m.id for m in models.data]


class RateLimiter:
    """Spaces requests to one model evenly at `rpm` per minute. Crusoe limits per model and
    also refuses short bursts (about 1-2 requests a second), so even spacing beats a window."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60 / rpm
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if self.next_at > now:
                events.emit(
                    "agent.waiting", reason="rate_limit_spacing", seconds=self.next_at - now
                )
                await asyncio.sleep(self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval


_limiters: dict[str, RateLimiter] = {}


def _limiter(model: str) -> RateLimiter:
    if model not in _limiters:
        _limiters[model] = RateLimiter(get_settings().crusoe_max_rpm)
    return _limiters[model]


async def chat_turn(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]] | None = None,
    **params: Any,
) -> tuple[ChatCompletionMessage, str | None, dict[str, int]]:
    """One agent turn. Returns (message, finish_reason, usage) where usage has
    prompt, cached and output token counts."""
    kwargs: dict[str, Any] = dict(params)
    if tools:
        kwargs["tools"] = tools
    # Retries go through the limiter too, with long jittered waits (~20 min in total), so
    # a burst of 429s slows agents down instead of stopping them.
    client = get_client().with_options(max_retries=0)
    attempts = 25
    for attempt in range(attempts):
        await _limiter(model).wait()
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, **kwargs)
            break
        except TRANSIENT as e:
            if isinstance(e, openai.RateLimitError):
                RATE_LIMITED[model] = RATE_LIMITED.get(model, 0) + 1
            if attempt == attempts - 1:
                raise
            delay = min(60.0, 5.0 * 2**attempt) * random.uniform(0.6, 1.0)
            events.emit(
                "agent.retry", error_type=type(e).__name__, attempt=attempt + 1, seconds=delay
            )
            log.info("%s from %s, retrying in %.0fs", type(e).__name__, model, delay)
            await asyncio.sleep(delay)
    choice = resp.choices[0]
    u = resp.usage
    details = getattr(u, "prompt_tokens_details", None) if u else None
    usage = {
        "prompt": u.prompt_tokens if u else 0,
        "cached": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
        "output": u.completion_tokens if u else 0,
    }
    return choice.message, choice.finish_reason, usage
