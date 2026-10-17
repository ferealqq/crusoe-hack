"""A plain tool-calling loop over Crusoe's OpenAI-compatible chat API."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app import events
from app.services import inference

log = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentResult:
    final: str
    messages: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)
    turns: int = 0


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for k, v in usage.items():
        total[k] = total.get(k, 0) + v
    total["calls"] = total.get("calls", 0) + 1


async def run_agent(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[Tool],
    params: dict[str, Any],
    max_turns: int,
    wrap_up: str = "You are out of turns. Reply now with your final answer and no tool calls.",
    public_info: dict[str, Any] | None = None,
) -> AgentResult:
    """Run until the model replies without a tool call, or `max_turns` is used up.

    An API or tool failure that survives the retries ends the agent early instead of
    crashing the generation; the reason becomes its final message."""
    usage: dict[str, int] = {}
    with events.scope(model=model):
        events.emit("agent.started", max_turns=max_turns, **(public_info or {}))
        try:
            result = await _loop(model, messages, tools, params, max_turns, wrap_up, usage)
        except asyncio.CancelledError:
            events.emit("agent.cancelled")
            raise
        except Exception as e:
            log.exception("agent on %s stopped early", model)
            events.emit("agent.failed", error_type=type(e).__name__, usage=usage)
            final = f"(stopped early: {type(e).__name__}: {str(e)[:300]})"
            return AgentResult(final, messages, usage, -1)
        events.emit("agent.completed", report=result.final[:2000], usage=usage, turns=result.turns)
        return result


async def _loop(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[Tool],
    params: dict[str, Any],
    max_turns: int,
    wrap_up: str,
    usage: dict[str, int],
) -> AgentResult:
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    for turn in range(1, max_turns + 1):
        with events.scope(turn=turn):
            events.emit("agent.turn.started")
            msg, finish, u = await inference.chat_turn(messages, model, schemas, **params)
            events.emit("agent.turn.completed", usage=u, finish_reason=finish)
        _add_usage(usage, u)
        content = msg.content or ""
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": content})
            if content.strip():
                return AgentResult(content, messages, usage, turn)
            # Thinking used the whole budget, or an empty reply: nudge and go on.
            log.info("empty reply from %s (finish=%s), nudging", model, finish)
            messages.append(
                {"role": "user", "content": "Your reply was empty. Call a tool or answer."}
            )
            continue
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )
        # Run tool calls one by one, in order, so budgets are enforced exactly.
        for tc in msg.tool_calls:
            with events.scope(turn=turn):
                name = tc.function.name
                # Never publish code, prompts, raw tool output, or model reasoning.
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except ValueError:
                    args = {}
                public = (
                    {
                        k: str(v)[:400]
                        for k, v in args.items()
                        if k in {"note", "query", "claim", "kind", "run_id", "id", "reason"}
                    }
                    if isinstance(args, dict)
                    else {}
                )
                events.emit("tool.started", tool=name, call_id=tc.id, arguments=public)
                if public.get("note") or public.get("claim"):
                    events.emit(
                        "agent.highlight",
                        source="tool_intent",
                        tool=name,
                        text=public.get("note") or public["claim"],
                    )
                try:
                    result = await _call(by_name, name, tc.function.arguments)
                except asyncio.CancelledError:
                    events.emit("tool.cancelled", tool=name, call_id=tc.id)
                    raise
                except Exception as e:
                    events.emit(
                        "tool.failed", tool=name, call_id=tc.id, error_type=type(e).__name__
                    )
                    raise
                safe_result = {
                    k: v
                    for k, v in result.items()
                    if k
                    in {
                        "status",
                        "val_loss",
                        "run_id",
                        "run_ids",
                        "runs_left",
                        "mean",
                        "spread",
                        "beats_best",
                        "id",
                        "tokens",
                        "cap",
                        "ok",
                        "seeds",
                        "val_losses",
                        "searches_left",
                    }
                }
                events.emit(
                    "tool.completed",
                    tool=name,
                    call_id=tc.id,
                    result=safe_result,
                    has_error="error" in result,
                )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)}
            )
    messages.append({"role": "user", "content": wrap_up})
    with events.scope(turn=max_turns + 1):
        events.emit("agent.turn.started", wrap_up=True)
        msg, finish, u = await inference.chat_turn(messages, model, None, **params)
        events.emit("agent.turn.completed", usage=u, finish_reason=finish)
    _add_usage(usage, u)
    final = msg.content or ""
    messages.append({"role": "assistant", "content": final})
    return AgentResult(final, messages, usage, max_turns + 1)


async def _call(by_name: dict[str, Tool], name: str, arguments: str) -> dict[str, Any]:
    tool = by_name.get(name)
    if tool is None:
        return {"error": f"unknown tool {name}; available: {sorted(by_name)}"}
    try:
        args = json.loads(arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except ValueError as e:
        return {"error": f"invalid JSON arguments: {e}"}
    try:
        return await tool.fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
