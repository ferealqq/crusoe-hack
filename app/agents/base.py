"""A minimal tool-calling loop over Crusoe's OpenAI-compatible API."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.schemas import Usage
from app.services import inference

ToolFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

NUDGE = "Continue: call a tool, or write your final report."
OUT_OF_TURNS = "You are out of turns. Write your final report now, with no tool calls."


@dataclass(frozen=True)
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
    report: str
    transcript: list[dict[str, Any]]
    usage: Usage
    turns: int


async def _call(tools: dict[str, Tool], name: str, arguments: str) -> dict[str, Any]:
    tool = tools.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}; use one of {sorted(tools)}"}
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"error": f"arguments are not valid JSON: {e}"}
    if not isinstance(args, dict):
        return {"error": "arguments must be a JSON object"}
    try:
        return await tool.fn(args)
    except Exception as e:  # tell the model, don't crash the agent
        return {"error": f"{type(e).__name__}: {e}"}


async def run_agent(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[Tool],
    max_turns: int,
    params: dict[str, Any],
) -> AgentResult:
    """Call the model until it replies without a tool call (its report) or runs out of turns."""
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    usage = Usage()
    for turn in range(1, max_turns + 1):
        reply = await inference.complete(messages, model, schemas, **params)
        usage += reply.usage
        if not reply.tool_calls:
            messages.append({"role": "assistant", "content": reply.content})
            if reply.content.strip():
                return AgentResult(reply.content.strip(), messages, usage, turn)
            messages.append({"role": "user", "content": NUDGE})
            continue
        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in reply.tool_calls
                ],
            }
        )
        for call in reply.tool_calls:
            result = await _call(by_name, call.name, call.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
    messages.append({"role": "user", "content": OUT_OF_TURNS})
    reply = await inference.complete(messages, model, None, **params)
    usage += reply.usage
    messages.append({"role": "assistant", "content": reply.content})
    return AgentResult(reply.content.strip(), messages, usage, max_turns + 1)
