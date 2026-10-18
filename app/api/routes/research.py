"""Telemetry for the demo UI: research runs under data/, as events (app/replay.py), and
POST to start a new run from a prompt (app/launch.py)."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import launch, replay
from app.config import get_settings

router = APIRouter(prefix="/research-runs", tags=["research"])

POLL_S = 2.0
KEEPALIVE_S = 15.0


def _snapshot(run_id: str) -> replay.Snapshot:
    snap = replay.snapshots().get(run_id)
    if snap is None:
        raise HTTPException(404, f"no research run {run_id}")
    return snap


@router.get("")
async def list_runs() -> list[dict[str, Any]]:
    return [s.info for s in (await asyncio.to_thread(replay.snapshots)).values()]


class LaunchRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    generations: int = Field(default=5, ge=1, le=10)


@router.post("", status_code=201)
async def start_run(body: LaunchRequest) -> dict[str, Any]:
    """Start Shared Brain and No memory side by side with `prompt` as the researchers' brief."""
    if not get_settings().research_launch_enabled:
        raise HTTPException(403, "starting runs is turned off on this server")
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(422, "the prompt is empty")
    runs = (await asyncio.to_thread(replay.snapshots)).values()
    busy = launch.active() or next((s for s in runs if s.info["status"] == "running"), None)
    if busy:
        run_id = busy.run_id if isinstance(busy, launch.Launch) else busy.info["run_id"]
        raise HTTPException(409, {"message": "a run is already in progress", "run_id": run_id})
    started = launch.start(prompt, body.generations)
    try:
        await asyncio.to_thread(launch.wait_started, started)
    except launch.LaunchError as e:
        raise HTTPException(502, str(e)) from e
    return {"run_id": started.run_id}


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    return (await asyncio.to_thread(_snapshot, run_id)).info


@router.get("/{run_id}/events")
async def get_events(run_id: str, after: int = 0, limit: int = 2000) -> dict[str, Any]:
    events = (await asyncio.to_thread(_snapshot, run_id)).events
    page = [e for e in events if e["seq"] > after][: max(1, min(limit, 5000))]
    return {
        "events": page,
        "next_after": page[-1]["seq"] if page else after,
        "has_more": bool(page) and page[-1]["seq"] < events[-1]["seq"],
    }


def _sse(event: str, data: Any, event_id: int | None = None) -> str:
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/{run_id}/stream")
async def stream(run_id: str, request: Request, after: int = 0) -> StreamingResponse:
    """Backlog after `after`, then new events as the run's files change.

    Events are rebuilt from files, so history can shift when an agent finishes (its trace is
    placed back in time). Then the stream sends `stream.reset` and the client reloads.
    """
    first = await asyncio.to_thread(_snapshot, run_id)

    async def gen() -> AsyncIterator[str]:
        sent = [e for e in first.events if e["seq"] <= after]
        idle = 0.0
        while not await request.is_disconnected():
            snap = await asyncio.to_thread(_snapshot, run_id)
            if snap.events[: len(sent)] != sent:
                yield _sse("stream.reset", {"reason": "history changed"})
                return
            for e in snap.events[len(sent) :]:
                yield _sse("research", e, e["seq"])
                idle = 0.0
            sent = snap.events
            if snap.info["status"] not in ("running", "stalled"):
                yield _sse("stream.end", {"status": snap.info["status"]})
                return
            await asyncio.sleep(POLL_S)
            idle += POLL_S
            if idle >= KEEPALIVE_S:
                yield ": keepalive\n\n"
                idle = 0.0

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )
