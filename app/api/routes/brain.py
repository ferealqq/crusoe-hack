"""Shared Brain demo API: recorded/live run event logs for the frontend (web/)."""

import asyncio
import json
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.runlog import store

router = APIRouter(prefix="/api/runs", tags=["shared-brain"])


@router.get("")
async def list_runs() -> list[dict]:
    return [asdict(r) for r in await store.list_runs()]


@router.get("/{run_id}/events", response_class=PlainTextResponse)
async def get_events(run_id: str) -> PlainTextResponse:
    """The whole event log as JSONL; the frontend paces replays itself."""
    lines = await store.read_lines(run_id)
    if lines is None:
        raise HTTPException(404, "run not found")
    return PlainTextResponse("\n".join(lines) + "\n", media_type="application/x-ndjson")


@router.get("/{run_id}/stream")
async def stream_events(run_id: str, request: Request, after: int = -1) -> StreamingResponse:
    """Server-sent events for a run being written locally; ends after run.finished."""
    path = store.local_path(run_id)
    if not path.exists():
        raise HTTPException(404, "no live run with that id")

    async def tail():
        sent = 0
        while not await request.is_disconnected():
            lines = path.read_text().splitlines()
            for line in lines[sent:]:
                sent += 1
                if not line.strip() or json.loads(line)["seq"] <= after:
                    continue
                yield f"data: {line}\n\n"
                if '"run.finished"' in line:
                    return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        tail(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )
