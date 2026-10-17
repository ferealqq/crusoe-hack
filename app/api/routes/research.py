"""Read-only dashboard API, independent of Postgres and inference providers."""

import asyncio
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.events import ResearchEvent, read_events
from app.services import research_events

router = APIRouter(prefix="/research-runs", tags=["research telemetry"])


class RunInfo(BaseModel):
    run_id: str
    dataset: str | None
    setups: list[str]
    status: Literal["unavailable", "running", "stalled", "completed", "failed", "cancelled"]
    events_available: bool
    last_seq: int
    duration_ms: int
    last_event_at: datetime | None
    replay_available: bool


class EventPage(BaseModel):
    events: list[ResearchEvent]
    next_after: int
    has_more: bool
    last_seq: int


def _root(run_id: str):
    try:
        return research_events.resolve_run(get_settings().data_dir, run_id)
    except FileNotFoundError:
        raise HTTPException(404, "research run not found") from None


@router.get("", response_model=list[RunInfo])
async def runs() -> list[dict]:
    return await asyncio.to_thread(research_events.list_runs, get_settings().data_dir)


@router.get("/{run_id}", response_model=RunInfo)
async def detail(run_id: str) -> dict:
    return await asyncio.to_thread(research_events.metadata, _root(run_id))


@router.get("/{run_id}/events", response_model=EventPage)
async def history(
    run_id: str,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
    until: Annotated[int | None, Query(ge=0)] = None,
) -> dict:
    return await asyncio.to_thread(
        research_events.event_page, _root(run_id) / "events.jsonl", after, limit, until
    )


@router.get("/{run_id}/state")
async def state(run_id: str, at: Annotated[int | None, Query(ge=0)] = None) -> dict:
    return await asyncio.to_thread(research_events.state_at, _root(run_id) / "events.jsonl", at)


@router.get("/{run_id}/stream", response_class=StreamingResponse)
async def stream(
    run_id: str,
    after: Annotated[int, Query(ge=0)] = 0,
    mode: Literal["live", "replay"] = "live",
    speed: Annotated[float, Query(ge=0.1, le=1000, allow_inf_nan=False)] = 1,
    max_gap: Annotated[float | None, Query(ge=0, le=60, allow_inf_nan=False)] = None,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    path = _root(run_id) / "events.jsonl"
    if not path.exists():
        raise HTTPException(
            409, "This older run has no event recording; record a new run for replay"
        )
    if last_event_id is not None:
        try:
            after = max(after, int(last_event_id))
        except ValueError:
            raise HTTPException(422, "Last-Event-ID must be a nonnegative integer") from None
        if int(last_event_id) < 0:
            raise HTTPException(422, "Last-Event-ID must be a nonnegative integer")
    events: list[ResearchEvent] = await asyncio.to_thread(read_events, path)
    end_seq = events[-1].seq if events else 0
    if after > end_seq:
        raise HTTPException(422, "cursor is beyond the last recorded event")
    return StreamingResponse(
        research_events.stream(
            path, after=after, mode=mode, speed=speed, max_gap=max_gap, end_seq=end_seq
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
