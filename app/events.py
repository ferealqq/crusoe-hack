"""Public research telemetry. One writer per run; CLI and API share only the JSONL file.

ContextVars keep concurrent setups/agents isolated without passing UI callbacks through
provider APIs. No prompts, code, raw reasoning, labels, or credentials belong here.
"""

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

TERMINAL = {"run.completed", "run.failed", "run.cancelled"}


class ResearchEvent(BaseModel):
    version: int = 1
    seq: int
    run_id: str
    timestamp: datetime
    elapsed_ms: int
    type: str
    setup: str | None = None
    gen: int | None = None
    agent: str | None = None
    role: str | None = None
    phase: str | None = None
    experiment_id: str | None = None
    seed: int | None = None
    turn: int | None = None
    model: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class EventLog:
    """Synchronous, flushed append: sequence assignment and write cannot interleave.

    A fresh run directory is required. Multiple OS processes must never write the same
    log. Readers may run in other processes and ignore an unfinished trailing line.
    """

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "events.jsonl"
        self.path.touch(exist_ok=False)
        self.run_id = root.name
        self.started = time.monotonic()
        self.seq = 0
        self.lock = threading.Lock()

    def append(self, kind: str, context: dict[str, Any], data: dict[str, Any]) -> ResearchEvent:
        with self.lock:
            event = ResearchEvent(
                seq=self.seq + 1,
                run_id=self.run_id,
                timestamp=datetime.now(UTC),
                elapsed_ms=round((time.monotonic() - self.started) * 1000),
                type=kind,
                data=data,
                **context,
            )
            with self.path.open("a") as f:
                f.write(event.model_dump_json() + "\n")
            self.seq = event.seq
            return event


_log: ContextVar[EventLog | None] = ContextVar("research_event_log", default=None)
_context: ContextVar[dict[str, Any] | None] = ContextVar("research_event_context", default=None)


@contextmanager
def bind(log: EventLog) -> Iterator[None]:
    token = _log.set(log)
    context_token = _context.set({})
    try:
        yield
    finally:
        _context.reset(context_token)
        _log.reset(token)


@contextmanager
def scope(**fields: Any) -> Iterator[None]:
    token = _context.set((_context.get() or {}) | fields)
    try:
        yield
    finally:
        _context.reset(token)


def emit(kind: str, **data: Any) -> ResearchEvent | None:
    log = _log.get()
    return log.append(kind, _context.get() or {}, data) if log is not None else None


class EventReader:
    """Incremental reader; never consumes an incomplete append."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.last: ResearchEvent | None = None

    def read(self, limit: int = 256) -> list[ResearchEvent]:
        if not self.path.exists():
            return []
        result = []
        with self.path.open("rb") as f:
            f.seek(self.offset)
            for _ in range(limit):
                line = f.readline()
                if not line.endswith(b"\n"):
                    break
                event = ResearchEvent.model_validate_json(line)
                expected = self.last.seq + 1 if self.last else 1
                if event.seq != expected:
                    raise ValueError("non-contiguous event sequence")
                self.offset = f.tell()
                self.last = event
                result.append(event)
        return result


def read_events(path: Path) -> list[ResearchEvent]:
    reader = EventReader(path)
    result = []
    while batch := reader.read():
        result.extend(batch)
    return result


def sse(event: ResearchEvent) -> str:
    return f"id: {event.seq}\nevent: research\ndata: {event.model_dump_json()}\n\n"


def control(kind: str, **data: Any) -> str:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"
