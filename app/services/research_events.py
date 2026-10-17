"""Read-only run discovery, prefix state, and paced/live event streams. No providers."""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.events import TERMINAL, EventReader, control, read_events, sse

POLL_SECONDS = 0.25
HEARTBEAT_SECONDS = 10
STALE_SECONDS = 30


def resolve_run(data_dir: str, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", run_id):
        raise FileNotFoundError(run_id)
    base = (Path(data_dir) / "runs").resolve()
    root = base / run_id
    if root.is_symlink() or not root.is_dir() or root.resolve().parent != base:
        raise FileNotFoundError(run_id)
    # Also reject an events file redirected outside this run directory.
    if (root / "events.jsonl").is_symlink():
        raise FileNotFoundError(run_id)
    return root


def metadata(root: Path) -> dict[str, Any]:
    events = read_events(root / "events.jsonl")
    last = events[-1] if events else None
    config_path = root / "config.json"
    start = next((e for e in events if e.type == "run.started"), None)
    config = start.data if start else {}
    if not config and config_path.is_file() and not config_path.is_symlink():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass  # A legacy runner may still be writing its config.
    # Deliberately whitelist public metadata rather than exposing config/code/transcripts.
    status = "unavailable"
    if last:
        terminal = next((e for e in reversed(events) if e.type in TERMINAL), None)
        status = terminal.type.split(".")[1] if terminal else "running"
        age = (datetime.now(UTC) - last.timestamp).total_seconds()
        if not terminal and age > STALE_SECONDS:
            status = "stalled"
    return {
        "run_id": root.name,
        "dataset": config.get("dataset"),
        "setups": config.get("setups", []),
        "status": status,
        "events_available": last is not None,
        "last_seq": last.seq if last else 0,
        "duration_ms": last.elapsed_ms if last else 0,
        "last_event_at": last.timestamp if last else None,
        "replay_available": any(e.type == "run.completed" for e in events),
    }


def list_runs(data_dir: str) -> list[dict[str, Any]]:
    base = Path(data_dir) / "runs"
    if not base.exists():
        return []
    result = []
    for path in sorted(base.iterdir(), reverse=True):
        try:
            result.append(metadata(resolve_run(data_dir, path.name)))
        except FileNotFoundError:
            continue
    return result


def event_page(path: Path, after: int, limit: int, until: int | None) -> dict[str, Any]:
    events = read_events(path)
    matches = [e for e in events if e.seq > after and (until is None or e.seq <= until)]
    page = matches[:limit]
    return {
        "events": page,
        "next_after": page[-1].seq if page else after,
        "has_more": len(matches) > limit,
        "last_seq": events[-1].seq if events else 0,
    }


def state_at(path: Path, at: int | None = None) -> dict[str, Any]:
    """A deterministic prefix-only reducer for seeking; never reads future results into state."""
    state: dict[str, Any] = {
        "seq": 0,
        "elapsed_ms": 0,
        "run_id": None,
        "run": None,
        "status": "pending",
        "baseline": None,
        "setups": {},
    }
    for e in read_events(path):
        if at is not None and e.seq > at:
            break
        state.update(seq=e.seq, elapsed_ms=e.elapsed_ms, run_id=e.run_id)
        if e.type == "run.started":
            state["status"] = "running"
            state["run"] = e.data
        if e.type in TERMINAL:
            state["status"] = e.type.split(".")[1]
        if e.type == "baseline.completed":
            state["baseline"] = e.data
        if not e.setup:
            continue
        setup = state["setups"].setdefault(
            e.setup,
            {
                "status": "pending",
                "gen": 0,
                "agents": {},
                "experiments": {},
                "best": None,
                "memory": None,
                "generations": [],
                "result": None,
            },
        )
        if e.gen is not None:
            setup["gen"] = e.gen
        if e.type.startswith("setup."):
            setup["status"] = e.type.split(".")[1]
        if e.type == "generation.started":
            setup["generation"] = e.data
            setup["agents"] = {}
        if e.type == "generation.completed":
            setup["generations"].append(e.data)
        if e.type == "best.updated":
            setup["best"] = e.data
        if e.type == "memory.updated":
            setup["memory"] = e.data
        if e.type == "setup.completed":
            setup["result"] = e.data
        if e.agent:
            # New short-lived agents replace prior-generation cards, not accumulate state.
            agent = setup["agents"].get(e.agent)
            if agent is None or agent["gen"] != e.gen:
                agent = {"gen": e.gen, "role": e.role, "status": "pending"}
                setup["agents"][e.agent] = agent
            agent.update(last_event=e.type, seq=e.seq)
            statuses = {
                "agent.started": "starting",
                "agent.turn.started": "thinking",
                "agent.waiting": "waiting",
                "agent.retry": "waiting",
                "tool.started": "working",
                "tool.completed": "working",
                "experiment.queued": "validating",
                "experiment.started": "validating",
                "experiment.completed": "working",
                "agent.completed": "completed",
                "agent.failed": "failed",
                "agent.cancelled": "cancelled",
            }
            if e.type in statuses:
                agent["status"] = statuses[e.type]
            if e.turn is not None:
                agent["turn"] = e.turn
            if e.type in {"agent.waiting", "agent.retry"}:
                agent["wait"] = e.data
            elif e.type == "agent.turn.completed":
                agent.pop("wait", None)
            if e.type == "agent.started":
                agent.update(e.data)
                agent["model"] = e.model
            if e.type == "tool.started":
                agent["current_tool"] = e.data.get("tool")
            if e.type == "tool.completed":
                agent.pop("current_tool", None)
                if "runs_left" in e.data.get("result", {}):
                    agent["runs_left"] = e.data["result"]["runs_left"]
            if e.type == "experiment.completed":
                agent["last_result"] = e.data | {"experiment_id": e.experiment_id}
                loss = e.data.get("val_loss")
                if loss is not None and loss < agent.get("best_val_loss", float("inf")):
                    agent["best_val_loss"] = loss
            if e.type == "agent.highlight":
                agent["highlight"] = e.data
            if e.type in {"agent.completed", "agent.failed"}:
                agent.update(e.data)
        if e.experiment_id:
            exp = setup["experiments"].setdefault(e.experiment_id, {})
            exp.update(e.data, agent=e.agent, seed=e.seed, gen=e.gen, phase=e.phase)
            exp["event"] = e.type
            if e.type in {
                "experiment.queued",
                "experiment.started",
                "experiment.cancelled",
                "experiment.failed",
            }:
                exp["status"] = e.type.split(".")[1]
    return state


def replay_delay(previous_ms: int, current_ms: int, speed: float, max_gap: float | None) -> float:
    delay = max(0, current_ms - previous_ms) / 1000 / speed
    return min(delay, max_gap) if max_gap is not None else delay


async def stream(
    path: Path,
    *,
    after: int,
    mode: Literal["live", "replay"],
    speed: float,
    max_gap: float | None,
    end_seq: int,
) -> AsyncIterator[str]:
    """Replay freezes its high-water mark on connection; live follows future appends.

    Browser closes the connection to pause; reconnects with after/Last-Event-ID to resume.
    Transport heartbeat/control frames are not research events and have no sequence ID.
    """
    reader = EventReader(path)
    cursor, previous_ms = after, 0
    heartbeat = time.monotonic()
    yield control(
        "stream.open", mode=mode, after=after, end_seq=end_seq if mode == "replay" else None
    )
    while True:
        batch = await asyncio.to_thread(reader.read)
        for e in batch:
            if e.seq <= after:
                previous_ms = e.elapsed_ms
                continue
            if mode == "replay" and e.seq > end_seq:
                break
            if mode == "replay":
                remaining = replay_delay(previous_ms, e.elapsed_ms, speed, max_gap)
                while remaining > 0:
                    step = min(remaining, HEARTBEAT_SECONDS)
                    await asyncio.sleep(step)
                    remaining -= step
                    if remaining > 0:
                        yield ": keep-alive\n\n"
            yield sse(e)
            cursor, previous_ms = e.seq, e.elapsed_ms
            heartbeat = time.monotonic()
        if mode == "replay" and (cursor >= end_seq or not batch):
            yield control("stream.end", mode=mode, cursor=cursor)
            return
        if mode == "live" and reader.last and reader.last.type in TERMINAL:
            yield control("stream.end", mode=mode, cursor=cursor)
            return
        if not batch:
            if time.monotonic() - heartbeat >= HEARTBEAT_SECONDS:
                yield ": keep-alive\n\n"
                heartbeat = time.monotonic()
            await asyncio.sleep(POLL_SECONDS)
