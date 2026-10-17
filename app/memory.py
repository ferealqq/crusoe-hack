"""Shared memory: entries, rendering, cap and staleness (README "Memory").

Stored in memory.jsonl, rewritten on every change. Only the curator writes entries; only the
orchestrator writes the `best` entry.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.store import read_jsonl, write_jsonl

Kind = Literal["best", "works", "failed", "try_next"]
ORDER: tuple[Kind, ...] = ("best", "works", "failed", "try_next")
MAX_CLAIM_CHARS = 200
STALE_AFTER_VERSIONS = 2


class Entry(BaseModel):
    id: str
    kind: Kind
    claim: str
    run_ids: list[str] = []
    val_change: float | None = None  # loss change vs the best it was tested on (negative = better)
    seeds: list[int] = []
    tested_on: int = 0  # best train.py version
    created_gen: int = 0
    updated_gen: int = 0


class MemoryFull(Exception):
    pass


def count_tokens(text: str) -> int:
    """Characters ÷ 4."""
    return len(text) // 4


def _num(entry: Entry) -> int:
    return int(entry.id.removeprefix("m"))


class Memory:
    def __init__(self, path: Path, cap_tokens: int) -> None:
        self.path = path
        self.cap = cap_tokens
        self.entries = [Entry(**row) for row in read_jsonl(path)]

    def render(self, entries: list[Entry] | None = None) -> str:
        """Stable text: best, works, failed, try_next, then by id, so the cache can reuse it."""
        entries = self.entries if entries is None else entries
        if not entries:
            return "(empty)"
        lines: list[str] = []
        kind: str | None = None
        for e in sorted(entries, key=lambda e: (ORDER.index(e.kind), _num(e))):
            if e.kind != kind:
                kind = e.kind
                lines.append(f"## {kind}")
            change = f"val {e.val_change:+.4f} " if e.val_change is not None else ""
            lines.append(f"- [{e.id}] {e.claim} ({change}on v{e.tested_on})")
        return "\n".join(lines)

    def tokens(self) -> int:
        return count_tokens(self.render())

    def get(self, entry_id: str) -> Entry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def upsert(
        self,
        *,
        kind: Kind,
        claim: str,
        run_ids: list[str],
        val_change: float | None,
        seeds: list[int],
        tested_on: int,
        generation: int,
        entry_id: str | None = None,
    ) -> Entry:
        """Add or edit an entry. Raises MemoryFull if the result would be over the cap."""
        if kind == "best":
            raise ValueError("only the orchestrator writes the best entry")
        if len(claim) > MAX_CLAIM_CHARS:
            raise ValueError(f"claim is {len(claim)} characters; the limit is {MAX_CLAIM_CHARS}")
        created = generation
        if entry_id is not None:
            old = self.get(entry_id)
            if old is None:
                raise ValueError(f"no entry {entry_id}")
            if old.kind == "best":
                raise ValueError("only the orchestrator writes the best entry")
            created = old.created_gen
        new = Entry(
            id=entry_id or self._next_id(),
            kind=kind,
            claim=claim,
            run_ids=run_ids,
            val_change=val_change,
            seeds=seeds,
            tested_on=tested_on,
            created_gen=created,
            updated_gen=generation,
        )
        candidate = [e for e in self.entries if e.id != new.id] + [new]
        used = count_tokens(self.render(candidate))
        if used > self.cap:
            raise MemoryFull(f"would use {used} tokens; the cap is {self.cap}")
        self.entries = candidate
        self.save()
        return new

    def remove(self, entry_id: str) -> None:
        entry = self.get(entry_id)
        if entry is None:
            raise ValueError(f"no entry {entry_id}")
        if entry.kind == "best":
            raise ValueError("the best entry can't be removed")
        self.entries.remove(entry)
        self.save()

    def set_best(
        self, *, claim: str, run_ids: list[str], val_change: float | None, version: int, gen: int
    ) -> None:
        old = next((e for e in self.entries if e.kind == "best"), None)
        best = Entry(
            id=old.id if old else self._next_id(),
            kind="best",
            claim=claim[:MAX_CLAIM_CHARS],
            run_ids=run_ids,
            val_change=val_change,
            tested_on=version,
            created_gen=old.created_gen if old else gen,
            updated_gen=gen,
        )
        self.entries = [e for e in self.entries if e.kind != "best"] + [best]
        self.save()

    def enforce_cap(self) -> list[Entry]:
        """Drop the oldest try_next, then the oldest failed, until under the cap. Never drops
        best (or works)."""
        dropped: list[Entry] = []
        for kind in ("try_next", "failed"):
            while self.tokens() > self.cap:
                victims = [e for e in self.entries if e.kind == kind]
                if not victims:
                    break
                oldest = min(victims, key=lambda e: (e.created_gen, _num(e)))
                self.entries.remove(oldest)
                dropped.append(oldest)
        if dropped:
            self.save()
        return dropped

    def stale(self, best_version: int) -> list[Entry]:
        """Entries tested on a best that is 2 or more versions old."""
        return [
            e
            for e in self.entries
            if e.kind != "best" and best_version - e.tested_on >= STALE_AFTER_VERSIONS
        ]

    def save(self) -> None:
        write_jsonl(self.path, [e.model_dump() for e in self.entries])

    def _next_id(self) -> str:
        return f"m{max((_num(e) for e in self.entries), default=0) + 1}"
