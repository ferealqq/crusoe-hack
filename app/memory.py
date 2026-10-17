"""What researchers read: Shared Brain's curated memory, or the Raw log setup's report log."""

from typing import Any, Literal

from pydantic import BaseModel

Kind = Literal["works", "failed", "try_next", "best"]
KIND_ORDER: tuple[Kind, ...] = ("best", "works", "failed", "try_next")
KIND_TITLES = {"best": "Best", "works": "Works", "failed": "Failed", "try_next": "Try next"}
CLAIM_MAX_CHARS = 200
STALE_AFTER_VERSIONS = 2


def count_tokens(text: str) -> int:
    """README: tokens are counted as characters / 4."""
    return len(text) // 4


class MemoryEntry(BaseModel):
    id: str
    kind: Kind
    claim: str
    run_ids: list[str] = []
    val_change: float | None = None
    seeds: list[int] = []
    tested_on: int  # best version the claim was tested against
    created_gen: int
    updated_gen: int
    active: bool = True


class SharedMemory:
    """Curated memory: lean (token cap), current (staleness), conflict-free (curator only)."""

    def __init__(self, cap_tokens: int = 2000) -> None:
        self.cap_tokens = cap_tokens
        self.entries: dict[str, MemoryEntry] = {}
        self.events: list[dict[str, Any]] = []
        self._next_id = 1

    def active(self) -> list[MemoryEntry]:
        def key(e: MemoryEntry) -> tuple[int, int]:
            return KIND_ORDER.index(e.kind), int(e.id[1:]) if e.id != "best" else 0

        return sorted((e for e in self.entries.values() if e.active), key=key)

    def is_stale(self, entry: MemoryEntry, best_version: int) -> bool:
        return entry.kind != "best" and best_version - entry.tested_on >= STALE_AFTER_VERSIONS

    def stale(self, best_version: int) -> list[MemoryEntry]:
        return [e for e in self.active() if self.is_stale(e, best_version)]

    def render(self, best_version: int) -> str:
        """Fixed order (best, works, failed, try_next, then id) so the prompt cache can reuse it."""
        lines: list[str] = []
        for kind in KIND_ORDER:
            group = [e for e in self.active() if e.kind == kind]
            if not group:
                continue
            lines.append(f"## {KIND_TITLES[kind]}")
            for e in group:
                if kind == "best":
                    lines.append(e.claim)
                    continue
                tags = [f"tested on v{e.tested_on}"]
                if e.val_change is not None:
                    tags.append(f"val {e.val_change:+.4f}")
                if e.run_ids:
                    tags.append("runs " + ",".join(e.run_ids))
                if self.is_stale(e, best_version):
                    tags.append("STALE")
                lines.append(f"- [{e.id}] ({'; '.join(tags)}) {e.claim}")
        return "\n".join(lines)

    def tokens(self, best_version: int) -> int:
        return count_tokens(self.render(best_version))

    def upsert(
        self,
        *,
        kind: Kind,
        claim: str,
        run_ids: list[str],
        gen: int,
        best_version: int,
        val_change: float | None = None,
        seeds: list[int] | None = None,
        id: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Curator write. Returns {id, tokens, cap} or {error, tokens, cap} (refused, unchanged)."""
        if kind == "best":
            return self._refuse("only the orchestrator sets best", best_version)
        if len(claim) > CLAIM_MAX_CHARS:
            return self._refuse(f"claim is over {CLAIM_MAX_CHARS} characters", best_version)
        if id is not None and (id not in self.entries or not self.entries[id].active):
            return self._refuse(f"no active entry {id}", best_version)
        old = self.entries.get(id) if id else None
        entry = MemoryEntry(
            id=id or f"m{self._next_id}",
            kind=kind,
            claim=claim,
            run_ids=run_ids,
            val_change=val_change,
            seeds=seeds or [],
            tested_on=best_version,
            created_gen=old.created_gen if old else gen,
            updated_gen=gen,
        )
        self.entries[entry.id] = entry
        tokens = self.tokens(best_version)
        if tokens > self.cap_tokens:
            if old:
                self.entries[entry.id] = old
            else:
                del self.entries[entry.id]
            return self._refuse("over the token cap; remove or shorten entries", best_version)
        if not old:
            self._next_id += 1
        self._log("upsert", entry, gen, reason or ("edit" if old else "add"))
        return {"id": entry.id, "tokens": tokens, "cap": self.cap_tokens}

    def remove(self, id: str, reason: str, gen: int) -> dict[str, Any]:
        entry = self.entries.get(id)
        if entry is None or not entry.active or entry.kind == "best":
            return {"error": f"no removable entry {id}"}
        entry.active = False
        self._log("remove", entry, gen, reason)
        return {"ok": True}

    def set_best(self, claim: str, run_ids: list[str], version: int, gen: int, val: float) -> None:
        entry = MemoryEntry(
            id="best",
            kind="best",
            claim=claim,
            run_ids=run_ids,
            val_change=None,
            seeds=[],
            tested_on=version,
            created_gen=gen,
            updated_gen=gen,
        )
        self.entries["best"] = entry
        self._log("set_best", entry, gen, f"new best v{version}, val {val:.4f}")

    def enforce_cap(self, gen: int, best_version: int) -> list[str]:
        """Drop the oldest try_next, then the oldest failed, until under the cap. Never best."""
        dropped: list[str] = []
        for kind in ("try_next", "failed"):
            while self.tokens(best_version) > self.cap_tokens:
                group = [e for e in self.active() if e.kind == kind]
                if not group:
                    break
                oldest = min(group, key=lambda e: int(e.id[1:]))  # ids grow with creation
                oldest.active = False
                self._log("drop", oldest, gen, "over the token cap")
                dropped.append(oldest.id)
        return dropped

    def _refuse(self, error: str, best_version: int) -> dict[str, Any]:
        return {"error": error, "tokens": self.tokens(best_version), "cap": self.cap_tokens}

    def _log(self, action: str, entry: MemoryEntry, gen: int, reason: str) -> None:
        self.events.append(
            {"gen": gen, "action": action, "reason": reason, "entry": entry.model_dump()}
        )


class RawLog:
    """Raw log setup: every report appended as-is; the oldest dropped past the token cap."""

    def __init__(self, cap_tokens: int = 100_000) -> None:
        self.cap_tokens = cap_tokens
        self.reports: list[str] = []

    def append(self, report: str) -> None:
        self.reports.append(report)
        while len(self.reports) > 1 and count_tokens(self.render()) > self.cap_tokens:
            self.reports.pop(0)

    def render(self) -> str:
        return "\n\n---\n\n".join(self.reports)

    def tokens(self) -> int:
        return count_tokens(self.render())
