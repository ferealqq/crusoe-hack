"""The curator: re-checks the top claim, settles conflicts and edits the shared memory."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app import events, scoring
from app.agents.base import Tool, run_agent
from app.config import ResearchConfig, get_settings
from app.lab import Lab
from app.memory import SharedMemory

CURATOR_PARAMS: dict[str, Any] = {
    "max_tokens": 8192,
    "extra_body": {"chat_template_kwargs": {"thinking": True}},
}


@dataclass
class Check:
    """A re-run of one train.py on several seeds."""

    source: str  # the run_id re-checked, or "code"
    code: str
    run_ids: list[str]
    seeds: list[int]
    losses: list[float | None]
    mean: float | None
    spread: float | None
    beats_best: bool
    by: str  # "curator" or "orchestrator" (fallback)


@dataclass
class CuratorState:
    runs_left: int
    checks: list[Check] = field(default_factory=list)


async def check_code(
    *,
    lab: Lab,
    code: str,
    source: str,
    seeds: list[int],
    gen: int,
    best_version: int,
    best_val: float,
    threshold: float,
    by: str,
) -> Check:
    with events.scope(agent=by, role="curator", phase="verification", gen=gen):
        return await _check_code(
            lab=lab,
            code=code,
            source=source,
            seeds=seeds,
            gen=gen,
            best_version=best_version,
            best_val=best_val,
            threshold=threshold,
            by=by,
        )


async def _check_code(
    *,
    lab: Lab,
    code: str,
    source: str,
    seeds: list[int],
    gen: int,
    best_version: int,
    best_val: float,
    threshold: float,
    by: str,
) -> Check:
    events.emit("verification.started", source=source, seeds=seeds)
    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(
                lab.run(code, seed, by, f"re-check {source} (seed {seed})", gen, best_version)
            )
            for seed in seeds
        ]
    exps = [task.result() for task in tasks]
    losses = [e.val_loss for e in exps]
    ok = [x for x in losses if x is not None]
    complete = len(ok) == len(losses)
    mean = sum(ok) / len(ok) if ok else None
    check = Check(
        source=source,
        code=code,
        run_ids=[e.id for e in exps],
        seeds=seeds,
        losses=losses,
        mean=mean,
        spread=scoring.spread(ok) if ok else None,
        beats_best=complete and mean is not None and scoring.beats(mean, best_val, threshold),
        by=by,
    )
    events.emit(
        "verification.completed", **{k: v for k, v in check.__dict__.items() if k != "code"}
    )
    return check


def curator_tools(
    *,
    lab: Lab,
    memory: SharedMemory,
    state: CuratorState,
    cfg: ResearchConfig,
    gen: int,
    best_version: int,
    best_val: float,
    threshold: float,
) -> list[Tool]:
    def val_change(run_ids: list[str]) -> tuple[float | None, list[int]]:
        """Val change vs this generation's starting best, from the cited runs."""
        for c in state.checks:
            if set(c.run_ids) & set(run_ids) or c.source in run_ids:
                return (c.mean - best_val if c.mean is not None else None), c.seeds
        exps = [lab.experiments[r] for r in run_ids if r in lab.experiments]
        ok = [e for e in exps if e.val_loss is not None]
        if not ok:
            return None, sorted({e.seed for e in exps})
        best = min(ok, key=lambda e: e.val_loss or 0.0)
        return (best.val_loss or 0.0) - best_val, [best.seed]

    async def get_memory() -> dict[str, Any]:
        return {"memory": memory.render(best_version) or "(empty)"}

    async def run_code(run_id: str | None = None, code: str | None = None) -> dict[str, Any]:
        if state.runs_left <= 0:
            return {"error": "no runs left this generation", "runs_left": 0}
        if run_id:
            exp = lab.experiments.get(run_id)
            if exp is None:
                return {"error": f"unknown run_id {run_id}"}
            code, source = exp.code, run_id
        elif code:
            source = "code"
        else:
            return {"error": "pass run_id or code"}
        assert code is not None
        seeds = cfg.curator_seeds[: min(len(cfg.curator_seeds), state.runs_left)]
        state.runs_left -= len(seeds)
        check = await check_code(
            lab=lab,
            code=code,
            source=source,
            seeds=seeds,
            gen=gen,
            best_version=best_version,
            best_val=best_val,
            threshold=threshold,
            by="curator",
        )
        state.checks.append(check)
        return {
            "run_ids": check.run_ids,
            "seeds": check.seeds,
            "val_losses": check.losses,
            "mean": check.mean,
            "spread": check.spread,
            "beats_best": check.beats_best,
            "best_val_loss": best_val,
            "noise_threshold": threshold,
            "runs_left": state.runs_left,
        }

    async def upsert_memory(
        kind: str, claim: str, run_ids: list[str] | None = None, id: str | None = None
    ) -> dict[str, Any]:
        run_ids = run_ids or []
        unknown = [r for r in run_ids if r not in lab.experiments]
        if unknown:
            return {"error": f"unknown run_ids {unknown}"}
        if kind in ("works", "failed") and not run_ids:
            return {"error": "works and failed entries need the run_ids that show it"}
        if kind not in ("works", "failed", "try_next"):
            return {"error": "kind must be works, failed or try_next"}
        change, seeds = val_change(run_ids) if run_ids else (None, [])
        result = memory.upsert(
            kind=kind,  # type: ignore[arg-type]
            claim=claim,
            run_ids=run_ids,
            gen=gen,
            best_version=best_version,
            val_change=change,
            seeds=seeds,
            id=id,
        )
        if "error" not in result:
            events.emit(
                "memory.updated",
                action="upsert",
                entry=memory.entries[result["id"]].model_dump(),
                text=memory.render(best_version),
                tokens=memory.tokens(best_version),
            )
        return result

    async def remove_memory(id: str, reason: str) -> dict[str, Any]:
        result = memory.remove(id, reason, gen)
        if "error" not in result:
            events.emit(
                "memory.updated",
                action="remove",
                id=id,
                reason=reason[:400],
                text=memory.render(best_version),
                tokens=memory.tokens(best_version),
            )
        return result

    return [
        Tool(
            "get_memory",
            "Return the memory with entry ids.",
            {"type": "object", "properties": {}},
            get_memory,
        ),
        Tool(
            "run_code",
            "Re-run a train.py (by run_id, or as code) on seeds 1, 2, 3. Returns mean, spread "
            "and beats_best. Uses one run per seed.",
            {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "A run to re-check."},
                    "code": {"type": "string", "description": "Or a whole train.py."},
                },
            },
            run_code,
        ),
        Tool(
            "upsert_memory",
            "Add an entry, or edit one by passing its id. Refused if the memory would go "
            "over the token cap.",
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["works", "failed", "try_next"]},
                    "claim": {"type": "string", "description": "At most 200 characters."},
                    "run_ids": {"type": "array", "items": {"type": "string"}},
                    "id": {"type": "string", "description": "Entry id to edit (optional)."},
                },
                "required": ["kind", "claim"],
            },
            upsert_memory,
        ),
        Tool(
            "remove_memory",
            "Remove an entry.",
            {
                "type": "object",
                "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["id", "reason"],
            },
            remove_memory,
        ),
    ]


@dataclass
class CuratorOutcome:
    summary: str
    checks: list[Check]
    runs_used: int
    usage: dict[str, int]
    turns: int
    messages: list[dict[str, Any]] = field(repr=False)


async def run_curator(
    *,
    lab: Lab,
    memory: SharedMemory,
    cfg: ResearchConfig,
    system: str,
    task: str,
    gen: int,
    best_version: int,
    best_val: float,
    threshold: float,
) -> CuratorOutcome:
    state = CuratorState(runs_left=cfg.curator_runs)
    tools = curator_tools(
        lab=lab,
        memory=memory,
        state=state,
        cfg=cfg,
        gen=gen,
        best_version=best_version,
        best_val=best_val,
        threshold=threshold,
    )
    with events.scope(agent="curator", role="curator", gen=gen, phase="curation"):
        result = await run_agent(
            model=get_settings().curator_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": task}],
            tools=tools,
            params=CURATOR_PARAMS,
            max_turns=cfg.curator_max_turns,
            public_info={"run_budget": cfg.curator_runs},
        )
    return CuratorOutcome(
        summary=result.final,
        checks=state.checks,
        runs_used=cfg.curator_runs - state.runs_left,
        usage=result.usage,
        turns=result.turns,
        messages=result.messages,
    )
