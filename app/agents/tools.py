"""Tools the agents call (README "Tools"). Budgets are enforced here, not by the prompts."""

import asyncio
import statistics
from dataclasses import dataclass
from typing import Any

from app import scoring
from app.agents.base import Tool
from app.lab import Lab
from app.memory import MemoryFull
from app.schemas import Experiment

CODE_PARAM = {"type": "string", "description": "The complete train.py"}


class Budget:
    def __init__(self, runs: int) -> None:
        self.left = runs

    def take(self, n: int = 1) -> bool:
        if n > self.left:
            return False
        self.left -= n
        return True


@dataclass(frozen=True)
class Verification:
    """One curator re-check. Only full (all verify seeds) checks can set a new best."""

    source_run_id: str | None
    code: str
    run_ids: list[str]
    losses: list[float | None]
    mean: float | None
    full: bool
    beats_best: bool


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 5)


def get_memory(lab: Lab) -> Tool:
    async def fn(_: dict[str, Any]) -> dict[str, Any]:
        return {"memory": lab.memory.render(), "tokens": lab.memory.tokens(), "cap": lab.memory.cap}

    return Tool("get_memory", "Read the team memory.", {"type": "object", "properties": {}}, fn)


def researcher_run_code(
    lab: Lab, *, agent: str, generation: int, budget: Budget, runs: list[Experiment]
) -> Tool:
    async def fn(args: dict[str, Any]) -> dict[str, Any]:
        code, note = args.get("code"), str(args.get("note", ""))[:200]
        if not isinstance(code, str) or not code.strip():
            return {"error": "code is required: the complete train.py"}
        if not budget.take():
            return {"error": "no runs left; write your report now", "runs_left": 0}
        exp = await lab.run_experiment(
            code,
            prefix=f"g{generation}-{agent}",
            agent=agent,
            generation=generation,
            seed=lab.config.research_seed,
            note=note,
        )
        runs.append(exp)
        return {
            "run_id": exp.run_id,
            "status": exp.status,
            "val_loss": _round(exp.loss),
            "output": exp.output_tail,
            "runs_left": budget.left,
        }

    return Tool(
        "run_code",
        "Run your complete train.py: fit on the training split, predict the validation inputs, "
        "and get the validation log loss (lower is better). Uses one run.",
        {
            "type": "object",
            "properties": {
                "code": CODE_PARAM,
                "note": {"type": "string", "description": "One line: what changed"},
            },
            "required": ["code", "note"],
        },
        fn,
    )


def curator_run_code(
    lab: Lab, *, generation: int, budget: Budget, verifications: list[Verification]
) -> Tool:
    all_seeds = lab.config.verify_seeds

    async def fn(args: dict[str, Any]) -> dict[str, Any]:
        run_id, code = args.get("run_id"), args.get("code")
        n = int(args.get("seeds", len(all_seeds)))
        if n not in (1, len(all_seeds)):
            return {"error": f"seeds must be 1 or {len(all_seeds)}"}
        if run_id:
            source = lab.find_run(str(run_id))
            if source is None:
                return {"error": f"no run {run_id} in this research run"}
            code = source.code
        if not isinstance(code, str) or not code.strip():
            return {"error": "give run_id (to re-check a run) or code"}
        if not budget.take(n):
            return {"error": f"this check needs {n} runs; {budget.left} left"}
        seeds = all_seeds[:n]
        exps = await asyncio.gather(
            *(
                lab.run_experiment(
                    code,
                    prefix=f"g{generation}-curator",
                    agent="curator",
                    generation=generation,
                    seed=s,
                    note=f"verify {run_id or 'code'} (seed {s})",
                )
                for s in seeds
            )
        )
        losses = [e.loss for e in exps]
        ok = [x for x in losses if x is not None]
        mean = statistics.mean(ok) if ok and len(ok) == len(losses) else None
        beats = mean is not None and scoring.beats(mean, lab.best.val_loss, lab.threshold)
        verifications.append(
            Verification(
                source_run_id=str(run_id) if run_id else None,
                code=code,
                run_ids=[e.run_id for e in exps],
                losses=losses,
                mean=mean,
                full=n == len(all_seeds),
                beats_best=beats,
            )
        )
        return {
            "run_ids": [e.run_id for e in exps],
            "losses": [_round(x) for x in losses],
            "mean": _round(mean),
            "spread": _round(scoring.spread(ok)) if mean is not None else None,
            "best_val_loss": _round(lab.best.val_loss),
            "noise_threshold": _round(lab.threshold),
            "beats_best": beats,
            "runs_left": budget.left,
            "errors": [e.output_tail[-500:] for e in exps if e.status != "ok"],
        }

    return Tool(
        "run_code",
        "Re-check a run (run_id) or new code on validation seeds. seeds=3 (default) uses 3 runs "
        "and returns mean, spread and beats_best; seeds=1 uses 1 run.",
        {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "A run to re-check"},
                "code": CODE_PARAM,
                "seeds": {"type": "integer", "enum": [1, len(all_seeds)]},
            },
        },
        fn,
    )


def upsert_memory(lab: Lab, *, generation: int) -> Tool:
    async def fn(args: dict[str, Any]) -> dict[str, Any]:
        kind, claim = args.get("kind"), str(args.get("claim", "")).strip()
        run_ids, entry_id = args.get("run_ids") or [], args.get("id")
        if kind not in ("works", "failed", "try_next"):
            return {"error": "kind must be works, failed or try_next"}
        runs = [lab.find_run(str(r)) for r in run_ids]
        if not runs or any(r is None for r in runs):
            return {
                "error": "run_ids must name runs from this research run; only results "
                "from run_code become memories"
            }
        ok = [r.loss for r in runs if r is not None and r.loss is not None]
        try:
            entry = lab.memory.upsert(
                kind=kind,
                claim=claim,
                run_ids=[str(r) for r in run_ids],
                val_change=statistics.mean(ok) - lab.best.val_loss if ok else None,
                seeds=sorted({r.seed for r in runs if r is not None}),
                tested_on=lab.best.version,
                generation=generation,
                entry_id=str(entry_id) if entry_id else None,
            )
        except MemoryFull as e:
            return {
                "error": f"refused: {e}. Merge or remove entries first.",
                "tokens": lab.memory.tokens(),
                "cap": lab.memory.cap,
            }
        except ValueError as e:
            return {"error": str(e)}
        verb = "edit" if entry_id else "add"
        await lab.checkpoint(f"g{generation} curator {verb} {entry.id} ({kind}): {claim}")
        return {"id": entry.id, "tokens": lab.memory.tokens(), "cap": lab.memory.cap}

    return Tool(
        "upsert_memory",
        "Add a memory entry, or edit one by passing its id. Refused if memory would go over the "
        "token cap.",
        {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["works", "failed", "try_next"]},
                "claim": {"type": "string", "description": "One testable sentence, max 200 chars"},
                "run_ids": {"type": "array", "items": {"type": "string"}},
                "id": {"type": "string", "description": "Entry to edit (omit to add)"},
            },
            "required": ["kind", "claim", "run_ids"],
        },
        fn,
    )


def remove_memory(lab: Lab, *, generation: int) -> Tool:
    async def fn(args: dict[str, Any]) -> dict[str, Any]:
        entry_id = str(args.get("id", ""))
        entry = lab.memory.get(entry_id)
        try:
            lab.memory.remove(entry_id)
        except ValueError as e:
            return {"error": str(e)}
        assert entry is not None
        await lab.checkpoint(
            f"g{generation} curator remove {entry_id} ({entry.kind}): {entry.claim}"
        )
        return {"ok": True, "tokens": lab.memory.tokens(), "cap": lab.memory.cap}

    return Tool(
        "remove_memory",
        "Remove a memory entry.",
        {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
        fn,
    )
