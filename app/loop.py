"""The generation loop: baseline and noise, researchers, ranking, memory, best tracking."""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import events, scoring
from app.agents import prompts
from app.agents.curator import Check, CuratorOutcome, check_code, run_curator
from app.agents.researcher import ResearcherOutcome, run_researcher
from app.config import ResearchConfig, Setup, get_settings
from app.lab import Dataset, Lab
from app.memory import RawLog, SharedMemory, count_tokens
from app.store import Store

log = logging.getLogger(__name__)


@dataclass
class Baseline:
    code: str
    val_losses: list[float]  # baseline seeds (0-4)
    l0: float
    threshold: float
    test_losses: list[float]  # test seeds (1-3)


@dataclass
class Best:
    code: str
    val: float
    version: int
    run_id: str | None
    note: str


async def measure_baseline(
    dataset: Dataset, code: str, cfg: ResearchConfig, store: Store
) -> Baseline:
    """Run the starting train.py on the baseline seeds: L0 and the noise threshold."""
    lab = Lab(dataset, store)
    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(lab.run(code, s, "orchestrator", "baseline", 0, 0))
            for s in cfg.baseline_seeds
        ]
    exps = [task.result() for task in tasks]
    losses = [e.val_loss for e in exps]
    if any(x is None for x in losses):
        raise RuntimeError(f"baseline failed: {[e.output_tail[-500:] for e in exps]}")
    val = [x for x in losses if x is not None]
    test = await dataset.test_losses(code, cfg.test_seeds)
    if any(x is None for x in test):
        raise RuntimeError("baseline failed on the test split")
    l0 = sum(val) / len(val)
    return Baseline(code, val, l0, scoring.noise_threshold(val), [x for x in test if x is not None])


def rank(losses: list[float | None], threshold: float) -> list[tuple[int, int]]:
    """(index, rank) ordered best first. Lower loss is better; gaps within the noise
    threshold are ties (same rank). Researchers with no successful run rank last."""

    def better(j: int, i: int) -> bool:
        lj, li = losses[j], losses[i]
        if lj is None:
            return False
        return li is None or lj < li - threshold

    order = sorted(range(len(losses)), key=lambda i: (losses[i] is None, losses[i] or 0.0, i))
    return [(i, 1 + sum(better(j, i) for j in range(len(losses)))) for i in order]


def ranking_text(
    outcomes: list[ResearcherOutcome], ranked: list[tuple[int, int]], start_val: float
) -> str:
    lines = []
    for pos, (i, r) in enumerate(ranked, 1):
        o = outcomes[i]
        b = o.best_run()
        if b is None:
            lines.append(f"{pos}. {o.name} ({o.focus}): no successful run. Rank {r}.")
            continue
        g = scoring.gain(start_val, b["val_loss"])
        lines.append(
            f"{pos}. {o.name} ({o.focus}): best run {b['run_id']}, val {b['val_loss']:.5f}, "
            f"gain {g:+.2%}. Rank {r}."
        )
    return "\n".join(lines)


def _sum_usage(usages: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for u in usages:
        for k, v in u.items():
            total[k] = total.get(k, 0) + v
    return total


class SetupRun:
    """One research run: one dataset, one setup, `generations` generations."""

    def __init__(
        self,
        setup: Setup,
        dataset: Dataset,
        baseline: Baseline,
        cfg: ResearchConfig,
        root: Path,
    ) -> None:
        self.setup: Setup = setup
        self.dataset = dataset
        self.baseline = baseline
        self.cfg = cfg
        self.store = Store(root / setup)
        self.lab = Lab(dataset, self.store)
        self.best = Best(baseline.code, baseline.l0, 0, None, "starting train.py")
        self.memory = SharedMemory(cfg.memory_cap_tokens) if setup == "shared_brain" else None
        self.raw_log = RawLog(cfg.raw_log_cap_tokens) if setup == "raw_log" else None
        self._events_saved = 0

    def memory_text(self) -> str | None:
        if self.memory is not None:
            return self.memory.render(self.best.version)
        if self.raw_log is not None:
            return self.raw_log.render()
        return None

    async def run(self) -> dict[str, Any]:
        with events.scope(setup=self.setup):
            events.emit("setup.started", generations=self.cfg.generations_for(self.setup))
            try:
                result = await self._run()
            except asyncio.CancelledError:
                events.emit("setup.cancelled")
                raise
            except Exception as e:
                events.emit("setup.failed", error_type=type(e).__name__)
                raise
            events.emit("setup.completed", **{k: v for k, v in result.items() if k != "best_code"})
            return result

    async def _run(self) -> dict[str, Any]:
        started = time.time()
        gens = self.cfg.generations_for(self.setup)
        self._snapshot(0, None)
        for gen in range(1, gens + 1):
            await self.generation(gen)
        test = await self.dataset.test_losses(self.best.code, self.cfg.test_seeds)
        fitness = self._fitness(test)
        result = {
            "setup": self.setup,
            "dataset": self.dataset.meta["name"],
            "l0": self.baseline.l0,
            "noise_threshold": self.baseline.threshold,
            "best_version": self.best.version,
            "best_val": self.best.val,
            "val_gain": scoring.gain(self.baseline.l0, self.best.val),
            "baseline_test_losses": self.baseline.test_losses,
            "best_test_losses": test,
            "fitness": fitness,
            "runs": len(self.lab.experiments),
            "minutes": round((time.time() - started) / 60, 1),
            "best_code": self.best.code,
        }
        self.store.write_json("result", result)
        log.info("[%s] done: fitness %s", self.setup, fitness)
        return result

    def _fitness(self, test: list[float | None]) -> float | None:
        if any(x is None for x in test):
            return None
        return scoring.fitness(self.baseline.test_losses, [x for x in test if x is not None])

    async def generation(self, gen: int) -> None:
        with events.scope(gen=gen):
            events.emit(
                "generation.started",
                best_val=self.best.val,
                best_version=self.best.version,
                researchers=self.cfg.researchers_for(self.setup),
            )
            await self._generation(gen)

    async def _generation(self, gen: int) -> None:
        cfg, setup, start = self.cfg, self.setup, self.best
        t0 = time.time()
        n = cfg.researchers_for(setup)
        runs = cfg.runs_per_researcher(setup)
        memory = self.memory_text()
        memory_title = "Team memory (curated)" if self.memory else "Team log (every report)"
        system = prompts.researcher_system(
            card=self.dataset.card(),
            loss=self.dataset.loss,
            memory=memory,
            memory_title=memory_title,
            best_code=start.code,
            best_val=start.val,
            best_version=start.version,
            runs=runs,
            max_turns=cfg.max_turns,
            timeout_s=get_settings().sandbox_timeout_seconds,
            web_search=bool(get_settings().ydc_api_key),
        )
        focuses = (
            ["any"] if setup == "one_shot" else [prompts.focus_for(gen, i, n) for i in range(n)]
        )
        log.info(
            "[%s] gen %d: %d researchers, best v%d val %.5f",
            setup,
            gen,
            n,
            start.version,
            start.val,
        )
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self._researcher(gen, i, f, system, runs, memory, start))
                for i, f in enumerate(focuses)
            ]
        outcomes = [task.result() for task in tasks]
        losses = [b["val_loss"] if (b := o.best_run()) else None for o in outcomes]
        ranked = rank(losses, self.baseline.threshold)

        curator: CuratorOutcome | None = None
        if self.memory is not None:
            curator = await self._curate(gen, outcomes, ranked, start)
        else:
            self._pick_best_by_val(gen, outcomes)
            if self.raw_log is not None:
                for o in outcomes:
                    self.raw_log.append(f"Generation {gen}, {o.name} ({o.focus}):\n{o.report}")

        test = await self.dataset.test_losses(self.best.code, cfg.test_seeds)
        gen_exps = [e for e in self.lab.experiments.values() if e.gen == gen]
        metrics = {
            "gen": gen,
            "start_best_val": start.val,
            "end_best_val": self.best.val,
            "best_version": self.best.version,
            "new_best": self.best.version != start.version,
            "best_run_id": self.best.run_id,
            "best_note": self.best.note,
            "val_gain": scoring.gain(self.baseline.l0, self.best.val),
            "test_losses": test,
            "test_gain": self._fitness(test),
            "memory_tokens": count_tokens(memory or ""),
            "runs": len(gen_exps),
            "errors": sum(e.status == "error" for e in gen_exps),
            "timeouts": sum(e.status == "timeout" for e in gen_exps),
            "researcher_best": [
                {"name": o.name, "focus": o.focus, "best": o.best_run(), "rank": r}
                for (i, r) in ranked
                for o in [outcomes[i]]
            ],
            "usage": {
                "researchers": _sum_usage([o.usage for o in outcomes]),
                "curator": curator.usage if curator else {},
            },
            "minutes": round((time.time() - t0) / 60, 1),
        }
        self.store.append("generations", metrics)
        events.emit("generation.completed", **metrics)
        for o in outcomes:
            self.store.append(
                "agents",
                {
                    "gen": gen,
                    "role": "researcher",
                    "name": o.name,
                    "focus": o.focus,
                    "model": get_settings().researcher_model,
                    "report": o.report,
                    "run_log": o.run_log,
                    "usage": o.usage,
                    "turns": o.turns,
                    "messages": o.messages,
                },
            )
        if curator:
            self.store.append(
                "agents",
                {
                    "gen": gen,
                    "role": "curator",
                    "name": "curator",
                    "model": get_settings().curator_model,
                    "report": curator.summary,
                    "checks": [c.__dict__ | {"code": None} for c in curator.checks],
                    "usage": curator.usage,
                    "turns": curator.turns,
                    "messages": curator.messages,
                },
            )
        self._snapshot(gen, curator)
        log.info(
            "[%s] gen %d done in %.1f min: best v%d val %.5f, test gain %s",
            setup, gen, (time.time() - t0) / 60, self.best.version, self.best.val,
            self._fitness(test),
        )  # fmt: skip

    async def _researcher(
        self, gen: int, i: int, focus: str, system: str, runs: int, memory: str | None, start: Best
    ) -> ResearcherOutcome:
        with events.scope(phase="candidate"):
            return await self._run_researcher(gen, i, focus, system, runs, memory, start)

    async def _run_researcher(
        self, gen: int, i: int, focus: str, system: str, runs: int, memory: str | None, start: Best
    ) -> ResearcherOutcome:
        description = prompts.ONE_SHOT_FOCUS if focus == "any" else prompts.FOCUSES[focus]
        return await run_researcher(
            lab=self.lab,
            name=f"researcher-{i + 1}",
            focus=focus,
            system=system,
            task=prompts.researcher_task(focus, description),
            gen=gen,
            best_version=start.version,
            runs=runs,
            memory=memory,
            cfg=self.cfg,
        )

    def _pick_best_by_val(self, gen: int, outcomes: list[ResearcherOutcome]) -> None:
        """No memory, Raw log, One-shot: the orchestrator takes the lowest val loss."""
        runs = [r for o in outcomes for r in o.run_log if r["val_loss"] is not None]
        if not runs:
            return
        top = min(runs, key=lambda r: r["val_loss"])
        if scoring.beats(top["val_loss"], self.best.val, self.baseline.threshold):
            exp = self.lab.experiments[top["run_id"]]
            self.best = Best(exp.code, top["val_loss"], self.best.version + 1, exp.id, exp.note)
            self._emit_best(verified=False)

    def _emit_best(self, *, verified: bool) -> None:
        events.emit(
            "best.updated",
            val_loss=self.best.val,
            version=self.best.version,
            run_id=self.best.run_id,
            note=self.best.note,
            verified=verified,
            val_gain=scoring.gain(self.baseline.l0, self.best.val),
        )

    async def _curate(
        self,
        gen: int,
        outcomes: list[ResearcherOutcome],
        ranked: list[tuple[int, int]],
        start: Best,
    ) -> CuratorOutcome:
        assert self.memory is not None
        cfg, threshold = self.cfg, self.baseline.threshold
        reports = []
        for pos, (i, r) in enumerate(ranked, 1):
            o = outcomes[i]
            reports.append(
                f"## {pos}. {o.name} (focus: {o.focus}, rank {r})\n"
                f"Run log:\n{o.run_log_text()}\n\nReport:\n{o.report}"
            )
        top_run = outcomes[ranked[0][0]].best_run() if ranked else None
        top_code = self.lab.experiments[top_run["run_id"]].code if top_run else None
        task = prompts.curator_task(
            gen=gen,
            card=self.dataset.card(),
            loss=self.dataset.loss,
            memory=self.memory.render(start.version),
            best_version=start.version,
            best_val=start.val,
            threshold=threshold,
            stale_ids=[e.id for e in self.memory.stale(start.version)],
            ranking=ranking_text(outcomes, ranked, start.val),
            reports="\n\n".join(reports),
            top_code=top_code,
        )
        system = prompts.CURATOR_SYSTEM.format(
            cap=cfg.memory_cap_tokens,
            runs=cfg.curator_runs,
            seeds=", ".join(map(str, cfg.curator_seeds)),
        )
        try:
            curator = await run_curator(
                lab=self.lab,
                memory=self.memory,
                cfg=cfg,
                system=system,
                task=task,
                gen=gen,
                best_version=start.version,
                best_val=start.val,
                threshold=threshold,
            )
        except Exception as e:
            log.exception("[%s] curator failed in gen %d", self.setup, gen)
            curator = CuratorOutcome(f"(curator failed: {e})", [], 0, {}, 0, [])

        # Fallback: if the curator never did a full re-check, the orchestrator re-checks the
        # top run itself, so a new best is never missed because of a tool-use slip.
        full = len(cfg.curator_seeds)
        runs_left = cfg.curator_runs - curator.runs_used
        if top_run and not any(len(c.seeds) == full for c in curator.checks) and runs_left >= full:
            log.info(
                "[%s] gen %d: curator skipped the re-check; orchestrator runs it", self.setup, gen
            )
            check = await check_code(
                lab=self.lab,
                code=self.lab.experiments[top_run["run_id"]].code,
                source=top_run["run_id"],
                seeds=cfg.curator_seeds,
                gen=gen,
                best_version=start.version,
                best_val=start.val,
                threshold=threshold,
                by="orchestrator",
            )
            curator.checks.append(check)
            curator.runs_used += full

        winners = [c for c in curator.checks if c.beats_best and len(c.seeds) == full]
        if winners:
            win: Check = min(winners, key=lambda c: c.mean or 0.0)
            assert win.mean is not None
            src = self.lab.experiments.get(win.source)
            note = src.note if src else "curator code"
            self.best = Best(win.code, win.mean, start.version + 1, win.source, note)
            self._emit_best(verified=True)
            self.memory.set_best(
                f"v{self.best.version}: val {self.dataset.loss} {win.mean:.5f} (mean of seeds "
                f"{', '.join(map(str, win.seeds))}), from run {win.source}: {note}"[:300],
                [win.source, *win.run_ids],
                self.best.version,
                gen,
                win.mean,
            )
        dropped = self.memory.enforce_cap(gen, self.best.version)
        events.emit(
            "memory.updated",
            action="curation_completed",
            text=self.memory.render(self.best.version),
            tokens=self.memory.tokens(self.best.version),
            best_version=self.best.version,
            dropped_ids=dropped,
        )
        return curator

    def _snapshot(self, gen: int, curator: CuratorOutcome | None) -> None:
        text = self.memory_text() or ""
        with events.scope(gen=gen):
            events.emit(
                "memory.updated",
                action="snapshot",
                text=text,
                tokens=count_tokens(text),
                best_version=self.best.version,
            )
        memory_events: list[dict[str, Any]] = []
        if self.memory is not None:
            memory_events = self.memory.events[self._events_saved :]
            self._events_saved = len(self.memory.events)
        self.store.append(
            "memory_snapshots",
            {
                "gen": gen,
                "best_version": self.best.version,
                "tokens": count_tokens(text),
                "text": text,
                "events": memory_events,
                "curator_summary": curator.summary if curator else None,
            },
        )
