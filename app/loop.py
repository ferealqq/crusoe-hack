"""One research run: baseline, generations (researchers, ranking, curator, best tracking),
then the final test (README "Generation loop")."""

import asyncio
import logging
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app import scoring
from app.agents import prompts
from app.agents.curator import run_curator
from app.agents.researcher import ResearcherOutcome, run_researcher
from app.config import ResearchConfig
from app.data import Dataset
from app.lab import Best, Lab, Setup, runs_per_generation
from app.schemas import Usage
from app.store import DATA_DIR, append_jsonl

BASELINES = Path(__file__).resolve().parents[1] / "baselines"

log = logging.getLogger(__name__)


def best_claim(lab: Lab) -> str:
    b = lab.best
    gain = scoring.gain(lab.l0, b.val_loss)
    claim = f"train.py v{b.version}: val loss {b.val_loss:.5f} ({gain:+.2%} vs start). {b.note}"
    return claim[:200]


async def run_baseline(lab: Lab) -> None:
    """Starting train.py on seeds 0-4: L0 is the mean, plus the noise threshold."""
    code = (BASELINES / lab.dataset.name / "train.py").read_text()
    exps = await asyncio.gather(
        *(
            lab.run_experiment(
                code,
                prefix="g0-base",
                agent="orchestrator",
                generation=0,
                seed=s,
                note="starting train.py",
            )
            for s in lab.config.baseline_seeds
        )
    )
    losses = [e.loss for e in exps if e.loss is not None]
    if len(losses) != len(exps):
        failed = next(e for e in exps if e.loss is None)
        raise RuntimeError(f"baseline failed: {failed.output_tail[-1000:]}")
    lab.baseline_code = code
    lab.l0 = statistics.mean(losses)
    lab.threshold = scoring.noise_threshold(losses)
    seed_run = next((e for e in exps if e.seed == lab.config.research_seed), exps[0])
    lab.best = Best(0, code, lab.l0, seed_run.run_id, "Starting train.py.", seed_run.loss or lab.l0)
    if lab.setup == "shared_brain":
        lab.memory.set_best(
            claim=best_claim(lab),
            run_ids=[e.run_id for e in exps],
            val_change=None,
            version=0,
            gen=0,
        )
    log.info("baseline: L0 %.5f, noise threshold %.5f", lab.l0, lab.threshold)
    await lab.checkpoint(f"start: L0 {lab.l0:.5f}, noise threshold {lab.threshold:.5f}")


def _focuses(lab: Lab, generation: int) -> list[str]:
    if lab.setup == "one_shot":
        return ["anything: you get one run, so make your single best change"]
    f = lab.config.focuses
    return [f[(generation - 1 + i) % len(f)] for i in range(lab.config.researchers)]


async def run_generation(lab: Lab, generation: int) -> tuple[Usage, int]:
    start = lab.best
    memory_tokens = lab.memory.tokens() if lab.setup == "shared_brain" else 0
    system = prompts.researcher_system(lab)  # identical for every researcher: cache-friendly
    log.info("g%d: %d researchers from v%d", generation, lab.config.researchers, start.version)
    outcomes: list[ResearcherOutcome] = await asyncio.gather(
        *(
            run_researcher(lab, generation=generation, index=i + 1, focus=focus, system=system)
            for i, focus in enumerate(_focuses(lab, generation))
        )
    )
    # Compare against the best's own score on the research seed, so an unchanged copy of the
    # best (or a lucky seed) can't look like a gain.
    ranked = scoring.rank_researchers(
        {o.agent: o.runs for o in outcomes}, start.seed_loss, lab.threshold
    )
    usage = sum((o.usage for o in outcomes), Usage())
    runs = sum(len(o.runs) for o in outcomes)

    if lab.setup == "shared_brain":
        curator = await run_curator(lab, generation=generation, outcomes=outcomes, ranked=ranked)
        usage += curator.usage
        runs += curator.runs_used
        winners = [
            v for v in curator.verifications if v.full and v.beats_best and v.mean is not None
        ]
        if winners:
            w = min(winners, key=lambda v: v.mean or 0.0)
            source = lab.find_run(w.source_run_id) if w.source_run_id else None
            on_seed = source and source.seed == lab.config.research_seed and source.loss
            lab.best = Best(
                start.version + 1,
                w.code,
                w.mean or 0.0,
                w.source_run_id or w.run_ids[0],
                source.note if source else "Curator's code.",
                source.loss if on_seed and source.loss is not None else w.mean or 0.0,
            )
    else:
        candidates = [r for o in outcomes for r in o.runs if r.loss is not None]
        top = min(candidates, key=lambda r: r.loss or 0.0, default=None)
        if top and top.loss is not None and scoring.beats(top.loss, start.seed_loss, lab.threshold):
            lab.best = Best(start.version + 1, top.code, top.loss, top.run_id, top.note, top.loss)

    changed = lab.best.version != start.version
    dropped: list[str] = []
    if lab.setup == "shared_brain":
        if changed:
            lab.memory.set_best(
                claim=best_claim(lab),
                run_ids=[lab.best.run_id],
                val_change=lab.best.val_loss - start.val_loss,
                version=lab.best.version,
                gen=generation,
            )
            await lab.checkpoint(
                f"g{generation} best -> v{lab.best.version}: val {lab.best.val_loss:.5f} "
                f"from {lab.best.run_id}"
            )
        dropped = [e.id for e in lab.memory.enforce_cap()]
        if dropped:
            await lab.checkpoint(
                f"g{generation} cap: dropped {', '.join(dropped)} ({lab.memory.tokens()} tokens)"
            )

    lab.log_generation(
        {
            "generation": generation,
            "start_best": start.val_loss,
            "start_best_seed_loss": start.seed_loss,
            "end_best": lab.best.val_loss,
            "best_version": lab.best.version,
            "best_run_id": lab.best.run_id,
            "best_changed": changed,
            "val_gain": scoring.gain(lab.l0, lab.best.val_loss),
            "memory_tokens": memory_tokens,
            "memory_tokens_after": lab.memory.tokens() if lab.setup == "shared_brain" else 0,
            "dropped": dropped,
            "runs": runs,
            "usage": usage.model_dump(),
            "ranking": [r.__dict__ for r in ranked],
        }
    )
    log.info(
        "g%d done: best v%d val %.5f (%+.2f%%), %d runs",
        generation,
        lab.best.version,
        lab.best.val_loss,
        100 * scoring.gain(lab.l0, lab.best.val_loss),
        runs,
    )
    await lab.checkpoint(
        f"g{generation} done: best v{lab.best.version} val {lab.best.val_loss:.5f} "
        f"({scoring.gain(lab.l0, lab.best.val_loss):+.2%}), {runs} runs"
    )
    return usage, runs


async def final_test(lab: Lab) -> tuple[list[float | None], list[float | None]]:
    """Score the starting and best train.py on the test split, seeds 1-3. Agents never see it."""
    gen = lab.config.generations + 1

    async def score(code: str, version: int) -> list[float | None]:
        exps = await asyncio.gather(
            *(
                lab.run_experiment(
                    code,
                    prefix=f"test-v{version}",
                    agent="orchestrator",
                    generation=gen,
                    seed=s,
                    note=f"test: train.py v{version}",
                    split="test",
                )
                for s in lab.config.test_seeds
            )
        )
        return [e.loss for e in exps]

    start = await score(lab.baseline_code, 0)
    best = start if lab.best.version == 0 else await score(lab.best.code, lab.best.version)
    return start, best


async def research_run(
    dataset: Dataset,
    setup: Setup,
    config: ResearchConfig,
    *,
    root: Path = DATA_DIR,
    commit: bool = False,
) -> dict[str, Any]:
    if setup == "one_shot":  # the floor: one researcher, one generation, one run
        config = config.model_copy(update={"generations": 1, "researchers": 1})
    lab = Lab(dataset, setup, config, root=root, commit=commit)
    started = datetime.now(UTC)
    log.info(
        "research run %s: %d generations, %d runs per generation",
        lab.run_id,
        config.generations,
        runs_per_generation(config, setup),
    )
    await run_baseline(lab)
    usage, runs = Usage(), len(config.baseline_seeds)
    for g in range(1, config.generations + 1):
        u, r = await run_generation(lab, g)
        usage, runs = usage + u, runs + r

    test_start, test_best = await final_test(lab)
    complete = all(x is not None for x in test_start + test_best)
    fitness = (
        scoring.fitness([x or 0.0 for x in test_start], [x or 0.0 for x in test_best])
        if complete
        else None
    )
    row = {
        "run_id": lab.run_id,
        "dataset": dataset.name,
        "setup": setup,
        "config": config.model_dump(),
        "l0": lab.l0,
        "noise_threshold": lab.threshold,
        "best_version": lab.best.version,
        "best_run_id": lab.best.run_id,
        "best_val_loss": lab.best.val_loss,
        "val_gain": scoring.gain(lab.l0, lab.best.val_loss),
        "test_start": test_start,
        "test_best": test_best,
        "fitness": fitness,
        "runs": runs,
        "memory_tokens": lab.memory.tokens() if setup == "shared_brain" else 0,
        "usage": usage.model_dump(),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    append_jsonl(root / "research_runs.jsonl", row)
    done = f"fitness {fitness:+.4f}" if fitness is not None else "fitness unknown (test run failed)"
    log.info("research run %s finished: %s", lab.run_id, done)
    await lab.checkpoint(f"finished: {done}")
    return row
