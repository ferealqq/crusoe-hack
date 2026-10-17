"""Generate recorded demo runs (one per setup) into app/runlog/samples/.

    uv run python scripts/make_sample_runs.py

Deterministic and synthetic, but shaped like a real Shared Brain run so the frontend can be
built and demoed before the orchestrator exists. Every event is validated against
app/runlog/events.py, so the samples double as a contract test.
"""

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.runlog.events import (  # noqa: E402
    AgentMessage,
    AgentReported,
    AgentSpawned,
    BestSoFar,
    CuratorRanked,
    CuratorVerified,
    EvalFinished,
    GenFinished,
    GenStarted,
    LlmUsage,
    MemoryOp,
    MemoryUpdated,
    RankEntry,
    ResearcherSpec,
    RunConfig,
    RunFinished,
    RunStarted,
    SeedScore,
    ToolCalled,
    ToolResult,
    event_adapter,
)

OUT = Path(__file__).resolve().parent.parent / "app/runlog/samples"
RESEARCHER_MODEL = "nvidia/Nemotron-3-Super-120B-A12B"
CURATOR_MODEL = "nvidia/Nemotron-3-Ultra-550B"
FOCUSES = ["optimizer", "architecture", "regularization", "data"]
L0 = 1.712
SEED_NOISE = 0.006
MEMORY_CAP = 1600

# (idea, effect on val loss; negative = better). Effects are what a run would show.
IDEAS = {
    "optimizer": [
        ("raise AdamW lr 1e-3 → 3e-3 with 100-step warmup", -0.041),
        ("cosine decay to 10% of peak lr instead of constant", -0.022),
        ("AdamW beta2 0.999 → 0.95", -0.011),
        ("switch to Lion at lr 3e-4", +0.019),
        ("gradient clipping at 0.5 instead of 1.0", 0.002),
    ],
    "architecture": [
        ("rotary position embeddings instead of learned", -0.027),
        ("n_layer 6 → 8 at the same width", -0.024),
        ("RMSNorm instead of LayerNorm", -0.003),
        ("12 heads of dim 32 instead of 6 of dim 64", +0.014),
        ("SwiGLU MLP at matched params", -0.015),
    ],
    "regularization": [
        ("dropout 0.2 → 0.05 (short budget underfits)", -0.031),
        ("weight decay 0.1 on matrices only, none on norms/biases", -0.012),
        ("label smoothing 0.1", +0.028),
        ("stochastic depth 0.1", +0.009),
        ("dropout 0.05 → 0.0", -0.004),
    ],
    "data": [
        ("block_size 256 → 384, fewer steps per token budget", -0.018),
        ("random sequence start offsets every epoch", -0.009),
        ("batch 64 → 128 with lr scaled by √2", -0.013),
        ("dedupe repeated lines in the train split", -0.002),
        ("byte-level instead of char-level vocab", +0.035),
    ],
}

SEARCHES = {
    "optimizer": "warmup schedules for small transformer LMs short budget",
    "architecture": "rotary embeddings vs learned positions small char-level models",
    "regularization": "dropout for undertrained transformers fixed compute",
    "data": "context length vs steps tradeoff fixed token budget",
}

SETUP_PROFILE = {
    # skill: how much of an idea's effect the agent captures; waste: chance a run repeats a
    # known dead end; cache: KV-cache hit rate of prompts; memory: how memory is kept
    "shared_brain": dict(skill=1.0, waste=0.05, cache=0.81, memory="curated"),
    "raw_log": dict(skill=0.8, waste=0.18, cache=0.62, memory="raw"),
    "no_memory": dict(skill=0.65, waste=0.4, cache=0.21, memory="none"),
    "one_shot": dict(skill=0.6, waste=0.0, cache=0.15, memory="none"),
}

FITNESS = {"shared_brain": 0.118, "raw_log": 0.079, "no_memory": 0.026, "one_shot": 0.011}


@dataclass
class Memory:
    entries: dict[str, MemoryOp] = field(default_factory=dict)
    version: int = 0
    next_id: int = 1

    def tokens(self) -> int:
        return 120 + sum(len(e.text.split()) * 2 + 18 for e in self.entries.values())

    def new_id(self) -> str:
        self.next_id += 1
        return f"m{self.next_id - 1}"


class RunWriter:
    def __init__(self, run_id: str, rng: random.Random):
        self.run_id, self.rng, self.events, self.calls = run_id, rng, [], 0

    def add(self, ts: float, event) -> None:
        event.ts = round(ts, 2)
        self.events.append(event)

    def call_id(self) -> str:
        self.calls += 1
        return f"c{self.calls}"

    def write(self) -> Path:
        self.events.sort(key=lambda e: e.ts)
        path = OUT / f"{self.run_id}.jsonl"
        with path.open("w") as f:
            for seq, e in enumerate(self.events):
                e.seq = seq
                event_adapter.validate_python(e.model_dump())  # contract check
                f.write(event_adapter.dump_json(e, exclude_none=True).decode() + "\n")
        return path


def usage(w: RunWriter, ts: float, agent_id: str, prompt: int, cache: float) -> None:
    cached = int(prompt * min(0.97, max(0.0, w.rng.gauss(cache, 0.04))))
    ttft = int(180 + (prompt - cached) * 0.045 + w.rng.uniform(-30, 30))
    w.add(
        ts,
        LlmUsage(
            agent_id=agent_id,
            prompt_tokens=prompt,
            cached_tokens=cached,
            completion_tokens=w.rng.randint(180, 900),
            ttft_ms=ttft,
        ),
    )


def train_diff(idea: str) -> str:
    return (
        "--- a/train.py\n+++ b/train.py\n@@\n"
        f"-# baseline\n+# {idea}\n"
        "  model = GPT(config)\n  ...\n"
        "run_code: 600 steps (budget), eval on val every 100 steps"
    )


def researcher(
    w: RunWriter,
    t0: float,
    gen: int,
    focus: str,
    start_best: float,
    profile: dict,
    memory: Memory,
    known_bad: set[str],
    avoid: set[str],
    max_runs: int = 3,
) -> tuple[float, float, str, list[str], int, int]:
    """Emit one researcher's lifetime. Returns (end_ts, best_loss, best_idea, tried, runs, waste)."""
    rng = w.rng
    aid = f"g{gen}-{focus}"
    t = t0 + rng.uniform(0, 4)
    w.add(
        t,
        AgentSpawned(gen=gen, agent_id=aid, role="researcher", focus=focus, model=RESEARCHER_MODEL),
    )
    mem_tokens = memory.tokens() if profile["memory"] != "none" else 0
    base_prompt = 2400 + mem_tokens

    if profile["memory"] != "none":
        cid = w.call_id()
        t += rng.uniform(1, 3)
        w.add(t, ToolCalled(agent_id=aid, call_id=cid, tool="get_memory", summary="get_memory()"))
        t += 0.4
        n = len(memory.entries)
        w.add(
            t,
            ToolResult(
                agent_id=aid,
                call_id=cid,
                tool="get_memory",
                summary=f"memory v{memory.version}: {n} entries, {mem_tokens} tokens",
                duration_ms=380,
            ),
        )
    usage(w, t + 1, aid, base_prompt, profile["cache"])
    t += rng.uniform(4, 9)
    w.add(
        t,
        AgentMessage(
            agent_id=aid,
            text=(
                f"Plan: focus on {focus}. "
                + (
                    "Memory lists what already worked and what failed; skipping known dead ends. "
                    if mem_tokens
                    else ""
                )
                + "Try the most promising untested change first."
            ),
        ),
    )

    if rng.random() < 0.6:
        cid = w.call_id()
        w.add(
            t,
            ToolCalled(
                agent_id=aid,
                call_id=cid,
                tool="web_search",
                summary=f'web_search("{SEARCHES[focus]}")',
            ),
        )
        t += rng.uniform(2, 5)
        w.add(
            t,
            ToolResult(
                agent_id=aid,
                call_id=cid,
                tool="web_search",
                summary="5 results (treated as hypotheses)",
                duration_ms=2100,
            ),
        )
        usage(w, t + 0.5, aid, base_prompt + 1800, profile["cache"])

    pool = IDEAS[focus][:]
    rng.shuffle(pool)
    pool.sort(key=lambda i: i[0] in avoid)  # dead ends the agent knows about go last
    unknown_bad = [i for i in IDEAS[focus] if i[0] in known_bad and i[0] not in avoid]
    best, best_idea, tried, waste = start_best, "", [], 0
    for run in range(max_runs):
        idea, effect = pool[run % len(pool)]
        if unknown_bad and rng.random() < profile["waste"]:
            idea, effect = rng.choice(unknown_bad)  # unknowingly repeats a known dead end
        waste += int(idea in known_bad)
        tried.append(idea)
        cid = w.call_id()
        t += rng.uniform(3, 8)
        w.add(
            t, ToolCalled(agent_id=aid, call_id=cid, tool="run_code", summary=f"run_code: {idea}")
        )
        dur = rng.uniform(70, 110)
        t += dur
        loss = round(
            best + effect * profile["skill"] * rng.uniform(0.7, 1.15) + rng.gauss(0, SEED_NOISE), 4
        )
        w.add(
            t,
            ToolResult(
                agent_id=aid,
                call_id=cid,
                tool="run_code",
                summary=f"val loss {loss:.4f} ({loss - best:+.4f})",
                val_loss=loss,
                duration_ms=int(dur * 1000),
                detail=train_diff(idea),
            ),
        )
        usage(w, t + 0.3, aid, base_prompt + 900 * (run + 1), profile["cache"])
        if loss < best:
            best, best_idea = loss, idea
    t += rng.uniform(3, 6)
    report = (
        f"Best: {best_idea or 'no improvement'} → val {best:.4f} ({best - start_best:+.4f} vs start {start_best:.4f}). "
        f"Tried: {'; '.join(tried)}."
    )
    w.add(t, AgentReported(gen=gen, agent_id=aid, best_val_loss=best, report=report))
    return t, best, best_idea, tried, len(tried), waste


def make_run(setup: str, seed: int) -> Path:
    rng = random.Random(seed)
    profile = SETUP_PROFILE[setup]
    run_id = f"sample-{setup.replace('_', '-')}"
    w = RunWriter(run_id, rng)
    gens = 1 if setup == "one_shot" else 5
    focuses = ["optimizer"] if setup == "one_shot" else FOCUSES
    config = RunConfig(
        generations=gens,
        researchers=[ResearcherSpec(focus=f, model=RESEARCHER_MODEL) for f in focuses],
        curator_model=CURATOR_MODEL,
        budget="600 steps per run_code (~90 s)",
        seeds=3,
        memory_cap_tokens=MEMORY_CAP,
        dataset="tinyshakespeare-char (train/val/test)",
    )
    w.add(0, RunStarted(run_id=run_id, setup=setup, config=config, baseline_val_loss=L0))

    memory, dead_ends, known_bad, raw_tokens = Memory(), set(), set(), 0
    works_by_idea: dict[str, str] = {}
    tested: set[str] = set()
    best_overall, best_summary, t = L0, "baseline train.py", 1.0
    for gen in range(1, gens + 1):
        start = best_overall if profile["memory"] != "none" else L0
        w.add(t, GenStarted(gen=gen, start_best_val_loss=start))
        results, ends = [], []
        if profile["memory"] == "curated":
            avoid = dead_ends
        elif profile["memory"] == "raw":  # it's in the log somewhere, but often missed
            avoid = {i for i in known_bad if rng.random() < 0.5}
        else:
            avoid = set()
        for focus in focuses:
            end, best, idea, tried, runs, waste = researcher(
                w,
                t,
                gen,
                focus,
                start,
                profile,
                memory,
                known_bad,
                avoid,
                max_runs=1 if setup == "one_shot" else 3,
            )
            results.append(
                dict(focus=focus, best=best, idea=idea, tried=tried, runs=runs, waste=waste)
            )
            ends.append(end)
        t = max(ends) + 2
        for r in results:
            known_bad |= {i for i in r["tried"] if dict(IDEAS[r["focus"]])[i] > 0}
            raw_tokens += 380 + 60 * r["runs"]
            if r["best"] < best_overall:
                best_overall, best_summary = r["best"], r["idea"]

        ops: list[MemoryOp] = []
        ranked = sorted(results, key=lambda r: r["best"])
        if profile["memory"] == "curated":
            cur = f"g{gen}-curator"
            w.add(t, AgentSpawned(gen=gen, agent_id=cur, role="curator", model=CURATOR_MODEL))
            usage(w, t + 1, cur, 3200 + memory.tokens() + 1600 * len(results), profile["cache"])
            t += rng.uniform(5, 9)
            ranking = []
            for i, r in enumerate(ranked):
                gain = (start - r["best"]) / start
                bucket = "top" if i < 2 else ("middle" if i < 3 else "bottom")
                tie = i + 1 < len(ranked) and abs(r["best"] - ranked[i + 1]["best"]) < SEED_NOISE
                ranking.append(
                    RankEntry(
                        agent_id=f"g{gen}-{r['focus']}",
                        best_val_loss=r["best"],
                        gain=round(gain, 4),
                        bucket=bucket,
                        tie=tie,
                    )
                )
            w.add(t, CuratorRanked(gen=gen, ranking=ranking))
            for entry, r in zip(ranking, ranked, strict=True):
                if entry.bucket == "top" and r["idea"]:
                    cid = w.call_id()
                    t += 2
                    w.add(
                        t,
                        ToolCalled(
                            agent_id=cur,
                            call_id=cid,
                            tool="run_code",
                            summary=f"verify: {r['idea']}",
                        ),
                    )
                    dur = rng.uniform(70, 100)
                    t += dur
                    rerun = round(r["best"] + rng.gauss(0.002, SEED_NOISE), 4)
                    held = rerun < start - SEED_NOISE
                    w.add(
                        t,
                        ToolResult(
                            agent_id=cur,
                            call_id=cid,
                            tool="run_code",
                            summary=f"re-run val {rerun:.4f}",
                            val_loss=rerun,
                            duration_ms=int(dur * 1000),
                            detail=train_diff(r["idea"]),
                        ),
                    )
                    w.add(
                        t + 0.5,
                        CuratorVerified(
                            gen=gen,
                            agent_id=entry.agent_id,
                            claim=r["idea"],
                            claimed_val_loss=r["best"],
                            rerun_val_loss=rerun,
                            held=held,
                        ),
                    )
                    if held:
                        # one entry per idea: re-confirming an idea edits its entry, it never duplicates
                        existing = works_by_idea.get(r["idea"])
                        ops.append(
                            MemoryOp(
                                op="edit" if existing else "add",
                                section="works",
                                entry_id=existing or memory.new_id(),
                                text=f"{r['idea']}: val {rerun:.4f} ({rerun - start:+.4f})",
                                reason="re-confirmed on the new best config"
                                if existing
                                else "re-run confirmed the claim",
                                proof=f"{entry.agent_id} run_code, curator re-run {rerun:.4f}",
                            )
                        )
                        works_by_idea[r["idea"]] = ops[-1].entry_id
                elif entry.bucket == "middle":
                    queued = {e.text for e in memory.entries.values() if e.section == "try_next"}
                    untried = [
                        i
                        for i, _ in IDEAS[r["focus"]]
                        if i not in tested and f"{r['focus']}: try {i}" not in queued
                    ]
                    if untried:
                        ops.append(
                            MemoryOp(
                                op="add",
                                section="try_next",
                                entry_id=memory.new_id(),
                                text=f"{r['focus']}: try {untried[0]}",
                                reason="new idea from a mid-ranked report",
                            )
                        )
                # failed runs from every report become dead ends, so they aren't retried
                for idea in r["tried"]:
                    if idea not in dead_ends and dict(IDEAS[r["focus"]])[idea] > 0:
                        dead_ends.add(idea)
                        ops.append(
                            MemoryOp(
                                op="add",
                                section="failed",
                                entry_id=memory.new_id(),
                                text=f"{idea}: worse or no gain",
                                reason=f"{entry.bucket}-ranked report; recorded so it isn't retried",
                                proof=f"{entry.agent_id} run_code",
                            )
                        )
            for r in results:
                tested.update(r["tried"])
            # keep it current: open questions that have now been tested are removed
            for e in [e for e in memory.entries.values() if e.section == "try_next"]:
                if e.text.split(": try ", 1)[-1] in tested:
                    ops.append(
                        MemoryOp(
                            op="remove",
                            section="try_next",
                            entry_id=e.entry_id,
                            text=e.text,
                            reason="tested this generation; result recorded above",
                        )
                    )
            # keep it lean: at most 5 findings; older ones are folded into the best config
            works_ids = [e.entry_id for e in memory.entries.values() if e.section == "works"]
            works_ids += [o.entry_id for o in ops if o.section == "works" and o.op == "add"]
            for old_id in works_ids[: max(0, len(works_ids) - 5)]:
                old = memory.entries[old_id]
                ops.append(
                    MemoryOp(
                        op="remove",
                        section="works",
                        entry_id=old_id,
                        text=old.text,
                        reason="folded into the best config (kept lean)",
                        proof=old.proof,
                    )
                )
                works_by_idea = {k: v for k, v in works_by_idea.items() if v != old_id}
        elif profile["memory"] == "raw":
            for r in results:
                ops.append(
                    MemoryOp(
                        op="add",
                        section="works" if r["idea"] else "failed",
                        entry_id=memory.new_id(),
                        text=f"[{r['focus']} g{gen}] {r['idea'] or 'no gain'} (tried: {', '.join(r['tried'])})",
                        reason="appended report verbatim (no curation)",
                    )
                )
        for op in ops:
            if op.op == "remove":
                memory.entries.pop(op.entry_id, None)
            else:
                memory.entries[op.entry_id] = op
        if profile["memory"] != "none" and ops:
            memory.version += 1
            w.add(
                t,
                MemoryUpdated(
                    gen=gen,
                    version=memory.version,
                    ops=ops,
                    best=BestSoFar(val_loss=best_overall, summary=best_summary),
                    size_tokens=memory.tokens(),
                    cap_tokens=MEMORY_CAP,
                ),
            )
        all_usage = [
            e for e in w.events if isinstance(e, LlmUsage) and e.agent_id.startswith(f"g{gen}-")
        ]
        w.add(
            t + 0.5,
            GenFinished(
                gen=gen,
                best_val_loss=best_overall,
                waste_runs=sum(r["waste"] for r in results),
                total_runs=sum(r["runs"] for r in results),
                memory_tokens=memory.tokens() if profile["memory"] != "none" else 0,
                raw_log_tokens=raw_tokens,
                prompt_tokens=sum(u.prompt_tokens for u in all_usage),
                cached_tokens=sum(u.cached_tokens for u in all_usage),
                mean_ttft_ms=int(sum(u.ttft_ms for u in all_usage) / max(1, len(all_usage))),
            ),
        )
        t += 3

    fit = FITNESS[setup]
    seeds = []
    for s in range(3):
        g = round(fit + rng.gauss(0, 0.006), 4)
        test_l0 = round(L0 + 0.011 + rng.gauss(0, 0.003), 4)
        seeds.append(
            SeedScore(seed=s, test_l0=test_l0, test_lbest=round(test_l0 * (1 - g), 4), gain=g)
        )
    w.add(
        t + 30,
        EvalFinished(setup=setup, seeds=seeds, fitness=round(sum(x.gain for x in seeds) / 3, 4)),
    )
    w.add(t + 31, RunFinished())
    return w.write()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for i, setup in enumerate(SETUP_PROFILE):
        path = make_run(setup, seed=17 + i)
        print(f"{path.relative_to(Path.cwd())}: {sum(1 for _ in path.open())} events")
