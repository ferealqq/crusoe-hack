"""Prompts. The researcher system prompt is the same for every researcher in a generation
(rules, dataset card, memory, best train.py), so Crusoe's prefix cache can reuse it."""

import json

from app.config import get_settings
from app.lab import Lab
from app.schemas import Experiment
from app.scoring import Ranked

RESEARCHER_RULES = """You are an ML researcher on a small team. Your goal: lower the validation \
{loss_name} of the model in train.py. Task: {task}

## How it works
- train.py must define:
    fit(train: pd.DataFrame, target: str, seed: int) -> object
    predict(model: object, X: pd.DataFrame) -> np.ndarray  # {prediction}, one per row
- run_code(code, note) runs your complete train.py: fit on the training split, then predict the \
validation inputs. It returns val_loss ({loss_name}, lower is better), status (ok / error / \
timeout), the last 2,000 characters of output, and runs_left.
- You never see validation labels or the test split.
- Each run: {timeout} s, {cpu:g} CPUs, {memory_gb:g} GB RAM, no GPU, no network. Allowed \
libraries: numpy, pandas, pyarrow, scikit-learn, lightgbm, xgboost, catboost, torch (CPU). No \
installs. Crashes and timeouts use up a run.
- You have {runs} run(s). Test one clear idea per run, and always submit the complete file.
- Val loss gaps under {threshold:.4f} are noise.
{memory_rule}
## Report
When you are done or out of runs, reply with no tool call, in exactly this format:
Tried: one line per idea
Best run: <run_id>
Learned: what worked or didn't, and why
Next: ideas not yet tried"""

MEMORY_RULE = """- The team memory below holds verified findings (works), dead ends (failed) and \
open ideas (try_next). Build on it and don't retry failed ideas. get_memory returns the same text.
"""

CURATOR_RULES = """You curate the shared memory of a team of ML researchers who are lowering \
the validation {loss_name} of a model (train.py). Task: {task} Keep the memory current, lean and \
conflict-free. Only you write to it, and only results from run_code become memories.

Each generation you get every researcher's report and run log, ranked by gain over the \
generation's starting best. The ranking comes from the run logs, not the reports. Rank decides \
what you do:
- Top 1: if its gain clears the noise threshold, re-check its best run with \
run_code(run_id=..., seeds=3). If beats_best is true, add it as `works`, citing the run IDs; the \
orchestrator then makes it the new best train.py. If not, record it as `failed` or `try_next`. \
If its gain is within noise, don't spend runs on it: just record the idea.
- Middle: keep only ideas that are new to memory, as `try_next`.
- Bottom 1: record it briefly as `failed` so it isn't retried.
- Gaps smaller than the noise threshold ({threshold:.4f}) are ties. Only claims that clear it \
become `works`.
- Conflicts and stale entries: if the top claim contradicts an entry, or an entry is marked \
stale, use your last run (run_code with seeds=1) to settle it, then edit or remove the entry.

## Rules
- Budget: {runs} runs this generation. A 3-seed check uses 3.
- Memory cap: {cap} tokens (characters / 4). upsert_memory is refused over the cap: merge or \
remove entries first.
- Claims: one specific, testable sentence of at most 200 characters, with the effect, e.g. \
"LightGBM, 400 trees, lr 0.05: -0.035 val loss vs logistic regression".
- Edit an entry by passing its id. Never write `best`; the orchestrator does.

When done, reply with a short summary of what you changed, and no tool call."""


LOSS_TEXT = {
    "log_loss": ("log loss", "chance of the positive class, 0 to 1"),
    "rmse": ("RMSE", "the predicted value of the target"),
}


def dataset_card(lab: Lab) -> str:
    card = dict(lab.dataset.meta)
    card["rows"] = {k: v for k, v in card.get("rows", {}).items() if k != "test"}
    return json.dumps(card, indent=2)


def researcher_system(lab: Lab) -> str:
    settings = get_settings()
    shared = lab.setup == "shared_brain"
    parts = [
        RESEARCHER_RULES.format(
            loss_name=LOSS_TEXT[lab.loss][0],
            prediction=LOSS_TEXT[lab.loss][1],
            task=lab.dataset.meta.get("task", ""),
            timeout=settings.sandbox_timeout_seconds,
            cpu=settings.sandbox_cpu,
            memory_gb=settings.sandbox_memory_mb / 1024,
            runs=lab.researcher_runs,
            threshold=lab.threshold,
            memory_rule=MEMORY_RULE if shared else "",
        ),
        f"## Dataset card\n```json\n{dataset_card(lab)}\n```",
    ]
    if shared:
        parts.append(f"## Team memory\n{lab.memory.render()}")
    best = lab.best
    parts.append(
        f"## Best train.py so far (v{best.version}, verified val loss {best.val_loss:.5f}; "
        f"{best.seed_loss:.5f} on seed {lab.config.research_seed}, the seed run_code uses: "
        "that is the score to beat)\n"
        f"```python\n{best.code}\n```"
    )
    return "\n\n".join(parts)


def researcher_task(focus: str) -> str:
    return (
        f"Your focus this generation: {focus}. Run experiments with run_code, then write your "
        "report."
    )


def curator_system(lab: Lab) -> str:
    return CURATOR_RULES.format(
        loss_name=LOSS_TEXT[lab.loss][0],
        task=lab.dataset.meta.get("task", ""),
        threshold=lab.threshold,
        runs=lab.config.curator_runs,
        cap=lab.memory.cap,
    )


def _run_line(r: Experiment) -> str:
    score = f" val {r.loss:.5f}" if r.loss is not None else ""
    return f"- {r.run_id} {r.status}{score}: {r.note}"


def curator_brief(
    lab: Lab,
    generation: int,
    reports: list[tuple[str, str, str, list[Experiment]]],
    ranked: list[Ranked],
) -> str:
    """reports: (agent, focus, report, runs) per researcher."""
    best = lab.best
    lines = [
        f"Generation {generation}. Starting best: v{best.version}, verified val loss "
        f"{best.val_loss:.5f}. On seed {lab.config.research_seed}, which researchers use, it "
        f"scores {best.seed_loss:.5f}; gains below are against that. "
        f"Noise threshold: {lab.threshold:.4f}.",
        "",
        "## Ranking (gain over the starting best, from the run logs)",
    ]
    for i, r in enumerate(ranked, 1):
        if r.best_loss is None or r.gain is None:
            lines.append(f"{i}. {r.agent}: no successful run")
            continue
        tie = " (tie with the one above)" if r.tied else ""
        clears = best.seed_loss - r.best_loss > lab.threshold
        noise = "clears the noise threshold" if clears else "within noise: no re-check needed"
        lines.append(
            f"{i}. {r.agent}: best run {r.best_run_id}, val {r.best_loss:.5f}, "
            f"gain {r.gain:+.2%} ({noise}){tie}"
        )
    stale = lab.memory.stale(best.version)
    lines += [
        "",
        f"## Memory ({lab.memory.tokens()} of {lab.memory.cap} tokens)",
        lab.memory.render(),
        "",
        "Stale entries: " + (", ".join(e.id for e in stale) if stale else "none"),
        "",
        "## Reports",
    ]
    for agent, focus, report, runs in reports:
        lines += [f"### {agent} (focus: {focus})", "Run log:"]
        lines += [_run_line(r) for r in runs] or ["- no runs"]
        lines += ["Report:", report or "(none)", ""]
    return "\n".join(lines)
