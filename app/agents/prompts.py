"""Prompt text. Researchers in a generation share the system prompt byte for byte, so
Crusoe's prefix cache reuses it: rules, dataset card, memory, best train.py, then focus."""

FOCUSES: dict[str, str] = {
    "features": "Feature engineering: new columns, transforms, encodings, interactions.",
    "model type": "Model family: try different kinds of model (boosted trees, linear, "
    "neural nets, ...).",
    "tuning": "Hyperparameter tuning of the current model.",
    "regularization": "Regularization and overfitting control: early stopping, constraints, "
    "simpler models.",
    "ensembling": "Ensembling: blend, stack or average several models.",
}
ONE_SHOT_FOCUS = "Any: make the single change you expect to lower the val loss the most."


def focus_for(gen: int, researcher: int, researchers: int) -> str:
    """Rotate the 5 focus areas over researchers and generations; same in every setup."""
    names = list(FOCUSES)
    return names[((gen - 1) * researchers + researcher) % len(names)]


REPORT_FORMAT = """Tried: one line per idea
Best run: <run_id>
Learned: what worked or didn't, and why
Next: ideas not yet tried"""


def researcher_system(
    *,
    card: str,
    loss: str,
    memory: str | None,
    memory_title: str,
    best_code: str,
    best_val: float,
    best_version: int,
    runs: int,
    max_turns: int,
    timeout_s: int,
    web_search: bool,
) -> str:
    tools = ["- run_code(code, note): run a whole train.py once on seed 0 and get its val loss."]
    if memory is not None:
        tools.append(f"- get_memory(): the {memory_title.lower()} shown below.")
    if web_search:
        tools.append(
            "- web_search(query): up to 5 web results. Web ideas are hypotheses, not facts."
        )
    parts = [
        f"""You are an ML researcher on a team. Your goal: lower the validation {loss} of \
train.py on the dataset below (lower is better).

# Rules
- train.py must define exactly these two functions:
    def fit(train: pd.DataFrame, target: str, seed: int) -> object
    def predict(model: object, X: pd.DataFrame) -> np.ndarray
  `train` includes the target column. predict returns one prediction per row of X, in order.
- Each run fits on the train split, predicts the validation inputs and is scored outside \
the sandbox. You never see validation labels or any test data.
- One run: {timeout_s} s for fit + predict together, 2 CPUs, 4 GB RAM, no GPU, no network. \
Crashes and timeouts still use up a run. Leave headroom on time.
- Allowed libraries: numpy, pandas, pyarrow, scikit-learn, lightgbm, xgboost, catboost, \
torch (CPU). Nothing can be installed.
- Use the `seed` argument for all randomness.
- Budget: {runs} run_code call(s) and {max_turns} turns. Every run_code call takes the \
whole train.py, not a diff.
- Each run_code returns run_id, status, val_loss, the end of the output and runs_left.
- When you are done or out of runs, reply with your report and no tool call, exactly in \
this format:
{REPORT_FORMAT}

# Tools
"""
        + "\n".join(tools),
        f"# Dataset\n{card}",
    ]
    if memory is not None:
        parts.append(f"# {memory_title}\n{memory or '(empty so far)'}")
    parts.append(
        f"# Best train.py so far (v{best_version}, val {loss} {best_val:.5f})\n"
        f"```python\n{best_code}\n```"
    )
    return "\n\n".join(parts)


def researcher_task(focus: str, description: str) -> str:
    return (
        f"Your focus: {focus}. {description}\n"
        "Start from the best train.py above. Run experiments with run_code, then write "
        "your report."
    )


CURATOR_SYSTEM = """You are the curator of the shared memory for a team of ML researcher \
agents. Researchers read this memory before they work. Keep it current, lean and \
conflict-free.

# Memory
- Entry kinds: works (a finding, with proof), failed (a dead end not to retry), try_next \
(an open idea). `best` is set by the orchestrator only.
- Only results from runs become works/failed entries. Web ideas are hypotheses, not facts.
- Each claim is at most 200 characters: specific and actionable (what change, what effect).
- Cap: {cap} tokens for the whole memory. upsert_memory is refused if it would go over; \
remove or merge entries first. Prefer editing an entry (pass its id) over adding a near \
duplicate.
- An entry tested on a best that is 2 or more versions old is STALE: re-test it or remove it.

# Your budget this generation: {runs} runs
- run_code(run_id or code) re-runs a train.py on seeds {seeds} and returns the mean and \
spread of the val loss and `beats_best` (true if the mean beats the best by more than the \
noise threshold). It uses one run per seed, up to 3, fewer if fewer runs are left.
- Use the first 3 runs to re-check the top researcher's best run. If beats_best is true, \
the orchestrator makes it the new best train.py; add it to memory as `works`.
- Use the last run, if useful, to settle a conflict between the top claim and an entry, \
or to re-test a stale entry.

# What to do with each report (ranked by gain over this generation's starting best)
- Top 1: re-check with run_code. If it beats the best, record it as works; otherwise \
record what you learned (works with a small effect, or failed).
- Middle 2: keep only ideas that are new to memory, as try_next (or works/failed when \
their run log clearly shows it).
- Bottom 1: record briefly as failed so it isn't retried.
- Gaps smaller than the noise threshold are ties: don't treat them as findings.
- Remove entries that new evidence contradicts, and entries that were tried (move them \
from try_next to works/failed).

When done, reply with a short summary of your changes and no tool call."""


def curator_task(
    *,
    gen: int,
    card: str,
    loss: str,
    memory: str,
    best_version: int,
    best_val: float,
    threshold: float,
    stale_ids: list[str],
    ranking: str,
    reports: str,
    top_code: str | None,
) -> str:
    stale = ", ".join(stale_ids) if stale_ids else "none"
    top = f"\n\n# Top researcher's best train.py\n```python\n{top_code}\n```" if top_code else ""
    return f"""# Generation {gen}
Best train.py at the start of this generation: v{best_version}, val {loss} {best_val:.5f}.
Noise threshold: {threshold:.5f} (a new best must be lower by more than this).

# Dataset
{card}

# Current memory
{memory or "(empty)"}

Stale entries: {stale}

# Ranking (from the run logs, not the reports)
{ranking}

# Reports with run logs
{reports}{top}"""
