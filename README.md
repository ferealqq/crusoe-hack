# Shared Brain

A shared memory layer for a team of ML research agents, built on Crusoe Managed Inference.

Every new agent on a team usually starts from zero and pays again for the same context. In Shared Brain, short-lived researcher agents read a shared memory before they work, and a curator keeps that memory **current, lean and conflict-free**. Each generation starts fresh but smarter. We measure the gain as the drop in loss on a held-out test split that no agent sees.

## How it works

```mermaid
flowchart TD
    A[Prepare data + measure noise] --> B[Spawn researchers]
    B --> C[Read memory + best train.py]
    C --> D[Run experiments]
    D --> E[Report results]
    E --> F[Curator verifies]
    F --> G[Update memory + best]
    G --> H{Generations left?}
    H -- Yes --> B
    H -- No --> I[Score on test split]
    I --> J[Compare setups]
```

1. Before generation 1, the orchestrator runs the starting `train.py` on 5 seeds to get the baseline loss and the noise level.
2. Each generation, 4 researchers start from the best `train.py` so far plus memory.
3. Each researcher runs experiments to lower val loss, then reports what it learned.
4. The curator re-checks the top claim, settles conflicts and updates memory.
5. At the end, the starting and best `train.py` are scored on the hidden test split.

## Setups

All setups use the same dataset, splits, models, sandbox and total number of runs. Every generation starts from the best `train.py` so far, so **memory is the only difference**.

| Setup | What it shows | What researchers read | Who picks the best `train.py` |
|---|---|---|---|
| One-shot | A single agent, a single edit; this is the floor | Nothing. One run | — |
| No memory | Every agent starts from zero | Nothing | Orchestrator: lowest val loss |
| Raw log | Memory without curation, which bloats and contradicts itself | Every report, appended as-is. Oldest dropped past 100k tokens | Orchestrator: lowest val loss |
| **Shared Brain** | Curated memory: lean and current | Curated memory, 2,000-token cap | Curator check on 3 seeds |

A new best must beat the old one by more than the noise threshold, in every setup.

## Results so far

| Dataset | One-shot | No memory | Raw log | Shared Brain | Details |
|---|---|---|---|---|---|
| Churn (test gain, 5 generations) | 4.1% | 20.0% | 20.0% | **21.0%** | [findings](data/runs/20261017-223035-churn/FINDINGS.md) |
| Churn (test accuracy) | 82.3% | 86.3% | 86.3% | **86.8%** | starting model 82.7% |
| Superconductivity (test gain, generation 1 only) | 44.1% | 45.7% | 46.0% | 46.2% | [partial findings](data/runs/20261017-234325-653156-superconductivity/FINDINGS.md) |

On churn, Shared Brain was the only setup that improved after generation 1. It's one research run per setup, so treat it as a promising signal, not proof. The superconductivity run was stopped after generation 1, before memory comes into play. Each run folder has a `report.html` with per-generation charts and memory snapshots.

## Budget

| Item | Value |
|---|---|
| Generations | 5 |
| Researchers per generation | 4 |
| `run_code` calls per researcher | 6 (Shared Brain), 7 (No memory, Raw log) |
| Curator runs per generation | 4: 3 seeds to verify the top claim, 1 for a conflict or stale entry |
| Total runs per generation | 28 in every setup |
| `web_search` per researcher | 2, 5 results each |
| Agent turns per researcher | 30 max |
| One run | 120 s, 2 CPUs, 4 GB RAM, no GPU, no network |
| Seeds | Researcher runs: 0. Curator checks: 1, 2, 3. Test scoring: 1, 2, 3 |

Crashes and timeouts use up a run.

## Data

Each run uses one dataset, split three ways. There are no other datasets in a run.

| Split | Who sees it | Use |
|---|---|---|
| Train | Researchers, curator | Training |
| Val | Researchers, curator (inputs only, never labels) | Scoring during research |
| Test | No agent | Final before-and-after comparison only, run by `test.py` |

- Researchers never see the test split: not its data, its loss, or any score built from it.
- The curator doesn't see it either. It verifies claims on val, because what it learns goes into memory, and researchers read memory.
- Agent code never sees val labels. `train.py` returns predictions and the orchestrator scores them outside the sandbox.

| Step | The sandbox gets | The sandbox never gets |
|---|---|---|
| Research (`run_code`) | Train, val inputs | Val labels, any test data |
| Final scoring (`test.py`) | Train + val (with labels), test inputs | Test labels |

### Data sources

We test on three different kinds of problem: a yes/no prediction, a time-series forecast and a numeric prediction. Each dataset gets its own run and is scored on its own test split. All datasets are free to download without a login.

| Problem | What it predicts | Dataset | Val loss | Split |
|---|---|---|---|---|
| Churn | Will this customer leave? (yes/no) | [OpenML 46371](https://www.openml.org/d/46371) (35k customers) | Log loss | Random 70/15/15, stratified, seed 42 |
| Regression | At what temperature does a material become a superconductor? | [Superconductivity](https://www.openml.org/d/44148) (21k rows, 79 features) | RMSE | Random 70/15/15, seed 42 |
| Inventory (not built yet) | Daily demand per product | [Walmart M5](https://zenodo.org/records/12636070), top 1,000 products by units sold | Pinball loss at 90% | By time: test = last 28 days, val = 28 days before, train = the rest |

Why these:

- **Churn:** clear room to improve (~20%) and stable between runs.
- **Regression:** most room to improve (~47%); 79 features give agents real tuning work.
- **Inventory:** real retail data; the score measures stock levels, not just the forecast. Room to improve not measured yet: `prepare_data.py` measures it, and we drop M5 if it's under 5%.

Order: churn first end-to-end, then superconductivity, then M5 only if there's time. Status: churn done; superconductivity built, first run stopped after generation 1; M5 not started.

`scripts/prepare_data.py <name>` (`churn` or `superconductivity`) downloads a dataset, splits it and saves it locally under `data/<name>/` (git-ignored): `train.parquet`, `val_X.parquet`, `val_y.parquet`, `test_X.parquet`, `test_y.parquet`, `trainval.parquet` (used only by `test.py`) and `meta.json` (target column, loss, split rule).

## Files

| File | Access |
|---|---|
| `train.py` | The only file agents can edit |
| `sandbox/harness.py` | Locked. Runs `train.py` on train, predicts val inputs, returns predictions |
| `sandbox/test.py` | Locked; agents can't see, edit or run it. Only the orchestrator runs it |

### The `train.py` contract

```python
import numpy as np
import pandas as pd


def fit(train: pd.DataFrame, target: str, seed: int) -> object: ...
def predict(model: object, X: pd.DataFrame) -> np.ndarray: ...
```

- Churn returns the chance of leaving (0 to 1). Regression returns the temperature. Inventory returns the 90th-percentile daily demand for each row.
- `test.py` fits on train + val, then predicts the test inputs, for seeds 1, 2 and 3.
- Allowed libraries: numpy, pandas, pyarrow, scikit-learn, lightgbm, xgboost, catboost, torch (CPU). No installs at runtime.

Starting `train.py` per dataset, in `baselines/<name>/train.py`:

| Dataset | Baseline |
|---|---|
| Churn | Logistic regression |
| Regression | Ridge regression |
| Inventory | Per-product 90th percentile of the last 28 days (not built yet) |

## Tools

| Tool | Input | Returns |
|---|---|---|
| `get_memory` | — | Memory as text |
| `run_code` (researcher) | `code` (the whole `train.py`), `note` (one line: what changed) | `run_id`, `status` (ok / error / timeout), `val_loss`, last 2,000 characters of output, `runs_left` |
| `run_code` (curator) | `run_id` to re-check, or `code` | Mean and spread of val loss over seeds 1, 2, 3, and `beats_best` (true if it beats the best by more than the noise threshold). Uses 3 runs |
| `web_search` | `query` | Up to 5 results: title, URL, description |
| `upsert_memory` | `kind` (works / failed / try_next), `claim` (200 characters max), `run_ids`, optional `id` to edit | `id`, tokens used, token cap. Refused if over the cap |
| `remove_memory` | `id`, `reason` | ok |

The memory tools are our own: memory lives in the orchestrator and is saved to the run's files every generation (see [Storage](#storage)). Crusoe has no memory API. `web_search` is offered only when `YDC_API_KEY` is set.

`web_search` uses the [You.com Search API](https://you.com/docs/get-started/quickstart): `POST https://ydc-index.io/v1/search` with header `X-API-Key: $YDC_API_KEY` and body `{"query": "...", "count": 5}`. Results are in `results.web[]` (`title`, `url`, `description`).

## Agents

| Agent | Model | Tools |
|---|---|---|
| **Researcher** (×4) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` | `get_memory`, `web_search`, `run_code` |
| **Curator** (×1) | `deepseek-ai/DeepSeek-V4-Pro` | `get_memory`, `run_code` (to re-check), `upsert_memory`, `remove_memory` |
| Waste checker (planned, not built) | `openai/gpt-oss-120b` | None |

- Researcher focus areas: features, model type, tuning, regularization and ensembling.
- Researcher settings: temperature 1.0, top_p 0.95, `chat_template_kwargs: {"enable_thinking": true, "force_nonempty_content": true}`, max_tokens 8192.
- Curator settings: `chat_template_kwargs: {"thinking": true}`, max_tokens 8192.
- At start, the orchestrator checks all model IDs against `/v1/models` and stops if one is missing. Nemotron 3 Ultra isn't on Crusoe's serverless list today; if it appears, it can replace the curator model.
- The orchestrator is a plain Python script, not an agent.

### Prompt order

Researchers in a generation share the same start of the prompt, so Crusoe's cache reuses it:

1. Rules and tool list (same for everyone)
2. Dataset card (`meta.json`)
3. Memory
4. Best `train.py` so far
5. This researcher's focus
6. The conversation

### Report

A researcher's final message is its report:

```
Tried: one line per idea
Best run: <run_id>
Learned: what worked or didn't, and why
Next: ideas not yet tried
```

The orchestrator attaches the researcher's run log (note, status and val loss per run) before passing it to the curator. **Ranking uses the run log, not the report.**

## Rules

- **Memory writes:** only the curator writes to memory.
- **`web_search`:** researchers only. Web ideas are hypotheses, not facts. Only results from `run_code` become memories.
- **Budget:** enforced inside `run_code` (time and run limits), so no one wins by training longer.
- **Test split:** no agent can read it or get a score on it, so nothing tuned on it leaks into memory or `train.py`.

## Generation loop

1. **Once per research run:** run the starting `train.py` on seeds 0 to 4. `L0` is the mean. The noise threshold is the larger of 2 × the spread and 0.5% of `L0`.
2. **Each generation:**
   1. Run 4 researchers in parallel from the best `train.py` and the current memory.
   2. Rank them by gain over the generation's starting best, from the run log.
   3. Update memory: the curator (Shared Brain), append reports (Raw log), or nothing (No memory).
   4. Update the best `train.py` if a run beats it by more than the noise threshold.
   5. Score this generation's best on the test split (seeds 1 to 3) for the chart only. Never shown to agents.
   6. Save metrics and a memory snapshot.
3. **At the end:** `test.py` scores the starting and the best `train.py`.

## Curation

Each generation, the 4 researchers are ranked by **gain over the generation's starting best** (not raw val loss). The curator reads every report; rank decides what it does with each one.

| Rank | Curator action |
|---|---|
| **Top 1** | Re-run with `run_code` on 3 seeds. If `beats_best`, add it to memory as a fact; the orchestrator updates the best `train.py` |
| **Middle 2** | Keep only new ideas, as `try_next` |
| **Bottom 1** | Record briefly as `failed` so it isn't retried |

- **Losers are kept on purpose:** failed runs are the "what failed" part of memory.
- **Noise:** gaps smaller than the noise threshold count as ties. Only claims that clear it become facts.
- **Conflicts:** if the top claim contradicts an entry, or an entry is stale, the curator's 4th run settles it.
- **Budget:** re-running is the curator's expensive step, so ranking decides where it's spent.

## Memory

The store holds four kinds of entry:

- **works:** findings, with proof.
- **failed:** dead ends not to retry.
- **try_next:** open questions.
- **best:** the top config and score (set by the orchestrator only).

Each entry has: `id`, `kind`, `claim`, `run_ids`, val change, seeds, `tested_on` (best version), created and updated generation.

Rules for the memory:

- **Lean:** 2,000-token cap (counted as characters ÷ 4). If still over the cap after curation, drop the oldest `try_next`, then the oldest `failed`. Never drop `best`.
- **Current:** an entry tested on a best that is 2 or more versions old is stale. The curator re-tests or removes it.
- **Conflict-free:** only the curator writes to it, and contradictions are settled by an experiment.
- **Versioned:** every change is logged, and memory is saved every generation.
- **Stable text:** memory is rendered in a fixed order (best, works, failed, try_next, then by `id`) so the cache can reuse it.

### Why Crusoe MemoryAlloy

Every researcher's prompt starts with the same memory (see [Prompt order](#prompt-order)), so Crusoe's [MemoryAlloy](https://crusoe.ai/resources/blog/crusoe-memoryalloy-reinventing-kv-caching-for-cluster-scale-inference) reads it once and reuses that work for the other researchers. Sharing memory stays cheap and fast: each researcher pays less and starts answering sooner.

## Scoring

```
gain    = (L0 − Lbest) / L0
fitness = average gain across seeds 1, 2, 3
```

- **`L0`:** loss of the starting `train.py`.
- **`Lbest`:** loss of the best `train.py` reached within the budget.
- During research, both are val loss. For the final score, `test.py` measures both on the test split, with the best `train.py` picked by val loss.

## Evaluation

All setups use the same dataset and budget, and are scored on the same test split. We track four metrics per generation:

- **Quality:** test gain of that generation's best `train.py`.
- **Waste (planned, not built):** runs that repeat a change that already failed earlier in the same research run. `openai/gpt-oss-120b` would check each change against the full run log, with the same rule in every setup.
- **Memory size:** tokens of what researchers read. Curated memory should stay flat while the raw log grows.
- **Cost:** prompt, cached and output tokens per agent, saved with each agent session; `report.html` turns them into dollars with Crusoe's prices. Time-to-first-token (which needs streamed calls; see [Why Crusoe MemoryAlloy](#why-crusoe-memoryalloy)) is planned, not built.

Measured size: on churn, about 135 sandbox runs per setup. With all 4 setups in parallel, the run took 52 minutes; most of that was waiting on Crusoe rate limits, and without them a generation takes about 5 minutes.

## Storage

No database and no bucket for research: everything is a file on local disk under `data/` (git-ignored), written with `app/store.py`. JSONL files are append-only. Finished run folders are committed with `git add -f`.

| File in `data/runs/<timestamp>-<dataset>/` | Holds |
|---|---|
| `config.json` | Dataset, loss, setups, budget, models, Crusoe prices |
| `baseline.json` | `L0`, noise threshold, the starting model's val and test losses |
| `summary.json`, `rate_limited.json` | Final result per setup; 429 counts per model |
| `events.jsonl` | Live research events for the telemetry API |
| `run.log`, `report.html`, `FINDINGS.md` | Console log; generated report; hand-written findings |
| `<setup>/generations.jsonl` | Per generation: start and end best val loss, best version, test losses and gain, memory tokens, runs, token usage |
| `<setup>/experiments.jsonl` | Every sandbox run: agent, seed, status, val loss, runtime, note, output tail, the full `train.py` |
| `<setup>/agents.jsonl` | Every agent session: role, focus, model, report, run log, curator re-checks, tokens, transcript |
| `<setup>/memory_snapshots.jsonl` | Memory text and token count after each generation, every memory change with its reason, the curator's summary |
| `<setup>/result.json` | Fitness, best version, best val loss, the best `train.py` |

Datasets live in `data/<name>/` and aren't committed; `prepare_data.py` rebuilds them. The older prompt-grading endpoints (`datasets`, `runs`, `evaluator.py`, the LLM judge) still use Neon Postgres and the bucket; the research code doesn't touch them.

## Code layout

```
app/
  orchestrator.py        # CLI: uv run python -m app.orchestrator --dataset churn --setup all
  loop.py                # baseline and noise, generation loop, ranking, best tracking
  lab.py                 # runs train.py in the sandbox, scores it outside, records experiments
  agents/                # base.py (tool-calling loop), researcher.py, curator.py, prompts.py
  memory.py              # curated memory (cap, staleness, drop order) and the raw log
  scoring.py             # losses, noise threshold, gain, fitness
  store.py               # JSON/JSONL files under data/runs/
  report.py              # report.html for a run folder
  events.py              # live research events (see docs/research-telemetry.md)
  services/              # inference.py, sandbox.py, search.py, research_events.py, model_exports.py
                         # (storage.py and evaluator.py serve the old prompt-grading endpoints)
  api/routes/            # research.py, exports.py, health.py, playground.py (+ old datasets.py, runs.py)
sandbox/                 # harness.py, test.py (locked, copied into the sandbox); export.py, predict.py for model exports
baselines/<dataset>/train.py
scripts/                 # prepare_data.py, preflight.py (+ setup.sh, check_env.py for the full service)
```

## API and dashboard

The backend now records live research events and serves SSE plus paced replay.
See [Research telemetry API](docs/research-telemetry.md) for endpoints, event types,
reconnect/seek semantics, and frontend integration. The dashboard UI is separate.

Research telemetry reads the CLI's files under `DATA_DIR/runs/`; the API is read-only
for research telemetry. The [trained model export API](docs/model-exports.md) can
refit a completed setup's winning code and download a fitted-model ZIP, including
a prediction helper, dependency versions and metrics. The older prompt-grading
endpoints still use Postgres.

| Endpoint | Returns |
|---|---|
| `GET /research-runs` | All research runs |
| `GET /research-runs/{id}` | One run's metadata |
| `GET /research-runs/{id}/events` | Recorded events, paged with `after` |
| `GET /research-runs/{id}/state` | Run state rebuilt from events, optionally at an event number (`at`) |
| `GET /research-runs/{id}/stream` | Live events over SSE, with paced replay |
| `POST /research-runs/{id}/setups/{setup}/export` | Build or reuse a trained model export |
| `GET /research-runs/{id}/setups/{setup}/export` | Export status and download URL |
| `GET /research-runs/{id}/setups/{setup}/export/download` | Download the trained model ZIP |
| `GET /health`, `GET /playground/models` | Unchanged |

There's no dashboard page in this repo yet; per-run charts are in each run's `report.html`.

## Settings

New settings go in `app/config.py` and `.env.example`:

| Setting | Default |
|---|---|
| `YDC_API_KEY` | — |
| `RESEARCHER_MODEL` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| `CURATOR_MODEL` | `deepseek-ai/DeepSeek-V4-Pro` |
| `CRUSOE_JUDGE_MODEL` (exists; for the planned waste check) | `openai/gpt-oss-120b` |
| `SANDBOX_TIMEOUT_SECONDS` (exists) | `120` |
| `SANDBOX_CPU` | `2` |
| `SANDBOX_MEMORY_MB` | `4096` |
| `SANDBOX_MAX_CONCURRENCY` (sandboxes at once, all setups) | `24` |
| `CRUSOE_MAX_RPM` (requests per minute per model, spaced evenly) | `30` |
| `MODAL_ENVIRONMENT` (overrides the `~/.modal.toml` profile's environment) | — |
| `DATA_DIR` | `data` |

Budget numbers live in a `ResearchConfig` model with the defaults above, and can be overridden from the CLI.

## Build order

1. Churn data prep, baseline, `harness.py`, `run_code`: the baseline gets a val score.
2. One researcher, one generation.
3. 4 researchers, ranking, best tracking: the No memory setup works.
4. Curator and memory: the Shared Brain setup works.
5. Raw log and One-shot setups.
6. `test.py`, per-generation test scores, metrics.
7. API and dashboard.
8. Superconductivity, then M5.

## Done when

- `uv run python -m app.orchestrator --dataset churn --setup shared_brain` finishes 5 generations and saves fitness.
- All 4 setups run on churn and show in the run's `report.html` (a dashboard UI can use the telemetry API).
- Tests (no network) cover: val and test labels never reach the research sandbox, run limits, memory cap and dropping order, staleness, ranking with ties, equal total runs across setups, scoring math, raw log trimming.
- `uv run ruff check . && uv run ruff format . && uv run pytest` passes.

## Out of scope

GPUs, fine-tuning, more than one dataset per research run, agents editing anything but `train.py`, repeating whole research runs (add only if time allows).

## Models (Crusoe Managed Inference)

| Role | Model |
|---|---|
| Researcher | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| Curator | `deepseek-ai/DeepSeek-V4-Pro` |
| Waste checker (planned) | `openai/gpt-oss-120b` |

Add a payment method to the Crusoe project: it raises the limit from 30 to 600 requests per minute per model.

## Tech stack

| Layer | Tool |
| --- | --- |
| API | [FastAPI](https://fastapi.tiangolo.com) (Python 3.12+, uv) |
| Database | [Neon Postgres](https://neon.com/docs) via SQLAlchemy async + asyncpg, Alembic migrations (old prompt-grading endpoints only) |
| File storage | Local files under `data/` (no object storage) |
| LLM inference | [Crusoe Intelligence Foundry](https://console.crusoecloud.com/foundry) (OpenAI-compatible API) |
| Code sandboxes | [Modal Sandboxes](https://modal.com/docs/guide/sandboxes) |
| Web search | [You.com Search API](https://you.com/docs/get-started/quickstart) |

## Setup

For research runs only, skip this and see [Running the experiments](#running-the-experiments): they need just `CRUSOE_API_KEY` and a Modal token. The steps below set up the full service, including the old prompt-grading endpoints that need Neon.

```bash
scripts/setup.sh                    # installs tooling, signs in to Modal/Neon/Crusoe, generates a verified .env
uv run python scripts/check_env.py  # re-verify .env anytime (--skip-sandbox for a fast check)
uv run fastapi dev app/main.py
```

Requires Node.js (for the Neon CLI) and access to the team Neon project `patient-art-99467040` (ask the owner to invite your Neon account).

## Running the experiments

Results are files under `data/runs/<timestamp>-<dataset>/` (see [Storage](#storage)). A run needs only `CRUSOE_API_KEY` and a Modal token in `.env`; `YDC_API_KEY` turns on `web_search`.

```bash
uv run python scripts/prepare_data.py churn                     # download + split into data/churn/ (or: superconductivity)
uv run python scripts/preflight.py                              # check Crusoe, You.com and Modal keys
uv run python -m app.orchestrator --dataset churn --setup all   # 4 setups in parallel, shared baseline
uv run python -m app.report data/runs/<run>                     # rebuild report.html (works mid-run)
```

`--generations`, `--researchers` and `--max-turns` override the budget for a quick smoke run. With `--setup all`, the setups share one baseline measurement, so they have the same `L0` and noise threshold. Crusoe refuses bursts of more than about 1–2 requests a second per model, so requests are spaced evenly (`CRUSOE_MAX_RPM`), and 429s are retried and counted in `rate_limited.json`.

Built at the Crusoe x Cerebral Valley Recursive Agents Hackathon, SF, Oct 2026.
