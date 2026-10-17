---
name: crusoe-hack-dev
description: Conventions, layout and commands for this repo (Shared Brain MVP - Crusoe + Modal, results as JSONL files, no database). Load before changing code - services, storage, settings, tests - or debugging how integrations are wired here.
---

# crusoe-hack development

The spec is `README.md` (Shared Brain). Build in its "Build order".

## Commands
```bash
uv sync
uv run python -m app.orchestrator --setup shared_brain        # a research run (also: no_memory)
uv run python -m app.orchestrator --generations 1 --researchers 1 --runs 2 --no-commit  # quick check
uv run python -m app.orchestrator --report                    # results table
uv run fastapi dev app/main.py                  # /docs: setup checks (/playground/models, /playground/sandbox)
uv run ruff check --fix . && uv run ruff format . && uv run pytest     # required before finishing
```

Setup is the "Setup" section in `README.md`: copy `.env.example` to `.env`, paste `CRUSOE_API_KEY`, `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Agents should assume `.env` exists.

## Layout
- `app/config.py`: `Settings` (pydantic-settings, reads `.env`). Only real knobs become settings; fixed values are module constants (`inference.BASE_URL`, `sandbox.APP_NAME`). Every new setting also goes in `.env.example`.
- `app/store.py`: `append_jsonl`, `read_jsonl`, `write_jsonl` (atomic rewrite). File layout under `data/` is "Storage" in `README.md`.
- `app/orchestrator.py` (CLI) → `app/loop.py` (baseline, generations, final test) → `app/agents/` (researcher, curator; `base.run_agent` is the tool loop; budgets live in `tools.py`).
- `app/lab.py`: one research run's state (`run_experiment` runs and scores, `best`, `memory`, `checkpoint()` commits to git). `app/memory.py`: entries, render, cap, staleness. `app/scoring.py`: losses, threshold, ranking. `app/data.py`: which files each sandbox gets.
- `app/schemas.py`: Pydantic I/O models.
- `app/api/routes/`: `health`, `playground` (setup checks only), registered in `main.py`.
- `app/services/`: provider clients.
  - `inference.py`: Crusoe via `AsyncOpenAI`, `max_retries=6`. `chat(messages, model, ...)` returns `(content, latency_ms)`.
  - `sandbox.py`: Modal. `run_train(code, files, target, seed)` runs `sandbox/harness.py`; `run_python` is for the playground. One image has every allowed library.

## Rules
- Async for provider calls. Match the surrounding style.
- No database. JSONL files are append-only, except `memory.jsonl`, which is rewritten whole with `write_jsonl`. Each research run is one process writing its own folder.
- Git trail: every memory change and generation calls `lab.checkpoint(message)`, which commits only that run's files (a lock in the git dir serializes parallel runs). Don't `git add -A` while runs are going.
- Agents never see labels: `data.Dataset.research_files` has train + val inputs only; scoring happens in `lab.run_experiment`, outside the sandbox.
- Tests never hit Crusoe or Modal: `monkeypatch.setattr(<module>, "fn", fake)` on the `app.services` module, and call it through the module (`from app.services import sandbox`; `sandbox.run_python(...)`). Tests that write files use pytest's `tmp_path`.
- Crusoe limits apply per model (30 RPM without a payment method). `inference.complete` retries 429/5xx for up to 15 minutes, and a crashed agent ends only itself, not the run.

## Wiring gotchas
| Symptom | Fix |
| --- | --- |
| Crusoe 401/404/429, empty output | See `crusoe-inference` (key quoting, ID casing, limits, `max_tokens`) |
| Modal `AuthError` | Set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` in `.env` (read via the bridge in `sandbox._get_app()`) |
| Modal `Environment '…' not found` | A `~/.modal.toml` profile's environment leaked in; the bridge sets `MODAL_ENVIRONMENT=""` |
| pytest warns about unknown config options | It found the parent checkout's `pyproject.toml`; keep `[tool.pytest.ini_options]` in this one |
