---
name: crusoe-hack-dev
description: Conventions, layout and commands for this repo (Shared Brain MVP - Crusoe + Modal, results as JSONL files, no database). Load before changing code - services, storage, settings, tests - or debugging how integrations are wired here.
---

# crusoe-hack development

The spec is `README.md` (Shared Brain). Build in its "Build order".

## Commands
```bash
uv sync
uv run fastapi dev app/main.py                  # /docs: setup checks (/playground/models, /playground/sandbox)
uv run ruff check --fix . && uv run ruff format . && uv run pytest     # required before finishing
```

Setup is the "Setup" section in `README.md`: copy `.env.example` to `.env`, paste `CRUSOE_API_KEY`, run `uv run modal token new` (opens a browser, so ask the user). Agents should assume `.env` exists.

## Layout
- `app/config.py`: `Settings` (pydantic-settings, reads `.env`). Only real knobs become settings; fixed values are module constants (`inference.BASE_URL`, `sandbox.APP_NAME`). Every new setting also goes in `.env.example`.
- `app/store.py`: `append_jsonl`, `read_jsonl`, `write_jsonl` (atomic rewrite). File layout under `data/` is "Storage" in `README.md`.
- `app/schemas.py`: Pydantic I/O models.
- `app/api/routes/`: `health`, `playground` (setup checks only), registered in `main.py`.
- `app/services/`: provider clients.
  - `inference.py`: Crusoe via `AsyncOpenAI`, `max_retries=6`. `chat(messages, model, ...)` returns `(content, latency_ms)`.
  - `sandbox.py`: Modal. `run_python(code, timeout)` returns `SandboxResult`.

## Rules
- Async for provider calls. Match the surrounding style.
- No database. JSONL files are append-only, except `memory.jsonl`, which is rewritten whole with `write_jsonl`. Writes happen in one process, so appends don't need locks.
- Tests never hit Crusoe or Modal: `monkeypatch.setattr(<module>, "fn", fake)` on the `app.services` module, and call it through the module (`from app.services import sandbox`; `sandbox.run_python(...)`). Tests that write files use pytest's `tmp_path`.
- Crusoe limits apply per model (30 RPM without a payment method). Parallel researchers on one model hit 429; the client retries.

## Wiring gotchas
| Symptom | Fix |
| --- | --- |
| Crusoe 401/404/429, empty output | See `crusoe-inference` (key quoting, ID casing, limits, `max_tokens`) |
| Modal `AuthError` | `uv run modal token new` (Modal doesn't read `.env`) |
| pytest warns about unknown config options | It found the parent checkout's `pyproject.toml`; keep `[tool.pytest.ini_options]` in this one |
