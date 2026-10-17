# AGENTS.md

Shared Brain (spec: `README.md`): ML research agents share a curated memory. Inference runs on Crusoe (OpenAI-compatible), agent-written code runs in Modal Sandboxes, and results are JSONL files in `data/` (no database).

## Skills (`.agents/skills/<name>/SKILL.md`)
Open the matching skill before working in that area. If your harness doesn't auto-load skills, read the file directly.

| Skill | When |
| --- | --- |
| `crusoe-hack-dev` | Any code change in this repo: layout, commands, conventions, wiring gotchas |
| `crusoe-inference` | Crusoe model IDs, limits, params, errors |
| `modal-sandboxes` | Sandbox code, isolation, exit codes, Modal auth |

## Rules
- Setup is the short "Setup" section in `README.md`: three keys in `.env`.
- Before finishing: `uv run ruff check . && uv run ruff format . && uv run pytest`.
- Tests never call Crusoe or Modal. Monkeypatch the `app.services.*` functions.
- No database. Results are JSONL files under `data/` (layout: "Storage" in `README.md`); use `app/store.py`.
- New settings go in `app/config.py` **and** `.env.example`. Never commit `.env`. Everything in `data/` is committed (see "Git trail" in `README.md`).
- While research runs are going, don't `git add -A` or `git add data/`: the orchestrator commits run files itself. Commit code paths by name.
- Platform facts (model IDs, APIs) change often. Trust the skills and live APIs (`GET /playground/models`) over memory.
