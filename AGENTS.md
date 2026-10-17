# AGENTS.md

Shared Brain: research agents on Crusoe (OpenAI-compatible) improve a `train.py`, which runs in Modal Sandboxes. Research datasets and results are local files under `data/` (see README "Running the experiments"). The older prompt-grading endpoints still use Neon Postgres and Neon Object Storage.

## Skills (`.agents/skills/<name>/SKILL.md`)
Open the matching skill before working in that area. If your harness doesn't auto-load skills, read the file directly. Each skill keeps long material in `references/`; open it only when needed.

| Skill | When |
| --- | --- |
| `crusoe-hack-dev` | Any code change in this repo: layout, commands, conventions, wiring gotchas |
| `llm-eval-loop` | Datasets, graders, judges, comparing runs, improve step, training exports |
| `crusoe-inference` | Crusoe model IDs, limits, params, errors |
| `modal-sandboxes` | Sandbox code, isolation, exit codes, Modal auth |
| `neon` | Postgres URLs/pooling/migrations, buckets, branches, `neon` CLI |

## Rules
- Local setup is `scripts/setup.sh` (interactive, opens browsers). Don't run it yourself; ask the user to.
- Before finishing: `uv run ruff check . && uv run ruff format . && uv run pytest`.
- Tests never call Crusoe, Modal or the bucket. Monkeypatch the `app.services.*` functions.
- Schema changes go through reviewed Alembic migrations, run against a Neon branch, never the main branch.
- New settings go in `app/config.py` **and** `.env.example`. Never commit `.env` or `.env.local`.
- Platform facts (model IDs, APIs) change often. Trust the skills and live APIs (`GET /playground/models`) over memory.
