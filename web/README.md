# Shared Brain demo frontend

Vite + React + TypeScript + Tailwind + Recharts. It replays or streams a run's event log
(contract: `src/lib/events.ts`, rebuilt from `data/research_runs/*` by `app/replay.py`).

```bash
uv run fastapi dev app/main.py     # from the repo root: serves /research-runs (port 8000)
cd web && npm install && npm run dev   # http://localhost:5173, proxies /research-runs to :8000
```

The app has two separate areas, switched by the Demo / Live toggle at the top right, and they never share data:

| Area | Routes | Data | Backend |
| --- | --- | --- | --- |
| **Demo** | `/demo`, `/demo/watch/:id`, `/demo/runs`, `/demo/run/:id`, `/demo/results` | static files in `public/samples` (exported from `data/` by `uv run python scripts/export_demo.py`) | not needed, never called |
| **Live** | `/app`, `/app/watch/:id`, `/app/runs`, `/app/run/:id`, `/app/results` | the API (`/research-runs`), live runs over SSE | required; shows "Backend not connected" without it |

- **Prompt → watch → result:** the home page (`/demo`, `/app`) shows only the prompt the research
  team starts from (`src/lib/prompt.ts`, mirrored from `app/agents/prompts.py` and the dataset card;
  `tests/test_web_prompt.py` keeps them in sync). Sending it morphs into the watch page (View
  Transitions API): a chat that narrates each generation on the left, the agents, loss chart and
  setups side by side on the right, then the result. Demo plays the recorded run in about 10 s.
- **Live starts a real run:** the prompt is editable, and sending it calls `POST /research-runs`
  (`app/launch.py`), which starts Shared Brain and No memory side by side
  (`app.orchestrator --goal <prompt> --stamp <ts>`). The prompt replaces the first paragraph of
  the researchers' system prompt (`ResearchConfig.goal`) and is saved in each run's
  `config.json`. One run at a time; logs go to `var/logs/`. Set `RESEARCH_LAUNCH_ENABLED=false`
  on a public deployment, since every run spends Crusoe and Modal credit.
- Demo data pages show a "recorded data" banner and the watch page a "Recorded replay" badge, so a replay is never mistaken for a live run.
- **Screens** (both areas): Runs, then Mission control and Memory per run (one tab per setup
  that ran side by side), and Results (setups compared within one run).
- **Replay:** paced client-side (play, pause, 1–100×, seek, jump to a generation). Long
  training waits are shortened to keep the story moving.
- **Live:** runs still being written stream over SSE (`/research-runs/:id/stream`).
- **One deployment:** `npm run build`, then FastAPI serves `web/dist` at `/`. `dist/` also works
  as a static site, with only the Demo area working.

Charts use the validated dark categorical palette (`src/lib/theme.ts`): color follows the
researcher focus or setup, scatter marks add shapes, and every chart has a legend and tooltip.
