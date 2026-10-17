# Shared Brain demo frontend

Vite + React + TypeScript + Tailwind + Recharts. It replays or streams a run's event log
(contract: `src/lib/events.ts`, mirrored by `app/runlog/events.py`).

```bash
uv run fastapi dev app/main.py     # from the repo root: serves /api/runs (port 8000)
cd web && npm install && npm run dev   # http://localhost:5173, proxies /api to :8000
```

- **Screens:** Runs (`/`), Mission control and Memory (`/run/:id`), Results (`/compare`).
- **Replay:** recorded runs are paced client-side (play, pause, 1–20×, seek, jump to a
  generation). Long training waits are shortened to keep the story moving.
- **Live:** runs being written by `RunLogger` stream over SSE (`/api/runs/:id/stream`).
  Rehearse with `uv run python scripts/simulate_live_run.py`.
- **No backend?** `npm run dev`/`build` copy the recorded samples into `public/samples`, and the
  app falls back to them, so `dist/` also works as a static site.
- **One deployment:** `npm run build`, then FastAPI serves `web/dist` at `/`.

Charts use the validated dark categorical palette (`src/lib/theme.ts`): color follows the
researcher focus or setup, scatter marks add shapes, and every chart has a legend and tooltip.
