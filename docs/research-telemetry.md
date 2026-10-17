# Research telemetry: frontend integration

The backend records public progress in `DATA_DIR/runs/<run_id>/events.jsonl`.
The CLI is the single writer; FastAPI serves it read-only. No database, bucket,
inference, or sandbox calls are made by the telemetry endpoints or during replay.
No frontend is included in this change.

## Start recording

Start the API and research CLI in separate terminals, in the same checkout with
the same `DATA_DIR` (absolute paths are safest if working directories differ):

```sh
uv run fastapi dev app/main.py
```

```sh
uv run python -m app.orchestrator --dataset churn --setup shared_brain --researchers 2
```

Every new CLI run automatically records events. Use `--setup all` to record setups
running in parallel. Recording does not add model calls or alter agent budgets.
The API and CLI must share the run directory; a separately hosted API needs a
shared filesystem. These endpoints have no authentication, like the existing API:
keep the demo private or put authentication in front before public deployment.
The existing application still imports the legacy database routes at startup,
so `DATABASE_URL` must be syntactically valid even though telemetry never connects
to Postgres. If your `.env` has an empty URL, prefix the API/CLI commands with
`DATABASE_URL=postgresql://localhost/crusoe_hack` for a file-only demo.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /research-runs` | Discover saved and active runs, newest folder first |
| `GET /research-runs/{id}` | Public run metadata, status, last sequence, recorded duration |
| `GET /research-runs/{id}/events?after=0&limit=500&until=1000` | Paginated history; `until` is optional and inclusive |
| `GET /research-runs/{id}/state` | Current reduced dashboard state |
| `GET /research-runs/{id}/state?at=42` | State using **only** events through sequence 42; `at=0` is empty |
| `GET /research-runs/{id}/stream?after=42` | Follow live events over SSE |
| `GET /research-runs/{id}/stream?mode=replay&after=0&speed=60&max_gap=1` | Replay saved events, 60× speed, at most 1 second per inter-event delay |

History returns `{events, next_after, has_more, last_seq}`. `after` is exclusive;
`seq` is contiguous and unique **within a run**, not across runs. `limit` is 1–2000.
Run metadata's `replay_available` means a complete recording exists. Partial
recordings can still be replayed, but must be identified as incomplete by the UI.

Unknown/unsafe run IDs return 404. Invalid cursors/stream parameters return 422.
Older runs without event recordings have status `unavailable` and streaming
returns 409: their historical files cannot reconstruct faithful activity timing.
The API never invents events or silently pretends those runs were recorded.

Run status is `running`, `completed`, `failed`, or `cancelled`. A nonterminal log
with no new event for 30 seconds is reported as `stalled` in metadata, not completed.
The CLI records a runner heartbeat every 5 seconds, including during slow requests.
SIGINT/SIGTERM trigger task cancellation and sandbox cleanup, then `run.cancelled`.
A hard kill/power loss cannot record cancellation; the stale status handles that.

## SSE protocol

Each research event uses the same envelope in history, live mode, and replay:

```text
id: 42
event: research
data: {"version":1,"seq":42,"run_id":"...","timestamp":"2026-10-18T10:00:00Z","elapsed_ms":12500,"type":"experiment.completed","setup":"shared_brain","gen":1,"agent":"researcher-2","role":"researcher","phase":"candidate","experiment_id":"e3","seed":0,"turn":2,"model":"...","data":{"status":"ok","val_loss":0.32,"runtime_s":8.1,"note":"Try gradient boosting","started_from":0}}

```

Nullable context fields are absent in meaning, not necessarily omitted from JSON.
Use `(run_id, setup, gen, agent)` to identify a short-lived agent and
`(run_id, setup, experiment_id)` to identify an experiment. Different setups can
each have `e1`. Baseline events have no setup. Timestamps are UTC; pacing uses the
monotonic `elapsed_ms` value rather than wall-clock differences.

Transport-only events:

- `stream.open`: `{mode, after, end_seq}`. Replay freezes `end_seq` on connection;
  it does not follow subsequently appended events. Live has `end_seq: null`.
- `stream.end`: `{mode, cursor}`. **Close EventSource on this event**, otherwise
  browsers automatically reconnect after the response ends.
- `: keep-alive` comments while idle. These do not advance the cursor and do not
  mean the research process is healthy; `run.heartbeat` is the runner heartbeat.

Research events use the named `research` SSE event, not the default `message`:

```js
const feed = new EventSource(`/research-runs/${id}/stream?after=${state.seq}`);
feed.addEventListener('research', ({data}) => {
  const event = JSON.parse(data);
  if (event.seq <= state.seq) return; // reconnects must be idempotent
  // Update cards/charts/feed using event.type and event.data.
  state.seq = event.seq;
});
feed.addEventListener('stream.end', () => feed.close());
```

Reconnect accepts `Last-Event-ID`; if both it and `after` are supplied, the higher
cursor wins. Live streams deliver backlog before following appends; file checks
occur every 250 ms. Split/incomplete trailing writes are retried, never parsed as
events. Slow readers use bounded batches rather than an unbounded in-memory queue.

### Live initialization without a race

1. Fetch `/state`.
2. Render that state.
3. Connect `/stream?after=<state.seq>`.

Events appended between steps 1 and 3 are delivered as backlog. Deduplicate by
sequence number. No final results file needs to be fetched separately.

### Replay, pause, seek, restart

1. Choose a completed run from `/research-runs`; visibly label the UI **Recorded replay**.
2. Initialize with `/state?at=0`, then open `stream?mode=replay&speed=60&max_gap=1`.
3. Pause by closing the stream and retaining the last **rendered** sequence.
4. Resume with a new stream using `after=<rendered sequence>` and the same pacing.
5. Seek by closing the old stream, loading `/state?at=<target sequence>`, then
   opening replay after that returned sequence. Ignore late callbacks from the old
   connection. Restart is seek to zero. Fetch history to map timeline times to sequences.

`speed` ranges from 0.1 to 1000. Default 1 preserves original timing. Optional
`max_gap` (0–60 seconds) caps the wall-clock delay **after** applying speed; 0
delivers immediately. A two-hour recording takes about two minutes at 60×.
Pause/resume at event boundaries restarts the remaining inter-event delay.
State replay is deterministic; scheduling and browser render timing are approximate.

Do not initialize replay with `/state` without `at`, or render current/final
metadata as if it had already happened. `/state?at=N` cannot reveal future scores,
memory updates, or best versions. There is no shared mutable replay session:
multiple viewers can independently pause/seek and live research keeps running.

## Event types and UI mapping

| Types | Important data / use |
| --- | --- |
| `run.started` | Dataset, setups, budgets, researcher/curator models |
| `run.heartbeat` | Runner liveness |
| `run.completed`, `run.failed`, `run.cancelled` | Terminal lifecycle; failures expose error type only |
| `baseline.started`, `baseline.completed` | Seeds; then `l0`, noise `threshold`, baseline losses |
| `setup.started`, `setup.completed`, `setup.failed`, `setup.cancelled` | Setup lifecycle; completion includes summary metrics without code |
| `generation.started`, `generation.completed` | Generation number, starting best; then metrics/rankings/usage |
| `agent.started` | Focus, model (envelope), run budget, turn budget |
| `agent.turn.started`, `agent.turn.completed` | Thinking indicator; then token usage and finish reason |
| `agent.waiting`, `agent.retry` | Rate-limit spacing or retry, wait seconds, error type, attempt |
| `agent.highlight` | Short `text` from experiment note or curator claim; `source: tool_intent` |
| `agent.completed` | Public final `report` (up to 2,000 characters), turns, usage |
| `agent.failed`, `agent.cancelled` | Agent lifecycle |
| `tool.started`, `tool.completed`, `tool.failed`, `tool.cancelled` | Tool name, call ID, safe argument/result subset, budget remaining |
| `experiment.queued` | Note, starting best version; waiting for provisioning/capacity |
| `experiment.started` | Sandbox is provisioned and code is about to execute |
| `sandbox.retry` | Infrastructure retry (not a failed model experiment) |
| `experiment.completed` | `status` (`ok`/`error`/`timeout`), `val_loss`, runtime, note |
| `experiment.failed`, `experiment.cancelled` | Infrastructure failure/cancellation, not a validation score |
| `verification.started`, `verification.completed` | Source run, seeds; then losses, mean, spread, `beats_best` |
| `best.updated` | Accepted version/loss/gain; `verified` distinguishes curator verification from raw selection |
| `memory.updated` | Current rendered `text`, `tokens`, action; individual edits also include entry/removal ID |
| `test.started`, `test.completed` | Orchestrator-only test scoring; never sent back into agent conversations |

Distinguish `phase: candidate`, `verification`, `baseline`, and `test`. Candidate
validation results can update an experiment chart immediately, but **only
`best.updated` changes the accepted best**. In Shared Brain this follows multi-seed
verification. `status != ok` or null losses must not be plotted as zero.

Highlights are declared tool intentions, not verified findings or raw chain of
thought. Render them as plain text, not HTML. No token-by-token reasoning stream,
extra summary-model calls, prompts, code, dataset rows/labels, raw tool stdout,
or credentials are exposed. This is observability only: no run-start/stop API,
model control, frontend implementation, or synthetic completed demo is included.
