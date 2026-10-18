// Run data access, as two separate sources with no fallback between them:
// - demo: static files exported from data/ by scripts/export_demo.py (public/samples), so the
//   demo also works as a static site and never touches the API;
// - live: the read-only telemetry API (/research-runs, app/api/routes/research.py), which
//   rebuilds events from data/ on every request, so runs in progress show up as they're written.
import { parseJsonl, type ResearchEvent, type RunInfo } from './events'

export type Source = 'demo' | 'live'

const API = '/research-runs'
const DEMO = '/samples'

async function get(url: string): Promise<Response> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status} ${url}`)
  if ((res.headers.get('content-type') ?? '').includes('text/html')) throw new Error(`no backend at ${url}`) // SPA fallback
  return res
}

// ---- demo: static recorded runs ----

const demoIndex = (): Promise<RunInfo[]> =>
  get(`${DEMO}/index.json`)
    .then((r) => r.json() as Promise<RunInfo[]>)
    .catch(() => [])

const demoCache = new Map<string, Promise<ResearchEvent[]>>()

/** Recorded runs bundled with the frontend, newest first. */
export const listDemoRuns = demoIndex

export async function getDemoRun(runId: string): Promise<RunInfo> {
  const run = (await demoIndex()).find((r) => r.run_id === runId)
  if (!run) throw new Error(`no recorded run ${runId}`)
  return run
}

export function getDemoEvents(runId: string): Promise<ResearchEvent[]> {
  if (!demoCache.has(runId)) {
    demoCache.set(runId, get(`${DEMO}/${encodeURIComponent(runId)}.jsonl`).then(async (r) => parseJsonl(await r.text())))
  }
  return demoCache.get(runId)!
}

// ---- live: the API ----

/** Runs under data/ as the API sees them now, newest first. Throws if there's no backend. */
export const listRuns = async (): Promise<RunInfo[]> => (await get(API)).json()

export async function getRun(runId: string): Promise<RunInfo> {
  return (await get(`${API}/${encodeURIComponent(runId)}`)).json()
}

/** The whole history, following `next_after` pages. */
export async function getEvents(run: RunInfo): Promise<ResearchEvent[]> {
  const out: ResearchEvent[] = []
  let after = 0
  for (;;) {
    const page = (await (await get(`${API}/${encodeURIComponent(run.run_id)}/events?after=${after}&limit=2000`)).json()) as {
      events: ResearchEvent[]
      next_after: number
      has_more: boolean
    }
    out.push(...page.events)
    if (!page.has_more) return out
    after = page.next_after
  }
}

export class StartError extends Error {
  runId?: string // set when another run is already in progress
  constructor(message: string, runId?: string) {
    super(message)
    this.runId = runId
  }
}

/** Start Shared Brain and No memory side by side, with `prompt` as the researchers' brief. */
export async function startRun(prompt: string, generations: number): Promise<string> {
  const res = await fetch(API, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt, generations }) })
  const body = await res.json().catch(() => ({}))
  if (res.ok) return body.run_id as string
  const detail = body.detail
  if (typeof detail === 'object' && detail?.run_id) throw new StartError(detail.message, detail.run_id)
  throw new StartError(typeof detail === 'string' ? detail : `${res.status} ${res.statusText}`)
}

/** Live SSE: backlog after `after`, then appends; `stream.reset` if history changed, `stream.end` when done. */
export const streamUrl = (runId: string, after: number) => `${API}/${encodeURIComponent(runId)}/stream?after=${after}`

// ---- either ----

export const runInfo = (runId: string, source: Source) => (source === 'demo' ? getDemoRun(runId) : getRun(runId))
export const runEvents = (run: RunInfo, source: Source) => (source === 'demo' ? getDemoEvents(run.run_id) : getEvents(run))
