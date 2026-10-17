// Run data access. Uses the FastAPI backend when it's running and falls back to the
// recorded runs bundled in public/samples, so the demo also works as a static site.
import { parseJsonl, type RunEvent, type Setup } from './events'

export interface RunSummary {
  run_id: string
  source: string
  setup: Setup | null
  status: 'live' | 'completed' | 'failed'
  best_val_loss: number | null
  baseline_val_loss: number | null
  fitness: number | null
  generations: number | null
}

async function getText(url: string): Promise<string> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status} ${url}`)
  const type = res.headers.get('content-type') ?? ''
  if (type.includes('text/html')) throw new Error(`no API at ${url}`) // Vite SPA fallback
  return res.text()
}

export function summarize(runId: string, source: string, events: RunEvent[]): RunSummary {
  const s: RunSummary = { run_id: runId, source, setup: null, status: 'live', best_val_loss: null, baseline_val_loss: null, fitness: null, generations: null }
  for (const e of events) {
    if (e.t === 'run.started') {
      s.setup = e.setup
      s.baseline_val_loss = e.baseline_val_loss
      s.generations = e.config.generations
    } else if (e.t === 'gen.finished') s.best_val_loss = e.best_val_loss
    else if (e.t === 'eval.finished') s.fitness = e.fitness
    else if (e.t === 'run.finished') s.status = e.status
  }
  return s
}

export async function listRuns(): Promise<RunSummary[]> {
  try {
    return JSON.parse(await getText('/api/runs')) as RunSummary[]
  } catch {
    const ids = JSON.parse(await getText('/samples/index.json')) as string[]
    return Promise.all(ids.map(async (id) => summarize(id, 'static', await getEvents(id))))
  }
}

export async function getEvents(runId: string): Promise<RunEvent[]> {
  try {
    return parseJsonl(await getText(`/api/runs/${encodeURIComponent(runId)}/events`))
  } catch {
    return parseJsonl(await getText(`/samples/${encodeURIComponent(runId)}.jsonl`))
  }
}

export function streamUrl(runId: string, after: number): string {
  return `/api/runs/${encodeURIComponent(runId)}/stream?after=${after}`
}
