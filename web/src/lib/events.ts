// Research telemetry contract: mirrors app/events.py (envelope) and the `events.emit(...)`
// payloads in app/ (see docs/research-telemetry.md). Change both together.

export type Setup = 'shared_brain' | 'raw_log' | 'no_memory' | 'one_shot'
export type Role = 'researcher' | 'curator' | 'orchestrator'
export type Phase = 'candidate' | 'verification' | 'curation' | 'baseline' | 'test'
export type Tool = 'get_memory' | 'upsert_memory' | 'remove_memory' | 'run_code' | 'web_search'
export type ExpStatus = 'ok' | 'error' | 'timeout'

export interface Usage {
  prompt: number
  cached: number
  output: number
  calls?: number
}

export interface ResearchConfig {
  generations: number
  researchers: number
  researcher_runs: number
  researcher_runs_shared_brain: number
  curator_runs: number
  curator_seeds: number[]
  test_seeds: number[]
  baseline_seeds: number[]
  max_turns: number
  curator_max_turns: number
  memory_cap_tokens: number
  raw_log_cap_tokens: number
}

export interface MemoryEntryData {
  id: string
  kind: 'works' | 'failed' | 'try_next' | 'best'
  claim: string
  run_ids: string[]
  val_change: number | null
  seeds: number[]
  tested_on: number
  created_gen: number
  updated_gen: number
  active: boolean
}

export interface ResearcherBest {
  name: string
  focus: string
  best: { run_id: string; note: string; status: ExpStatus; val_loss: number | null } | null
  rank: number
}

export interface GenerationMetrics {
  gen: number
  start_best_val: number
  end_best_val: number
  best_version: number
  new_best: boolean
  best_run_id: string | null
  best_note: string
  val_gain: number
  test_losses: (number | null)[]
  test_gain: number | null
  memory_tokens: number
  runs: number
  errors: number
  timeouts: number
  researcher_best: ResearcherBest[]
  usage: { researchers: Usage; curator: Usage | Record<string, never> }
  minutes: number
}

export interface SetupResult {
  setup: Setup
  dataset: string
  l0: number
  noise_threshold: number
  best_version: number
  best_val: number
  val_gain: number
  baseline_test_losses: number[]
  best_test_losses: (number | null)[]
  // Share of test rows classified right (p >= 0.5); absent on runs scored before it was tracked.
  baseline_test_accuracies?: (number | null)[]
  best_test_accuracies?: (number | null)[]
  /** Always predicting the most common test label: the floor a model must beat. */
  majority_test_accuracy?: number | null
  fitness: number | null
  runs: number
  minutes: number
}

export interface Check {
  source: string
  run_ids: string[]
  seeds: number[]
  losses: (number | null)[]
  mean: number | null
  spread: number | null
  beats_best: boolean
  by: 'curator' | 'orchestrator'
}

/** Payload per event type (`data`). Types not listed carry no data the UI needs. */
export interface EventData {
  'run.started': { dataset: string; setups: Setup[]; research: ResearchConfig; researcher_model: string; curator_model: string; prompt?: string }
  'run.heartbeat': Record<string, never>
  'run.completed': Record<string, never>
  'run.failed': { error_type?: string }
  'run.cancelled': Record<string, never>
  'baseline.started': { seeds: number[] }
  'baseline.completed': { l0: number; threshold: number; val_losses: number[]; test_losses: number[] }
  'setup.started': { generations: number }
  'setup.completed': SetupResult
  'setup.failed': { error_type: string }
  'setup.cancelled': Record<string, never>
  'generation.started': { best_val: number; best_version: number; researchers: number }
  'generation.completed': GenerationMetrics
  'agent.started': { max_turns: number; run_budget: number; focus?: string; best_version?: number }
  'agent.turn.started': { wrap_up?: boolean }
  'agent.turn.completed': { usage: Usage; finish_reason: string | null }
  'agent.waiting': { reason: string; seconds: number }
  'agent.retry': { error_type: string; attempt: number; seconds: number }
  'agent.highlight': { source: 'tool_intent'; tool: Tool; text: string }
  'agent.completed': { report: string; usage: Usage; turns: number }
  'agent.failed': { error_type: string; usage: Usage }
  'agent.cancelled': Record<string, never>
  'tool.started': { tool: Tool; call_id: string; arguments: Record<string, string> }
  'tool.completed': { tool: Tool; call_id: string; result: Record<string, unknown>; has_error: boolean }
  'tool.failed': { tool: Tool; call_id: string; error_type: string }
  'tool.cancelled': { tool: Tool; call_id: string }
  'experiment.queued': { note: string; started_from: number }
  'experiment.started': { sandbox_mode: string }
  'sandbox.retry': { error_type: string; attempt: number }
  'experiment.completed': { status: ExpStatus; val_loss: number | null; runtime_s: number; note: string; started_from: number }
  'experiment.failed': { error_type: string }
  'experiment.cancelled': Record<string, never>
  'verification.started': { source: string; seeds: number[] }
  'verification.completed': Check
  'best.updated': { val_loss: number; version: number; run_id: string | null; note: string; verified: boolean; val_gain: number }
  'memory.updated': {
    action: 'snapshot' | 'upsert' | 'remove' | 'curation_completed'
    text: string
    tokens: number
    best_version?: number
    entry?: MemoryEntryData
    id?: string
    reason?: string
    dropped_ids?: string[]
  }
  'test.started': { seeds: number[] }
  'test.completed': { seeds: number[]; losses: (number | null)[] }
}

export type EventType = keyof EventData

interface Envelope {
  version: number
  seq: number
  run_id: string
  timestamp: string
  elapsed_ms: number
  setup: Setup | null
  gen: number | null
  agent: string | null
  role: Role | null
  phase: Phase | null
  experiment_id: string | null
  seed: number | null
  turn: number | null
  model: string | null
}

/** One recorded event; narrow on `type` to get a typed `data`. */
export type ResearchEvent = { [K in EventType]: Envelope & { type: K; data: EventData[K] } }[EventType]
export type EventOf<K extends EventType> = Extract<ResearchEvent, { type: K }>

export const TERMINAL = new Set<string>(['run.completed', 'run.failed', 'run.cancelled'])

/** Per-setup headline numbers for run lists (null until known). */
export interface SetupSummary {
  status: 'running' | 'completed' | 'stalled'
  generations_done: number
  l0: number | null
  best_val: number | null
  fitness: number | null
}

/** Run metadata: GET /research-runs (app/replay.py run_info) and public/samples/index.json. */
export interface RunInfo {
  run_id: string
  dataset: string | null
  setups: Setup[]
  status: 'unavailable' | 'running' | 'stalled' | 'completed' | 'failed' | 'cancelled'
  events_available: boolean
  last_seq: number
  duration_ms: number
  last_event_at: string | null
  replay_available: boolean
  generations: number
  started_at: string
  summary: Partial<Record<Setup, SetupSummary>>
}

export function parseJsonl(text: string): ResearchEvent[] {
  return text
    .split('\n')
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as ResearchEvent)
}
