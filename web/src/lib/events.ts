// Event contract between the orchestrator and this frontend.
// Mirrors app/runlog/events.py - change both together.

export type Setup = 'shared_brain' | 'raw_log' | 'no_memory' | 'one_shot'
export type Role = 'researcher' | 'curator'
export type Tool = 'get_memory' | 'upsert_memory' | 'remove_memory' | 'run_code' | 'web_search'
export type Section = 'works' | 'failed' | 'try_next'

interface Base {
  seq: number
  ts: number // seconds since run start
}

export interface RunConfig {
  generations: number
  researchers: { focus: string; model: string }[]
  curator_model: string
  budget: string
  seeds: number
  memory_cap_tokens: number
  dataset: string
}

export interface RankEntry {
  agent_id: string
  best_val_loss: number
  gain: number
  bucket: 'top' | 'middle' | 'bottom'
  tie?: boolean
}

export interface MemoryOp {
  op: 'add' | 'edit' | 'remove'
  section: Section
  entry_id: string
  text: string
  reason: string
  proof?: string
}

export interface SeedScore {
  seed: number
  test_l0: number
  test_lbest: number
  gain: number
}

export type RunEvent =
  | (Base & { t: 'run.started'; run_id: string; setup: Setup; config: RunConfig; baseline_val_loss: number })
  | (Base & { t: 'gen.started'; gen: number; start_best_val_loss: number })
  | (Base & { t: 'agent.spawned'; gen: number; agent_id: string; role: Role; focus?: string; model: string })
  | (Base & { t: 'tool.called'; agent_id: string; call_id: string; tool: Tool; summary: string })
  | (Base & {
      t: 'tool.result'
      agent_id: string
      call_id: string
      tool: Tool
      ok?: boolean
      summary: string
      val_loss?: number
      duration_ms?: number
      detail?: string
    })
  | (Base & {
      t: 'llm.usage'
      agent_id: string
      prompt_tokens: number
      cached_tokens: number
      completion_tokens: number
      ttft_ms: number
    })
  | (Base & { t: 'agent.message'; agent_id: string; text: string })
  | (Base & { t: 'agent.reported'; gen: number; agent_id: string; best_val_loss: number; report: string })
  | (Base & { t: 'curator.ranked'; gen: number; ranking: RankEntry[] })
  | (Base & {
      t: 'curator.verified'
      gen: number
      agent_id: string
      claim: string
      claimed_val_loss: number
      rerun_val_loss: number
      held: boolean
    })
  | (Base & {
      t: 'memory.updated'
      gen: number
      version: number
      ops: MemoryOp[]
      best: { val_loss: number; summary: string }
      size_tokens: number
      cap_tokens: number
    })
  | (Base & {
      t: 'gen.finished'
      gen: number
      best_val_loss: number
      waste_runs: number
      total_runs: number
      memory_tokens: number
      raw_log_tokens: number
      prompt_tokens: number
      cached_tokens: number
      mean_ttft_ms: number
    })
  | (Base & { t: 'eval.finished'; setup: Setup; seeds: SeedScore[]; fitness: number })
  | (Base & { t: 'run.finished'; status: 'completed' | 'failed'; error?: string })

export type EventOf<T extends RunEvent['t']> = Extract<RunEvent, { t: T }>

export function parseJsonl(text: string): RunEvent[] {
  return text
    .split('\n')
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as RunEvent)
}
