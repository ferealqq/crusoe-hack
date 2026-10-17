// Folds the event stream into everything the UI shows. Pure: replay, seek and live
// streaming all go through `applyEvent`.
import { produce } from 'immer'
import type { EventOf, MemoryOp, RankEntry, RunConfig, RunEvent, Section, Setup, Tool } from './events'

export type AgentStatus = 'thinking' | 'tool' | 'reported'

export interface TraceItem {
  seq: number
  ts: number
  kind: 'call' | 'result' | 'message' | 'report'
  tool?: Tool
  text: string
  detail?: string
  valLoss?: number
}

export interface AgentState {
  id: string
  gen: number
  role: 'researcher' | 'curator'
  focus?: string
  model: string
  status: AgentStatus
  currentTool?: { tool: Tool; summary: string; since: number }
  lastValLoss?: number
  bestValLoss?: number
  runs: number
  searches: number
  promptTokens: number
  cachedTokens: number
  ttfts: number[]
  trace: TraceItem[]
}

export interface Verification {
  agentId: string
  claim: string
  claimed: number
  rerun: number
  held: boolean
}

export interface GenState {
  gen: number
  startTs: number
  endTs?: number
  startBest: number
  phase: 'research' | 'curate' | 'done'
  agentIds: string[]
  ranking?: RankEntry[]
  verifications: Verification[]
  memoryVersion?: number
  metrics?: EventOf<'gen.finished'>
}

export interface MemoryEntry {
  id: string
  section: Section
  text: string
  reason: string
  proof?: string
  gen: number
}

export interface MemoryVersion {
  version: number
  gen: number
  ts: number
  ops: MemoryOp[]
  entries: MemoryEntry[]
  best: { val_loss: number; summary: string }
  size: number
  cap: number
}

export interface LossPoint {
  ts: number
  gen: number
  agentId: string
  focus: string
  valLoss: number
  verify: boolean
}

export interface FeedItem {
  seq: number
  ts: number
  agentId?: string
  focus?: string
  text: string
  tone: 'neutral' | 'good' | 'bad' | 'accent'
}

export interface RunState {
  runId?: string
  setup?: Setup
  config?: RunConfig
  baseline?: number
  ts: number
  lastSeq: number
  gens: GenState[]
  agents: Record<string, AgentState>
  lossPoints: LossPoint[]
  memory: MemoryVersion[]
  feed: FeedItem[]
  eval?: EventOf<'eval.finished'>
  finished?: EventOf<'run.finished'>
}

export const emptyRun = (): RunState => ({
  ts: 0,
  lastSeq: -1,
  gens: [],
  agents: {},
  lossPoints: [],
  memory: [],
  feed: [],
})

const FEED_MAX = 300

function feed(s: RunState, e: RunEvent, text: string, tone: FeedItem['tone'] = 'neutral', agentId?: string) {
  const focus = agentId ? s.agents[agentId]?.focus ?? (s.agents[agentId]?.role === 'curator' ? 'curator' : undefined) : undefined
  s.feed.push({ seq: e.seq, ts: e.ts, agentId, focus, text, tone })
  if (s.feed.length > FEED_MAX) s.feed.splice(0, s.feed.length - FEED_MAX)
}

const genOf = (s: RunState, gen: number) => s.gens.find((g) => g.gen === gen)

export const applyEvent = (state: RunState, e: RunEvent): RunState =>
  produce(state, (s) => {
    s.ts = e.ts
    s.lastSeq = e.seq
    switch (e.t) {
      case 'run.started':
        s.runId = e.run_id
        s.setup = e.setup
        s.config = e.config
        s.baseline = e.baseline_val_loss
        feed(s, e, `Run started: ${e.config.generations} generations, ${e.config.researchers.length} researchers`, 'accent')
        break
      case 'gen.started':
        s.gens.push({ gen: e.gen, startTs: e.ts, startBest: e.start_best_val_loss, phase: 'research', agentIds: [], verifications: [] })
        feed(s, e, `Generation ${e.gen} started from best val ${e.start_best_val_loss.toFixed(4)}`, 'accent')
        break
      case 'agent.spawned': {
        s.agents[e.agent_id] = {
          id: e.agent_id,
          gen: e.gen,
          role: e.role,
          focus: e.focus,
          model: e.model,
          status: 'thinking',
          runs: 0,
          searches: 0,
          promptTokens: 0,
          cachedTokens: 0,
          ttfts: [],
          trace: [],
        }
        const g = genOf(s, e.gen)
        if (g) {
          g.agentIds.push(e.agent_id)
          if (e.role === 'curator') g.phase = 'curate'
        }
        feed(s, e, e.role === 'curator' ? 'Curator spawned' : `Researcher spawned (${e.focus})`, 'neutral', e.agent_id)
        break
      }
      case 'tool.called': {
        const a = s.agents[e.agent_id]
        if (!a) break
        a.status = 'tool'
        a.currentTool = { tool: e.tool, summary: e.summary, since: e.ts }
        if (e.tool === 'web_search') a.searches++
        a.trace.push({ seq: e.seq, ts: e.ts, kind: 'call', tool: e.tool, text: e.summary })
        feed(s, e, e.summary, 'neutral', e.agent_id)
        break
      }
      case 'tool.result': {
        const a = s.agents[e.agent_id]
        if (!a) break
        a.status = 'thinking'
        a.currentTool = undefined
        a.trace.push({ seq: e.seq, ts: e.ts, kind: 'result', tool: e.tool, text: e.summary, detail: e.detail, valLoss: e.val_loss })
        if (e.val_loss !== undefined) {
          a.runs++
          a.lastValLoss = e.val_loss
          if (a.role === 'researcher') a.bestValLoss = Math.min(a.bestValLoss ?? Infinity, e.val_loss)
          s.lossPoints.push({
            ts: e.ts,
            gen: a.gen,
            agentId: a.id,
            focus: a.focus ?? 'curator',
            valLoss: e.val_loss,
            verify: a.role === 'curator',
          })
          const start = genOf(s, a.gen)?.startBest ?? Infinity
          feed(s, e, `${e.summary}`, e.val_loss < start ? 'good' : 'bad', e.agent_id)
        }
        break
      }
      case 'llm.usage': {
        const a = s.agents[e.agent_id]
        if (!a) break
        a.promptTokens += e.prompt_tokens
        a.cachedTokens += e.cached_tokens
        a.ttfts.push(e.ttft_ms)
        break
      }
      case 'agent.message': {
        const a = s.agents[e.agent_id]
        a?.trace.push({ seq: e.seq, ts: e.ts, kind: 'message', text: e.text })
        break
      }
      case 'agent.reported': {
        const a = s.agents[e.agent_id]
        if (!a) break
        a.status = 'reported'
        a.currentTool = undefined
        a.bestValLoss = e.best_val_loss
        a.trace.push({ seq: e.seq, ts: e.ts, kind: 'report', text: e.report })
        feed(s, e, `Reported best val ${e.best_val_loss.toFixed(4)}`, 'neutral', e.agent_id)
        break
      }
      case 'curator.ranked': {
        const g = genOf(s, e.gen)
        if (g) g.ranking = e.ranking
        feed(s, e, `Curator ranked ${e.ranking.length} reports by gain`, 'accent', `g${e.gen}-curator`)
        break
      }
      case 'curator.verified': {
        const g = genOf(s, e.gen)
        g?.verifications.push({ agentId: e.agent_id, claim: e.claim, claimed: e.claimed_val_loss, rerun: e.rerun_val_loss, held: e.held })
        feed(s, e, `${e.held ? 'Verified' : 'Rejected'}: ${e.claim} (re-run ${e.rerun_val_loss.toFixed(4)})`, e.held ? 'good' : 'bad', `g${e.gen}-curator`)
        break
      }
      case 'memory.updated': {
        const prev = s.memory.at(-1)?.entries ?? []
        const entries = new Map(prev.map((m) => [m.id, m]))
        for (const op of e.ops) {
          if (op.op === 'remove') entries.delete(op.entry_id)
          else entries.set(op.entry_id, { id: op.entry_id, section: op.section, text: op.text, reason: op.reason, proof: op.proof, gen: e.gen })
        }
        s.memory.push({
          version: e.version,
          gen: e.gen,
          ts: e.ts,
          ops: e.ops,
          entries: [...entries.values()],
          best: e.best,
          size: e.size_tokens,
          cap: e.cap_tokens,
        })
        const g = genOf(s, e.gen)
        if (g) g.memoryVersion = e.version
        const n = (op: MemoryOp['op']) => e.ops.filter((o) => o.op === op).length
        feed(s, e, `Memory v${e.version}: +${n('add')} added, ${n('edit')} edited, −${n('remove')} removed (${e.size_tokens}/${e.cap_tokens} tokens)`, 'accent')
        break
      }
      case 'gen.finished': {
        const g = genOf(s, e.gen)
        if (g) {
          g.phase = 'done'
          g.endTs = e.ts
          g.metrics = e
        }
        feed(s, e, `Generation ${e.gen} done: best val ${e.best_val_loss.toFixed(4)}, ${e.waste_runs} wasted runs`, 'accent')
        break
      }
      case 'eval.finished':
        s.eval = e
        feed(s, e, `Test split: fitness ${(e.fitness * 100).toFixed(1)}% (mean gain over ${e.seeds.length} seeds)`, 'good')
        break
      case 'run.finished':
        s.finished = e
        feed(s, e, e.status === 'completed' ? 'Run complete' : `Run failed: ${e.error ?? ''}`, e.status === 'completed' ? 'accent' : 'bad')
        break
    }
  })

export const reduceAll = (events: RunEvent[]): RunState => events.reduce(applyEvent, emptyRun())

/** Best val loss reached so far, as of the current state. */
export function bestSoFar(s: RunState): number | undefined {
  const done = s.gens.filter((g) => g.metrics).map((g) => g.metrics!.best_val_loss)
  const memBest = s.memory.at(-1)?.best.val_loss
  const vals = [...done, ...(memBest !== undefined ? [memBest] : [])]
  return vals.length ? Math.min(...vals) : s.baseline
}
