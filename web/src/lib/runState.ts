// Folds research events into everything the UI shows. Pure and prefix-only (like the
// backend's /state?at=N): replay, seek and live streaming all go through `mutate`.
import { produce, type Draft } from 'immer'
import type { Check, EventData, EventOf, GenerationMetrics, ResearchEvent, Setup, SetupResult, Tool, Usage } from './events'

export type AgentStatus = 'starting' | 'thinking' | 'tool' | 'validating' | 'waiting' | 'completed' | 'failed' | 'cancelled'

export interface TraceItem {
  seq: number
  ts: number
  kind: 'turn' | 'tool' | 'result' | 'experiment' | 'wait' | 'report' | 'error'
  tool?: Tool
  text: string
  detail?: string
  valLoss?: number | null
}

export interface AgentState {
  key: string // `${gen}:${name}`: agent names repeat every generation
  name: string
  gen: number
  role: 'researcher' | 'curator'
  focus?: string
  model: string
  status: AgentStatus
  currentTool?: { tool: Tool; summary: string }
  highlight?: string
  turn: number
  maxTurns: number
  runBudget: number
  runsLeft?: number
  runs: number // experiments finished
  searches: number
  usage: Usage
  bestValLoss?: number
  wait?: string
  trace: TraceItem[]
  report?: string
}

export interface Verification extends Check {
  seq: number
  ts: number
}

/** Generation steps in order; a setup without a curator skips the curator ones. */
export const PHASES = ['research', 'reported', 'curate', 'verify', 'memory', 'pick', 'test', 'done'] as const
export type GenPhase = (typeof PHASES)[number]

export interface GenState {
  gen: number
  startTs: number
  endTs?: number
  startBest: number
  startVersion: number
  phase: GenPhase
  agentKeys: string[]
  verifications: Verification[]
  memoryEdits: number
  metrics?: GenerationMetrics
}

export interface MemoryVersion {
  seq: number
  ts: number
  gen: number
  action: EventData['memory.updated']['action']
  text: string
  tokens: number
  changedId?: string
  removed?: { id: string; reason: string }[]
}

export interface LossPoint {
  ts: number
  gen: number
  agentKey: string
  focus: string
  valLoss: number
  verify: boolean
  runId: string
  note: string
}

export interface BestPoint {
  ts: number
  val: number
  version: number
  verified: boolean
  note: string
  runId: string | null
}

export interface FeedItem {
  seq: number
  ts: number
  focus?: string
  text: string
  tone: 'neutral' | 'good' | 'bad' | 'accent'
}

export interface SetupState {
  setup: Setup
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  generations: number
  gens: GenState[]
  agents: Record<string, AgentState>
  lossPoints: LossPoint[]
  best: BestPoint[]
  memory: MemoryVersion[]
  feed: FeedItem[]
  tests: { ts: number; gen: number | null; losses: (number | null)[] }[]
  result?: SetupResult
}

export interface RunState {
  runId?: string
  config?: EventData['run.started']
  baseline?: EventData['baseline.completed']
  baselineRunning: boolean
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  ts: number // seconds since run start (monotonic elapsed_ms)
  lastSeq: number
  lastTimestamp?: string
  setups: Partial<Record<Setup, SetupState>>
  feed: FeedItem[] // run-level: baseline, lifecycle
}

export const emptyRun = (): RunState => ({ baselineRunning: false, status: 'pending', ts: 0, lastSeq: 0, setups: {}, feed: [] })

const FEED_MAX = 300
const fmt = (v?: number | null) => (v === undefined || v === null ? '—' : v.toFixed(5))

function push(feed: FeedItem[], e: ResearchEvent, text: string, tone: FeedItem['tone'] = 'neutral', focus?: string) {
  feed.push({ seq: e.seq, ts: e.elapsed_ms / 1000, focus, text, tone })
  if (feed.length > FEED_MAX) feed.splice(0, feed.length - FEED_MAX)
}

function setupOf(s: Draft<RunState>, setup: Setup): Draft<SetupState> {
  return (s.setups[setup] ??= {
    setup,
    status: 'pending',
    generations: s.config?.research.generations ?? 0,
    gens: [],
    agents: {},
    lossPoints: [],
    best: [],
    memory: [],
    feed: [],
    tests: [],
  })
}

const advance = (g: Draft<GenState>, phase: GenPhase) => {
  if (PHASES.indexOf(phase) > PHASES.indexOf(g.phase)) g.phase = phase
}

export const agentLabel = (a: { role: string; focus?: string }) => (a.role === 'curator' ? 'curator' : a.focus)

function toolSummary(tool: Tool, args: Record<string, string>): string {
  switch (tool) {
    case 'run_code':
      return args.run_id ? `re-check ${args.run_id}` : args.note || 'run train.py'
    case 'web_search':
      return `search: ${args.query ?? ''}`
    case 'get_memory':
      return 'reading memory'
    case 'upsert_memory':
      return `${args.id ? `edit ${args.id}` : `add ${args.kind ?? ''}`}: ${args.claim ?? ''}`
    case 'remove_memory':
      return `remove ${args.id ?? ''}: ${args.reason ?? ''}`
  }
}

function mutate(s: Draft<RunState>, e: ResearchEvent) {
  s.ts = e.elapsed_ms / 1000
  s.lastSeq = e.seq
  s.lastTimestamp = e.timestamp
  s.runId = e.run_id

  // Run-level events
  switch (e.type) {
    case 'run.started':
      s.status = 'running'
      s.config = e.data
      for (const setup of e.data.setups) setupOf(s, setup)
      push(s.feed, e, `Run started: ${e.data.dataset}, ${e.data.setups.length} setups, ${e.data.research.generations} generations`, 'accent')
      return
    case 'baseline.started':
      s.baselineRunning = true
      push(s.feed, e, `Measuring the starting train.py on seeds ${e.data.seeds.join(', ')}`)
      return
    case 'baseline.completed':
      s.baselineRunning = false
      s.baseline = e.data
      push(s.feed, e, `Baseline L0 ${fmt(e.data.l0)}, noise threshold ${fmt(e.data.threshold)}`, 'accent')
      return
    case 'run.completed':
    case 'run.failed':
    case 'run.cancelled':
      s.status = e.type === 'run.completed' ? 'completed' : e.type === 'run.failed' ? 'failed' : 'cancelled'
      push(s.feed, e, e.type === 'run.failed' ? `Run failed${e.data.error_type ? `: ${e.data.error_type}` : ''}` : `Run ${s.status}`, e.type === 'run.completed' ? 'accent' : 'bad')
      return
    case 'run.heartbeat':
      return
  }
  if (!e.setup) return
  const st = setupOf(s, e.setup)
  const gen = e.gen !== null ? st.gens.find((g) => g.gen === e.gen) : undefined
  const key = e.agent && e.gen !== null ? `${e.gen}:${e.agent}` : undefined
  const agent = key ? st.agents[key] : undefined
  const focus = agent ? agentLabel(agent) : undefined
  const ts = e.elapsed_ms / 1000
  const trace = (item: Omit<TraceItem, 'seq' | 'ts'>) => agent?.trace.push({ seq: e.seq, ts, ...item })

  switch (e.type) {
    case 'setup.started':
      st.status = 'running'
      st.generations = e.data.generations
      break
    case 'setup.completed':
      st.status = 'completed'
      st.result = e.data
      push(st.feed, e, `Setup done: test fitness ${e.data.fitness === null ? '—' : `${(e.data.fitness * 100).toFixed(1)}%`}`, 'good')
      break
    case 'setup.failed':
      st.status = 'failed'
      push(st.feed, e, `Setup failed: ${e.data.error_type}`, 'bad')
      break
    case 'setup.cancelled':
      st.status = 'cancelled'
      push(st.feed, e, 'Setup cancelled', 'bad')
      break

    case 'generation.started':
      if (e.gen === null) break
      st.gens.push({ gen: e.gen, startTs: ts, startBest: e.data.best_val, startVersion: e.data.best_version, phase: 'research', agentKeys: [], verifications: [], memoryEdits: 0 })
      push(st.feed, e, `Generation ${e.gen} started from best v${e.data.best_version} (val ${fmt(e.data.best_val)})`, 'accent')
      break
    case 'generation.completed':
      if (gen) {
        gen.phase = 'done'
        gen.endTs = ts
        gen.metrics = e.data
      }
      push(st.feed, e, `Generation ${e.data.gen} done: best val ${fmt(e.data.end_best_val)} (${(e.data.val_gain * 100).toFixed(1)}% vs L0), ${e.data.runs} runs, ${e.data.errors + e.data.timeouts} failed`, 'accent')
      break

    case 'agent.started': {
      if (!key || e.gen === null || (e.role !== 'researcher' && e.role !== 'curator')) break
      st.agents[key] = {
        key,
        name: e.agent!,
        gen: e.gen,
        role: e.role,
        focus: e.data.focus,
        model: e.model ?? '',
        status: 'starting',
        turn: 0,
        maxTurns: e.data.max_turns,
        runBudget: e.data.run_budget,
        runs: 0,
        searches: 0,
        usage: { prompt: 0, cached: 0, output: 0 },
        trace: [],
      }
      gen?.agentKeys.push(key)
      if (gen && e.role === 'curator') advance(gen, 'curate')
      push(st.feed, e, e.role === 'curator' ? `Curator started (${e.data.run_budget} re-check runs)` : `Researcher started: ${e.data.focus}, ${e.data.run_budget} runs`, 'neutral', agentLabel(st.agents[key]))
      break
    }
    case 'agent.turn.started':
      if (!agent) break
      agent.status = 'thinking'
      agent.turn = e.turn ?? agent.turn
      agent.wait = undefined
      if (e.data.wrap_up) trace({ kind: 'turn', text: 'Out of turns: writing the report' })
      break
    case 'agent.turn.completed':
      if (!agent) break
      agent.usage.prompt += e.data.usage.prompt
      agent.usage.cached += e.data.usage.cached
      agent.usage.output += e.data.usage.output
      trace({ kind: 'turn', text: `Turn ${e.turn ?? ''}: ${e.data.usage.prompt} prompt tokens (${e.data.usage.cached} cached), ${e.data.usage.output} out · ${e.data.finish_reason ?? ''}` })
      break
    case 'agent.waiting':
    case 'agent.retry':
      if (!agent) break
      agent.status = 'waiting'
      agent.wait = e.type === 'agent.retry' ? `retry ${e.data.attempt} after ${e.data.error_type}` : 'rate-limit spacing'
      trace({ kind: 'wait', text: `${agent.wait} (${e.data.seconds.toFixed(1)} s)` })
      if (e.type === 'agent.retry') push(st.feed, e, `Retrying after ${e.data.error_type} (attempt ${e.data.attempt})`, 'bad', focus)
      break
    case 'agent.highlight':
      if (agent) agent.highlight = e.data.text
      break
    case 'agent.completed':
    case 'agent.failed':
    case 'agent.cancelled':
      if (!agent) break
      agent.status = e.type === 'agent.completed' ? 'completed' : e.type === 'agent.failed' ? 'failed' : 'cancelled'
      agent.currentTool = undefined
      agent.wait = undefined
      if (e.type === 'agent.completed') {
        agent.report = e.data.report
        trace({ kind: 'report', text: e.data.report })
        push(st.feed, e, agent.role === 'curator' ? 'Curator finished' : `Reported, best val ${fmt(agent.bestValLoss)}`, 'neutral', focus)
      } else {
        trace({ kind: 'error', text: e.type === 'agent.failed' ? `Stopped early: ${e.data.error_type}` : 'Cancelled' })
        push(st.feed, e, e.type === 'agent.failed' ? `Agent failed: ${e.data.error_type}` : 'Agent cancelled', 'bad', focus)
      }
      if (gen && agent.role === 'researcher') {
        const researchers = gen.agentKeys.map((k) => st.agents[k]).filter((a) => a?.role === 'researcher')
        if (researchers.every((a) => a.status === 'completed' || a.status === 'failed' || a.status === 'cancelled')) advance(gen, 'reported')
      }
      break

    case 'tool.started': {
      if (!agent) break
      const summary = toolSummary(e.data.tool, e.data.arguments)
      agent.status = 'tool'
      agent.currentTool = { tool: e.data.tool, summary }
      if (e.data.tool === 'web_search') {
        agent.searches++
        push(st.feed, e, summary, 'neutral', focus)
      }
      trace({ kind: 'tool', tool: e.data.tool, text: summary })
      break
    }
    case 'tool.completed':
    case 'tool.failed':
    case 'tool.cancelled': {
      if (!agent) break
      agent.currentTool = undefined
      agent.status = 'thinking'
      if (e.type === 'tool.completed') {
        const r = e.data.result
        if (typeof r.runs_left === 'number') agent.runsLeft = r.runs_left
        const parts = Object.entries(r).map(([k, v]) => `${k}: ${typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(5) : JSON.stringify(v)}`)
        trace({ kind: 'result', tool: e.data.tool, text: e.data.has_error ? 'error' : parts.join(' · ') || 'ok' })
      } else {
        trace({ kind: 'error', tool: e.data.tool, text: e.type === 'tool.failed' ? `${e.data.tool} failed: ${e.data.error_type}` : `${e.data.tool} cancelled` })
      }
      break
    }

    case 'experiment.queued':
    case 'experiment.started':
      if (agent) agent.status = 'validating'
      break
    case 'experiment.completed': {
      const { status, val_loss, note, runtime_s } = e.data
      if (agent) {
        agent.runs++
        agent.status = 'thinking'
        if (val_loss !== null && e.phase === 'candidate') agent.bestValLoss = Math.min(agent.bestValLoss ?? Infinity, val_loss)
      }
      trace({ kind: 'experiment', text: `${e.experiment_id} ${status}${val_loss !== null ? `: val ${fmt(val_loss)}` : ''} in ${runtime_s.toFixed(0)} s · ${note}`, valLoss: val_loss })
      // Failed runs have no loss: never plot them as zero.
      if (val_loss !== null && status === 'ok' && e.gen !== null) {
        st.lossPoints.push({ ts, gen: e.gen, agentKey: key ?? '', focus: focus ?? 'curator', valLoss: val_loss, verify: e.phase === 'verification', runId: e.experiment_id ?? '', note })
      }
      if (e.phase === 'candidate') {
        const start = gen?.startBest ?? Infinity
        if (val_loss === null) push(st.feed, e, `${e.experiment_id} ${status}: ${note}`, 'bad', focus)
        else push(st.feed, e, `${e.experiment_id} val ${fmt(val_loss)}: ${note}`, val_loss < start ? 'good' : 'neutral', focus)
      }
      break
    }
    case 'experiment.failed':
      trace({ kind: 'error', text: `${e.experiment_id} infrastructure failure: ${e.data.error_type}` })
      break
    case 'sandbox.retry':
      trace({ kind: 'wait', text: `sandbox retry ${e.data.attempt} after ${e.data.error_type}` })
      break

    case 'verification.started':
      if (gen) advance(gen, 'verify')
      push(st.feed, e, `Re-checking ${e.data.source} on seeds ${e.data.seeds.join(', ')}`, 'neutral', 'curator')
      break
    case 'verification.completed':
      gen?.verifications.push({ ...e.data, seq: e.seq, ts })
      push(
        st.feed,
        e,
        `${e.data.source} ${e.data.beats_best ? 'holds' : 'did not hold'}: mean ${fmt(e.data.mean)}${e.data.spread !== null ? ` ± ${e.data.spread.toFixed(5)}` : ''}${e.data.by === 'orchestrator' ? ' (orchestrator fallback)' : ''}`,
        e.data.beats_best ? 'good' : 'bad',
        'curator',
      )
      break

    case 'best.updated':
      st.best.push({ ts, val: e.data.val_loss, version: e.data.version, verified: e.data.verified, note: e.data.note, runId: e.data.run_id })
      if (gen) advance(gen, 'pick')
      push(st.feed, e, `New best v${e.data.version}: val ${fmt(e.data.val_loss)} (${(e.data.val_gain * 100).toFixed(1)}% vs L0), ${e.data.verified ? 'verified on 3 seeds' : 'single seed, unverified'}`, 'good')
      break

    case 'memory.updated': {
      const d = e.data
      const prev = st.memory.at(-1)
      // Snapshots repeat the curated text; keep them only when something changed.
      if (d.action === 'snapshot' && prev && prev.text === d.text) break
      const removed =
        d.action === 'remove' && d.id
          ? [{ id: d.id, reason: d.reason ?? '' }]
          : d.action === 'curation_completed' && d.dropped_ids?.length
            ? d.dropped_ids.map((id) => ({ id, reason: 'dropped to stay under the token cap' }))
            : undefined
      st.memory.push({ seq: e.seq, ts, gen: e.gen ?? 0, action: d.action, text: d.text, tokens: d.tokens, changedId: d.entry?.id, removed })
      if (d.action === 'upsert' || d.action === 'remove') {
        if (gen) {
          gen.memoryEdits++
          advance(gen, 'memory')
        }
        push(st.feed, e, d.action === 'upsert' ? `Memory ${d.entry?.id} (${d.entry?.kind}): ${d.entry?.claim}` : `Memory − ${d.id}: ${d.reason}`, 'accent', 'curator')
      } else if (d.action === 'curation_completed') {
        push(st.feed, e, `Memory curated: ${d.tokens} tokens${d.dropped_ids?.length ? `, dropped ${d.dropped_ids.join(', ')}` : ''}`, 'accent', 'curator')
      }
      break
    }

    case 'test.started':
      if (gen) advance(gen, 'test')
      break
    case 'test.completed': {
      st.tests.push({ ts, gen: e.gen, losses: e.data.losses })
      const ok = e.data.losses.filter((x): x is number => x !== null)
      if (ok.length) push(st.feed, e, `Test split (orchestrator only): mean loss ${fmt(ok.reduce((a, b) => a + b, 0) / ok.length)}`)
      break
    }
  }
}

export const applyEvents = (state: RunState, events: ResearchEvent[]): RunState =>
  events.length ? produce(state, (d) => events.forEach((e) => mutate(d, e))) : state
export const applyEvent = (state: RunState, e: ResearchEvent) => applyEvents(state, [e])
export const reduceAll = (events: ResearchEvent[]): RunState => applyEvents(emptyRun(), events)

/** The accepted best as of now: only best.updated changes it. */
export const currentBest = (run: RunState, st?: SetupState): number | undefined => st?.best.at(-1)?.val ?? run.baseline?.l0

export const usesMemory = (s?: Setup) => s === 'shared_brain' || s === 'raw_log'

export type { EventOf }
