import { BookOpen, CheckCircle2, FlaskConical, Loader2, Search, ShieldCheck } from 'lucide-react'
import type { Tool } from '@/lib/events'
import { cn, loss, pct, tokens } from '@/lib/format'
import type { AgentState, RunState } from '@/lib/runState'
import { FOCUS_ORDER, FOCUS_SHAPE, focusColor } from '@/lib/theme'
import { Swatch } from './ui'

const TOOL_ICON: Record<Tool, typeof Search> = {
  get_memory: BookOpen,
  web_search: Search,
  run_code: FlaskConical,
  upsert_memory: ShieldCheck,
  remove_memory: ShieldCheck,
}

function Activity({ a }: { a: AgentState }) {
  if (a.status === 'reported')
    return (
      <span className="flex items-center gap-1.5 text-good">
        <CheckCircle2 size={14} /> Reported
      </span>
    )
  if (a.currentTool) {
    const Icon = TOOL_ICON[a.currentTool.tool]
    return (
      <span className="flex min-w-0 items-center gap-1.5 text-ink">
        <Icon size={14} className="shrink-0 text-[#86b6ef]" />
        <span className="truncate font-mono text-xs">{a.currentTool.summary}</span>
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1.5 text-ink-2">
      <Loader2 size={14} className="animate-spin" /> Thinking
    </span>
  )
}

export function AgentCard({ a, startBest, onOpen }: { a: AgentState; startBest?: number; onOpen: (id: string) => void }) {
  const label = a.role === 'curator' ? 'Curator' : a.focus
  const color = a.role === 'curator' ? '#f0efec' : focusColor(a.focus)
  const cacheRate = a.promptTokens ? a.cachedTokens / a.promptTokens : undefined
  const improved = a.bestValLoss !== undefined && startBest !== undefined && a.bestValLoss < startBest
  const active = a.status !== 'reported'

  return (
    <button
      onClick={() => onOpen(a.id)}
      className={cn(
        'flex flex-col gap-3 rounded-xl border bg-surface p-3.5 text-left transition-colors hover:border-[#5598e7]/60',
        active ? 'border-line' : 'border-line opacity-80',
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2 text-sm font-semibold capitalize">
          <Swatch color={color} shape={a.focus ? FOCUS_SHAPE[a.focus] : 'circle'} />
          {label}
        </span>
        <span className="text-[10px] text-muted">{a.model.split('/').at(-1)}</span>
      </div>
      <div className="h-5 text-sm">
        <Activity a={a} />
      </div>
      <div className="grid grid-cols-3 gap-2 border-t border-line pt-2.5 text-xs">
        <div>
          <div className="text-muted">{a.role === 'curator' ? 'Re-runs' : 'Best val'}</div>
          <div className={cn('font-semibold tabular', improved ? 'text-good' : 'text-ink')}>
            {a.role === 'curator' ? a.runs : loss(a.bestValLoss)}
          </div>
        </div>
        <div>
          <div className="text-muted">Runs · web</div>
          <div className="font-semibold text-ink tabular">
            {a.runs} · {a.searches}
          </div>
        </div>
        <div title={`${tokens(a.cachedTokens)} of ${tokens(a.promptTokens)} prompt tokens served from Crusoe's KV cache`}>
          <div className="text-muted">Cached</div>
          <div className="font-semibold text-ink tabular">{pct(cacheRate, 0)}</div>
        </div>
      </div>
    </button>
  )
}

export function AgentGrid({ state, onOpen }: { state: RunState; onOpen: (id: string) => void }) {
  const g = state.gens.at(-1)
  if (!g) return <div className="rounded-xl border border-dashed border-line p-8 text-center text-sm text-muted">Press play to start the run.</div>
  const agents = g.agentIds.map((id) => state.agents[id]).filter(Boolean)
  // fixed order, so a focus keeps its slot across generations
  const rank = (f?: string) => FOCUS_ORDER.indexOf(f as (typeof FOCUS_ORDER)[number])
  const researchers = agents.filter((a) => a.role === 'researcher').sort((x, y) => rank(x.focus) - rank(y.focus))
  const curator = agents.find((a) => a.role === 'curator')
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h2 className="text-xs font-semibold tracking-wide text-ink-2 uppercase">Generation {g.gen} agents</h2>
        <span className="text-xs text-muted">Click an agent for its full trace</span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {researchers.map((a) => (
          <AgentCard key={a.id} a={a} startBest={g.startBest} onOpen={onOpen} />
        ))}
      </div>
      {curator && <AgentCard a={curator} onOpen={onOpen} />}
    </div>
  )
}
