import { bestSoFar, type RunState } from '@/lib/runState'
import { cn, loss, signedPct } from '@/lib/format'

const STEPS = ['Read memory', 'Run experiments', 'Report', 'Curator verifies', 'Update memory'] as const

function activeStep(s: RunState): number {
  const g = s.gens.at(-1)
  if (!g) return -1
  if (g.phase === 'done') return 5
  if (g.phase === 'curate') return g.memoryVersion !== undefined ? 4 : 3
  const researchers = g.agentIds.map((id) => s.agents[id]).filter((a) => a?.role === 'researcher')
  if (researchers.length && researchers.every((a) => a.status === 'reported')) return 2
  if (researchers.some((a) => a.runs > 0 || a.currentTool?.tool === 'run_code')) return 1
  return 0
}

export function GenTimeline({ state }: { state: RunState }) {
  const total = state.config?.generations ?? 0
  const current = state.gens.at(-1)
  const step = activeStep(state)
  const best = bestSoFar(state)
  const gain = state.baseline && best ? (best - state.baseline) / state.baseline : 0
  const usesMemory = state.setup === 'shared_brain' || state.setup === 'raw_log'

  return (
    <div className="flex flex-wrap items-center gap-x-8 gap-y-3 rounded-xl border border-line bg-surface px-4 py-3">
      <div className="flex items-center gap-2" aria-label="Generations">
        {Array.from({ length: total }, (_, i) => {
          const g = state.gens.find((x) => x.gen === i + 1)
          return (
            <div
              key={i}
              className={cn(
                'flex h-7 min-w-12 items-center justify-center rounded-md border px-2 text-xs font-medium',
                !g && 'border-line text-muted',
                g?.phase === 'done' && 'border-series-1/40 bg-series-1/15 text-[#86b6ef]',
                g && g.phase !== 'done' && 'pulse border-series-1 bg-series-1/25 text-ink',
              )}
            >
              Gen {i + 1}
            </div>
          )
        })}
      </div>
      <ol className="flex flex-1 items-center gap-1 text-xs" aria-label="Generation steps">
        {STEPS.map((label, i) => {
          const skip = !usesMemory && (i === 0 || i >= 3)
          return (
            <li key={label} className="flex items-center gap-1">
              <span
                className={cn(
                  'rounded px-2 py-1',
                  skip && 'text-axis line-through',
                  !skip && current && i < step && 'text-ink-2',
                  !skip && current && i === step && 'bg-series-1/20 font-semibold text-ink',
                  !skip && (!current || i > step) && 'text-muted',
                )}
              >
                {label}
              </span>
              {i < STEPS.length - 1 && <span className="text-axis">→</span>}
            </li>
          )
        })}
      </ol>
      <div className="text-right">
        <div className="text-[11px] tracking-wide text-muted uppercase">Best val loss</div>
        <div className="text-xl font-semibold tabular">
          <span className="text-ink-2">{loss(state.baseline)}</span>
          <span className="mx-1.5 text-muted">→</span>
          <span>{loss(best)}</span>
          {gain < 0 && <span className="ml-2 text-sm text-good">{signedPct(gain)}</span>}
        </div>
      </div>
    </div>
  )
}
