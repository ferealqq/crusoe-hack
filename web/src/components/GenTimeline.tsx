import { cn, loss, signedPct } from '@/lib/format'
import { currentBest, usesMemory, type GenPhase, type RunState, type SetupState } from '@/lib/runState'

// Generation steps and the reducer phases each one covers.
const STEPS: { label: string; phases: GenPhase[]; memory?: boolean }[] = [
  { label: 'Run experiments', phases: ['research'] },
  { label: 'Report', phases: ['reported'] },
  { label: 'Curator verifies', phases: ['curate', 'verify'], memory: true },
  { label: 'Update memory', phases: ['memory'], memory: true },
  { label: 'Pick best', phases: ['pick', 'test'] },
]

export function GenTimeline({ run, st }: { run: RunState; st: SetupState }) {
  const current = st.gens.at(-1)
  const phase = current?.phase
  const step = !phase ? -1 : phase === 'done' ? STEPS.length : STEPS.findIndex((s) => s.phases.includes(phase))
  const best = currentBest(run, st)
  const l0 = run.baseline?.l0
  const gain = l0 && best ? (best - l0) / l0 : 0
  const memory = usesMemory(st.setup)

  return (
    <div className="flex flex-wrap items-center gap-x-8 gap-y-3 rounded-xl border border-line bg-surface px-4 py-3">
      <div className="flex items-center gap-2" aria-label="Generations">
        {Array.from({ length: st.generations }, (_, i) => {
          const g = st.gens.find((x) => x.gen === i + 1)
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
        {run.baselineRunning && <span className="text-xs text-muted">measuring baseline…</span>}
      </div>
      <ol className="flex flex-1 items-center gap-1 text-xs" aria-label="Generation steps">
        {STEPS.map((s, i) => {
          const skip = s.memory && st.setup !== 'shared_brain'
          return (
            <li key={s.label} className="flex items-center gap-1">
              <span
                className={cn(
                  'rounded px-2 py-1',
                  skip && 'text-axis line-through',
                  !skip && current && i < step && 'text-ink-2',
                  !skip && current && i === step && 'bg-series-1/20 font-semibold text-ink',
                  !skip && (!current || i > step) && 'text-muted',
                )}
              >
                {i === 0 && memory ? 'Read memory, run experiments' : s.label}
              </span>
              {i < STEPS.length - 1 && <span className="text-axis">→</span>}
            </li>
          )
        })}
      </ol>
      <div className="text-right">
        <div className="text-[11px] tracking-wide text-muted uppercase">Best val loss</div>
        <div className="text-xl font-semibold tabular">
          <span className="text-ink-2">{loss(l0)}</span>
          <span className="mx-1.5 text-muted">→</span>
          <span>{loss(best)}</span>
          {gain < 0 && <span className="ml-2 text-sm text-good">{signedPct(gain)}</span>}
        </div>
      </div>
    </div>
  )
}
