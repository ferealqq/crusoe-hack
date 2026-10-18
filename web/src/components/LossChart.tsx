import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'
import { clock, loss } from '@/lib/format'
import type { LossPoint, RunState, SetupState } from '@/lib/runState'
import { CHROME, FOCUS_ORDER, focusColor, focusShape } from '@/lib/theme'
import { Card, Swatch } from './ui'

interface TipProps {
  active?: boolean
  payload?: { payload: Partial<LossPoint> & { best?: number } }[]
}

function Tip({ active, payload }: TipProps) {
  const p = payload?.[0]?.payload
  if (!active || !p) return null
  return (
    <div className="max-w-72 rounded-lg border border-line bg-surface-2 px-3 py-2 text-xs shadow-lg">
      {p.valLoss === undefined ? (
        <div className="text-ink">
          Best so far <span className="font-semibold tabular">{loss(p.best)}</span>
        </div>
      ) : (
        <>
          <div className="flex items-center gap-1.5 font-semibold text-ink capitalize">
            {p.verify ? <span className="inline-block h-2.5 w-2.5 rounded-full border-2 border-ink" /> : <Swatch color={focusColor(p.focus)} shape={focusShape(p.focus)} />}
            {p.verify ? 'Curator re-check' : p.focus} · gen {p.gen}
          </div>
          <div className="mt-1 text-ink-2">
            val loss <span className="font-semibold text-ink tabular">{loss(p.valLoss)}</span> at {clock(p.ts ?? 0)}
          </div>
          {p.note && <div className="mt-1 text-ink-2">{p.note}</div>}
        </>
      )}
    </div>
  )
}

const VerifyRing = (props: { cx?: number; cy?: number }) => <circle cx={props.cx} cy={props.cy} r={6} fill="none" stroke={CHROME.ink} strokeWidth={2} />

export function LossChart({ run, st }: { run: RunState; st: SetupState }) {
  const research = st.lossPoints.filter((p) => !p.verify)
  const verify = st.lossPoints.filter((p) => p.verify)
  const focuses = FOCUS_ORDER.filter((f) => research.some((p) => p.focus === f))
  const l0 = run.baseline?.l0

  // Best-so-far step line: baseline, then every accepted best.
  const best = [
    ...(l0 !== undefined ? [{ ts: 0, best: l0 }] : []),
    ...st.best.map((b) => ({ ts: b.ts, best: b.val })),
    ...(run.ts > 0 && (st.best.length || l0 !== undefined) ? [{ ts: run.ts, best: st.best.at(-1)?.val ?? l0 }] : []),
  ]
  const xMax = Math.max(run.ts, 60)
  const ys = [...st.lossPoints.map((p) => p.valLoss), ...best.flatMap((b) => (b.best !== undefined ? [b.best] : []))]
  const yDomain: [number, number] = ys.length ? [Math.floor((Math.min(...ys) - 0.01) * 50) / 50, Math.ceil((Math.max(...ys) + 0.01) * 50) / 50] : [0, 1]

  return (
    <Card
      title="Validation loss"
      action={
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-ink-2" aria-label="Legend">
          {focuses.map((f) => (
            <span key={f} className="flex items-center gap-1.5 capitalize">
              <Swatch color={focusColor(f)} shape={focusShape(f)} />
              {f}
            </span>
          ))}
          {verify.length > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full border-2 border-ink" /> curator re-check
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4 bg-ink" /> best so far
          </span>
        </div>
      }
    >
      <div className="h-72" role="img" aria-label={`Validation loss over time; best so far ${loss(best.at(-1)?.best)}`}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={CHROME.grid} vertical={false} />
            <XAxis
              type="number"
              dataKey="ts"
              domain={[0, xMax]}
              tickFormatter={clock}
              stroke={CHROME.axis}
              tick={{ fill: CHROME.muted, fontSize: 11 }}
              tickLine={false}
              allowDuplicatedCategory={false}
            />
            <YAxis
              type="number"
              domain={yDomain}
              allowDataOverflow
              tickFormatter={(v: number) => v.toFixed(2)}
              stroke={CHROME.axis}
              tick={{ fill: CHROME.muted, fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={44}
            />
            <Tooltip content={<Tip />} cursor={{ stroke: CHROME.axis }} isAnimationActive={false} />
            {st.gens.map((g) => (
              <ReferenceLine
                key={g.gen}
                x={g.startTs}
                stroke={CHROME.axis}
                strokeDasharray="3 3"
                label={{ value: `Gen ${g.gen}`, position: 'insideTopLeft', fill: CHROME.muted, fontSize: 10 }}
              />
            ))}
            <Line data={best} dataKey="best" type="stepAfter" stroke={CHROME.ink} strokeWidth={2} dot={false} isAnimationActive={false} />
            {focuses.map((f) => (
              <Scatter
                key={f}
                data={research.filter((p) => p.focus === f)}
                dataKey="valLoss"
                fill={focusColor(f)}
                shape={focusShape(f)}
                isAnimationActive={false}
                legendType="none"
              />
            ))}
            <Scatter data={verify} dataKey="valLoss" shape={<VerifyRing />} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}
