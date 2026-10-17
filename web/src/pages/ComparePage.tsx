import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ChartTip } from '@/components/ChartTip'
import { Card, Swatch } from '@/components/ui'
import { getEvents, listRuns } from '@/lib/api'
import type { Setup } from '@/lib/events'
import { loss, pct, tokens } from '@/lib/format'
import { reduceAll, type RunState } from '@/lib/runState'
import { CHROME, SETUP_BLURB, SETUP_LABEL, SETUP_ORDER, setupColor } from '@/lib/theme'

async function loadComparison(): Promise<Partial<Record<Setup, RunState>>> {
  const runs = await listRuns()
  const out: Partial<Record<Setup, RunState>> = {}
  for (const setup of SETUP_ORDER) {
    // Prefer a real finished run over the bundled sample for each setup.
    const candidates = runs.filter((r) => r.setup === setup && r.status === 'completed' && r.fitness !== null)
    const pick = candidates.find((r) => !r.run_id.startsWith('sample-')) ?? candidates[0]
    if (pick) out[setup] = reduceAll(await getEvents(pick.run_id))
  }
  return out
}

const axis = { stroke: CHROME.axis, tick: { fill: CHROME.muted, fontSize: 11 }, tickLine: false } as const

function Legend({ setups }: { setups: Setup[] }) {
  return (
    <div className="flex flex-wrap gap-3 text-[11px] text-ink-2" aria-label="Legend">
      {setups.map((s) => (
        <span key={s} className="flex items-center gap-1.5">
          <Swatch color={setupColor(s)} />
          {SETUP_LABEL[s]}
        </span>
      ))}
    </div>
  )
}

function perGen(runs: Partial<Record<Setup, RunState>>, setups: Setup[], pick: (s: RunState, gen: number) => number | undefined, from = 1) {
  const maxGen = Math.max(...setups.map((s) => runs[s]?.gens.length ?? 0))
  return Array.from({ length: maxGen - from + 1 }, (_, i) => {
    const gen = i + from
    const row: Record<string, number | undefined> = { gen }
    for (const s of setups) if (runs[s]) row[s] = pick(runs[s]!, gen)
    return row
  })
}

export function ComparePage() {
  const { data: runs, isLoading, error } = useQuery({ queryKey: ['compare'], queryFn: loadComparison })
  const [asTable, setAsTable] = useState(false)
  if (isLoading) return <p className="p-8 text-muted">Loading runs…</p>
  if (error || !runs) return <p className="p-8 text-[#e66767]">Could not load runs: {String(error)}</p>

  const setups = SETUP_ORDER.filter((s) => runs[s]?.eval)
  const fitness = setups.map((s) => {
    const gains = runs[s]!.eval!.seeds.map((x) => x.gain)
    const f = runs[s]!.eval!.fitness
    return { setup: s, name: SETUP_LABEL[s], fitness: f, err: [f - Math.min(...gains), Math.max(...gains) - f], fill: setupColor(s) }
  })
  const sb = runs.shared_brain?.eval?.fitness
  const baselines = fitness.filter((f) => f.setup !== 'shared_brain')
  const bestBaseline = baselines.reduce((a, b) => (b.fitness > a.fitness ? b : a), baselines[0])
  const multi = setups.filter((s) => s !== 'one_shot')

  const genMetric = (s: RunState, gen: number) => s.gens.find((g) => g.gen === gen)?.metrics
  const bestRows = [
    { gen: 0, ...Object.fromEntries(multi.map((s) => [s, runs[s]?.baseline])) },
    ...perGen(runs, multi, (s, g) => genMetric(s, g)?.best_val_loss),
  ]
  const memRows = perGen(runs, ['shared_brain', 'raw_log'], (s, g) => genMetric(s, g)?.memory_tokens)
  const wasteRows = perGen(runs, multi, (s, g) => s.gens.filter((x) => x.gen <= g && x.metrics).reduce((n, x) => n + x.metrics!.waste_runs, 0))
  const cost = setups.map((s) => {
    const ms = runs[s]!.gens.flatMap((g) => (g.metrics ? [g.metrics] : []))
    const prompt = ms.reduce((n, m) => n + m.prompt_tokens, 0)
    const cached = ms.reduce((n, m) => n + m.cached_tokens, 0)
    const ttft = ms.reduce((n, m) => n + m.mean_ttft_ms, 0) / Math.max(1, ms.length)
    return { setup: s, name: SETUP_LABEL[s], cache: prompt ? cached / prompt : 0, ttft: Math.round(ttft), prompt, waste: ms.reduce((n, m) => n + m.waste_runs, 0), fill: setupColor(s) }
  })
  const memCap = runs.shared_brain?.config?.memory_cap_tokens

  return (
    <div className="mx-auto max-w-7xl space-y-5 p-6">
      <div className="grid gap-5 lg:grid-cols-[1.1fr_2fr]">
        <section className="flex flex-col justify-center rounded-xl border border-series-1/40 bg-series-1/10 p-6">
          <div className="text-xs font-semibold tracking-wide text-ink-2 uppercase">Fitness on the hidden test split</div>
          <div className="mt-2 text-6xl font-semibold text-ink">{pct(sb)}</div>
          <div className="mt-1 text-sm text-ink-2">mean loss reduction vs the starting train.py, Shared Brain</div>
          {sb !== undefined && bestBaseline && (
            <div className="mt-4 text-lg text-ink">
              <span className="font-semibold text-good">{(sb / bestBaseline.fitness).toFixed(1)}×</span> the best baseline ({bestBaseline.name}, {pct(bestBaseline.fitness)})
            </div>
          )}
          <div className="mt-4 text-xs text-muted">Same dataset, same budget, same test split. Test data is never shown to any agent.</div>
        </section>
        <Card title="Fitness by setup" action={<span className="text-[11px] text-muted">whiskers: min–max over {runs.shared_brain?.eval?.seeds.length ?? 3} seeds</span>}>
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={fitness} margin={{ top: 16, right: 8, left: 0, bottom: 0 }} barCategoryGap="28%">
                <CartesianGrid stroke={CHROME.grid} vertical={false} />
                <XAxis dataKey="name" {...axis} />
                <YAxis {...axis} axisLine={false} tickFormatter={(v: number) => pct(v, 0)} width={44} />
                <Tooltip content={<ChartTip valueFmt={(v) => pct(v)} />} cursor={{ fill: 'rgb(255 255 255 / 0.04)' }} />
                <Bar dataKey="fitness" name="Fitness" radius={[4, 4, 0, 0]} isAnimationActive={false} label={{ position: 'top', fill: CHROME.ink2, fontSize: 12, formatter: (v: unknown) => pct(v as number) }}>
                  {fitness.map((f) => (
                    <Cell key={f.setup} fill={f.fill} />
                  ))}
                  <ErrorBar dataKey="err" stroke={CHROME.ink2} width={6} strokeWidth={1.5} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 grid grid-cols-4 gap-2 text-[11px] text-muted">
            {setups.map((s) => (
              <div key={s}>{SETUP_BLURB[s]}</div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <Card title="Best val loss per generation" action={<Legend setups={multi} />}>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart data={bestRows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={CHROME.grid} vertical={false} />
                <XAxis dataKey="gen" {...axis} tickFormatter={(g: number) => (g === 0 ? 'start' : `G${g}`)} />
                <YAxis {...axis} axisLine={false} domain={['auto', 'auto']} tickFormatter={(v: number) => v.toFixed(2)} width={44} />
                <Tooltip content={<ChartTip valueFmt={loss} labelFmt={(g) => (g === 0 ? 'Start' : `Generation ${g}`)} />} cursor={{ stroke: CHROME.axis }} />
                {multi.map((s) => (
                  <Line key={s} dataKey={s} name={SETUP_LABEL[s]} stroke={setupColor(s)} strokeWidth={2} dot={{ r: 4, strokeWidth: 0, fill: setupColor(s) }} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Memory size" action={<Legend setups={['shared_brain', 'raw_log']} />}>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart data={memRows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={CHROME.grid} vertical={false} />
                <XAxis dataKey="gen" {...axis} tickFormatter={(g: number) => `G${g}`} />
                <YAxis {...axis} axisLine={false} tickFormatter={tokens} width={44} />
                <Tooltip content={<ChartTip valueFmt={(v) => `${tokens(v)} tokens`} labelFmt={(g) => `Generation ${g}`} />} cursor={{ stroke: CHROME.axis }} />
                {memCap && <ReferenceLine y={memCap} stroke={CHROME.ink2} strokeDasharray="4 4" label={{ value: 'cap', position: 'insideTopRight', fill: CHROME.muted, fontSize: 10 }} />}
                {(['shared_brain', 'raw_log'] as Setup[]).map((s) => (
                  <Line key={s} dataKey={s} name={SETUP_LABEL[s]} stroke={setupColor(s)} strokeWidth={2} dot={{ r: 4, strokeWidth: 0, fill: setupColor(s) }} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Wasted runs (cumulative)" action={<Legend setups={multi} />}>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart data={wasteRows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={CHROME.grid} vertical={false} />
                <XAxis dataKey="gen" {...axis} tickFormatter={(g: number) => `G${g}`} />
                <YAxis {...axis} axisLine={false} allowDecimals={false} width={44} />
                <Tooltip content={<ChartTip valueFmt={(v) => `${v} runs`} labelFmt={(g) => `Through generation ${g}`} />} cursor={{ stroke: CHROME.axis }} />
                {multi.map((s) => (
                  <Line key={s} dataKey={s} name={SETUP_LABEL[s]} stroke={setupColor(s)} strokeWidth={2} dot={{ r: 4, strokeWidth: 0, fill: setupColor(s) }} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-[11px] text-muted">Runs that repeat a known dead end.</p>
        </Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Prompt tokens served from Crusoe's KV cache">
          <div className="h-48">
            <ResponsiveContainer>
              <BarChart data={cost} layout="vertical" margin={{ top: 0, right: 48, left: 8, bottom: 0 }} barCategoryGap="30%">
                <CartesianGrid stroke={CHROME.grid} horizontal={false} />
                <XAxis type="number" domain={[0, 1]} {...axis} tickFormatter={(v: number) => pct(v, 0)} />
                <YAxis type="category" dataKey="name" {...axis} axisLine={false} width={96} />
                <Tooltip content={<ChartTip valueFmt={(v) => pct(v)} />} cursor={{ fill: 'rgb(255 255 255 / 0.04)' }} />
                <Bar dataKey="cache" name="Cache hit rate" radius={[0, 4, 4, 0]} isAnimationActive={false} label={{ position: 'right', fill: CHROME.ink2, fontSize: 12, formatter: (v: unknown) => pct(v as number, 0) }}>
                  {cost.map((c) => (
                    <Cell key={c.setup} fill={c.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-[11px] text-muted">The curated memory is a stable shared prompt prefix, so Crusoe's KV cache reuses it across agents.</p>
        </Card>
        <Card title="Mean time to first token">
          <div className="h-48">
            <ResponsiveContainer>
              <BarChart data={cost} layout="vertical" margin={{ top: 0, right: 56, left: 8, bottom: 0 }} barCategoryGap="30%">
                <CartesianGrid stroke={CHROME.grid} horizontal={false} />
                <XAxis type="number" {...axis} tickFormatter={(v: number) => `${v} ms`} />
                <YAxis type="category" dataKey="name" {...axis} axisLine={false} width={96} />
                <Tooltip content={<ChartTip valueFmt={(v) => `${v} ms`} />} cursor={{ fill: 'rgb(255 255 255 / 0.04)' }} />
                <Bar dataKey="ttft" name="TTFT" radius={[0, 4, 4, 0]} isAnimationActive={false} label={{ position: 'right', fill: CHROME.ink2, fontSize: 12, formatter: (v: unknown) => `${v} ms` }}>
                  {cost.map((c) => (
                    <Cell key={c.setup} fill={c.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div>
        <button className="text-xs text-[#86b6ef] hover:underline" onClick={() => setAsTable((v) => !v)}>
          {asTable ? 'Hide table' : 'View as table'}
        </button>
        {asTable && (
          <table className="mt-3 w-full rounded-xl border border-line bg-surface text-sm tabular">
            <thead>
              <tr className="text-left text-[11px] tracking-wide text-muted uppercase">
                {['Setup', 'Fitness (test)', 'Best val loss', 'Wasted runs', 'Final memory', 'Cache hit', 'Mean TTFT', 'Prompt tokens'].map((h) => (
                  <th key={h} className="px-3 py-2 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cost.map((c) => {
                const r = runs[c.setup]!
                const last = r.gens.at(-1)?.metrics
                return (
                  <tr key={c.setup} className="border-t border-line">
                    <td className="px-3 py-2">{c.name}</td>
                    <td className="px-3 py-2">{pct(r.eval?.fitness)}</td>
                    <td className="px-3 py-2">{loss(last?.best_val_loss)}</td>
                    <td className="px-3 py-2">{c.waste}</td>
                    <td className="px-3 py-2">{last ? tokens(last.memory_tokens) : '—'}</td>
                    <td className="px-3 py-2">{pct(c.cache, 0)}</td>
                    <td className="px-3 py-2">{c.ttft} ms</td>
                    <td className="px-3 py-2">{tokens(c.prompt)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
