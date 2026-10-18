import { useQuery } from '@tanstack/react-query'
import { useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ChartTip } from '@/components/ChartTip'
import { useSource } from '@/components/sources'
import { Badge, Swatch } from '@/components/ui'
import type { GenerationMetrics, RunInfo, Setup } from '@/lib/events'
import { cn, gain, loss, pct, signedPct, tokens } from '@/lib/format'
import { reduceAll, usesMemory, type RunState, type SetupState } from '@/lib/runState'
import { CHROME, SETUP_BLURB, SETUP_LABEL, focusColor, focusShape, setupColor, sortSetups } from '@/lib/theme'

const axis = { stroke: CHROME.axis, tick: { fill: CHROME.muted, fontSize: 11 }, tickLine: false } as const
const finished = (r: RunInfo) => Object.values(r.summary).some((s) => s?.fitness != null)
const metrics = (st: SetupState): GenerationMetrics[] => st.gens.flatMap((g) => (g.metrics ? [g.metrics] : []))
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined)
/** Round an axis max up to a whole step (380 → 400, 6879 → 7000). */
const niceCeil = (max: number) => {
  const step = 10 ** Math.floor(Math.log10(Math.max(max, 1)))
  return Math.ceil(max / step) * step
}
const genLabel = (g: string | number) => (Number(g) === 0 ? 'Start' : `Generation ${g}`)

const known = (xs?: (number | null)[]) => (xs ?? []).filter((x): x is number => x != null)
/** Mean test accuracy of the final best train.py (undefined for runs scored before it was tracked). */
const testAccuracy = (st: SetupState) => mean(known(st.result?.best_test_accuracies))

/** Per-seed test gains of the final best train.py vs the starting one. */
function seedGains(st: SetupState) {
  const r = st.result
  if (!r) return []
  return r.baseline_test_losses.flatMap((b, i) => (r.best_test_losses[i] != null ? [gain(b, r.best_test_losses[i]!)] : []))
}

function Section({ title, children, note }: { title: string; children: ReactNode; note?: ReactNode }) {
  return (
    <section className="grid min-w-0 gap-4">
      <div className="grid gap-1">
        <h2 className="text-xl font-semibold text-ink">{title}</h2>
        {note && <p className="max-w-[70ch] text-sm text-muted">{note}</p>}
      </div>
      {children}
    </section>
  )
}

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

function Figure({ caption, setups, children }: { caption: ReactNode; setups: Setup[]; children: ReactNode }) {
  return (
    <figure className="m-0 grid min-w-0 gap-2">
      <div className="rounded-xl border border-line bg-surface p-3">
        <div className="h-60">{children}</div>
        <div className="mt-2 flex justify-center">
          <Legend setups={setups} />
        </div>
      </div>
      <figcaption className="text-xs text-muted">{caption}</figcaption>
    </figure>
  )
}

const Th = ({ children, className }: { children?: ReactNode; className?: string }) => (
  <th className={cn('px-3 py-2 text-left text-[11px] font-medium tracking-wide text-muted uppercase', className)}>{children}</th>
)
const Td = ({ children, className }: { children?: ReactNode; className?: string }) => <td className={cn('px-3 py-2 align-top font-mono text-[13px] whitespace-nowrap', className)}>{children}</td>

function Headline({ run, sts }: { run: RunState; sts: SetupState[] }) {
  const cfg = run.config
  const sb = run.setups.shared_brain?.result?.fitness
  const rivals = sts.filter((s) => s.setup !== 'shared_brain' && s.result?.fitness != null)
  const best = rivals.reduce<SetupState | undefined>((a, b) => (!a || b.result!.fitness! > a.result!.fitness! ? b : a), undefined)
  const bf = best?.result?.fitness
  let lede: ReactNode = 'Shared Brain has no test score in this run.'
  if (sb != null && best && bf != null) {
    const diff = Math.abs(sb - bf) * 100
    lede =
      sb >= bf ? (
        <>
          Shared Brain improved test log loss by <b className="text-good">{pct(sb, 2)}</b> over the starting model, <b className="text-ink">{diff.toFixed(2)} points</b> ahead of {SETUP_LABEL[best.setup]} ({pct(bf, 2)}) on the same budget.
        </>
      ) : (
        <>
          Shared Brain improved test log loss by <b className="text-ink">{pct(sb, 2)}</b> over the starting model, {diff.toFixed(2)} points behind {SETUP_LABEL[best.setup]} ({pct(bf, 2)}) on the same budget.
        </>
      )
  } else if (sb != null) {
    lede = (
      <>
        Shared Brain improved test log loss by <b className="text-good">{pct(sb, 2)}</b> over the starting model.
      </>
    )
  }
  const r = cfg?.research
  const testStart = mean(run.baseline?.test_losses ?? [])
  const meta = [
    r && `${r.generations} generations`,
    r && `${r.researchers} researchers each`,
    r && `${r.researchers * r.researcher_runs} runs per generation`,
    cfg && `researcher ${cfg.researcher_model}`,
    cfg && sts.some((s) => s.setup === 'shared_brain') && `curator ${cfg.curator_model}`,
    run.baseline && `start val log loss L0 ${loss(run.baseline.l0, 5)}`,
    run.baseline && `noise threshold ${loss(run.baseline.threshold, 5)}`,
    testStart != null && `start test log loss ${loss(testStart, 5)}`,
  ].filter(Boolean) as string[]

  return (
    <header className="grid gap-3">
      <div className="font-mono text-xs font-semibold tracking-widest text-series-1 uppercase">
        Shared Brain · {cfg?.dataset ?? 'run'} · {run.runId}
      </div>
      <h1 className="text-3xl leading-tight font-semibold text-balance text-ink sm:text-4xl">Does curated memory make research agents better?</h1>
      <p className="max-w-[65ch] text-lg text-ink-2">{lede}</p>
      <div className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-muted">
        {meta.map((m) => (
          <span key={m}>{m}</span>
        ))}
      </div>
    </header>
  )
}

function ResultsTable({ sts }: { sts: SetupState[] }) {
  const lead = Math.max(...sts.map((s) => s.result?.fitness ?? -Infinity))
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface">
      <table className="w-full tabular">
        <thead>
          <tr className="border-b border-line">
            <Th>Setup</Th>
            <Th>Test gain</Th>
            <Th>Test accuracy</Th>
            <Th>Val gain</Th>
            <Th>Best</Th>
            <Th>Gens</Th>
            <Th>Runs</Th>
            <Th>Failed</Th>
            <Th>Prompt tokens</Th>
            <Th>Time</Th>
          </tr>
        </thead>
        <tbody>
          {sts.map((st) => {
            const ms = metrics(st)
            const r = st.result
            const gains = seedGains(st)
            const prompt = ms.reduce((n, m) => n + m.usage.researchers.prompt + (m.usage.curator.prompt ?? 0), 0)
            const cached = ms.reduce((n, m) => n + m.usage.researchers.cached + (m.usage.curator.cached ?? 0), 0)
            const failed = ms.reduce((n, m) => n + m.errors + m.timeouts, 0)
            const newBests = ms.filter((m) => m.new_best).length
            const minutes = r?.minutes ?? ms.reduce((n, m) => n + m.minutes, 0)
            const top = r?.fitness != null && r.fitness === lead
            return (
              <tr key={st.setup} className="border-b border-line last:border-0">
                <th scope="row" className="px-3 py-2.5 text-left align-top font-normal">
                  <div className="flex items-center gap-2 text-sm font-semibold text-ink">
                    <Swatch color={setupColor(st.setup)} shape="square" />
                    {SETUP_LABEL[st.setup]}
                  </div>
                  <div className="pl-[18px] text-[11px] text-muted">{SETUP_BLURB[st.setup]}</div>
                </th>
                <Td className={cn(top ? 'font-semibold text-good' : 'text-ink')}>
                  {r?.fitness != null ? signedPct(r.fitness, 2) : '—'}
                  {gains.length > 1 && (
                    <div className="text-[11px] font-normal text-muted">
                      {pct(Math.min(...gains), 1)}–{pct(Math.max(...gains), 1)} over {gains.length} seeds
                    </div>
                  )}
                </Td>
                <Td>{pct(testAccuracy(st), 1)}</Td>
                <Td>{r ? signedPct(r.val_gain, 2) : ms.length ? signedPct(ms.at(-1)!.val_gain, 2) : '—'}</Td>
                <Td>
                  v{r?.best_version ?? ms.at(-1)?.best_version ?? 0} <span className="text-muted">({newBests} new)</span>
                </Td>
                <Td>{ms.length}</Td>
                <Td>{r?.runs ?? ms.reduce((n, m) => n + m.runs, 0)}</Td>
                <Td className={failed ? 'text-serious' : undefined}>{failed}</Td>
                <Td>
                  {tokens(prompt)} {prompt > 0 && <span className="text-muted">({pct(cached / prompt, 0)} cached)</span>}
                </Td>
                <Td>{minutes.toFixed(1)} min</Td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function AccuracyChart({ run, sts }: { run: RunState; sts: SetupState[] }) {
  const setups = sts.flatMap((st) => {
    const value = testAccuracy(st)
    return value == null ? [] : [{ name: SETUP_LABEL[st.setup], value, fill: st.setup === 'shared_brain' ? setupColor(st.setup) : CHROME.muted }]
  })
  if (!setups.length) return null
  const start = mean(sts.flatMap((st) => known(st.result?.baseline_test_accuracies)))
  const majority = sts.map((st) => st.result?.majority_test_accuracy).find((x) => x != null)
  const refs = [
    ...(start != null ? [{ name: 'Starting model', value: start, fill: '#5c5b56' }] : []),
    ...(majority != null ? [{ name: run.config?.dataset === 'churn' ? 'Always "no churn"' : 'Always the majority class', value: majority, fill: '#3b3b38' }] : []),
  ]
  const bars = [...setups, ...refs].sort((a, b) => b.value - a.value)
  const values = bars.map((b) => b.value)
  // Start the axis just below the lowest bar (on a 5% step) so sub-point gaps stay visible.
  const lo = Math.max(0, Math.floor((Math.min(...values) - 0.01) * 20) / 20)
  const hi = Math.min(1, Math.ceil((Math.max(...values) + 0.005) * 50) / 50)
  const ticks = [lo]
  for (let t = Math.floor(lo * 50 + 1) / 50; t <= hi + 1e-9; t += 0.02) ticks.push(t)
  const seeds = run.config?.research.test_seeds

  return (
    <figure className="m-0 grid min-w-0 gap-2">
      <div className="rounded-xl border border-line bg-surface p-4">
        <div className="h-80">
          <ResponsiveContainer>
            <BarChart data={bars} margin={{ top: 24, right: 8, left: 8, bottom: 0 }} barCategoryGap="32%">
              <CartesianGrid stroke={CHROME.grid} vertical={false} />
              <XAxis dataKey="name" {...axis} tick={{ fill: CHROME.ink2, fontSize: 13 }} interval={0} />
              <YAxis
                {...axis}
                axisLine={false}
                domain={[lo, hi]}
                ticks={ticks}
                interval={0}
                allowDataOverflow
                tickFormatter={(v: number) => pct(v, 0)}
                width={56}
                label={{ value: run.config?.dataset ? `${run.config.dataset[0].toUpperCase()}${run.config.dataset.slice(1)} test accuracy` : 'Test accuracy', angle: -90, position: 'insideLeft', fill: CHROME.muted, fontSize: 12, dx: -2, dy: 60 }}
              />
              <Tooltip content={<ChartTip valueFmt={(v) => pct(v, 2)} />} cursor={{ fill: 'rgb(255 255 255 / 0.04)' }} />
              <Bar dataKey="value" name="Test accuracy" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                {bars.map((b) => (
                  <Cell key={b.name} fill={b.fill} />
                ))}
                <LabelList dataKey="value" position="top" fill={CHROME.ink} fontSize={13} formatter={(v: unknown) => pct(v as number, 1)} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <figcaption className="text-xs text-muted">
        Axis starts at {pct(lo, 0)} so the gaps are visible. Accuracy on the held-back test split (positive class when p ≥ 0.5), averaged over{' '}
        {seeds?.length ? `seeds ${seeds[0]}–${seeds.at(-1)}` : 'the test seeds'}.
      </figcaption>
    </figure>
  )
}

function Charts({ run, sts }: { run: RunState; sts: SetupState[] }) {
  const l0 = run.baseline?.l0
  const threshold = run.baseline?.threshold
  const all = sts.map((s) => s.setup)
  const maxGen = Math.max(0, ...sts.map((s) => metrics(s).length))
  const gens = Array.from({ length: maxGen + 1 }, (_, g) => g)
  // Per-generation test scores exist only if the orchestrator tested every generation's best.
  const perGenTest = sts.every((s) => metrics(s).every((m) => m.test_gain != null))
  const at = (st: SetupState, g: number) => metrics(st).find((m) => m.gen === g)

  const gainRows = gens.map((g) => ({
    gen: g,
    ...Object.fromEntries(sts.map((st) => [st.setup, g === 0 ? 0 : perGenTest ? at(st, g)?.test_gain : at(st, g)?.val_gain])),
  }))
  // One-shot runs a single generation; carry its result forward so it reads as a flat reference.
  const carry = (rows: Record<string, number | undefined>[]) =>
    rows.map((row, i) => (row.one_shot === undefined && i > 0 ? { ...row, one_shot: rows[i - 1].one_shot } : row))
  const valRows = gens.map((g) => ({ gen: g, ...Object.fromEntries(sts.map((st) => [st.setup, g === 0 ? l0 : at(st, g)?.end_best_val])) }))

  const memSts = sts.filter((s) => usesMemory(s.setup))
  // What researchers read at the start of generation g: the memory left by generation g-1.
  const memRows = gens.slice(1).map((g) => ({ gen: g, ...Object.fromEntries(memSts.map((st) => [st.setup, g === 1 ? 0 : at(st, g - 1)?.memory_tokens])) }))
  const memCap = run.config?.research.memory_cap_tokens

  const line = (s: Setup, stepped = false) => (
    <Line
      key={s}
      dataKey={s}
      name={SETUP_LABEL[s]}
      type={stepped ? 'stepAfter' : 'linear'}
      stroke={setupColor(s)}
      strokeWidth={s === 'shared_brain' ? 3 : 2}
      strokeDasharray={s === 'one_shot' ? '5 4' : undefined}
      dot={{ r: 3, strokeWidth: 0, fill: setupColor(s) }}
      isAnimationActive={false}
      connectNulls
    />
  )
  const xAxis = <XAxis dataKey="gen" {...axis} tickFormatter={(g: number) => (g === 0 ? 'start' : `G${g}`)} />
  const grid = <CartesianGrid stroke={CHROME.grid} vertical={false} />
  const margin = { top: 8, right: 12, left: 0, bottom: 0 }

  return (
    <div className="grid gap-6 [grid-template-columns:repeat(auto-fit,minmax(min(100%,22rem),1fr))]">
      <Figure
        setups={all}
        caption={
          perGenTest
            ? "Test gain of each generation's best train.py. Higher is better."
            : "Val gain of each generation's best train.py over L0. Higher is better. Only the final best is scored on test (table above)."
        }
      >
        <ResponsiveContainer>
          <LineChart data={carry(gainRows)} margin={margin}>
            {grid}
            {xAxis}
            <YAxis {...axis} axisLine={false} tickFormatter={(v: number) => pct(v, 0)} width={44} />
            <Tooltip content={<ChartTip valueFmt={(v) => pct(v, 2)} labelFmt={genLabel} />} cursor={{ stroke: CHROME.axis }} />
            {all.map((s) => line(s))}
          </LineChart>
        </ResponsiveContainer>
      </Figure>
      <Figure setups={all} caption="Best val log loss so far. A step down is a new best that beat the old one by more than the noise threshold (dashed).">
        <ResponsiveContainer>
          <LineChart data={carry(valRows)} margin={margin}>
            {grid}
            {xAxis}
            <YAxis {...axis} axisLine={false} domain={['auto', 'auto']} tickFormatter={(v: number) => v.toFixed(3)} width={48} />
            <Tooltip content={<ChartTip valueFmt={(v) => loss(v, 5)} labelFmt={genLabel} />} cursor={{ stroke: CHROME.axis }} />
            {l0 != null && threshold != null && (
              <ReferenceLine y={l0 - threshold} stroke={CHROME.ink2} strokeDasharray="2 3" label={{ value: 'L0 − noise', position: 'insideTopRight', fill: CHROME.muted, fontSize: 10 }} />
            )}
            {all.map((s) => line(s, true))}
          </LineChart>
        </ResponsiveContainer>
      </Figure>
      {memSts.length > 0 && (
        <Figure
          setups={memSts.map((s) => s.setup)}
          caption={
            <>
              What researchers read at the start of each generation.
              {memCap ? ` Curated memory is capped at ${memCap.toLocaleString()} tokens` : ' Curated memory is capped'}
              {memSts.some((s) => s.setup === 'raw_log') ? '; the raw log keeps growing.' : '.'}
            </>
          }
        >
          <ResponsiveContainer>
            <LineChart data={memRows} margin={margin}>
              {grid}
              <XAxis dataKey="gen" {...axis} tickFormatter={(g: number) => `G${g}`} />
              <YAxis {...axis} axisLine={false} domain={[0, niceCeil]} tickCount={5} tickFormatter={tokens} width={44} />
              <Tooltip content={<ChartTip valueFmt={(v) => `${Math.round(v).toLocaleString()} tokens`} labelFmt={genLabel} />} cursor={{ stroke: CHROME.axis }} />
              {memCap && <ReferenceLine y={memCap} stroke={CHROME.ink2} strokeDasharray="4 4" label={{ value: 'cap', position: 'insideTopRight', fill: CHROME.muted, fontSize: 10 }} />}
              {memSts.map((s) => line(s.setup))}
            </LineChart>
          </ResponsiveContainer>
        </Figure>
      )}
    </div>
  )
}

function Generation({ st, m, read }: { st: SetupState; m: GenerationMetrics; read: number }) {
  const mem = usesMemory(st.setup) ? st.memory.filter((v) => v.gen === m.gen).at(-1) : undefined
  const rows = [...m.researcher_best].sort((a, b) => a.rank - b.rank || (a.best?.val_loss ?? Infinity) - (b.best?.val_loss ?? Infinity))
  const failed = m.errors + m.timeouts
  return (
    <article className="grid min-w-0 gap-3 border-t border-line py-5">
      <header className="grid gap-1.5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <h3 className="text-base font-semibold text-ink">Generation {m.gen}</h3>
          {m.new_best ? <Badge tone="good">new best v{m.best_version}</Badge> : <Badge>no new best</Badge>}
          {m.new_best && m.best_note && <span className="text-sm text-ink-2">{m.best_note}</span>}
        </div>
        <div className="flex flex-wrap gap-x-2 font-mono text-xs text-muted">
          <span>val {loss(m.end_best_val, 5)}</span>·<span>val gain {signedPct(m.val_gain, 2)}</span>
          {m.test_gain != null && (
            <>
              ·<span>test gain {signedPct(m.test_gain, 2)}</span>
            </>
          )}
          ·
          <span>
            {m.runs} runs, <span className={failed ? 'text-serious' : undefined}>{failed} failed</span>
          </span>
          {usesMemory(st.setup) && (
            <>
              ·<span>read {read.toLocaleString()} tokens</span>
            </>
          )}
          ·<span>{m.minutes.toFixed(1)} min</span>
        </div>
      </header>
      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[40rem] table-fixed tabular">
            <thead>
              <tr className="border-b border-line">
                <Th className="w-36 pl-0">Focus</Th>
                <Th className="w-16">Rank</Th>
                <Th className="w-44">Best val</Th>
                <Th>Change</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const isBest = !!r.best && r.best.run_id === m.best_run_id && m.new_best
                return (
                  <tr key={r.name} className="border-b border-line/60 last:border-0">
                    <td className="py-1.5 pr-3 align-top text-[13px] whitespace-nowrap text-ink-2">
                      <span className="flex items-center gap-2">
                        <Swatch color={focusColor(r.focus)} shape={focusShape(r.focus)} />
                        {r.focus || r.name}
                      </span>
                    </td>
                    <Td className="py-1.5">{r.rank}</Td>
                    <Td className={cn('py-1.5 whitespace-nowrap', isBest && 'text-good')}>
                      {loss(r.best?.val_loss, 5)} <span className="text-muted">{r.best?.run_id}</span>
                    </Td>
                    <td className="py-1.5 pl-3 align-top text-[13px] text-ink-2">{r.best?.note || <span className="text-muted">no successful run</span>}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      {mem && (
        <details className="group">
          <summary className="cursor-pointer text-xs text-[#86b6ef] hover:underline">
            Memory after generation {m.gen} ({mem.tokens.toLocaleString()} tokens)
          </summary>
          <pre className="mt-2 max-h-96 overflow-auto rounded-lg border border-line bg-surface p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap text-ink-2">{mem.text}</pre>
        </details>
      )}
    </article>
  )
}

function Evolution({ sts }: { sts: SetupState[] }) {
  const [pick, setPick] = useState<Setup>(sts.find((s) => s.setup === 'shared_brain')?.setup ?? sts[0].setup)
  const st = sts.find((s) => s.setup === pick) ?? sts[0]
  const ms = metrics(st)
  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Setup">
        {sts.map((s) => (
          <button
            key={s.setup}
            role="tab"
            aria-selected={s.setup === st.setup}
            onClick={() => setPick(s.setup)}
            className={cn(
              'flex items-center gap-2 rounded-full border px-3 py-1 text-sm transition-colors',
              s.setup === st.setup ? 'border-ink-2 bg-surface-2 text-ink' : 'border-line text-ink-2 hover:text-ink',
            )}
          >
            <Swatch color={setupColor(s.setup)} shape="square" />
            {SETUP_LABEL[s.setup]}
          </button>
        ))}
      </div>
      <div role="tabpanel">
        {ms.length ? ms.map((m, i) => <Generation key={m.gen} st={st} m={m} read={i ? ms[i - 1].memory_tokens : 0} />) : <p className="py-4 text-sm text-muted">No finished generation.</p>}
      </div>
    </div>
  )
}

function Results({ run }: { run: RunState }) {
  const sts = sortSetups(Object.keys(run.setups) as Setup[]).map((s) => run.setups[s]!)
  const seeds = run.config?.research.test_seeds
  return (
    <div className="grid gap-10">
      <Headline run={run} sts={sts} />
      {sts.some((st) => testAccuracy(st) != null) && (
        <Section title="Test accuracy" note="How often each setup's final model classifies a held-back row correctly, next to the model the agents started from and the floor of always predicting the most common class.">
          <AccuracyChart run={run} sts={sts} />
        </Section>
      )}
      <Section
        title="Results by setup"
        note={
          <>
            Test gain is the drop in test log loss of the final best train.py versus the starting one, averaged over {seeds?.length ? `seeds ${seeds.join(', ')}` : 'the test seeds'}. Agents never see the test split.
          </>
        }
      >
        <ResultsTable sts={sts} />
      </Section>
      <Section title="How it evolved">
        <Charts run={run} sts={sts} />
      </Section>
      <Section title="Generation by generation" note="Each researcher's best run, ranked, and the memory the next generation starts from.">
        <Evolution sts={sts} />
      </Section>
    </div>
  )
}

export function ComparePage() {
  const src = useSource()
  const [params, setParams] = useSearchParams()
  const runs = useQuery({ queryKey: ['runs', src.source], queryFn: src.listRuns, retry: src.source === 'live' ? 1 : 3 })
  const options = (runs.data ?? []).filter(finished)
  const runId = params.get('run') ?? options[0]?.run_id
  const info = options.find((r) => r.run_id === runId)
  const state = useQuery({
    queryKey: ['results', src.source, runId],
    queryFn: async () => reduceAll(await src.getEvents(info!)),
    enabled: !!info,
  })

  let body
  if (runs.isLoading || state.isLoading) body = <p className="text-muted">Loading runs…</p>
  else if (runs.isError) body = <p className="text-[#e66767]">Backend not connected: results come from the API in the live product.</p>
  else if (!options.length) body = <p className="text-muted">No finished run with test scores yet.</p>
  else if (state.isError || !state.data) body = <p className="text-[#e66767]">Could not load {runId}: {String(state.error)}</p>
  else body = <Results run={state.data} />

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-6 py-8">
      <div className="flex flex-wrap items-center gap-3">
        {src.source === 'demo' ? <Badge tone="warn">recorded data</Badge> : <Badge tone="accent">from this server</Badge>}
        {options.length > 1 && (
          <select
            value={runId}
            onChange={(e) => setParams({ run: e.target.value })}
            className="rounded-md border border-line bg-surface-2 px-2 py-1 font-mono text-xs text-ink"
            aria-label="Run"
          >
            {options.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id}
              </option>
            ))}
          </select>
        )}
        {info && (
          <Link to={`${src.base}/run/${info.run_id}?autoplay=1`} className="ml-auto text-xs text-[#86b6ef] hover:underline">
            Replay this run →
          </Link>
        )}
      </div>
      {body}
    </div>
  )
}
