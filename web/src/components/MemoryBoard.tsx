import { useState } from 'react'
import { Ban, CheckCircle2, HelpCircle, Trophy } from 'lucide-react'
import type { MemoryEntryData } from '@/lib/events'
import { cn, loss, tokens } from '@/lib/format'
import type { MemoryVersion, RunState, SetupState } from '@/lib/runState'
import { Badge, Card } from './ui'

type Kind = MemoryEntryData['kind']

const SECTIONS: { key: Kind; title: string; icon: typeof CheckCircle2; blurb: string }[] = [
  { key: 'works', title: 'What works', icon: CheckCircle2, blurb: 'findings, with proof' },
  { key: 'failed', title: 'What failed', icon: Ban, blurb: 'dead ends not to retry' },
  { key: 'try_next', title: 'What to try next', icon: HelpCircle, blurb: 'open questions' },
]

interface Line {
  kind: Kind
  id: string
  claim: string
  meta: string
}

/** Parses app/memory.py render(): `## kind` headers, then `- [id] claim (val +0.0012 on v3)`. */
function parse(text: string): Line[] {
  const out: Line[] = []
  let kind: Kind | undefined
  for (const raw of text.split('\n')) {
    const h = raw.match(/^## (\w+)/)
    if (h) {
      kind = h[1] as Kind
      continue
    }
    const m = raw.match(/^- \[([^\]]+)\] (.*?)(?: \(([^()]*)\))?$/)
    if (m && kind) out.push({ kind, id: m[1], claim: m[2], meta: m[3] ?? '' })
  }
  return out
}

export function SizeGauge({ run, st, version }: { run: RunState; st: SetupState; version?: MemoryVersion }) {
  const v = version ?? st.memory.at(-1)
  const research = run.config?.research
  const isRaw = st.setup === 'raw_log'
  const cap = (isRaw ? research?.raw_log_cap_tokens : research?.memory_cap_tokens) ?? 0
  const size = v?.tokens ?? 0
  const max = Math.max(cap, size, 1)
  return (
    <div className="space-y-1 text-xs">
      <div className="flex justify-between text-ink-2">
        <span>{isRaw ? 'Raw log memory' : 'Curated memory'}</span>
        <span className="tabular">
          {tokens(size)} / {tokens(cap)} token cap
        </span>
      </div>
      <div className="relative h-2.5 rounded-full bg-surface-2">
        <div className={cn('h-full rounded-full', size > cap ? 'bg-critical' : 'bg-series-1')} style={{ width: `${Math.min(100, (size / max) * 100)}%` }} />
        <div className="absolute top-[-3px] h-4 w-0.5 bg-ink-2" style={{ left: `${(cap / max) * 100}%` }} title="size cap" />
      </div>
    </div>
  )
}

export function MemoryBoard({ run, st }: { run: RunState; st: SetupState }) {
  const versions = st.memory
  const [picked, setPicked] = useState<number | null>(null) // index; null = follow latest
  const idx = picked !== null && picked < versions.length ? picked : versions.length - 1
  const current = versions[idx]

  if (st.setup === 'no_memory' || st.setup === 'one_shot') {
    return (
      <Card title="Memory">
        <p className="text-sm text-ink-2">This setup has no memory: every agent starts from zero.</p>
      </Card>
    )
  }
  if (!current) {
    return (
      <Card title="Memory">
        <p className="text-sm text-muted">Memory is empty until the first generation is curated.</p>
      </Card>
    )
  }
  const lines = parse(current.text)
  const best = lines.find((l) => l.kind === 'best')
  const bestPoint = [...st.best].reverse().find((b) => b.ts <= current.ts)
  const l0 = run.baseline?.l0
  const ACTION: Record<MemoryVersion['action'], string> = {
    snapshot: 'snapshot',
    upsert: `edited ${current.changedId ?? ''}`,
    remove: 'entry removed',
    curation_completed: 'curation done',
  }

  return (
    <div className="space-y-4">
      <Card
        title={`Memory · generation ${current.gen} · ${ACTION[current.action]}`}
        action={
          picked !== null && (
            <button className="text-xs text-[#86b6ef] hover:underline" onClick={() => setPicked(null)}>
              Follow latest
            </button>
          )
        }
      >
        <div className="grid gap-6 md:grid-cols-[1fr_280px]">
          <div>
            <label className="mb-1 block text-xs text-muted" htmlFor="mem-version">
              History: drag to see how memory changed ({idx + 1} of {versions.length} changes)
            </label>
            <input
              id="mem-version"
              type="range"
              min={0}
              max={versions.length - 1}
              value={idx}
              onChange={(e) => setPicked(Number(e.target.value))}
              className="w-full accent-[#3987e5]"
            />
          </div>
          <SizeGauge run={run} st={st} version={current} />
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-4">
        <section className="rounded-xl border border-series-1/40 bg-series-1/10 p-4 lg:order-last">
          <h3 className="flex items-center gap-2 text-xs font-semibold tracking-wide text-ink-2 uppercase">
            <Trophy size={14} /> Best so far
          </h3>
          <div className="mt-2 text-3xl font-semibold tabular">{loss(bestPoint?.val ?? l0)}</div>
          <div className="mt-1 text-sm text-ink-2">{best?.claim ?? bestPoint?.note ?? 'the starting train.py'}</div>
          {bestPoint && l0 && <div className="mt-2 text-xs text-good">{(((bestPoint.val - l0) / l0) * 100).toFixed(1)}% vs baseline {loss(l0)}</div>}
        </section>
        {SECTIONS.map(({ key, title, icon: Icon, blurb }) => {
          const entries = lines.filter((l) => l.kind === key)
          return (
            <section key={key} className="rounded-xl border border-line bg-surface p-4">
              <h3 className="flex items-center gap-2 text-xs font-semibold tracking-wide text-ink-2 uppercase">
                <Icon size={14} /> {title}
                <span className="ml-auto font-normal text-muted normal-case">{entries.length}</span>
              </h3>
              <p className="mb-3 text-[11px] text-muted">{blurb}</p>
              <ul className="space-y-2">
                {entries.map((e) => {
                  const changed = e.id === current.changedId
                  return (
                    <li key={e.id} className={cn('rounded-lg border px-3 py-2 text-sm', changed ? 'flash-add border-good/50' : 'border-line')}>
                      <div className="text-ink">{e.claim}</div>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-muted">
                        {changed && <Badge tone="good">changed</Badge>}
                        <span className="font-mono">{e.id}</span>
                        {e.meta && <span>· {e.meta}</span>}
                      </div>
                    </li>
                  )
                })}
                {!entries.length && <li className="text-xs text-muted">Nothing yet</li>}
              </ul>
            </section>
          )
        })}
      </div>
      {current.removed?.length ? (
        <Card title="Removed in this change">
          <ul className="space-y-1 text-sm">
            {current.removed.map((r) => (
              <li key={r.id} className="text-ink-2">
                <span className="font-mono text-xs text-muted line-through">{r.id}</span> {r.reason}
              </li>
            ))}
          </ul>
        </Card>
      ) : null}
    </div>
  )
}
