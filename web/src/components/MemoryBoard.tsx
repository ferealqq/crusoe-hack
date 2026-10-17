import { useState } from 'react'
import { Ban, CheckCircle2, HelpCircle, Trophy } from 'lucide-react'
import type { MemoryOp, Section } from '@/lib/events'
import { cn, loss, tokens } from '@/lib/format'
import type { MemoryEntry, RunState } from '@/lib/runState'
import { Badge, Card } from './ui'

const SECTIONS: { key: Section; title: string; icon: typeof CheckCircle2; blurb: string }[] = [
  { key: 'works', title: 'What works', icon: CheckCircle2, blurb: 'findings, with proof' },
  { key: 'failed', title: 'What failed', icon: Ban, blurb: "dead ends not to retry" },
  { key: 'try_next', title: 'What to try next', icon: HelpCircle, blurb: 'open questions' },
]

function Entry({ e, op }: { e: MemoryEntry; op?: MemoryOp }) {
  return (
    <li
      key={`${e.id}-${op?.op ?? ''}`}
      className={cn('rounded-lg border px-3 py-2 text-sm', op?.op === 'add' ? 'flash-add border-good/50' : op?.op === 'edit' ? 'border-warning/50' : 'border-line')}
    >
      <div className="text-ink">{e.text}</div>
      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-muted">
        {op?.op === 'add' && <Badge tone="good">added</Badge>}
        {op?.op === 'edit' && <Badge tone="warn">edited</Badge>}
        <span>gen {e.gen}</span>
        {e.proof && <span title="proof">· {e.proof}</span>}
      </div>
      {op && <div className="mt-1 text-[11px] text-ink-2 italic">{op.reason}</div>}
    </li>
  )
}

export function SizeGauge({ state, version }: { state: RunState; version?: number }) {
  const v = state.memory.find((m) => m.version === version) ?? state.memory.at(-1)
  const cap = state.config?.memory_cap_tokens ?? v?.cap ?? 0
  const size = v?.size ?? 0
  const raw = state.gens.filter((g) => g.metrics && (!v || g.gen <= v.gen)).at(-1)?.metrics?.raw_log_tokens ?? 0
  const max = Math.max(cap, raw, 1)
  const isRaw = state.setup === 'raw_log'
  return (
    <div className="space-y-2.5 text-xs">
      <div>
        <div className="mb-1 flex justify-between text-ink-2">
          <span>{isRaw ? 'Raw log memory' : 'Curated memory'}</span>
          <span className="tabular">
            {tokens(size)} / {tokens(cap)} cap
          </span>
        </div>
        <div className="relative h-2.5 rounded-full bg-surface-2">
          <div className={cn('h-full rounded-full', size > cap ? 'bg-critical' : 'bg-series-1')} style={{ width: `${Math.min(100, (size / max) * 100)}%` }} />
          <div className="absolute top-[-3px] h-4 w-0.5 bg-ink-2" style={{ left: `${(cap / max) * 100}%` }} title="size cap" />
        </div>
      </div>
      {!isRaw && raw > 0 && (
        <div>
          <div className="mb-1 flex justify-between text-ink-2">
            <span>Uncurated log, same reports</span>
            <span className="tabular">{tokens(raw)}</span>
          </div>
          <div className="h-2.5 rounded-full bg-surface-2">
            <div className="h-full rounded-full bg-muted" style={{ width: `${Math.min(100, (raw / max) * 100)}%` }} />
          </div>
        </div>
      )}
    </div>
  )
}

export function MemoryBoard({ state }: { state: RunState }) {
  const versions = state.memory
  const [picked, setPicked] = useState<number | null>(null) // null = follow latest
  const current = (picked !== null ? versions.find((m) => m.version === picked) : undefined) ?? versions.at(-1)

  if (state.setup === 'no_memory' || state.setup === 'one_shot') {
    return <Card title="Memory"><p className="text-sm text-ink-2">This setup has no memory: every agent starts from zero.</p></Card>
  }
  if (!current) {
    return <Card title="Memory"><p className="text-sm text-muted">Memory is empty until the first generation is curated. Press play.</p></Card>
  }
  const opFor = (id: string) => current.ops.find((o) => o.entry_id === id && o.op !== 'remove')
  const removed = current.ops.filter((o) => o.op === 'remove')

  return (
    <div className="space-y-4">
      <Card
        title={`Memory v${current.version} · after generation ${current.gen}`}
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
              Version history: drag to see how memory changed
            </label>
            <input
              id="mem-version"
              type="range"
              min={versions[0].version}
              max={versions.at(-1)!.version}
              value={current.version}
              onChange={(e) => setPicked(Number(e.target.value))}
              className="w-full accent-[#3987e5]"
            />
            <div className="mt-1 flex justify-between text-[10px] text-muted">
              {versions.map((v) => (
                <span key={v.version}>v{v.version}</span>
              ))}
            </div>
          </div>
          <SizeGauge state={state} version={current.version} />
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-4">
        <section className="rounded-xl border border-series-1/40 bg-series-1/10 p-4 lg:order-last">
          <h3 className="flex items-center gap-2 text-xs font-semibold tracking-wide text-ink-2 uppercase">
            <Trophy size={14} /> Best so far
          </h3>
          <div className="mt-2 text-3xl font-semibold tabular">{loss(current.best.val_loss)}</div>
          <div className="mt-1 text-sm text-ink-2">{current.best.summary}</div>
          {state.baseline && (
            <div className="mt-2 text-xs text-good">
              {(((current.best.val_loss - state.baseline) / state.baseline) * 100).toFixed(1)}% vs baseline {loss(state.baseline)}
            </div>
          )}
        </section>
        {SECTIONS.map(({ key, title, icon: Icon, blurb }) => {
          const entries = current.entries.filter((e) => e.section === key)
          const gone = removed.filter((o) => o.section === key)
          return (
            <section key={key} className="rounded-xl border border-line bg-surface p-4">
              <h3 className="flex items-center gap-2 text-xs font-semibold tracking-wide text-ink-2 uppercase">
                <Icon size={14} /> {title}
                <span className="ml-auto font-normal text-muted normal-case">{entries.length}</span>
              </h3>
              <p className="mb-3 text-[11px] text-muted">{blurb}</p>
              <ul className="space-y-2">
                {entries.map((e) => (
                  <Entry key={`${e.id}-${current.version}`} e={e} op={opFor(e.id)} />
                ))}
                {gone.map((o) => (
                  <li key={`rm-${o.entry_id}`} className="rounded-lg border border-dashed border-critical/40 px-3 py-2 text-sm">
                    <div className="text-muted line-through">{o.text}</div>
                    <div className="mt-1 text-[11px] text-ink-2 italic">removed: {o.reason}</div>
                  </li>
                ))}
                {!entries.length && !gone.length && <li className="text-xs text-muted">Nothing yet</li>}
              </ul>
            </section>
          )
        })}
      </div>
    </div>
  )
}
