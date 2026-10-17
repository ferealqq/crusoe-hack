import { X } from 'lucide-react'
import { useEffect } from 'react'
import { clock, cn, loss, pct, tokens } from '@/lib/format'
import type { RunState } from '@/lib/runState'
import { FOCUS_SHAPE, focusColor } from '@/lib/theme'
import { Badge, Swatch } from './ui'

export function TraceDrawer({ state, agentId, onClose }: { state: RunState; agentId: string | null; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const a = agentId ? state.agents[agentId] : undefined
  if (!a) return null
  const ttft = a.ttfts.length ? Math.round(a.ttfts.reduce((x, y) => x + y, 0) / a.ttfts.length) : undefined

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/50" onClick={onClose}>
      <aside
        className="flex h-full w-full max-w-2xl flex-col border-l border-line bg-page"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label={`Trace for ${a.id}`}
      >
        <header className="flex items-start justify-between border-b border-line p-4">
          <div>
            <div className="flex items-center gap-2 text-lg font-semibold capitalize">
              <Swatch color={a.role === 'curator' ? '#f0efec' : focusColor(a.focus)} shape={a.focus ? FOCUS_SHAPE[a.focus] : 'circle'} />
              {a.role === 'curator' ? 'Curator' : `${a.focus} researcher`} · gen {a.gen}
            </div>
            <div className="mt-1 text-xs text-muted">{a.model}</div>
            <div className="mt-2 flex flex-wrap gap-2 text-xs">
              <Badge>{a.runs} runs</Badge>
              <Badge>{a.searches} searches</Badge>
              {a.role === 'researcher' && <Badge tone="accent">best val {loss(a.bestValLoss)}</Badge>}
              <Badge>
                {tokens(a.promptTokens)} prompt tokens · {pct(a.promptTokens ? a.cachedTokens / a.promptTokens : undefined, 0)} cached
              </Badge>
              {ttft !== undefined && <Badge>TTFT {ttft} ms</Badge>}
            </div>
          </div>
          <button onClick={onClose} className="rounded-md p-1 text-ink-2 hover:bg-surface-2" aria-label="Close">
            <X size={18} />
          </button>
        </header>
        <ol className="flex-1 space-y-3 overflow-y-auto p-4">
          {a.trace.map((t) => (
            <li key={t.seq} className="flex gap-3">
              <span className="w-10 shrink-0 pt-0.5 text-xs text-muted tabular">{clock(t.ts)}</span>
              <div
                className={cn(
                  'min-w-0 flex-1 rounded-lg border px-3 py-2 text-sm',
                  t.kind === 'call' && 'border-line bg-surface font-mono text-xs text-ink',
                  t.kind === 'result' && 'border-line bg-surface-2 font-mono text-xs text-ink-2',
                  t.kind === 'message' && 'border-transparent text-ink-2 italic',
                  t.kind === 'report' && 'border-series-1/40 bg-series-1/10 text-ink',
                )}
              >
                {t.kind === 'call' && <span className="text-[#86b6ef]">→ </span>}
                {t.kind === 'result' && <span className="text-muted">← </span>}
                {t.kind === 'report' && <div className="mb-1 text-[11px] font-semibold tracking-wide text-ink-2 uppercase not-italic">Final report</div>}
                {t.text}
                {t.detail && <pre className="mt-2 overflow-x-auto rounded bg-page p-2 text-[11px] whitespace-pre text-ink-2">{t.detail}</pre>}
              </div>
            </li>
          ))}
        </ol>
      </aside>
    </div>
  )
}
