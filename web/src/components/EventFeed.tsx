import { clock, cn } from '@/lib/format'
import type { RunState, SetupState } from '@/lib/runState'
import { CHROME, focusColor, focusShape } from '@/lib/theme'
import { Card, Swatch } from './ui'

export function EventFeed({ run, st, limit = 60 }: { run: RunState; st: SetupState; limit?: number }) {
  // Run-level items (baseline, lifecycle) interleaved with this setup's.
  const items = [...run.feed, ...st.feed].sort((a, b) => b.seq - a.seq).slice(0, limit)
  return (
    <Card title="Event feed" className="flex min-h-0 flex-col" action={<span className="text-[11px] text-muted">{run.lastSeq} events</span>}>
      <ol className="-my-1 max-h-80 space-y-0.5 overflow-y-auto pr-1 font-mono text-xs" aria-live="polite">
        {items.map((f) => (
          <li key={f.seq} className="flex items-start gap-2 py-1">
            <span className="w-10 shrink-0 text-muted tabular">{clock(f.ts)}</span>
            <span className="mt-0.5">
              <Swatch color={f.focus === 'curator' ? CHROME.ink : f.focus ? focusColor(f.focus) : CHROME.axis} shape={focusShape(f.focus)} />
            </span>
            <span
              className={cn(
                'min-w-0 break-words',
                f.tone === 'neutral' && 'text-ink-2',
                f.tone === 'good' && 'text-good',
                f.tone === 'bad' && 'text-[#e66767]',
                f.tone === 'accent' && 'text-ink',
              )}
            >
              {f.focus && <span className="mr-1 text-muted capitalize">{f.focus}:</span>}
              {f.text}
            </span>
          </li>
        ))}
        {!items.length && <li className="py-4 text-center text-muted">No events yet</li>}
      </ol>
    </Card>
  )
}
