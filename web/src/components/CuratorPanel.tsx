import { CheckCircle2, XCircle } from 'lucide-react'
import { loss, signedPct } from '@/lib/format'
import type { SetupState } from '@/lib/runState'
import { focusColor, focusShape } from '@/lib/theme'
import { Badge, Card, Swatch } from './ui'

export function CuratorPanel({ st }: { st: SetupState }) {
  if (st.setup !== 'shared_brain') {
    return (
      <Card title="Curator">
        <p className="text-sm text-ink-2">
          No curator in this setup. {st.setup === 'raw_log' ? 'Reports are appended to memory verbatim.' : 'Nothing is remembered between generations.'}
        </p>
      </Card>
    )
  }
  // The latest generation the curator has something to show for: re-checks while it runs, the ranking once done.
  const g = [...st.gens].reverse().find((x) => x.verifications.length || x.metrics)
  const ranking = g?.metrics?.researcher_best ?? []

  return (
    <Card title={g ? `Curator · generation ${g.gen}` : 'Curator'} action={g && <span className="text-[11px] text-muted">gain vs gen start {loss(g.startBest)}</span>}>
      {!g ? (
        <p className="text-sm text-muted">Waiting for researchers to report…</p>
      ) : (
        <div className="space-y-4">
          {g.verifications.length > 0 && (
            <div>
              <h3 className="mb-2 text-[11px] tracking-wide text-muted uppercase">Re-checks on fresh seeds</h3>
              <ul className="space-y-1.5 text-sm">
                {g.verifications.map((v) => (
                  <li key={v.seq} className="flex items-start gap-2">
                    {v.beats_best ? <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-good" /> : <XCircle size={15} className="mt-0.5 shrink-0 text-[#e66767]" />}
                    <span className="min-w-0">
                      <span className="font-mono text-xs">{v.source}</span>{' '}
                      <span className="text-ink-2 tabular">
                        mean {loss(v.mean)}
                        {v.spread !== null && ` ± ${v.spread.toFixed(4)}`} on seeds {v.seeds.join(', ')}
                      </span>{' '}
                      <span className={v.beats_best ? 'text-good' : 'text-[#e66767]'}>{v.beats_best ? 'holds → memory' : 'did not hold'}</span>
                      {v.by === 'orchestrator' && (
                        <span className="ml-1.5">
                          <Badge>orchestrator fallback</Badge>
                        </span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {ranking.length > 0 && (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[11px] tracking-wide text-muted uppercase">
                  <th className="pb-2 font-medium">#</th>
                  <th className="pb-2 font-medium">Researcher</th>
                  <th className="pb-2 font-medium">Best idea</th>
                  <th className="pb-2 text-right font-medium">Val</th>
                  <th className="pb-2 text-right font-medium">Gain</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {[...ranking]
                  .sort((a, b) => a.rank - b.rank)
                  .map((r) => {
                    const val = r.best?.val_loss
                    const gain = val != null ? (g.startBest - val) / g.startBest : undefined
                    return (
                      <tr key={r.name} className="border-t border-line align-top">
                        <td className="py-2 text-muted">{r.rank}</td>
                        <td className="py-2">
                          <span className="flex items-center gap-2 capitalize">
                            <Swatch color={focusColor(r.focus)} shape={focusShape(r.focus)} />
                            {r.focus}
                          </span>
                        </td>
                        <td className="max-w-64 truncate py-2 text-xs text-ink-2" title={r.best?.note}>
                          {r.best ? r.best.note : 'no successful run'}
                        </td>
                        <td className="py-2 text-right">{loss(val)}</td>
                        <td className={`py-2 text-right ${gain !== undefined && gain > 0 ? 'text-good' : 'text-ink-2'}`}>{gain !== undefined ? signedPct(gain) : '—'}</td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </Card>
  )
}
