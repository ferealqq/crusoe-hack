import { CheckCircle2, CircleSlash, Lightbulb, XCircle } from 'lucide-react'
import { loss, signedPct } from '@/lib/format'
import type { RunState } from '@/lib/runState'
import { FOCUS_SHAPE, focusColor } from '@/lib/theme'
import { Badge, Card, Swatch } from './ui'

const ACTION = {
  top: 'Re-run; fact if it holds',
  middle: 'Keep new ideas as open questions',
  bottom: 'Record as dead ends',
} as const

export function CuratorPanel({ state }: { state: RunState }) {
  const g = [...state.gens].reverse().find((x) => x.ranking)
  if (state.setup !== 'shared_brain') {
    return (
      <Card title="Curator">
        <p className="text-sm text-ink-2">
          No curator in this setup.{' '}
          {state.setup === 'raw_log' ? 'Reports are appended to memory verbatim.' : 'Nothing is remembered between generations.'}
        </p>
      </Card>
    )
  }
  return (
    <Card title={g ? `Curator · generation ${g.gen}` : 'Curator'} action={g && <span className="text-[11px] text-muted">ranked by gain over gen start {loss(g.startBest)}</span>}>
      {!g ? (
        <p className="text-sm text-muted">Waiting for researchers to report…</p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-[11px] tracking-wide text-muted uppercase">
              <th className="pb-2 font-medium">#</th>
              <th className="pb-2 font-medium">Researcher</th>
              <th className="pb-2 text-right font-medium">Val</th>
              <th className="pb-2 text-right font-medium">Gain</th>
              <th className="pb-2 pl-4 font-medium">Action</th>
            </tr>
          </thead>
          <tbody className="tabular">
            {g.ranking!.map((r, i) => {
              const focus = state.agents[r.agent_id]?.focus
              const v = g.verifications.find((x) => x.agentId === r.agent_id)
              return (
                <tr key={r.agent_id} className="border-t border-line align-top">
                  <td className="py-2 text-muted">{i + 1}</td>
                  <td className="py-2">
                    <span className="flex items-center gap-2 capitalize">
                      <Swatch color={focusColor(focus)} shape={focus ? FOCUS_SHAPE[focus] : 'circle'} />
                      {focus}
                      {r.tie && <Badge>tie (noise)</Badge>}
                    </span>
                  </td>
                  <td className="py-2 text-right">{loss(r.best_val_loss)}</td>
                  <td className={`py-2 text-right ${r.gain > 0 ? 'text-good' : 'text-ink-2'}`}>{signedPct(r.gain)}</td>
                  <td className="py-2 pl-4 text-xs">
                    <div className="flex items-center gap-1.5 text-ink-2">
                      {r.bucket === 'top' && <CheckCircle2 size={13} className="text-[#86b6ef]" />}
                      {r.bucket === 'middle' && <Lightbulb size={13} className="text-warning" />}
                      {r.bucket === 'bottom' && <CircleSlash size={13} className="text-muted" />}
                      {ACTION[r.bucket]}
                    </div>
                    {v && (
                      <div className={`mt-1 flex items-center gap-1.5 ${v.held ? 'text-good' : 'text-[#e66767]'}`}>
                        {v.held ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
                        re-run {loss(v.rerun)} · {v.held ? 'holds → memory' : 'did not hold'}
                      </div>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </Card>
  )
}
