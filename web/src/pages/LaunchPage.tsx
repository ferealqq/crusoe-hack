import { useQuery } from '@tanstack/react-query'
import { Play, Radio } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import { Badge, Button, Card } from '@/components/ui'
import { listRuns } from '@/lib/api'
import { loss, pct } from '@/lib/format'
import { SETUP_BLURB, SETUP_LABEL, SETUP_ORDER } from '@/lib/theme'

export function LaunchPage() {
  const navigate = useNavigate()
  const { data: runs = [], isLoading } = useQuery({ queryKey: ['runs'], queryFn: listRuns, refetchInterval: 5000 })
  const live = runs.filter((r) => r.status === 'live')
  const featured = runs.find((r) => r.setup === 'shared_brain' && !r.run_id.startsWith('sample-') && r.status === 'completed') ?? runs.find((r) => r.run_id === 'sample-shared-brain')

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <section className="rounded-2xl border border-line bg-surface p-8">
        <h1 className="text-3xl font-semibold">Shared Brain</h1>
        <p className="mt-2 max-w-2xl text-ink-2">
          Short-lived researcher agents read a shared memory before they work. A curator keeps that memory current, lean and conflict-free.
          Each generation starts fresh but smarter, and we measure the gain as validation loss.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          {featured && (
            <Button variant="primary" onClick={() => navigate(`/run/${featured.run_id}?autoplay=1`)}>
              <Play size={16} /> Replay the Shared Brain run
            </Button>
          )}
          <Button onClick={() => navigate('/compare')}>Compare setups</Button>
          {live.map((r) => (
            <Button key={r.run_id} onClick={() => navigate(`/run/${r.run_id}?live=1`)}>
              <Radio size={16} className="animate-pulse text-[#e66767]" /> Watch live: {r.run_id}
            </Button>
          ))}
        </div>
        {!live.length && <p className="mt-3 text-xs text-muted">No live run right now. Live runs appear here as soon as the orchestrator starts writing events.</p>}
      </section>

      <Card title="Runs">
        {isLoading ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : (
          <table className="w-full text-sm tabular">
            <thead>
              <tr className="text-left text-[11px] tracking-wide text-muted uppercase">
                {['Run', 'Setup', 'Status', 'Generations', 'Best val loss', 'Test fitness', ''].map((h) => (
                  <th key={h} className="pb-2 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...runs]
                .sort((a, b) => SETUP_ORDER.indexOf(a.setup ?? 'one_shot') - SETUP_ORDER.indexOf(b.setup ?? 'one_shot'))
                .map((r) => (
                  <tr key={r.run_id} className="border-t border-line">
                    <td className="py-2.5 font-mono text-xs">
                      <Link className="hover:text-[#86b6ef]" to={`/run/${r.run_id}${r.status === 'live' ? '?live=1' : ''}`}>
                        {r.run_id}
                      </Link>
                      {r.run_id.startsWith('sample-') && <span className="ml-2"><Badge>recorded sample</Badge></span>}
                    </td>
                    <td className="py-2.5">
                      {r.setup ? SETUP_LABEL[r.setup] : '—'}
                      {r.setup && <div className="text-[11px] text-muted">{SETUP_BLURB[r.setup]}</div>}
                    </td>
                    <td className="py-2.5">
                      <Badge tone={r.status === 'live' ? 'bad' : r.status === 'completed' ? 'good' : 'warn'}>{r.status}</Badge>
                    </td>
                    <td className="py-2.5">{r.generations ?? '—'}</td>
                    <td className="py-2.5">
                      {loss(r.baseline_val_loss)} → {loss(r.best_val_loss)}
                    </td>
                    <td className="py-2.5">{pct(r.fitness)}</td>
                    <td className="py-2.5 text-right">
                      <Link to={`/run/${r.run_id}${r.status === 'live' ? '?live=1' : '?autoplay=1'}`} className="text-xs text-[#86b6ef] hover:underline">
                        {r.status === 'live' ? 'Watch' : 'Replay'} →
                      </Link>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card title="Start a new run">
        <p className="text-sm text-ink-2">
          Runs are started by the orchestrator (<code className="font-mono text-xs">RunLogger</code> in <code className="font-mono text-xs">app/runlog</code>). As soon as it
          writes events, the run shows up above as live and can be watched here in real time.
        </p>
      </Card>
    </div>
  )
}
