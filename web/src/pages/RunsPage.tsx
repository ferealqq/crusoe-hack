import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useSource } from '@/components/sources'
import { Badge, Card, Swatch } from '@/components/ui'
import type { RunInfo } from '@/lib/events'
import { loss, pct } from '@/lib/format'
import { SETUP_LABEL, setupColor, sortSetups } from '@/lib/theme'

const isLive = (r: RunInfo) => r.status === 'running' || r.status === 'stalled'

const STATUS_TONE: Record<RunInfo['status'], 'good' | 'bad' | 'warn' | 'neutral'> = {
  completed: 'good',
  running: 'bad',
  stalled: 'warn',
  failed: 'bad',
  cancelled: 'warn',
  unavailable: 'neutral',
}

function RunsTable({ runs }: { runs: RunInfo[] }) {
  const { base, source } = useSource()
  return (
    <table className="w-full text-sm tabular">
      <thead>
        <tr className="text-left text-[11px] tracking-wide text-muted uppercase">
          {['Run', 'Started', 'Status', 'Setup', 'Generations', 'Val loss', 'Test fitness', ''].map((h) => (
            <th key={h} className="pb-2 font-medium">
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {runs.map((r) => {
          const setups = sortSetups(r.setups)
          const href = `${base}/run/${r.run_id}${isLive(r) ? '' : '?autoplay=1'}`
          return setups.map((s, i) => {
            const sum = r.summary[s]
            return (
              <tr key={`${r.run_id}-${s}`} className={i === 0 ? 'border-t border-line' : ''}>
                {i === 0 && (
                  <>
                    <td className="py-2 align-top font-mono text-xs" rowSpan={setups.length}>
                      <Link className="hover:text-[#86b6ef]" to={href}>
                        {r.run_id}
                      </Link>
                      <div className="mt-1 font-sans text-[11px] text-muted">{r.dataset}</div>
                    </td>
                    <td className="py-2 align-top text-xs text-ink-2" rowSpan={setups.length}>
                      {r.started_at ? new Date(r.started_at).toLocaleString() : '—'}
                    </td>
                    <td className="py-2 align-top" rowSpan={setups.length}>
                      <Badge tone={source === 'demo' ? 'warn' : STATUS_TONE[r.status]}>{source === 'demo' ? 'recorded' : isLive(r) ? `● ${r.status}` : r.status}</Badge>
                    </td>
                  </>
                )}
                <td className="py-2">
                  <span className="flex items-center gap-2">
                    <Swatch color={setupColor(s)} /> {SETUP_LABEL[s]}
                  </span>
                </td>
                <td className="py-2">
                  {sum?.generations_done ?? 0} / {r.generations}
                </td>
                <td className="py-2">
                  {loss(sum?.l0)} → {loss(sum?.best_val)}
                </td>
                <td className="py-2">{pct(sum?.fitness)}</td>
                {i === 0 && (
                  <td className="py-2 text-right align-top" rowSpan={setups.length}>
                    <Link to={href} className="text-xs text-[#86b6ef] hover:underline">
                      {isLive(r) && source === 'live' ? 'Watch live' : 'Replay'} →
                    </Link>
                  </td>
                )}
              </tr>
            )
          })
        })}
      </tbody>
    </table>
  )
}

export function RunsPage() {
  const src = useSource()
  const { data: runs = [], isLoading, isError } = useQuery({
    queryKey: ['runs', src.source],
    queryFn: src.listRuns,
    refetchInterval: src.source === 'live' ? 5000 : false,
    retry: src.source === 'live' ? 1 : 3,
  })

  if (src.source === 'demo') {
    return (
      <div className="mx-auto max-w-[1500px] space-y-4 p-6">
        <Card title="Recorded runs" action={<span className="text-[11px] text-muted">static files bundled with the site</span>}>
          {isLoading ? (
            <p className="text-sm text-muted">Loading…</p>
          ) : runs.length ? (
            <RunsTable runs={runs} />
          ) : (
            <p className="text-sm text-muted">
              No recorded runs bundled. Export them with <code className="font-mono text-xs">uv run python scripts/export_demo.py</code>.
            </p>
          )}
        </Card>
      </div>
    )
  }

  const live = runs.filter(isLive)
  return (
    <div className="mx-auto max-w-[1500px] space-y-4 p-6">
      {isError ? (
        <div className="rounded-xl border border-critical/40 bg-critical/10 p-5 text-sm">
          <div className="font-semibold text-[#e66767]">Backend not connected</div>
          <p className="mt-1 text-ink-2">
            The live product reads runs from the API (<code className="font-mono text-xs">/research-runs</code>). Start it with{' '}
            <code className="font-mono text-xs">uv run fastapi dev app/main.py</code>, or open the{' '}
            <Link to="/demo" className="text-[#86b6ef] hover:underline">
              recorded demo
            </Link>{' '}
            instead.
          </p>
        </div>
      ) : (
        <>
          {live.length > 0 && (
            <Card title="Live now">
              <RunsTable runs={live} />
            </Card>
          )}
          <Card title="Runs on this server" action={<span className="text-[11px] text-muted">refreshes every 5 s</span>}>
            {isLoading ? (
              <p className="text-sm text-muted">Loading…</p>
            ) : runs.length ? (
              <RunsTable runs={runs.filter((r) => !isLive(r))} />
            ) : (
              <p className="text-sm text-muted">No runs yet.</p>
            )}
          </Card>
        </>
      )}
      <Card title="Start a new run">
        <p className="text-sm text-ink-2">Runs are started by the orchestrator. As soon as it writes results, the run shows up above as live and can be watched in real time.</p>
        <pre className="mt-3 overflow-x-auto rounded-lg bg-page p-3 font-mono text-xs text-ink-2">uv run python -m app.orchestrator --setup shared_brain</pre>
      </Card>
    </div>
  )
}
