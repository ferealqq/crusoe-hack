import { useEffect, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { AgentGrid } from '@/components/AgentCards'
import { CuratorPanel } from '@/components/CuratorPanel'
import { EventFeed } from '@/components/EventFeed'
import { GenTimeline } from '@/components/GenTimeline'
import { LossChart } from '@/components/LossChart'
import { MemoryBoard, SizeGauge } from '@/components/MemoryBoard'
import { PlayerBar } from '@/components/PlayerBar'
import { TraceDrawer } from '@/components/TraceDrawer'
import { Badge, Card } from '@/components/ui'
import { cn, pct } from '@/lib/format'
import { usePlayer } from '@/lib/player'
import { SETUP_LABEL } from '@/lib/theme'

export function RunPage() {
  const { runId = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const tab = params.get('tab') === 'memory' ? 'memory' : 'mission'
  const live = params.get('live') === '1'
  const { load, goLive, play, stop, state, mode, error, runId: loaded } = usePlayer()
  const [openAgent, setOpenAgent] = useState<string | null>(null)

  useEffect(() => {
    if (loaded === runId && mode !== 'idle') return
    if (live) goLive(runId)
    else load(runId).then(() => params.get('autoplay') === '1' && play())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, live])
  useEffect(() => () => stop(), [stop])

  return (
    <div className="mx-auto max-w-[1500px] space-y-4 p-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-lg font-semibold">{state.setup ? SETUP_LABEL[state.setup] : 'Run'}</h1>
        <span className="font-mono text-xs text-muted">{runId}</span>
        {state.config && (
          <span className="text-xs text-ink-2">
            {state.config.dataset} · {state.config.budget}
          </span>
        )}
        {state.eval && <Badge tone="good">test fitness {pct(state.eval.fitness)}</Badge>}
        <nav className="ml-auto flex rounded-lg border border-line p-0.5 text-sm">
          {(['mission', 'memory'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setParams((p) => (t === 'memory' ? p.set('tab', 'memory') : p.delete('tab'), p))}
              className={cn('rounded-md px-3 py-1', tab === t ? 'bg-surface-2 text-ink' : 'text-muted hover:text-ink')}
            >
              {t === 'mission' ? 'Mission control' : `Memory${state.memory.length ? ` v${state.memory.at(-1)!.version}` : ''}`}
            </button>
          ))}
        </nav>
      </div>
      {error && <div className="rounded-lg border border-critical/40 bg-critical/10 px-3 py-2 text-sm text-[#e66767]">{error}</div>}
      <PlayerBar />
      <GenTimeline state={state} />
      {tab === 'mission' ? (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
          <div className="space-y-4">
            <AgentGrid state={state} onOpen={setOpenAgent} />
            <EventFeed state={state} />
          </div>
          <div className="space-y-4">
            <LossChart state={state} />
            <CuratorPanel state={state} />
            {(state.setup === 'shared_brain' || state.setup === 'raw_log') && (
              <Card title="Memory size" action={<Link to="?tab=memory" className="text-xs text-[#86b6ef] hover:underline">Open memory →</Link>}>
                <SizeGauge state={state} />
              </Card>
            )}
          </div>
        </div>
      ) : (
        <MemoryBoard state={state} />
      )}
      <TraceDrawer state={state} agentId={openAgent} onClose={() => setOpenAgent(null)} />
    </div>
  )
}
