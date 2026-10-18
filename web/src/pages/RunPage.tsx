import { useEffect, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { AgentGrid } from '@/components/AgentCards'
import { CuratorPanel } from '@/components/CuratorPanel'
import { EventFeed } from '@/components/EventFeed'
import { GenTimeline } from '@/components/GenTimeline'
import { LossChart } from '@/components/LossChart'
import { MemoryBoard, SizeGauge } from '@/components/MemoryBoard'
import { PlayerBar } from '@/components/PlayerBar'
import { useSource } from '@/components/sources'
import { TraceDrawer } from '@/components/TraceDrawer'
import { Badge, Card, Swatch } from '@/components/ui'
import type { Setup } from '@/lib/events'
import { cn, pct } from '@/lib/format'
import { usePlayer } from '@/lib/player'
import { usesMemory } from '@/lib/runState'
import { SETUP_LABEL, setupColor, sortSetups } from '@/lib/theme'

export function RunPage() {
  const { runId = '' } = useParams()
  const { source, base } = useSource()
  const [params, setParams] = useSearchParams()
  const tab = params.get('tab') === 'memory' ? 'memory' : 'mission'
  const { open, stop, state, mode, error, run } = usePlayer()
  const loadedSource = usePlayer((s) => s.source)
  const [openAgent, setOpenAgent] = useState<string | null>(null)

  useEffect(() => {
    if (run?.run_id === runId && loadedSource === source && mode !== 'idle') return
    open(runId, source, { autoplay: params.get('autoplay') === '1' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, source])
  useEffect(() => () => stop(), [stop])

  const setups = sortSetups((state.config?.setups ?? run?.setups ?? []) as Setup[])
  const picked = params.get('setup') as Setup | null
  const setup = picked && setups.includes(picked) ? picked : setups[0]
  const st = setup ? state.setups[setup] : undefined
  const set = (k: string, v?: string) => setParams((p) => (v ? p.set(k, v) : p.delete(k), p), { replace: true })

  return (
    <div className="mx-auto max-w-[1500px] space-y-4 p-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link to={`${base}/runs`} className="text-sm text-ink-2 hover:text-ink">
          ← {source === 'demo' ? 'Recorded runs' : 'Runs'}
        </Link>
        <span className="font-mono text-xs text-muted">{runId}</span>
        {source === 'demo' ? (
          <Badge tone="warn">recorded replay</Badge>
        ) : mode === 'live' ? (
          <Badge tone="bad">● live</Badge>
        ) : (
          mode === 'replay' && <Badge>replay</Badge>
        )}
        {state.config && (
          <span className="text-xs text-ink-2">
            {state.config.dataset} · {state.config.research.generations} generations · {state.config.research.researchers} researchers
          </span>
        )}
        <nav className="ml-auto flex rounded-lg border border-line p-0.5 text-sm">
          {(['mission', 'memory'] as const).map((t) => (
            <button
              key={t}
              onClick={() => set('tab', t === 'memory' ? 'memory' : undefined)}
              className={cn('rounded-md px-3 py-1', tab === t ? 'bg-surface-2 text-ink' : 'text-muted hover:text-ink')}
            >
              {t === 'mission' ? 'Mission control' : 'Memory'}
            </button>
          ))}
        </nav>
      </div>

      {setups.length > 1 && (
        <div className="flex flex-wrap gap-2" role="tablist" aria-label="Setup">
          {setups.map((s) => {
            const x = state.setups[s]
            return (
              <button
                key={s}
                role="tab"
                aria-selected={s === setup}
                onClick={() => set('setup', s)}
                className={cn(
                  'flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm',
                  s === setup ? 'border-ink-2 bg-surface-2 text-ink' : 'border-line text-ink-2 hover:text-ink',
                )}
              >
                <Swatch color={setupColor(s)} />
                {SETUP_LABEL[s]}
                {x?.result?.fitness != null && <span className="text-xs text-good">{pct(x.result.fitness)}</span>}
                {x?.status === 'running' && <span className="text-xs text-muted">gen {x.gens.at(-1)?.gen ?? 0}</span>}
              </button>
            )
          })}
          <span className="self-center text-xs text-muted">Setups ran side by side on the same data and budget.</span>
        </div>
      )}

      {error && <div className="rounded-lg border border-critical/40 bg-critical/10 px-3 py-2 text-sm text-[#e66767]">{error}</div>}
      <PlayerBar setup={setup} />
      {st ? (
        <>
          <GenTimeline run={state} st={st} />
          {tab === 'mission' ? (
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
              <div className="space-y-4">
                <AgentGrid st={st} onOpen={setOpenAgent} />
                <EventFeed run={state} st={st} />
              </div>
              <div className="space-y-4">
                <LossChart run={state} st={st} />
                <CuratorPanel st={st} />
                {usesMemory(st.setup) && (
                  <Card title="Memory size" action={<button onClick={() => set('tab', 'memory')} className="text-xs text-[#86b6ef] hover:underline">Open memory →</button>}>
                    <SizeGauge run={state} st={st} />
                  </Card>
                )}
              </div>
            </div>
          ) : (
            <MemoryBoard run={state} st={st} />
          )}
          <TraceDrawer st={st} agentKey={openAgent} onClose={() => setOpenAgent(null)} />
        </>
      ) : (
        <div className="rounded-xl border border-dashed border-line p-10 text-center text-sm text-muted">
          {mode === 'loading' ? 'Loading run…' : mode === 'replay' ? 'Press play to start the replay.' : 'Waiting for the first events…'}
        </div>
      )}
    </div>
  )
}
