// Prompt → run → result, in one screen. The chat narrates the run from the same RunState the
// mission control uses: demo plays a recorded run in ~10 s, live follows the run over SSE.
import { ArrowRight, BarChart3, Bot, CheckCircle2, FastForward, Loader2, LayoutDashboard, Radio, RotateCcw, Sparkles, Trophy } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { AgentGrid } from '@/components/AgentCards'
import { GenTimeline } from '@/components/GenTimeline'
import { LossChart } from '@/components/LossChart'
import { useSource } from '@/components/sources'
import { TraceDrawer } from '@/components/TraceDrawer'
import { Badge, Button, Card, Swatch } from '@/components/ui'
import type { Setup } from '@/lib/events'
import { clock, cn, loss, pct } from '@/lib/format'
import { usePlayer } from '@/lib/player'
import { DATASET, INITIAL_PROMPT, TASK } from '@/lib/prompt'
import { currentBest, PHASES, type GenState, type RunState, type SetupState } from '@/lib/runState'
import { focusColor, focusShape, SETUP_LABEL, setupColor, sortSetups } from '@/lib/theme'

const STORY_SECONDS = 10

const PHASE_TEXT: Record<(typeof PHASES)[number], string> = {
  research: 'Researchers are running experiments',
  reported: 'Researchers reported back',
  curate: 'Curator is reading the reports',
  verify: 'Curator re-checks the top claim on 3 seeds',
  memory: 'Curator is updating the shared memory',
  pick: 'Picking the new best train.py',
  test: 'Scoring on the held-out test split',
  done: 'Done',
}

const gainOf = (run: RunState, st: SetupState) => {
  const l0 = run.baseline?.l0
  const best = currentBest(run, st)
  return l0 && best ? (l0 - best) / l0 : 0
}

function Msg({ icon, title, children, pending }: { icon: ReactNode; title: ReactNode; children?: ReactNode; pending?: boolean }) {
  return (
    <div className="anim-rise flex gap-3">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-surface-2 text-ink-2">{icon}</div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-sm font-medium text-ink">
          {title}
          {pending && <Loader2 size={13} className="animate-spin text-muted" />}
        </div>
        {children && <div className="mt-1 space-y-1.5 text-sm text-ink-2">{children}</div>}
      </div>
    </div>
  )
}

function GenMsg({ g, st, run }: { g: GenState; st: SetupState; run: RunState }) {
  const m = g.metrics
  const researchers = g.agentKeys.map((k) => st.agents[k]).filter((a) => a?.role === 'researcher')
  const l0 = run.baseline?.l0
  return (
    <Msg icon={<span className="text-[11px] font-semibold">G{g.gen}</span>} title={`Generation ${g.gen}`} pending={!m}>
      <div className="flex flex-wrap gap-1.5">
        {researchers.map((a) => {
          const finished = a.status === 'completed' || a.status === 'failed'
          return (
            <span key={a.key} className="flex items-center gap-1.5 rounded-md border border-line px-1.5 py-0.5 text-xs capitalize">
              <Swatch color={focusColor(a.focus)} shape={focusShape(a.focus)} />
              {a.focus}
              {finished ? <CheckCircle2 size={11} className="text-good" /> : <Loader2 size={11} className="animate-spin text-muted" />}
            </span>
          )
        })}
      </div>
      {m ? (
        <p className="anim-fade">
          {m.runs} experiments{g.memoryEdits > 0 && `, ${g.memoryEdits} memory edits`}. Best val{' '}
          <span className="font-semibold text-ink tabular">{loss(m.end_best_val)}</span>
          {l0 && <span className="text-good"> ({pct((l0 - m.end_best_val) / l0)} better than baseline)</span>}
          {m.new_best && <Badge tone="good">new best v{m.best_version}</Badge>}
        </p>
      ) : (
        <p className="text-muted">{PHASE_TEXT[g.phase]}…</p>
      )}
    </Msg>
  )
}

/** Assistant side of the chat, derived from the run state so seek, replay and live all agree. */
function Narration({ run, primary, live }: { run: RunState; primary?: SetupState; live: boolean }) {
  const cfg = run.config
  const setups = sortSetups(Object.keys(run.setups) as Setup[]).map((s) => run.setups[s]!)
  return (
    <>
      {!cfg && <Msg icon={<Bot size={14} />} title={live ? 'Connecting to the run…' : 'Loading the recorded run…'} pending />}
      {cfg && (
        <Msg icon={<Bot size={14} />} title="Starting the research run">
          <p>
            {cfg.setups.map((s) => SETUP_LABEL[s]).join(' vs ')} on the same data and budget: {cfg.research.generations} generation{cfg.research.generations === 1 ? '' : 's'} of {cfg.research.researchers}{' '}
            researchers, each with its own focus{cfg.setups.includes('shared_brain') && ', plus a curator who keeps the shared memory'}.
          </p>
          <p className="text-xs text-muted">
            Agents on Crusoe{cfg.researcher_model && ` (${cfg.researcher_model})`}, every train.py in a Modal sandbox.
          </p>
        </Msg>
      )}
      {(run.baselineRunning || run.baseline) && (
        <Msg icon={<Bot size={14} />} title="Measuring the starting train.py" pending={run.baselineRunning}>
          {run.baseline && (
            <p className="anim-fade">
              Baseline val loss <span className="font-semibold text-ink tabular">{loss(run.baseline.l0)}</span> on {run.baseline.val_losses.length} seeds. Gaps under{' '}
              {loss(run.baseline.threshold)} count as noise.
            </p>
          )}
        </Msg>
      )}
      {primary?.gens.map((g) => <GenMsg key={g.gen} g={g} st={primary} run={run} />)}
      {run.status === 'completed' && (
        <Msg icon={<Trophy size={14} className="text-warning" />} title="Done">
          {setups.map((st) => (
            <p key={st.setup} className="flex items-center gap-2">
              <Swatch color={setupColor(st.setup)} /> {SETUP_LABEL[st.setup]}:{' '}
              {st.result?.fitness != null ? `test loss ${pct(st.result.fitness)} better than baseline` : `val loss ${pct(st.result?.val_gain)} better (test run failed)`}
            </p>
          ))}
        </Msg>
      )}
      {(run.status === 'failed' || run.status === 'cancelled') && <Msg icon={<Bot size={14} />} title={`Run ${run.status}`} />}
    </>
  )
}

function SetupRace({ run }: { run: RunState }) {
  const setups = sortSetups(Object.keys(run.setups) as Setup[]).map((s) => run.setups[s]!)
  const gains = setups.map((st) => gainOf(run, st))
  const top = Math.max(0.01, ...gains)
  return (
    <Card title="Setups side by side" action={<span className="text-[11px] text-muted">val loss gain over the baseline</span>}>
      <div className="space-y-3">
        {setups.map((st, i) => (
          <div key={st.setup} className="grid grid-cols-[120px_1fr_150px] items-center gap-3 text-sm">
            <span className="flex items-center gap-2">
              <Swatch color={setupColor(st.setup)} /> {SETUP_LABEL[st.setup]}
            </span>
            <div className="h-2.5 overflow-hidden rounded-full bg-surface-2">
              <div className="h-full rounded-full transition-[width] duration-500 ease-out" style={{ width: `${(gains[i] / top) * 100}%`, background: setupColor(st.setup) }} />
            </div>
            <span className="text-right text-xs text-ink-2 tabular">
              gen {st.gens.filter((g) => g.metrics).length}/{st.generations} · <span className="font-semibold text-ink">{loss(currentBest(run, st))}</span>{' '}
              <span className="text-good">{gains[i] > 0 && pct(gains[i])}</span>
            </span>
          </div>
        ))}
      </div>
    </Card>
  )
}

function Results({ run, onReplay }: { run: RunState; onReplay?: () => void }) {
  const { base } = useSource()
  const navigate = useNavigate()
  const setups = sortSetups(Object.keys(run.setups) as Setup[]).map((s) => run.setups[s]!)
  // Test fitness when every setup has one, else the validation gain, so setups compare like for like.
  const score = (st: SetupState) => (setups.every((x) => x.result?.fitness != null) ? st.result?.fitness : st.result?.val_gain)
  const best = Math.max(...setups.map((st) => score(st) ?? -Infinity))
  return (
    <section className="anim-rise rounded-2xl border border-good/40 bg-gradient-to-br from-good/10 to-surface p-5">
      <div className="flex items-center gap-2 text-sm font-semibold text-good">
        <Trophy size={16} /> Result
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-2">
        {setups.map((st) => (
          <div key={st.setup} className={cn('rounded-xl border bg-surface p-4', score(st) === best ? 'border-good/50' : 'border-line')}>
            <div className="flex items-center gap-2 text-sm text-ink-2">
              <Swatch color={setupColor(st.setup)} /> {SETUP_LABEL[st.setup]}
              {score(st) === best && setups.length > 1 && <Badge tone="good">best</Badge>}
            </div>
            <div className="mt-2 text-3xl font-semibold tabular">{pct(st.result?.fitness ?? st.result?.val_gain)}</div>
            <div className="text-xs text-muted">
              {st.result?.fitness != null ? 'lower test loss than the starting train.py' : 'lower validation loss (the test run failed)'}
            </div>
            <div className="mt-2 text-xs text-ink-2 tabular">
              val {loss(st.result?.l0)} → {loss(st.result?.best_val)} · {st.result?.runs ?? 0} experiments · {Math.round(st.result?.minutes ?? 0)} min
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button variant="primary" onClick={() => navigate(`${base}/run/${run.runId}`)}>
          <LayoutDashboard size={15} /> Open mission control
        </Button>
        <Button onClick={() => navigate(`${base}/results`)}>
          <BarChart3 size={15} /> Full results
        </Button>
        {onReplay && (
          <Button variant="ghost" onClick={onReplay}>
            <RotateCcw size={15} /> Replay
          </Button>
        )}
      </div>
    </section>
  )
}

export function WatchPage() {
  const { runId = '' } = useParams()
  const sentPrompt = (useLocation().state as { prompt?: string } | null)?.prompt
  const { source, base } = useSource()
  const navigate = useNavigate()
  const { open, stop, play, pause, seek, state, mode, error, cursor, events, connection, run } = usePlayer()
  const [openAgent, setOpenAgent] = useState<string | null>(null)
  const chat = useRef<HTMLDivElement>(null)

  useEffect(() => {
    open(runId, source, { autoplay: true, fitSeconds: STORY_SECONDS })
  }, [runId, source, open])
  useEffect(() => () => stop(), [stop])

  // The run's own brief once its events say so; older recordings don't carry it.
  const prompt = state.config?.prompt ?? sentPrompt ?? INITIAL_PROMPT
  const setups = sortSetups(Object.keys(state.setups) as Setup[])
  const primarySetup = setups.includes('shared_brain') ? 'shared_brain' : setups[0]
  const primary = primarySetup ? state.setups[primarySetup] : undefined
  const finished = state.status === 'completed'
  const progress = events.length ? cursor / events.length : 0
  const genCount = primary?.gens.length ?? 0
  const phase = primary?.gens.at(-1)?.phase

  // Keep the newest message in view.
  useEffect(() => {
    chat.current?.scrollTo({ top: chat.current.scrollHeight, behavior: 'smooth' })
  }, [genCount, finished, state.baseline, phase])

  const replay = () => {
    seek(0)
    play()
  }

  return (
    <div className="grid h-[calc(100vh-57px)] grid-cols-1 lg:grid-cols-[400px_minmax(0,1fr)]">
      <aside className="flex min-h-0 flex-col border-r border-line bg-surface/40">
        <div ref={chat} className="min-h-0 flex-1 space-y-5 overflow-y-auto p-5">
          <div className="prompt-box ml-auto max-w-[92%] rounded-2xl rounded-br-md border border-line bg-surface-2 px-4 py-3 text-sm leading-relaxed">
            <span className="whitespace-pre-wrap">{prompt}</span>
            <p className="mt-2 text-ink-2">Task: {TASK}</p>
            <p className="mt-2 text-xs text-muted">
              {DATASET.name} · {DATASET.source}
            </p>
          </div>
          <Narration run={state} primary={primary} live={source === 'live'} />
          {error && <div className="rounded-lg border border-critical/40 bg-critical/10 px-3 py-2 text-sm text-[#e66767]">{error}</div>}
        </div>
        <div className="border-t border-line p-3">
          <button
            onClick={() => navigate(base, { viewTransition: true })}
            className="flex w-full items-center gap-2 rounded-xl border border-line bg-surface px-3 py-2.5 text-left text-sm text-muted hover:text-ink"
          >
            <Sparkles size={14} /> New prompt <ArrowRight size={14} className="ml-auto" />
          </button>
        </div>
      </aside>

      <main className="min-h-0 space-y-4 overflow-y-auto p-5">
        <div className="flex flex-wrap items-center gap-3">
          {mode === 'live' ? (
            <span className="flex items-center gap-2 text-sm font-semibold text-[#e66767]">
              <Radio size={15} className={cn(connection === 'open' && 'animate-pulse')} /> LIVE
              <span className="font-normal text-ink-2">
                {events.length} events · run time {clock(state.ts)}
                {connection === 'reconnecting' && ' · reconnecting…'}
                {connection === 'ended' && ' · stream ended'}
              </span>
            </span>
          ) : (
            <Badge tone={source === 'demo' ? 'warn' : 'neutral'}>{source === 'demo' ? 'Recorded replay' : 'Replay of a finished run'}</Badge>
          )}
          <span className="font-mono text-xs text-muted">{runId}</span>
          {mode === 'replay' && (
            <div className="ml-auto flex items-center gap-2">
              <div className="h-1.5 w-40 overflow-hidden rounded-full bg-surface-2" aria-label="Replay progress">
                <div className="h-full rounded-full bg-series-1" style={{ width: `${progress * 100}%` }} />
              </div>
              {!finished && (
                <Button
                  variant="ghost"
                  onClick={() => {
                    pause()
                    seek(events.length)
                  }}
                >
                  <FastForward size={15} /> Skip to result
                </Button>
              )}
            </div>
          )}
        </div>

        {finished && <Results run={state} onReplay={mode === 'replay' ? replay : undefined} />}
        {primary ? (
          <>
            <GenTimeline run={state} st={primary} />
            {setups.length > 1 && <SetupRace run={state} />}
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
              <AgentGrid st={primary} onOpen={setOpenAgent} />
              <LossChart run={state} st={primary} />
            </div>
            <TraceDrawer st={primary} agentKey={openAgent} onClose={() => setOpenAgent(null)} />
          </>
        ) : (
          <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-line text-sm text-muted">
            <Loader2 size={16} className="mr-2 animate-spin" /> {mode === 'loading' ? 'Loading run…' : 'Waiting for the first events…'}
          </div>
        )}
        {run && mode === 'replay' && !finished && cursor >= events.length && <p className="text-sm text-muted">This run ended with status {run.status}.</p>}
      </main>
    </div>
  )
}
