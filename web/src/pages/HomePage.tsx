import { useQuery } from '@tanstack/react-query'
import { ArrowUp, Database, Layers, Loader2, Sparkles } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useSource } from '@/components/sources'
import { StartError, startRun } from '@/lib/api'
import type { RunInfo } from '@/lib/events'
import { cn } from '@/lib/format'
import { DATASET, INITIAL_PROMPT, TASK } from '@/lib/prompt'

const TYPE_MS = 1200 // the prompt types itself in, then waits for send

const isLive = (r: RunInfo) => r.status === 'running' || r.status === 'stalled'

const GENERATIONS = [1, 2, 3, 5] as const

export function HomePage() {
  const navigate = useNavigate()
  const src = useSource()
  const runs = useQuery({
    queryKey: ['runs', src.source],
    queryFn: src.listRuns,
    refetchInterval: src.source === 'live' ? 5000 : false,
    retry: src.source === 'live' ? 1 : 3,
  })
  const [typed, setTyped] = useState(0)
  const done = typed >= INITIAL_PROMPT.length
  // Live: once typed in, the prompt is an editable text box; demo keeps it read-only.
  const editable = src.source === 'live'
  const [text, setText] = useState(INITIAL_PROMPT)
  const input = useRef<HTMLTextAreaElement>(null)
  // Demo plays the recorded run; live starts a new run with the prompt (one at a time).
  const recorded = runs.data?.find((r) => r.replay_available) ?? runs.data?.[0]
  const inProgress = editable ? runs.data?.find(isLive) : undefined
  const [generations, setGenerations] = useState<number>(5)
  const [starting, setStarting] = useState(false)
  const [failure, setFailure] = useState<{ message: string; runId?: string }>()
  const canSend = editable ? !runs.isError && !runs.isLoading && !starting : !!recorded

  useEffect(() => {
    if (editable && done) {
      const el = input.current
      el?.focus()
      el?.setSelectionRange(el.value.length, el.value.length)
    }
  }, [editable, done])

  useEffect(() => {
    const start = performance.now()
    let raf = 0
    const step = (now: number) => {
      const n = Math.ceil(((now - start) / TYPE_MS) * INITIAL_PROMPT.length)
      setTyped(Math.min(n, INITIAL_PROMPT.length))
      if (n < INITIAL_PROMPT.length) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [])

  const send = async () => {
    if (!canSend) return
    if (!done) setTyped(INITIAL_PROMPT.length)
    if (!editable) {
      navigate(`${src.base}/watch/${recorded!.run_id}`, { viewTransition: true, state: { prompt: INITIAL_PROMPT } })
      return
    }
    const prompt = text.trim()
    if (!prompt) return
    setStarting(true)
    setFailure(undefined)
    try {
      const runId = await startRun(prompt, generations)
      navigate(`${src.base}/watch/${runId}`, { viewTransition: true, state: { prompt } })
    } catch (e) {
      setFailure(e instanceof StartError ? { message: e.message, runId: e.runId } : { message: String(e) })
      setStarting(false)
    }
  }

  const status =
    src.source === 'demo'
      ? 'Replays a recorded run in about 10 seconds. No backend, nothing is running.'
      : runs.isError
        ? 'Backend not connected. Start it with: uv run fastapi dev app/main.py'
        : runs.isLoading
          ? 'Connecting to the server…'
          : starting
            ? 'Starting the agents on Crusoe and the sandboxes on Modal…'
            : failure
              ? failure.message
              : inProgress
                ? 'A run is already in progress. Watch it, or send when it has finished.'
                : 'Sending starts a real run: researcher agents on Crusoe, every experiment in a Modal sandbox.'
  const bad = runs.isError || (!!failure && !starting)
  const watchId = failure?.runId ?? inProgress?.run_id

  return (
    <div className="relative flex min-h-[calc(100vh-57px)] flex-col items-center justify-center overflow-hidden px-6 pb-24">
      <div aria-hidden className="hero-glow pointer-events-none absolute inset-0" />
      <div className="relative w-full max-w-2xl">
        <h1 className="text-center text-4xl font-semibold tracking-tight md:text-5xl">What should the team research?</h1>
        <p className="mx-auto mt-3 max-w-xl text-center text-ink-2">
          Short-lived researcher agents share a curated memory. Each generation starts fresh, but smarter.
        </p>

        <form
          className="prompt-box mt-10 rounded-2xl border border-line bg-surface/90 p-4 transition-colors focus-within:border-series-1/60 shadow-2xl shadow-black/40 backdrop-blur"
          onSubmit={(e) => {
            e.preventDefault()
            send()
          }}
        >
          <div className="min-h-24 text-[15px] leading-relaxed text-ink" aria-label="Prompt">
            {editable && done ? (
              <textarea
                ref={input}
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={starting}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    send()
                  }
                }}
                rows={3}
                placeholder="Describe what the team should research…"
                className="block w-full resize-none bg-transparent outline-none placeholder:text-muted"
              />
            ) : (
              <>
                {INITIAL_PROMPT.slice(0, typed)}
                {!done && <span className="caret" />}
              </>
            )}
            {done && <p className="anim-fade mt-2 text-ink-2">Task: {TASK}</p>}
          </div>
          <div className="mt-3 flex items-center gap-2">
            <span className="flex items-center gap-1.5 rounded-lg border border-line px-2 py-1 text-xs text-ink-2">
              <Database size={13} /> {DATASET.name} · {DATASET.source} · {DATASET.trainRows.toLocaleString()} rows
            </span>
            <span className="flex items-center gap-1.5 rounded-lg border border-line px-2 py-1 text-xs text-ink-2">
              <Layers size={13} /> Shared Brain vs No memory
            </span>
            {editable && (
              <label className="flex items-center gap-1.5 rounded-lg border border-line px-2 py-1 text-xs text-ink-2">
                Generations
                <select
                  value={generations}
                  onChange={(e) => setGenerations(Number(e.target.value))}
                  disabled={starting}
                  className="bg-transparent text-ink outline-none"
                >
                  {GENERATIONS.map((g) => (
                    <option key={g} value={g} className="bg-surface">
                      {g}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <button
              type="submit"
              disabled={!canSend}
              aria-label="Send"
              className={cn(
                'ml-auto flex h-9 w-9 items-center justify-center rounded-full bg-ink text-page transition disabled:opacity-30',
                done && canSend && !inProgress && 'send-ready',
              )}
            >
              {starting ? <Loader2 size={18} className="animate-spin" /> : <ArrowUp size={18} />}
            </button>
          </div>
        </form>
        <p className={cn('mt-4 flex items-center justify-center gap-1.5 text-center text-xs whitespace-pre-line', bad ? 'text-[#e66767]' : 'text-muted')}>
          <Sparkles size={12} className="shrink-0" /> {status}
          {editable && watchId && (
            <Link to={`${src.base}/watch/${watchId}`} viewTransition className="ml-1 text-[#86b6ef] hover:underline">
              Watch it live →
            </Link>
          )}
        </p>
      </div>
    </div>
  )
}
