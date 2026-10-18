// Drives the UI from a run's event log, following docs/research-telemetry.md:
// - replay: the whole history is fetched once and paced here (play, pause, speed, seek);
//   state at a cursor uses only the events before it, like /state?at=N.
// - live: fetch the history, render it, then follow /stream?after=<last seq> over SSE
//   (named `research` events, deduplicated by seq, closed on `stream.end`). The server
//   rebuilds events from data/ files; `stream.reset` means earlier history changed, so reload.
import { create } from 'zustand'
import { runEvents, runInfo, streamUrl, type Source } from './api'
import type { ResearchEvent, RunInfo } from './events'
import { applyEvents, emptyRun, reduceAll, type RunState } from './runState'

/** Long waits (training runs, slow turns) are shortened to keep the story moving. */
const MAX_GAP_S = 3
const SILENT = new Set(['run.heartbeat']) // advance no replay time

export const SPEEDS = [1, 5, 10, 30, 100] as const

interface PlayerStore {
  run?: RunInfo
  source?: Source
  mode: 'idle' | 'loading' | 'replay' | 'live'
  events: ResearchEvent[]
  playTs: number[] // replay timeline in seconds, gaps compressed
  cursor: number // number of events applied
  speed: number
  playing: boolean
  connection?: 'open' | 'reconnecting' | 'ended'
  state: RunState
  error?: string
  /** demo: static recorded files, always replayed; live: the API, followed over SSE while running. */
  /** `fitSeconds`: pick the replay speed so the whole run plays in about that long. */
  open: (runId: string, source: Source, opts?: { autoplay?: boolean; fitSeconds?: number }) => Promise<void>
  play: () => void
  pause: () => void
  setSpeed: (speed: number) => void
  seek: (cursor: number) => void
  seekToSeq: (seq: number) => void
  toReplay: () => void
  stop: () => void
}

let raf = 0
let lastFrame = 0
let clock = 0 // position on the playTs timeline; kept outside React state on purpose
let source: EventSource | undefined

function compress(events: ResearchEvent[]): number[] {
  const out: number[] = []
  let prevMs = 0
  events.forEach((e, i) => {
    const gap = SILENT.has(e.type) ? 0 : Math.min(MAX_GAP_S, Math.max(0, (e.elapsed_ms - prevMs) / 1000))
    out.push(i === 0 ? 0 : out[i - 1] + gap)
    if (!SILENT.has(e.type)) prevMs = e.elapsed_ms
  })
  return out
}

export const usePlayer = create<PlayerStore>((set, get) => {
  const tick = (now: number) => {
    const s = get()
    if (!s.playing || s.mode !== 'replay') return
    const dt = lastFrame ? (now - lastFrame) / 1000 : 0
    lastFrame = now
    clock += dt * s.speed
    let cursor = s.cursor
    while (cursor < s.events.length && s.playTs[cursor] <= clock) cursor++
    const done = cursor >= s.events.length
    // Only touch React state when events were applied, so charts aren't re-rendered every frame.
    if (cursor !== s.cursor || done) set({ cursor, state: applyEvents(s.state, s.events.slice(s.cursor, cursor)), playing: !done })
    if (!done) raf = requestAnimationFrame(tick)
  }

  const stopLoops = () => {
    cancelAnimationFrame(raf)
    lastFrame = 0
    source?.close()
    source = undefined
  }

  const follow = (runId: string) => {
    source = new EventSource(streamUrl(runId, get().state.lastSeq))
    set({ connection: 'open' })
    source.addEventListener('research', (msg) => {
      const e = JSON.parse((msg as MessageEvent).data) as ResearchEvent
      const s = get()
      if (e.seq <= s.state.lastSeq) return // reconnects must be idempotent
      set({ events: [...s.events, e], cursor: s.cursor + 1, state: applyEvents(s.state, [e]), connection: 'open' })
    })
    // Close on stream.end, otherwise the browser reconnects after the response ends.
    source.addEventListener('stream.reset', () => {
      stopLoops()
      get().open(runId, 'live')
    })
    source.addEventListener('stream.end', () => {
      source?.close()
      set({ connection: 'ended', playing: false })
    })
    source.onerror = () => {
      // EventSource retries by itself (the server resumes from Last-Event-ID); CLOSED means it gave up.
      if (source?.readyState === EventSource.CLOSED) set({ connection: 'ended', playing: false, error: 'Live stream closed by the server' })
      else set({ connection: 'reconnecting' })
    }
  }

  return {
    mode: 'idle',
    events: [],
    playTs: [],
    cursor: 0,
    speed: 10,
    playing: false,
    state: emptyRun(),

    async open(runId, src, opts) {
      stopLoops()
      clock = 0
      set({ run: undefined, source: src, mode: 'loading', events: [], playTs: [], cursor: 0, playing: false, connection: undefined, state: emptyRun(), error: undefined })
      try {
        const run = await runInfo(runId, src)
        if (run.status === 'unavailable') {
          set({ run, mode: 'idle', error: 'This older run has no event recording; record a new run to replay it.' })
          return
        }
        const events = await runEvents(run, src)
        if (get().mode !== 'loading') return // another open() won
        const live = src === 'live' && (run.status === 'running' || run.status === 'stalled')
        if (live) {
          set({ run, mode: 'live', events, cursor: events.length, state: reduceAll(events), playing: true })
          follow(runId)
        } else {
          const playTs = compress(events)
          const speed = opts?.fitSeconds ? Math.max(1, (playTs.at(-1) ?? 0) / opts.fitSeconds) : 10
          set({ run, mode: 'replay', events, playTs, speed })
          if (opts?.autoplay) get().play()
        }
      } catch (e) {
        set({ mode: 'idle', error: String(e) })
      }
    },

    play() {
      const s = get()
      if (s.mode !== 'replay') return
      if (s.cursor >= s.events.length) get().seek(0)
      set({ playing: true })
      lastFrame = 0
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(tick)
    },

    pause() {
      cancelAnimationFrame(raf)
      set({ playing: false })
    },

    setSpeed(speed) {
      set({ speed })
    },

    seek(cursor) {
      const s = get()
      if (s.mode !== 'replay') return
      const c = Math.max(0, Math.min(cursor, s.events.length))
      clock = c === 0 ? 0 : s.playTs[c - 1]
      set({ cursor: c, state: reduceAll(s.events.slice(0, c)) })
    },

    seekToSeq(seq) {
      const i = get().events.findIndex((e) => e.seq === seq)
      if (i >= 0) get().seek(i + 1)
    },

    toReplay() {
      stopLoops()
      const s = get()
      set({ mode: 'replay', playTs: compress(s.events), connection: undefined, playing: false })
      get().seek(0)
      get().play()
    },

    stop() {
      stopLoops()
      set({ playing: false })
    },
  }
})
