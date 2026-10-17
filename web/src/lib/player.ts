// Drives the UI from an event log: replay (pace recorded events, with pause/speed/seek)
// or live (Server-Sent Events from a run being written). Both feed the same reducer.
import { create } from 'zustand'
import { getEvents, streamUrl } from './api'
import type { RunEvent } from './events'
import { applyEvent, emptyRun, reduceAll, type RunState } from './runState'

/** Long waits (training runs) are shortened during replay so the story keeps moving. */
const MAX_GAP_S = 4

export const SPEEDS = [1, 2, 5, 10, 20] as const

interface PlayerStore {
  runId?: string
  mode: 'idle' | 'loading' | 'replay' | 'live'
  events: RunEvent[]
  playTs: number[] // replay timeline with gaps compressed
  cursor: number // number of events applied
  speed: number
  playing: boolean
  state: RunState
  error?: string
  load: (runId: string) => Promise<void>
  play: () => void
  pause: () => void
  setSpeed: (speed: number) => void
  seek: (cursor: number) => void
  seekToGen: (gen: number) => void
  goLive: (runId: string) => void
  stop: () => void
}

let raf = 0
let lastFrame = 0
let clock = 0 // position on the playTs timeline; kept outside React state on purpose
let source: EventSource | undefined

function compress(events: RunEvent[]): number[] {
  const out: number[] = []
  events.forEach((e, i) => {
    out.push(i === 0 ? 0 : out[i - 1] + Math.min(MAX_GAP_S, Math.max(0, e.ts - events[i - 1].ts)))
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
    let state = s.state
    while (cursor < s.events.length && s.playTs[cursor] <= clock) {
      state = applyEvent(state, s.events[cursor])
      cursor++
    }
    const done = cursor >= s.events.length
    // Only touch React state when events were applied; otherwise just advance the clock,
    // so the page isn't re-rendered (charts included) on every animation frame.
    if (cursor !== s.cursor || done) set({ cursor, state, playing: !done })
    if (!done) raf = requestAnimationFrame(tick)
  }

  const stopLoops = () => {
    cancelAnimationFrame(raf)
    lastFrame = 0
    source?.close()
    source = undefined
  }

  return {
    mode: 'idle',
    events: [],
    playTs: [],
    cursor: 0,
    speed: 5,
    playing: false,
    state: emptyRun(),

    async load(runId) {
      stopLoops()
      clock = 0
      set({ runId, mode: 'loading', events: [], playTs: [], cursor: 0, playing: false, state: emptyRun(), error: undefined })
      try {
        const events = await getEvents(runId)
        if (get().runId !== runId) return
        set({ mode: 'replay', events, playTs: compress(events) })
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
      const c = Math.max(0, Math.min(cursor, s.events.length))
      clock = c === 0 ? 0 : s.playTs[c - 1]
      set({ cursor: c, state: reduceAll(s.events.slice(0, c)) })
    },

    seekToGen(gen) {
      const s = get()
      const i = s.events.findIndex((e) => e.t === 'gen.started' && e.gen === gen)
      if (i >= 0) get().seek(i + 1)
    },

    goLive(runId) {
      stopLoops()
      set({ runId, mode: 'live', events: [], playTs: [], cursor: 0, playing: true, state: emptyRun(), error: undefined })
      source = new EventSource(streamUrl(runId, -1))
      source.onmessage = (msg) => {
        const e = JSON.parse(msg.data) as RunEvent
        const s = get()
        set({ events: [...s.events, e], cursor: s.cursor + 1, state: applyEvent(s.state, e) })
        if (e.t === 'run.finished') {
          source?.close()
          set({ playing: false })
        }
      }
      source.onerror = () => {
        if (get().state.finished) return
        set({ error: 'Live stream disconnected' })
      }
    },

    stop() {
      stopLoops()
      set({ playing: false })
    },
  }
})
