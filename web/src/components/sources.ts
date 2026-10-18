// The two halves of the app, kept apart on purpose:
// - demo: recorded runs bundled as static files (web/public/samples, exported from data/).
//   Never calls the API, works with no backend.
// - live: the product. Only the API (/research-runs): live runs over SSE, nothing bundled.
import { createContext, useContext } from 'react'
import { getDemoEvents, getDemoRun, getEvents, getRun, listDemoRuns, listRuns, type Source } from '@/lib/api'
import type { ResearchEvent, RunInfo } from '@/lib/events'

export type { Source }

export interface SourceInfo {
  source: Source
  base: string // route prefix
  label: string
  listRuns: () => Promise<RunInfo[]>
  getRun: (runId: string) => Promise<RunInfo>
  getEvents: (run: RunInfo) => Promise<ResearchEvent[]>
}

export const SOURCES: Record<Source, SourceInfo> = {
  demo: {
    source: 'demo',
    base: '/demo',
    label: 'Demo',
    listRuns: listDemoRuns,
    getRun: getDemoRun,
    getEvents: (run) => getDemoEvents(run.run_id),
  },
  live: {
    source: 'live',
    base: '/app',
    label: 'Live',
    listRuns,
    getRun,
    getEvents,
  },
}

export const SourceCtx = createContext<SourceInfo>(SOURCES.demo)
export const useSource = () => useContext(SourceCtx)
