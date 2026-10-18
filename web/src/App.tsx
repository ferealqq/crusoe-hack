import { Archive, Radio } from 'lucide-react'
import { Navigate, NavLink, Outlet, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { SourceProvider } from '@/components/source'
import { SOURCES, type Source } from '@/components/sources'
import { cn } from '@/lib/format'
import { ComparePage } from '@/pages/ComparePage'
import { HomePage } from '@/pages/HomePage'
import { RunPage } from '@/pages/RunPage'
import { RunsPage } from '@/pages/RunsPage'
import { WatchPage } from '@/pages/WatchPage'

const link = ({ isActive }: { isActive: boolean }) => cn('rounded-md px-3 py-1.5 text-sm', isActive ? 'bg-surface-2 text-ink' : 'text-ink-2 hover:text-ink')

/** Which half of the app the URL is in; drives the header switch. */
function useArea(): Source {
  const { pathname } = useLocation()
  return pathname.startsWith(SOURCES.live.base) ? 'live' : 'demo'
}

/** Top-right switch: each side opens on its own prompt page. */
function AreaSwitch() {
  const area = useArea()
  const navigate = useNavigate()
  const go = (s: Source) => s !== area && navigate(SOURCES[s].base, { viewTransition: true })
  return (
    <div className="flex rounded-full border border-line bg-surface p-0.5 text-sm" role="group" aria-label="Data source">
      <button
        onClick={() => go('demo')}
        aria-pressed={area === 'demo'}
        className={cn('flex items-center gap-1.5 rounded-full px-3 py-1 transition-colors', area === 'demo' ? 'bg-warning/20 text-warning' : 'text-ink-2 hover:text-ink')}
      >
        <Archive size={14} /> Demo
      </button>
      <button
        onClick={() => go('live')}
        aria-pressed={area === 'live'}
        className={cn('flex items-center gap-1.5 rounded-full px-3 py-1 transition-colors', area === 'live' ? 'bg-series-1/20 text-[#86b6ef]' : 'text-ink-2 hover:text-ink')}
      >
        <Radio size={14} /> Live
      </button>
    </div>
  )
}

/** Always visible on demo data pages, so nobody mistakes a recording for a live run. */
function DemoBanner() {
  return (
    <div className="border-b border-warning/30 bg-warning/10 px-6 py-1.5 text-xs text-warning">
      <span className="font-semibold">Demo: recorded data.</span>{' '}
      <span className="text-ink-2">
        Finished research runs committed to <code className="font-mono">data/research_runs</code>, replayed in the browser. Nothing here calls Crusoe or Modal.
      </span>
    </div>
  )
}

function AreaLayout({ source, banner }: { source: Source; banner?: boolean }) {
  return (
    <SourceProvider source={source}>
      {banner && source === 'demo' && <DemoBanner />}
      <Outlet />
    </SourceProvider>
  )
}

function AreaNav() {
  const base = SOURCES[useArea()].base
  return (
    <nav className="flex gap-1">
      <NavLink to={`${base}/runs`} className={link}>
        Runs
      </NavLink>
      <NavLink to={`${base}/results`} className={link}>
        Results
      </NavLink>
    </nav>
  )
}

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 flex h-[57px] items-center gap-4 border-b border-line bg-page/90 px-6 backdrop-blur">
        <NavLink to={SOURCES[useArea()].base} viewTransition className="flex items-center gap-2 font-semibold">
          <span className="inline-block h-3 w-3 rounded-full bg-series-1" /> Shared Brain
        </NavLink>
        <AreaNav />
        <div className="ml-auto">
          <AreaSwitch />
        </div>
      </header>
      <Routes>
        <Route path="/" element={<Navigate to={SOURCES.demo.base} replace />} />
        {(['demo', 'live'] as const).map((s) => (
          <Route key={s} path={SOURCES[s].base}>
            {/* prompt → watch: full-screen, no banner (the watch page labels the replay itself) */}
            <Route element={<AreaLayout source={s} />}>
              <Route index element={<HomePage />} />
              <Route path="watch/:runId" element={<WatchPage />} />
            </Route>
            <Route element={<AreaLayout source={s} banner />}>
              <Route path="runs" element={<RunsPage />} />
              <Route path="run/:runId" element={<RunPage />} />
              <Route path="results" element={<ComparePage />} />
            </Route>
          </Route>
        ))}
        {/* old links */}
        <Route path="/compare" element={<Navigate to="/demo/results" replace />} />
        <Route path="/run/:runId" element={<Navigate to="/demo/runs" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  )
}
