import { NavLink, Route, Routes } from 'react-router-dom'
import { cn } from '@/lib/format'
import { ComparePage } from '@/pages/ComparePage'
import { LaunchPage } from '@/pages/LaunchPage'
import { RunPage } from '@/pages/RunPage'

const link = ({ isActive }: { isActive: boolean }) => cn('rounded-md px-3 py-1.5 text-sm', isActive ? 'bg-surface-2 text-ink' : 'text-ink-2 hover:text-ink')

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 flex items-center gap-4 border-b border-line bg-page/90 px-6 py-3 backdrop-blur">
        <NavLink to="/" className="flex items-center gap-2 font-semibold">
          <span className="inline-block h-3 w-3 rounded-full bg-series-1" /> Shared Brain
        </NavLink>
        <nav className="flex gap-1">
          <NavLink to="/" end className={link}>
            Runs
          </NavLink>
          <NavLink to="/compare" className={link}>
            Results
          </NavLink>
        </nav>
        <span className="ml-auto text-xs text-muted">Crusoe Managed Inference · Recursive Agents Hackathon</span>
      </header>
      <Routes>
        <Route path="/" element={<LaunchPage />} />
        <Route path="/run/:runId" element={<RunPage />} />
        <Route path="/compare" element={<ComparePage />} />
      </Routes>
    </div>
  )
}
