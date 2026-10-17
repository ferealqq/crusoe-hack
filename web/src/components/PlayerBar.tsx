import { Pause, Play, Radio, RotateCcw } from 'lucide-react'
import { clock } from '@/lib/format'
import { SPEEDS, usePlayer } from '@/lib/player'
import { cn } from '@/lib/format'
import { Button } from './ui'

export function PlayerBar() {
  const { mode, playing, speed, cursor, events, state, play, pause, setSpeed, seek, seekToGen } = usePlayer()
  const total = events.length
  const gens = events.flatMap((e, i) => (e.t === 'gen.started' ? [{ gen: e.gen, i }] : []))

  if (mode === 'live') {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-line bg-surface px-4 py-2 text-sm">
        <span className="flex items-center gap-2 font-semibold text-[#e66767]">
          <Radio size={16} className={cn(playing && 'animate-pulse')} /> LIVE
        </span>
        <span className="text-ink-2">{total} events · run time {clock(state.ts)}</span>
        {!playing && <span className="text-muted">stream ended</span>}
      </div>
    )
  }

  return (
    <div className="flex items-center gap-3 rounded-xl border border-line bg-surface px-3 py-2">
      <Button variant="primary" onClick={playing ? pause : play} disabled={mode !== 'replay'} aria-label={playing ? 'Pause' : 'Play'}>
        {playing ? <Pause size={16} /> : <Play size={16} />}
        {playing ? 'Pause' : cursor >= total && total > 0 ? 'Replay' : 'Play'}
      </Button>
      <Button variant="ghost" onClick={() => seek(0)} aria-label="Restart">
        <RotateCcw size={15} />
      </Button>
      <div className="flex items-center rounded-lg border border-line p-0.5" role="group" aria-label="Replay speed">
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            className={cn('rounded-md px-2 py-0.5 text-xs tabular', s === speed ? 'bg-surface-2 text-ink' : 'text-muted hover:text-ink')}
          >
            {s}×
          </button>
        ))}
      </div>
      <div className="relative flex-1">
        <input
          type="range"
          min={0}
          max={total}
          value={cursor}
          onChange={(e) => seek(Number(e.target.value))}
          className="w-full accent-[#3987e5]"
          aria-label="Seek"
        />
        <div className="pointer-events-none absolute inset-x-0 -bottom-3 h-3">
          {gens.map((g) => (
            <span key={g.gen} className="absolute text-[10px] text-muted" style={{ left: `${(g.i / Math.max(1, total)) * 100}%` }}>
              G{g.gen}
            </span>
          ))}
        </div>
      </div>
      <div className="flex gap-1">
        {gens.map((g) => (
          <button key={g.gen} onClick={() => seekToGen(g.gen)} className="rounded px-1.5 py-0.5 text-xs text-ink-2 hover:bg-surface-2" title={`Jump to generation ${g.gen}`}>
            G{g.gen}
          </button>
        ))}
      </div>
      <span className="w-24 text-right text-xs text-muted tabular" title="Run time (training waits shortened during replay)">
        {clock(state.ts)} / {clock(events.at(-1)?.ts ?? 0)}
      </span>
    </div>
  )
}
