import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/format'

export function Card({ title, action, children, className }: { title?: ReactNode; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn('rounded-xl border border-line bg-surface', className)}>
      {(title || action) && (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
          <h2 className="text-xs font-semibold tracking-wide text-ink-2 uppercase">{title}</h2>
          {action}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

export function Button({ variant = 'default', className, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'default' | 'primary' | 'ghost' }) {
  return (
    <button
      {...props}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
        variant === 'primary' && 'bg-series-1 text-white hover:bg-[#2a78d6]',
        variant === 'default' && 'border border-line bg-surface-2 text-ink hover:bg-[#2c2c2a]',
        variant === 'ghost' && 'text-ink-2 hover:bg-surface-2 hover:text-ink',
        className,
      )}
    />
  )
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'good' | 'bad' | 'warn' | 'accent' }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium',
        tone === 'neutral' && 'bg-surface-2 text-ink-2',
        tone === 'good' && 'bg-good/15 text-good',
        tone === 'bad' && 'bg-critical/15 text-[#e66767]',
        tone === 'warn' && 'bg-warning/15 text-warning',
        tone === 'accent' && 'bg-series-1/15 text-[#86b6ef]',
      )}
    >
      {children}
    </span>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-[11px] tracking-wide text-muted uppercase">{label}</div>
      <div className="mt-0.5 text-2xl font-semibold text-ink">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-2">{sub}</div>}
    </div>
  )
}

/** Identity swatch: a colored mark beside text that stays in ink colors. */
export function Swatch({ color, shape = 'circle' }: { color: string; shape?: 'circle' | 'square' | 'triangle' | 'diamond' }) {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden className="shrink-0">
      {shape === 'circle' && <circle cx="5" cy="5" r="4" fill={color} />}
      {shape === 'square' && <rect x="1" y="1" width="8" height="8" rx="1.5" fill={color} />}
      {shape === 'triangle' && <path d="M5 1 L9 9 L1 9 Z" fill={color} />}
      {shape === 'diamond' && <path d="M5 0.5 L9.5 5 L5 9.5 L0.5 5 Z" fill={color} />}
    </svg>
  )
}
