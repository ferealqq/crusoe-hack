export const loss = (v?: number | null, digits = 4) => (v === undefined || v === null ? '—' : v.toFixed(digits))
export const pct = (v?: number | null, digits = 1) =>
  v === undefined || v === null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`
export const signedPct = (v: number, digits = 1) => `${v > 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`
export const tokens = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n))
export const clock = (s: number) => {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(Math.floor(s % 60)).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`
}
/** Relative loss reduction, positive = better (app/scoring.py gain). */
export const gain = (from: number, to: number) => (from - to) / from
export const cn = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(' ')
