export const loss = (v?: number | null) => (v === undefined || v === null ? '—' : v.toFixed(4))
export const pct = (v?: number | null, digits = 1) =>
  v === undefined || v === null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`
export const signedPct = (v: number, digits = 1) => `${v > 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`
export const tokens = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n))
export const clock = (s: number) => {
  const m = Math.floor(s / 60)
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}
export const cn = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(' ')
