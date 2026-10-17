import { Swatch } from './ui'

interface Item {
  name?: string | number
  value?: number | string | (number | string)[]
  color?: string
  payload?: Record<string, unknown>
}

/** Shared tooltip: label row, then one swatch + name + value row per series. */
export function ChartTip({ active, payload, label, labelFmt, valueFmt }: {
  active?: boolean
  payload?: Item[]
  label?: string | number
  labelFmt?: (l: string | number) => string
  valueFmt: (v: number) => string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-line bg-surface-2 px-3 py-2 text-xs shadow-lg">
      {label !== undefined && <div className="mb-1 font-semibold text-ink">{labelFmt ? labelFmt(label) : label}</div>}
      {payload
        .filter((p) => typeof p.value === 'number')
        .map((p) => (
          <div key={String(p.name)} className="flex items-center gap-2 text-ink-2">
            <Swatch color={(p.payload?.fill as string) ?? p.color ?? '#c3c2b7'} />
            <span>{p.name}</span>
            <span className="ml-auto pl-4 font-semibold text-ink tabular">{valueFmt(p.value as number)}</span>
          </div>
        ))}
    </div>
  )
}
