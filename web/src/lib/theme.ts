import type { Setup } from './events'

// Categorical slots 1-5 of the validated dark palette, in fixed order (validated: CVD
// worst adjacent ΔE 8.4). Color follows the entity (a researcher focus, or a setup), never
// its rank; scatter marks add a shape per focus as secondary encoding.
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'] as const

export const CHROME = {
  surface: '#1a1a19',
  ink: '#f0efec',
  ink2: '#c3c2b7',
  muted: '#898781',
  grid: '#2c2c2a',
  axis: '#383835',
  good: '#0ca30c',
  critical: '#d03b3b',
} as const

// The five focus areas of app/agents/prompts.py FOCUSES (one-shot uses "any").
export const FOCUS_ORDER = ['features', 'model type', 'tuning', 'regularization', 'ensembling'] as const
export type Shape = 'circle' | 'square' | 'triangle' | 'diamond' | 'cross'
const SHAPES: Shape[] = ['circle', 'square', 'triangle', 'diamond', 'cross']

const focusIndex = (focus?: string) => FOCUS_ORDER.indexOf(focus as (typeof FOCUS_ORDER)[number])
export const focusColor = (focus?: string) => {
  if (focus === 'curator') return CHROME.ink
  const i = focusIndex(focus)
  return i >= 0 ? SERIES[i] : CHROME.ink2
}
export const focusShape = (focus?: string): Shape => SHAPES[focusIndex(focus)] ?? 'circle'

export const SETUP_ORDER: Setup[] = ['shared_brain', 'raw_log', 'no_memory', 'one_shot']
export const SETUP_LABEL: Record<Setup, string> = {
  shared_brain: 'Shared Brain',
  raw_log: 'Raw log',
  no_memory: 'No memory',
  one_shot: 'One-shot',
}
export const SETUP_BLURB: Record<Setup, string> = {
  shared_brain: 'Curated, verified memory',
  raw_log: 'Every report appended, uncurated',
  no_memory: 'Every agent starts from zero',
  one_shot: 'A single agent, a single run',
}
export const setupColor = (s: Setup) => SERIES[SETUP_ORDER.indexOf(s)]
export const sortSetups = (setups: Setup[]) => [...setups].sort((a, b) => SETUP_ORDER.indexOf(a) - SETUP_ORDER.indexOf(b))
