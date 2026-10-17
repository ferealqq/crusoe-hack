import type { Setup } from './events'

// Categorical slots 1-4 of the validated dark palette, in fixed order. Color follows the
// entity (a researcher focus, or a setup), never its rank.
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500'] as const

export const FOCUS_ORDER = ['optimizer', 'architecture', 'regularization', 'data'] as const
export const focusColor = (focus?: string) => {
  const i = FOCUS_ORDER.indexOf(focus as (typeof FOCUS_ORDER)[number])
  return i >= 0 ? SERIES[i] : CHROME.ink2
}
// Secondary encoding for scatter marks, where four hues alone aren't CVD-safe.
export const FOCUS_SHAPE: Record<string, 'circle' | 'square' | 'triangle' | 'diamond'> = {
  optimizer: 'circle',
  architecture: 'square',
  regularization: 'triangle',
  data: 'diamond',
}

export const SETUP_ORDER: Setup[] = ['shared_brain', 'raw_log', 'no_memory', 'one_shot']
export const SETUP_LABEL: Record<Setup, string> = {
  shared_brain: 'Shared Brain',
  raw_log: 'Raw log',
  no_memory: 'No memory',
  one_shot: 'One-shot',
}
export const SETUP_BLURB: Record<Setup, string> = {
  shared_brain: 'Curated memory: lean and current',
  raw_log: 'Memory without curation',
  no_memory: 'Every agent starts from zero',
  one_shot: 'A single agent, a single edit',
}
export const setupColor = (s: Setup) => SERIES[SETUP_ORDER.indexOf(s)]

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
