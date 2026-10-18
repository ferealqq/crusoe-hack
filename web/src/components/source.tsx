import type { ReactNode } from 'react'
import { SourceCtx, SOURCES, type Source } from './sources'

export function SourceProvider({ source, children }: { source: Source; children: ReactNode }) {
  return <SourceCtx.Provider value={SOURCES[source]}>{children}</SourceCtx.Provider>
}
