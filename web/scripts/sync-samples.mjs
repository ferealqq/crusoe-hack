// Copies the recorded runs from the backend into public/samples so the demo also works
// as a static site, with no API running (see src/lib/api.ts fallback).
import { copyFileSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs'

const src = new URL('../../app/runlog/samples/', import.meta.url)
const dest = new URL('../public/samples/', import.meta.url)
mkdirSync(dest, { recursive: true })
const ids = readdirSync(src)
  .filter((f) => f.endsWith('.jsonl'))
  .map((f) => {
    copyFileSync(new URL(f, src), new URL(f, dest))
    return f.replace(/\.jsonl$/, '')
  })
writeFileSync(new URL('index.json', dest), JSON.stringify(ids, null, 2) + '\n')
console.log(`synced ${ids.length} sample runs to public/samples`)
