// Propagate measured benchmark numbers from bench/results.json into the docs,
// so the figures have a single source of truth and stop drifting.
//
//   node bench/sync-docs.mjs            # dry run — print what would change
//   node bench/sync-docs.mjs --write    # apply
//
// `npm run bench` writes results.json (from a real-GPU run) and calls this
// with --write. You can also re-run it standalone after editing results.json.
//
// What it updates automatically:
//   - BENCH.md, README.md, index.html, docs.html
//     — numbers wrapped in <!--bench:KEY-->…<!--/bench:KEY--> markers
//   - src/webllm-bench/main.ts — the `// bench:zt` constant
// Prose mentions in README/docs/demo are reported for manual review, not
// rewritten (auto-editing prose is how numbers get mangled) — prose should
// cite the stable ratio range, not exact medians.

import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const WRITE = process.argv.includes('--write')
const RESULTS = resolve(ROOT, 'bench/results.json')

if (!existsSync(RESULTS)) {
  console.error(`No ${RESULTS}. Run \`npm run bench\` on a GPU machine first, or write it by hand:`)
  console.error('  { "ztDecode": 69.55, "webllmDecode": 59.95, "hardware": "Apple M2 Max", "date": "2026-07-30" }')
  console.error('  (ztDecode/webllmDecode are TOTAL wall-clock medians — names predate the 2026-07-30 metric split)')
  process.exit(1)
}

const r = JSON.parse(readFileSync(RESULTS, 'utf8'))
const gap = Math.round(((r.webllmDecode - r.ztDecode) / r.webllmDecode) * 100)
const markerVals = {
  zt: r.ztDecode.toFixed(2),
  webllm: r.webllmDecode.toFixed(1),
  gap: String(gap),
}

const changes = []

/** Replace <!--bench:KEY-->…<!--/bench:KEY--> spans in a markdown/HTML file. */
function syncMarkers(rel) {
  const p = resolve(ROOT, rel)
  if (!existsSync(p)) return
  const orig = readFileSync(p, 'utf8')
  let src = orig
  for (const [key, val] of Object.entries(markerVals)) {
    src = src.replace(
      new RegExp(`(<!--bench:${key}-->)(.*?)(<!--/bench:${key}-->)`, 'g'),
      (_m, a, old, b) => {
        if (old !== val) changes.push({ rel, key, old, val })
        return a + val + b
      },
    )
  }
  if (src !== orig && WRITE) writeFileSync(p, src)
}

/** Replace the number on a `// bench:zt` line in a TS/JS file. */
function syncTsConst(rel) {
  const p = resolve(ROOT, rel)
  if (!existsSync(p)) return
  const orig = readFileSync(p, 'utf8')
  const src = orig.replace(/(=\s*)([\d.]+)(\s*\/\/\s*bench:zt\b)/g, (_m, a, old, b) => {
    if (old !== markerVals.zt) changes.push({ rel, key: 'zt', old, val: markerVals.zt })
    return a + markerVals.zt + b
  })
  if (src !== orig && WRITE) writeFileSync(p, src)
}

/** List tok/s / gap mentions in prose files for manual review. */
function reportProse() {
  const hits = []
  for (const rel of ['README.md', 'docs.html', 'demo.html', 'index.html']) {
    const p = resolve(ROOT, rel)
    if (!existsSync(p)) continue
    readFileSync(p, 'utf8')
      .split('\n')
      .forEach((line, i) => {
        if (line.includes('<!--bench:')) return // marker-managed, not prose
        // The % pattern skips en-dash range phrasing like "28–31% faster".
        // (That particular band was retired on 2026-07-30 — prose should now
        // cite a dated same-session pair with BOTH metrics; see bench/README.)
        if (/\b\d+(\.\d+)?\s*tok\/s/i.test(line) || /(?<![\d.–-])\b\d+\s*%\s*(faster|slower|behind)/i.test(line))
          hits.push(`  ${rel}:${i + 1}  ${line.trim().slice(0, 96)}`)
      })
  }
  return hits
}

console.log(`bench/results.json: Zero-TVM ${markerVals.zt} tok/s · WebLLM ${markerVals.webllm} tok/s `
  + `· gap ${gap}% · ${r.hardware ?? 'unknown HW'} · ${r.date ?? ''}`)

syncMarkers('BENCH.md')
syncMarkers('README.md')
syncMarkers('index.html')
syncMarkers('docs.html')
syncTsConst('src/webllm-bench/main.ts')

console.log(`\n${WRITE ? 'Applied' : 'Would change'} ${changes.length} marked value(s):`)
for (const c of changes) console.log(`  ${c.rel}  [${c.key}]  ${c.old} → ${c.val}`)
if (!changes.length) console.log('  (everything already up to date)')

const prose = reportProse()
if (prose.length) {
  console.log(`\nProse mentions to review by hand (not auto-edited):`)
  for (const h of prose) console.log(h)
}
if (!WRITE) console.log('\nDry run — re-run with --write to apply.')
