#!/usr/bin/env node
/**
 * Download Phi-3-mini-4k-instruct-q4f16_1-MLC weights from HuggingFace
 * to public/weights/ so Vite serves them locally. Run once, never re-download.
 *
 * Usage:
 *   node scripts/download-weights.mjs
 *
 * Downloads to: public/weights/Phi-3-mini-4k-instruct-q4f16_1-MLC/
 * Served at:    http://localhost:5173/weights/Phi-3-mini-4k-instruct-q4f16_1-MLC/
 */

import fs from 'fs'
import path from 'path'
import { pipeline } from 'stream/promises'

const MODEL_ID   = 'mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC'
const BASE_URL   = `https://huggingface.co/${MODEL_ID}/resolve/main`
const OUT_DIR    = `public/weights/Phi-3-mini-4k-instruct-q4f16_1-MLC`

// Extra files needed besides the shards
const EXTRA_FILES = [
  'ndarray-cache.json',
  'tokenizer.json',
  'tokenizer_config.json',
]

// ── helpers ──────────────────────────────────────────────────────────────────

function fmtMB(bytes) { return (bytes / 1024 / 1024).toFixed(1) + ' MB' }

async function downloadFile(url, destPath) {
  if (fs.existsSync(destPath)) {
    const size = fs.statSync(destPath).size
    if (size > 0) {
      console.log(`  [skip]  ${path.basename(destPath)} (${fmtMB(size)})`)
      return false
    }
    fs.unlinkSync(destPath) // zero-byte file — redownload
  }

  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`)

  const total   = Number(res.headers.get('content-length') || 0)
  const tmpPath = destPath + '.tmp'
  const out     = fs.createWriteStream(tmpPath)

  let downloaded = 0
  const reader   = res.body.getReader()

  process.stdout.write(`  [fetch] ${path.basename(destPath)} `)

  // Stream with progress dots
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    out.write(Buffer.from(value))
    downloaded += value.length
    if (total > 0) {
      const pct = Math.floor((downloaded / total) * 20)
      process.stdout.write(`\r  [fetch] ${path.basename(destPath)} ${fmtMB(downloaded)}/${fmtMB(total)} [${'='.repeat(pct)}${' '.repeat(20 - pct)}]`)
    }
  }

  await new Promise((res, rej) => out.end(err => err ? rej(err) : res()))
  fs.renameSync(tmpPath, destPath)
  console.log(`\r  [done]  ${path.basename(destPath)} (${fmtMB(downloaded)})                          `)
  return true
}

// ── main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`\nDownloading ${MODEL_ID}`)
  console.log(`→ ${OUT_DIR}\n`)

  fs.mkdirSync(OUT_DIR, { recursive: true })

  // 1) Fetch ndarray-cache.json first (needed to know shard list)
  const cacheJsonPath = path.join(OUT_DIR, 'ndarray-cache.json')
  await downloadFile(`${BASE_URL}/ndarray-cache.json`, cacheJsonPath)
  const ndarray = JSON.parse(fs.readFileSync(cacheJsonPath, 'utf8'))

  // 2) Collect unique shard filenames
  const shards = new Set()
  for (const entry of ndarray.records) {
    if (entry.dataPath) shards.add(entry.dataPath)
  }
  const shardList = [...shards].sort()
  console.log(`Found ${shardList.length} unique shards\n`)

  // 3) Download shards
  let fetched = 0
  let skipped = 0
  for (let i = 0; i < shardList.length; i++) {
    const shard = shardList[i]
    console.log(`[${i + 1}/${shardList.length}] ${shard}`)
    const dest = path.join(OUT_DIR, shard)
    const downloaded = await downloadFile(`${BASE_URL}/${shard}`, dest)
    downloaded ? fetched++ : skipped++
  }

  // 4) Download extra files
  console.log('\nExtra files:')
  for (const f of EXTRA_FILES) {
    if (f === 'ndarray-cache.json') continue // already done
    const dest = path.join(OUT_DIR, f)
    await downloadFile(`${BASE_URL}/${f}`, dest)
  }

  // 5) Summary
  const totalBytes = [...shardList, ...EXTRA_FILES]
    .map(f => path.join(OUT_DIR, f))
    .filter(p => fs.existsSync(p))
    .reduce((s, p) => s + fs.statSync(p).size, 0)

  console.log(`\n✓ Done — ${fetched} fetched, ${skipped} skipped`)
  console.log(`  Total on disk: ${(totalBytes / 1024 / 1024 / 1024).toFixed(2)} GB`)
  console.log(`  Served at:     /weights/Phi-3-mini-4k-instruct-q4f16_1-MLC/\n`)
}

main().catch(e => { console.error('\nFATAL:', e.message); process.exit(1) })
