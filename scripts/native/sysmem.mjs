/**
 * SYSTEM MEMORY — measured, not estimated.
 *
 * The station's fit estimate says what a model SHOULD cost. This says what
 * the machine is actually doing, which is the input that was missing every
 * time a run went mysteriously slow: two resident models on a 32 GB Mac page
 * to disk and decode collapses to a third of its rate, with nothing in the
 * engine's own numbers to explain it.
 *
 * The load-bearing signal is not "free memory" — macOS reports that in a way
 * that looks alarming and usually isn't. It is SWAPOUTS INCREASING: the
 * kernel writing pages to disk means the working set no longer fits, and that
 * is exactly when tok/s falls off a cliff.
 *
 * macOS only (vm_stat + memory_pressure). Anywhere else this reports nulls
 * and the UI simply omits the row — a missing measurement is not a zero.
 */

import { execFile } from 'node:child_process'
import { totalmem } from 'node:os'

const run = (cmd, args) => new Promise((resolve) => {
  execFile(cmd, args, { timeout: 4000 }, (err, stdout) => resolve(err ? '' : stdout))
})

let prevSwapouts = null
let lastSwapAt = 0

export async function sampleMemory() {
  const [vm, pressure] = await Promise.all([run('vm_stat', []), run('memory_pressure', [])])
  if (!vm) return null

  // Page size is 16 KiB on Apple silicon, 4 KiB on Intel — read it, never assume.
  const pageSize = Number(/page size of (\d+) bytes/.exec(vm)?.[1] ?? 4096)
  const stat = (label) => {
    const m = new RegExp(`${label}:\\s+(\\d+)`).exec(vm)
    return m ? Number(m[1]) : null
  }
  const gb = (pages) => (pages == null ? null : (pages * pageSize) / 1024 ** 3)

  const wired = stat('Pages wired down')
  const compressed = stat('Pages occupied by compressor')
  const active = stat('Pages active')
  const swapouts = stat('Swapouts')

  // "Used" the way Activity Monitor means it: what cannot simply be dropped.
  const usedGb = gb((active ?? 0) + (wired ?? 0) + (compressed ?? 0))
  const totalGb = totalmem() / 1024 ** 3
  const freePct = Number(/free percentage:\s+(\d+)/.exec(pressure)?.[1] ?? NaN)

  // Swapping RIGHT NOW: swapouts climbing between samples. Cumulative
  // swapouts are useless on their own — every long-lived Mac has millions
  // from something that happened days ago.
  let swappingNow = false
  if (swapouts != null) {
    if (prevSwapouts != null && swapouts > prevSwapouts) lastSwapAt = Date.now()
    prevSwapouts = swapouts
    swappingNow = Date.now() - lastSwapAt < 30_000   // "in the last 30s"
  }

  return {
    totalGb: +totalGb.toFixed(1),
    usedGb: usedGb == null ? null : +usedGb.toFixed(1),
    compressedGb: compressed == null ? null : +gb(compressed).toFixed(1),
    freePct: Number.isFinite(freePct) ? freePct : null,
    swappingNow,
    // A model's working set is GPU buffers in unified memory: it counts here
    // like anything else, which is why two loaded models thrash a 32 GB Mac.
    note: swappingNow ? 'the machine is swapping — throughput will be a fraction of normal' : '',
  }
}
