#!/usr/bin/env node
// PAGING-MEASURE — the three numbers docs/PAGING_PLAN.md §0.3 requires before
// any threshold is written into code.
//
//   node scripts/paging-measure.mjs [--size 256] [--full]
//
// (a) adapter.limits.maxStorageBufferBindingSize / maxBufferSize. The machinery
//     to read these exists (loading-ui.ts:191-194) but nobody has ever recorded
//     what it returns on this machine. It decides whether Phase 3 can hold two
//     sequences in the existing per-layer buffer.
// (b) OPFS write and read throughput through createSyncAccessHandle in a worker.
//     Every "seconds, not 99 seconds" claim in the plan rests on a GUESSED
//     ~300 MB/s. This is the number that decides whether Phase 1 — the whole
//     useful subset — is worth shipping.
// (c) GPU->CPU readback throughput. The engine has never read the KV cache back
//     in its life; a pool must. Measured at three chunk sizes, because the
//     right one is not obvious and peer-weights' 240 KB is tuned for a
//     DataChannel, not for mapAsync.
//
// The reference workload is Qwen3.5 at 20k tokens: 32 KiB/token x 20000 =
// 625 MiB (tests/unit/kv-budget.test.ts). --full measures that exact size;
// the default 256 MiB is enough to establish the rate without holding 625 MiB
// of GPU, CPU and disk at once on a 32 GB machine.
//
// Reports every run, not a median. BENCH.md:153 records Qwen3.5 drifting
// downward WITHIN one invocation, so a single number here would be the same
// mistake in a new place.

import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'
import { execSync } from 'node:child_process'

const args = process.argv.slice(2)
const sizeIdx = args.indexOf('--size')
const MIB = args.includes('--full') ? 625 : (sizeIdx >= 0 ? Number(args[sizeIdx + 1]) : 256)
const REPS = 3

const mem = () => {
  try {
    const out = execSync('memory_pressure', { encoding: 'utf8' })
    return (out.match(/free percentage:\s*(\d+)%/) ?? [])[1] ?? '?'
  } catch { return '?' }
}

const fmt = (bytes, ms) => `${(bytes / 1e6 / (ms / 1000)).toFixed(0)} MB/s`
const rate = (r) => r.map((x) => x.toFixed(0)).join(', ')

console.log(`paging §0.3 — ${MIB} MiB working set, ${REPS} runs each`)
console.log(`free RAM before: ${mem()}%\n`)

await startHarness()
try {
  const page = await newPage('/docs.html')
  page.on('console', (m) => { if (m.text().startsWith('[measure]')) console.log(`  ${m.text()}`) })

  const out = await page.evaluate(async (MIB, REPS) => {
    const BYTES = MIB * 1024 * 1024
    const res = { limits: null, opfs: null, readback: null, errors: [] }

    // ---- (a) limits -------------------------------------------------------
    if (!navigator.gpu) return { ...res, errors: ['no WebGPU'] }
    const adapter = await navigator.gpu.requestAdapter()
    res.limits = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      subgroups: adapter.features.has('subgroups'),
      info: `${adapter.info?.vendor ?? '?'} / ${adapter.info?.architecture ?? '?'} / ${adapter.info?.description ?? ''}`,
    }

    // ---- (b) OPFS, in a worker -------------------------------------------
    // createSyncAccessHandle is worker-only. Written as a blob worker so this
    // needs no build step and no route on the dev server.
    const src = `
      self.onmessage = async (e) => {
        const { bytes, reps } = e.data
        try {
          const root = await navigator.storage.getDirectory()
          const dir = await root.getDirectoryHandle('zero-tvm-paging-measure', { create: true })
          const w = [], r = []
          // One buffer reused: this measures OPFS, not allocation.
          const buf = new Uint8Array(bytes)
          for (let i = 0; i < bytes; i += 4096) buf[i] = i & 255
          for (let k = 0; k < reps; k++) {
            const fh = await dir.getFileHandle('probe.bin', { create: true })
            const h = await fh.createSyncAccessHandle()
            let t = performance.now()
            h.write(buf, { at: 0 })
            h.flush()
            w.push(bytes / 1e6 / ((performance.now() - t) / 1000))
            const back = new Uint8Array(bytes)
            t = performance.now()
            h.read(back, { at: 0 })
            r.push(bytes / 1e6 / ((performance.now() - t) / 1000))
            h.close()
            if (back[4096] !== buf[4096] || back[bytes - 1] !== buf[bytes - 1]) {
              self.postMessage({ error: 'OPFS read did not return what was written' }); return
            }
          }
          await dir.removeEntry('probe.bin').catch(() => {})
          self.postMessage({ write: w, read: r })
        } catch (err) { self.postMessage({ error: String(err && err.message || err) }) }
      }`
    const worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })))
    res.opfs = await new Promise((resolve) => {
      worker.onmessage = (e) => resolve(e.data)
      worker.onerror = (e) => resolve({ error: `worker: ${e.message}` })
      worker.postMessage({ bytes: BYTES, reps: REPS })
    })
    worker.terminate()

    // ---- (c) GPU -> CPU readback -----------------------------------------
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    })
    device.addEventListener?.('uncapturederror', (e) => res.errors.push(String(e.error?.message ?? e)))

    if (BYTES > adapter.limits.maxBufferSize) {
      res.readback = { error: `${MIB} MiB exceeds maxBufferSize ${adapter.limits.maxBufferSize}` }
    } else {
      // A STORAGE buffer, as the KV cache is, with the same usage flags: the
      // engine's STORAGE constant is already
      // STORAGE | COPY_SRC | COPY_DST (engine-core.ts:36), so every KV buffer
      // can already be copied out and written back. No allocation change is
      // needed for Phase 1.
      const gpu = device.createBuffer({ size: BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
      device.queue.writeBuffer(gpu, 0, new Uint8Array(1024 * 1024))   // touch it

      const chunkSizes = [16, 64, MIB].filter((m, i, a) => m <= MIB && a.indexOf(m) === i)
      const byChunk = {}
      for (const cm of chunkSizes) {
        const chunk = cm * 1024 * 1024
        const staging = device.createBuffer({ size: chunk, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
        const runs = []
        for (let k = 0; k < REPS; k++) {
          const t = performance.now()
          let done = 0
          while (done < BYTES) {
            const n = Math.min(chunk, BYTES - done)
            const enc = device.createCommandEncoder()
            enc.copyBufferToBuffer(gpu, done, staging, 0, n)
            device.queue.submit([enc.finish()])
            await staging.mapAsync(GPUMapMode.READ, 0, n)
            // COPY the bytes out. Reading the mapped range without copying
            // measures mapAsync alone, and a pool has to own the bytes.
            new Uint8Array(BYTES === 0 ? 0 : n).set(new Uint8Array(staging.getMappedRange(0, n)))
            staging.unmap()
            done += n
          }
          runs.push(BYTES / 1e6 / ((performance.now() - t) / 1000))
        }
        byChunk[`${cm} MiB`] = runs
        staging.destroy()
        console.log(`[measure] readback chunk ${cm} MiB: ${runs.map((x) => x.toFixed(0)).join(', ')} MB/s`)
      }
      // ---- (d) CPU -> GPU, the actual Phase 1 restore path ----------------
      // Restoring a prefix under the identity table is one writeBuffer per
      // layer. This is the number the "cold restart" claim rests on, and the
      // plan left it unmeasured.
      const host = new Uint8Array(BYTES)
      const wruns = []
      for (let k = 0; k < REPS; k++) {
        const t = performance.now()
        device.queue.writeBuffer(gpu, 0, host)
        await device.queue.onSubmittedWorkDone()
        wruns.push(BYTES / 1e6 / ((performance.now() - t) / 1000))
      }
      res.upload = wruns
      console.log(`[measure] writeBuffer (CPU->GPU): ${wruns.map((x) => x.toFixed(0)).join(', ')} MB/s`)

      gpu.destroy()
      res.readback = byChunk
    }
    return res
  }, MIB, REPS)

  console.log(`\n(a) adapter limits`)
  if (out.limits) {
    const L = out.limits
    console.log(`    ${L.info}`)
    console.log(`    maxStorageBufferBindingSize  ${(L.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`)
    console.log(`    maxBufferSize                ${(L.maxBufferSize / 2 ** 20).toFixed(0)} MiB`)
    console.log(`    subgroups                    ${L.subgroups}`)
  }

  console.log(`\n(b) OPFS ${MIB} MiB, createSyncAccessHandle in a worker`)
  if (out.opfs?.error) console.log(`    FAILED: ${out.opfs.error}`)
  else {
    console.log(`    write  ${rate(out.opfs.write)} MB/s`)
    console.log(`    read   ${rate(out.opfs.read)} MB/s`)
  }

  console.log(`\n(c) GPU -> CPU readback of ${MIB} MiB (copyBufferToBuffer -> mapAsync -> copy out)`)
  if (out.readback?.error) console.log(`    FAILED: ${out.readback.error}`)
  else for (const [k, v] of Object.entries(out.readback ?? {})) console.log(`    chunk ${k.padEnd(9)} ${rate(v)} MB/s`)

  if (out.upload) console.log(`\n(d) CPU -> GPU writeBuffer of ${MIB} MiB — the Phase 1 restore path\n    ${rate(out.upload)} MB/s`)

  if (out.errors?.length) console.log(`\n    WebGPU errors: ${out.errors.join(' | ')}`)

  // What the numbers mean for the plan, computed rather than asserted.
  const best = (o) => Math.max(...Object.values(o ?? {}).flat().filter(Number.isFinite))
  if (out.opfs?.read && out.readback && !out.readback.error) {
    const KV = 625 * 1024 * 1024
    const rd = best(out.readback)
    const wr = Math.max(...out.opfs.write)
    const rdOpfs = Math.max(...out.opfs.read)
    // Extrapolated from the SLOWEST run, not the best. The rates climb across
    // reps — 5434 -> 7255 -> 7294 MB/s on the first OPFS run — which is the
    // page cache warming, not the device getting faster. Reporting the best
    // would repeat the bandwidth-bench mistake this repo already made once.
    const slow = (xs) => Math.min(...xs.filter(Number.isFinite))
    const rdS = slow(Object.values(out.readback).flat())
    const wrS = slow(out.opfs.write)
    const rdOpfsS = slow(out.opfs.read)
    const upS = out.upload ? slow(out.upload) : null
    const KVmb = 625 * 1024 * 1024 / 1e6
    console.log(`\n  extrapolated to Qwen3.5's 20k KV (625 MiB), at the SLOWEST run of each:`)
    console.log(`    save    = readback ${(KVmb / rdS).toFixed(2)} s + OPFS write ${(KVmb / wrS).toFixed(2)} s`
      + ` = ${(KVmb / rdS + KVmb / wrS).toFixed(2)} s`)
    console.log(`    restore = OPFS read ${(KVmb / rdOpfsS).toFixed(2)} s`
      + (upS ? ` + writeBuffer ${(KVmb / upS).toFixed(2)} s = ${(KVmb / rdOpfsS + KVmb / upS).toFixed(2)} s` : ' + writeBuffer (unmeasured)'))
    console.log(`    threshold: restore must beat 1/5 of a cold re-prefill (~99 s) = under ~20 s`)
    if (upS) {
      const margin = 20 / (KVmb / rdOpfsS + KVmb / upS)
      console.log(`    margin: ${margin.toFixed(0)}x under the threshold — so the conclusion survives`)
      console.log(`            these numbers being an order of magnitude optimistic, which they may be.`)
    }
    console.log(`\n  CAVEATS, and they are not small:`)
    console.log(`    - OPFS at ${MIB} MiB on a 32 GB machine is measured THROUGH THE PAGE CACHE.`)
    console.log(`      h.flush() does not guarantee a write to media on macOS. Treat as an upper bound.`)
    console.log(`    - Apple Silicon is unified memory: readback and writeBuffer are memcpy, no PCIe.`)
    console.log(`      Neither number transfers to a discrete GPU.`)
    console.log(`    - A real restore is a COLD read in a fresh tab. This is warm.`)
  }
} finally {
  await stopHarness()
}
console.log(`\nfree RAM after: ${mem()}%`)
