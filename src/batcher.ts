/**
 * SUBMIT BATCHER — Accumulate command buffers, submit once per token.
 *
 * Key insight from the recorder: TVM allocates SEPARATE uniform buffers
 * per dispatch. Each dispatch's bind group references its own buffer.
 * So writeBuffer calls don't conflict — we can let them all through
 * and batch the submits.
 *
 * TVM does per token:
 *   writeBuffer(buf1) → submit(cmd1) → writeBuffer(buf2) → submit(cmd2) → ... × 342
 *
 * We do:
 *   writeBuffer(buf1) → writeBuffer(buf2) → ... → submit([cmd1, cmd2, ...]) at mapAsync
 *
 * Queue ordering guarantees: all writeBuffers execute before the batched submit.
 * Each command buffer reads its own unique uniform buffer. No conflicts.
 */

let enabled = false
let pendingBuffers: GPUCommandBuffer[] = []
let deferredDestroys: Array<() => void> = []
let originalSubmits = 0
let batchedSubmits = 0
let tokensProcessed = 0

export function setBatcherEnabled(on: boolean): void {
  enabled = on
  tokensProcessed = 0
  originalSubmits = 0
  batchedSubmits = 0
}

export function getBatcherStats(): string {
  if (tokensProcessed === 0) return 'Batcher: no tokens yet'
  return [
    `Batcher: ${enabled ? 'ON' : 'OFF'}`,
    `Tokens: ${tokensProcessed}`,
    `Original submits: ${originalSubmits} (${(originalSubmits / tokensProcessed).toFixed(0)}/tok)`,
    `Batched submits: ${batchedSubmits} (${(batchedSubmits / tokensProcessed).toFixed(0)}/tok)`,
    `Reduction: ${originalSubmits > 0 ? (originalSubmits / batchedSubmits).toFixed(1) : 0}x`,
  ].join('\n')
}

export function patchDeviceBatcher(device: GPUDevice): void {
  const origSubmit = device.queue.submit.bind(device.queue)

  function flush(): void {
    if (pendingBuffers.length > 0) {
      origSubmit(pendingBuffers)
      batchedSubmits++
      pendingBuffers = []
    }
    // Now safe to destroy deferred buffers
    for (const destroyFn of deferredDestroys) {
      destroyFn()
    }
    deferredDestroys = []
  }

  // Intercept submit — accumulate instead of submitting
  const prevSubmit = device.queue.submit
  device.queue.submit = function(buffers: Iterable<GPUCommandBuffer>) {
    if (!enabled) return prevSubmit.call(device.queue, buffers)

    originalSubmits++
    for (const buf of buffers) pendingBuffers.push(buf)
    // Don't submit yet — wait for mapAsync
  }

  // Intercept buffer creation — defer destroy + flush at mapAsync
  const prevCreateBuffer = device.createBuffer
  device.createBuffer = function(descriptor: GPUBufferDescriptor) {
    const buffer = prevCreateBuffer.call(device, descriptor)

    // Defer destroy while we have pending work
    const origDestroy = buffer.destroy.bind(buffer)
    buffer.destroy = function() {
      if (enabled && pendingBuffers.length > 0) {
        deferredDestroys.push(origDestroy)
        return
      }
      return origDestroy()
    }

    // Flush before any GPU read-back
    if (descriptor.usage & GPUBufferUsage.MAP_READ) {
      const origMapAsync = buffer.mapAsync.bind(buffer)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(buffer as any).mapAsync = function(...args: Parameters<GPUBuffer['mapAsync']>) {
        if (enabled) {
          flush()
          tokensProcessed++
        }
        return origMapAsync(...args)
      }
    }

    return buffer
  }

  // Also flush on onSubmittedWorkDone (used by some WebLLM flows)
  const origOnSubmittedWorkDone = device.queue.onSubmittedWorkDone.bind(device.queue)
  device.queue.onSubmittedWorkDone = function() {
    if (enabled) flush()
    return origOnSubmittedWorkDone()
  }
}
