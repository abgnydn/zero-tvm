/**
 * GPU DISPATCH PROFILER
 *
 * Monkey-patches WebGPU to count dispatches, measure time per dispatch,
 * and identify the hot path for fusion optimization.
 */

export interface DispatchRecord {
  pipelineLabel: string
  workgroups: [number, number, number]
  timestamp: number
}

export interface ProfileReport {
  totalDispatches: number
  totalSubmits: number
  dispatches: DispatchRecord[]
  elapsedMs: number
}

let recording = false
let dispatches: DispatchRecord[] = []
let submitCount = 0
let startTime = 0

/** Start recording GPU dispatches */
export function startProfile(): void {
  dispatches = []
  submitCount = 0
  recording = true
  startTime = performance.now()
}

/** Stop recording and return report */
export function stopProfile(): ProfileReport {
  recording = false
  return {
    totalDispatches: dispatches.length,
    totalSubmits: submitCount,
    dispatches: [...dispatches],
    elapsedMs: performance.now() - startTime,
  }
}

/**
 * Patch a GPUDevice to intercept all dispatches.
 * Call this before any GPU work happens.
 */
export function patchDevice(device: GPUDevice): void {
  const origCreateEncoder = device.createCommandEncoder.bind(device)
  const origSubmit = device.queue.submit.bind(device.queue)

  device.createCommandEncoder = function(desc?: GPUCommandEncoderDescriptor) {
    const encoder = origCreateEncoder(desc)
    const origBeginPass = encoder.beginComputePass.bind(encoder)

    encoder.beginComputePass = function(passDesc?: GPUComputePassDescriptor) {
      const pass = origBeginPass(passDesc)
      const origDispatch = pass.dispatchWorkgroups.bind(pass)
      let currentPipelineLabel = 'unknown'

      const origSetPipeline = pass.setPipeline.bind(pass)
      pass.setPipeline = function(pipeline: GPUComputePipeline) {
        currentPipelineLabel = pipeline.label || 'unlabeled'
        return origSetPipeline(pipeline)
      }

      pass.dispatchWorkgroups = function(x: number, y?: number, z?: number) {
        if (recording) {
          dispatches.push({
            pipelineLabel: currentPipelineLabel,
            workgroups: [x, y ?? 1, z ?? 1],
            timestamp: performance.now(),
          })
        }
        return origDispatch(x, y, z)
      }

      return pass
    }

    return encoder
  }

  device.queue.submit = function(buffers: Iterable<GPUCommandBuffer>) {
    if (recording) submitCount++
    return origSubmit(buffers)
  }
}

/** Summarize a profile report */
export function summarizeProfile(report: ProfileReport): string {
  const lines: string[] = []
  lines.push(`=== GPU Profile ===`)
  lines.push(`Total dispatches: ${report.totalDispatches}`)
  lines.push(`Total submits: ${report.totalSubmits}`)
  lines.push(`Elapsed: ${report.elapsedMs.toFixed(0)} ms`)
  lines.push(`Dispatches/ms: ${(report.totalDispatches / report.elapsedMs).toFixed(1)}`)

  // Group by pipeline
  const byPipeline = new Map<string, number>()
  for (const d of report.dispatches) {
    byPipeline.set(d.pipelineLabel, (byPipeline.get(d.pipelineLabel) || 0) + 1)
  }
  lines.push(`\nDispatches by pipeline:`)
  const sorted = [...byPipeline.entries()].sort((a, b) => b[1] - a[1])
  for (const [name, count] of sorted) {
    lines.push(`  ${name}: ${count} (${(count / report.totalDispatches * 100).toFixed(0)}%)`)
  }

  return lines.join('\n')
}
