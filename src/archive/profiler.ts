/**
 * GPU DISPATCH PROFILER + SHADER CAPTURE
 *
 * Monkey-patches WebGPU to:
 * 1. Count and profile dispatches per token
 * 2. Capture all WGSL shader source code from createShaderModule
 * 3. Map pipelines to their shaders
 * 4. Build a complete picture of TVM's compiled execution graph
 */

export interface CapturedShader {
  index: number
  label: string
  code: string
  lineCount: number
  hasBindings: number  // number of @binding declarations
}

export interface CapturedPipeline {
  label: string
  shaderIndex: number
  entryPoint: string
}

export interface DispatchRecord {
  pipelineLabel: string
  pipelineIndex: number
  workgroups: [number, number, number]
  timestamp: number
  writeBuffersBefore: number
}

export interface TokenSequence {
  dispatches: DispatchRecord[]
  writeBufferCalls: number
}

export interface ProfileReport {
  totalDispatches: number
  totalSubmits: number
  totalWriteBuffers: number
  dispatches: DispatchRecord[]
  tokenSequences: TokenSequence[]
  elapsedMs: number
}

/** All captured shaders and pipelines — available globally after model load */
export const captured = {
  shaders: [] as CapturedShader[],
  pipelines: new Map<string, CapturedPipeline>(),
  shaderByCode: new Map<string, number>(),  // code hash → shader index
}

let recording = false
let dispatches: DispatchRecord[] = []
let submitCount = 0
let writeBufferCount = 0
let writeBuffersSinceLastDispatch = 0
let startTime = 0

// Token boundary detection: mapAsync signals end of a token decode
let currentTokenDispatches: DispatchRecord[] = []
let currentTokenWriteBuffers = 0
let tokenSequences: TokenSequence[] = []

/** Start recording GPU dispatches */
export function startProfile(): void {
  dispatches = []
  submitCount = 0
  writeBufferCount = 0
  writeBuffersSinceLastDispatch = 0
  currentTokenDispatches = []
  currentTokenWriteBuffers = 0
  tokenSequences = []
  recording = true
  startTime = performance.now()
}

/** Stop recording and return report */
export function stopProfile(): ProfileReport {
  recording = false
  if (currentTokenDispatches.length > 0) {
    tokenSequences.push({
      dispatches: [...currentTokenDispatches],
      writeBufferCalls: currentTokenWriteBuffers,
    })
  }
  return {
    totalDispatches: dispatches.length,
    totalSubmits: submitCount,
    totalWriteBuffers: writeBufferCount,
    dispatches: [...dispatches],
    tokenSequences: [...tokenSequences],
    elapsedMs: performance.now() - startTime,
  }
}

/** Get summary of captured shaders */
export function getShaderSummary(): string {
  const lines: string[] = []
  lines.push(`\n=== Captured Shaders: ${captured.shaders.length} modules ===`)
  for (const s of captured.shaders) {
    lines.push(`  [${s.index}] ${s.label || 'unlabeled'} — ${s.lineCount} lines, ${s.hasBindings} bindings`)
  }
  lines.push(`\n=== Pipelines: ${captured.pipelines.size} ===`)
  for (const [label, p] of captured.pipelines) {
    lines.push(`  ${label} → shader[${p.shaderIndex}] entry="${p.entryPoint}"`)
  }
  return lines.join('\n')
}

/**
 * Patch a GPUDevice to:
 * 1. Capture all shader source code from createShaderModule
 * 2. Map compute pipelines to their shaders
 * 3. Profile dispatches with full shader traceability
 */
export function patchDevice(device: GPUDevice): void {
  const origCreateShaderModule = device.createShaderModule.bind(device)
  const origCreateComputePipeline = device.createComputePipeline.bind(device)
  const origCreateEncoder = device.createCommandEncoder.bind(device)
  const origSubmit = device.queue.submit.bind(device.queue)
  const origWriteBuffer = device.queue.writeBuffer.bind(device.queue)

  // Track pipeline → shader index mapping at runtime
  const pipelineToIndex = new WeakMap<GPUComputePipeline, number>()

  // === SHADER CAPTURE ===
  device.createShaderModule = function(desc: GPUShaderModuleDescriptor) {
    const module = origCreateShaderModule(desc)
    const code = desc.code
    const index = captured.shaders.length
    const bindingCount = (code.match(/@binding\(/g) || []).length

    captured.shaders.push({
      index,
      label: desc.label || module.label || `shader_${index}`,
      code,
      lineCount: code.split('\n').length,
      hasBindings: bindingCount,
    })
    captured.shaderByCode.set(code, index)

    // Tag the module so pipelines can reference it
    ;(module as unknown as Record<string, unknown>).__capturedIndex = index
    return module
  }

  // === PIPELINE CAPTURE (sync + async) ===
  function tagPipeline(pipeline: GPUComputePipeline, desc: GPUComputePipelineDescriptor): void {
    const shaderModule = desc.compute.module as unknown as Record<string, unknown>
    const shaderIndex = (shaderModule.__capturedIndex as number) ?? -1
    const entryPoint = desc.compute.entryPoint ?? 'main'
    const label = desc.label || pipeline.label || `pipeline_${captured.pipelines.size}`

    captured.pipelines.set(label, { label, shaderIndex, entryPoint })
    pipelineToIndex.set(pipeline, shaderIndex)
  }

  device.createComputePipeline = function(desc: GPUComputePipelineDescriptor) {
    const pipeline = origCreateComputePipeline(desc)
    tagPipeline(pipeline, desc)
    return pipeline
  }

  const origCreateComputePipelineAsync = device.createComputePipelineAsync.bind(device)
  device.createComputePipelineAsync = async function(desc: GPUComputePipelineDescriptor) {
    const pipeline = await origCreateComputePipelineAsync(desc)
    tagPipeline(pipeline, desc)
    return pipeline
  }

  // === DISPATCH PROFILING ===
  device.createCommandEncoder = function(desc?: GPUCommandEncoderDescriptor) {
    const encoder = origCreateEncoder(desc)
    const origBeginPass = encoder.beginComputePass.bind(encoder)

    encoder.beginComputePass = function(passDesc?: GPUComputePassDescriptor) {
      const pass = origBeginPass(passDesc)
      const origDispatch = pass.dispatchWorkgroups.bind(pass)
      let currentPipelineLabel = 'unknown'
      let currentPipelineIndex = -1

      const origSetPipeline = pass.setPipeline.bind(pass)
      pass.setPipeline = function(pipeline: GPUComputePipeline) {
        currentPipelineLabel = pipeline.label || 'unlabeled'
        currentPipelineIndex = pipelineToIndex.get(pipeline) ?? -1
        return origSetPipeline(pipeline)
      }

      pass.dispatchWorkgroups = function(x: number, y?: number, z?: number) {
        if (recording) {
          const record: DispatchRecord = {
            pipelineLabel: currentPipelineLabel,
            pipelineIndex: currentPipelineIndex,
            workgroups: [x, y ?? 1, z ?? 1],
            timestamp: performance.now(),
            writeBuffersBefore: writeBuffersSinceLastDispatch,
          }
          dispatches.push(record)
          currentTokenDispatches.push(record)
          writeBuffersSinceLastDispatch = 0
        }
        return origDispatch(x, y, z)
      }

      return pass
    }

    return encoder
  }

  // === SUBMIT COUNTING ===
  device.queue.submit = function(buffers: Iterable<GPUCommandBuffer>) {
    if (recording) submitCount++
    return origSubmit(buffers)
  }

  // === WRITEBUFFER COUNTING ===
  device.queue.writeBuffer = function(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: BufferSource | SharedArrayBuffer,
    dataOffset?: number,
    size?: number,
  ) {
    if (recording) {
      writeBufferCount++
      writeBuffersSinceLastDispatch++
      currentTokenWriteBuffers++
    }
    return origWriteBuffer(buffer, bufferOffset, data, dataOffset, size)
  }

  // === TOKEN BOUNDARY DETECTION ===
  const origCreateBuffer = device.createBuffer.bind(device)
  device.createBuffer = function(descriptor: GPUBufferDescriptor) {
    const buffer = origCreateBuffer(descriptor)

    if (descriptor.usage & GPUBufferUsage.MAP_READ) {
      const origMapAsync = buffer.mapAsync.bind(buffer)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(buffer as any).mapAsync = function(...args: Parameters<GPUBuffer['mapAsync']>) {
        if (recording && currentTokenDispatches.length > 0) {
          tokenSequences.push({
            dispatches: [...currentTokenDispatches],
            writeBufferCalls: currentTokenWriteBuffers,
          })
          currentTokenDispatches = []
          currentTokenWriteBuffers = 0
        }
        return origMapAsync(...args)
      }
    }

    return buffer
  }
}

/** Summarize a profile report with shader-mapped dispatch analysis */
export function summarizeProfile(report: ProfileReport): string {
  const lines: string[] = []
  lines.push(`=== GPU Profile ===`)
  lines.push(`Shaders captured: ${captured.shaders.length}`)
  lines.push(`Pipelines captured: ${captured.pipelines.size}`)
  lines.push(`Total dispatches: ${report.totalDispatches}`)
  lines.push(`Total submits: ${report.totalSubmits}`)
  lines.push(`Total writeBuffers: ${report.totalWriteBuffers}`)
  lines.push(`Elapsed: ${report.elapsedMs.toFixed(0)} ms`)

  // Group dispatches by shader index
  const byShader = new Map<number, number>()
  for (const d of report.dispatches) {
    byShader.set(d.pipelineIndex, (byShader.get(d.pipelineIndex) || 0) + 1)
  }
  lines.push(`\nDispatches by shader:`)
  const sorted = [...byShader.entries()].sort((a, b) => b[1] - a[1])
  for (const [idx, count] of sorted) {
    const shader = captured.shaders[idx]
    const name = shader ? `shader[${idx}] ${shader.label}` : `unknown[${idx}]`
    const info = shader ? `${shader.lineCount}L ${shader.hasBindings}B` : ''
    lines.push(`  ${name} (${info}): ${count}× (${(count / report.totalDispatches * 100).toFixed(0)}%)`)
  }

  // Per-token sequence with shader IDs
  if (report.tokenSequences.length > 1) {
    const decodeSeqs = report.tokenSequences.slice(1)
    const repToken = decodeSeqs[Math.min(2, decodeSeqs.length - 1)]

    lines.push(`\n=== Decode Token Dispatch Sequence (${repToken.dispatches.length} dispatches) ===`)

    // Show compact sequence: shader index + workgroup
    // Identify the repeating layer pattern
    const seq = repToken.dispatches.map(d => d.pipelineIndex)
    lines.push(`Shader sequence: [${seq.join(', ')}]`)

    // Find repeating pattern
    const layerLen = findRepeatLength(seq)
    if (layerLen > 0) {
      const layerPattern = seq.slice(0, layerLen)
      const numRepeats = Math.floor(seq.length / layerLen)
      const tail = seq.slice(numRepeats * layerLen)

      lines.push(`\nRepeating pattern detected: ${layerLen} dispatches × ${numRepeats} layers`)
      lines.push(`Layer pattern: [${layerPattern.join(', ')}]`)

      // Label each dispatch in the pattern
      for (let i = 0; i < layerPattern.length; i++) {
        const d = repToken.dispatches[i]
        const shader = captured.shaders[d.pipelineIndex]
        const wg = `${d.workgroups[0]}×${d.workgroups[1]}×${d.workgroups[2]}`
        lines.push(`  ${i}: shader[${d.pipelineIndex}] ${shader?.label || '?'} — ${wg}`)
      }

      if (tail.length > 0) {
        lines.push(`\nTail (LM head + sampling): ${tail.length} dispatches`)
        lines.push(`  [${tail.join(', ')}]`)
        for (let i = 0; i < tail.length; i++) {
          const d = repToken.dispatches[numRepeats * layerLen + i]
          const shader = captured.shaders[d.pipelineIndex]
          const wg = `${d.workgroups[0]}×${d.workgroups[1]}×${d.workgroups[2]}`
          lines.push(`  ${i}: shader[${d.pipelineIndex}] ${shader?.label || '?'} — ${wg}`)
        }
      }
    }

    lines.push(`\nTokens: ${report.tokenSequences.length} (${decodeSeqs.length} decode)`)
    lines.push(`Dispatches/token: ${repToken.dispatches.length}`)
  }

  return lines.join('\n')
}

/** Find the shortest repeating prefix in a number sequence */
function findRepeatLength(seq: number[]): number {
  for (let len = 8; len <= seq.length / 2; len++) {
    let match = true
    // Check at least 2 full repeats
    const repeats = Math.floor(seq.length / len)
    if (repeats < 2) continue
    for (let r = 1; r < repeats && match; r++) {
      for (let i = 0; i < len; i++) {
        if (seq[r * len + i] !== seq[i]) { match = false; break }
      }
    }
    if (match) return len
  }
  return 0
}
