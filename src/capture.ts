/**
 * CAPTURE — Intercepts TVM's model load to extract shaders, pipelines, and weights.
 *
 * During WebLLM model load, TVM:
 * 1. Creates 85 shader modules (WGSL source)
 * 2. Creates 85 compute pipelines (async)
 * 3. Writes ~2GB of weight data to GPU buffers
 *
 * We intercept all three and store the results for building our own engine.
 */

export interface CapturedShader {
  index: number
  code: string
  module: GPUShaderModule
}

export interface CapturedPipeline {
  index: number
  shaderIndex: number
  entryPoint: string
  pipeline: GPUComputePipeline
  // Bind group layout extracted from the pipeline (via getBindGroupLayout)
  layout: GPUBindGroupLayout
}

export interface CapturedWeight {
  buffer: GPUBuffer
  offset: number
  size: number
}

/** A bind group entry as TVM created it */
export interface CapturedBGEntry {
  binding: number
  buffer: GPUBuffer
  offset: number
  size: number
}

/** A full dispatch as TVM executed it (during first decode token) */
export interface CapturedDispatch {
  pipeline: GPUComputePipeline
  pipelineIndex: number
  entries: CapturedBGEntry[]
  workgroups: [number, number, number]
}

/** A writeBuffer call from the captured decode token */
export interface CapturedWrite {
  buffer: GPUBuffer
  offset: number
  data: Uint8Array
}

export interface CapturedCopy {
  src: GPUBuffer
  srcOffset: number
  dst: GPUBuffer
  dstOffset: number
  size: number
}

export interface CaptureResult {
  shaders: CapturedShader[]
  pipelines: CapturedPipeline[]
  weights: CapturedWeight[]
  dispatches: CapturedDispatch[]
  writes: CapturedWrite[]
  copy: CapturedCopy | null  // the copyBufferToBuffer from captured decode token
  device: GPUDevice
}

let shaders: CapturedShader[] = []
let pipelines: CapturedPipeline[] = []
let weights: CapturedWeight[] = []
let dispatches: CapturedDispatch[] = []
let capturedWrites: CapturedWrite[] = []
let capturedCopy: CapturedCopy | null = null
let capturedDevice: GPUDevice | null = null

// Dispatch capture state
type CapturePhase = 'loading' | 'prefill_done' | 'capturing' | 'done'
let capturePhase: CapturePhase = 'loading'
let tokenBoundaryCount = 0

// Map bind group objects to their entries
const bgToEntries = new WeakMap<GPUBindGroup, CapturedBGEntry[]>()

export function getCaptureResult(): CaptureResult | null {
  if (!capturedDevice) return null
  return { shaders, pipelines, weights, dispatches, writes: capturedWrites, copy: capturedCopy, device: capturedDevice }
}

/**
 * Patch a GPUDevice to capture shaders, pipelines, and weight writes.
 * Call before model load.
 */
export function patchForCapture(device: GPUDevice): void {
  capturedDevice = device
  shaders = []
  pipelines = []
  weights = []

  let shaderIndex = 0
  const shaderModuleToIndex = new WeakMap<GPUShaderModule, number>()

  // Capture shader modules
  const origCreateShader = device.createShaderModule.bind(device)
  device.createShaderModule = function(desc: GPUShaderModuleDescriptor): GPUShaderModule {
    const module = origCreateShader(desc)
    const idx = shaderIndex++
    shaders.push({ index: idx, code: desc.code, module })
    shaderModuleToIndex.set(module, idx)
    return module
  }

  // Capture compute pipelines (async — TVM uses createComputePipelineAsync)
  const origCreatePipelineAsync = device.createComputePipelineAsync.bind(device)
  device.createComputePipelineAsync = async function(desc: GPUComputePipelineDescriptor): Promise<GPUComputePipeline> {
    const pipeline = await origCreatePipelineAsync(desc)
    const sIdx = shaderModuleToIndex.get(desc.compute.module) ?? -1
    const entry = desc.compute.entryPoint ?? 'main'
    const layout = pipeline.getBindGroupLayout(0)
    pipelines.push({ index: pipelines.length, shaderIndex: sIdx, entryPoint: entry, pipeline, layout })
    return pipeline
  }

  // Also capture sync pipeline creation
  const origCreatePipeline = device.createComputePipeline.bind(device)
  device.createComputePipeline = function(desc: GPUComputePipelineDescriptor): GPUComputePipeline {
    const pipeline = origCreatePipeline(desc)
    const sIdx = shaderModuleToIndex.get(desc.compute.module) ?? -1
    const entry = desc.compute.entryPoint ?? 'main'
    const layout = pipeline.getBindGroupLayout(0)
    pipelines.push({ index: pipelines.length, shaderIndex: sIdx, entryPoint: entry, pipeline, layout })
    return pipeline
  }

  // Capture large weight writes (>1KB)
  const origWriteBuffer = device.queue.writeBuffer.bind(device.queue)
  device.queue.writeBuffer = function(
    buffer: GPUBuffer, offset: number, data: BufferSource | SharedArrayBuffer,
    dataOffset?: number, size?: number,
  ): undefined {
    const dataSize = size ?? (data instanceof ArrayBuffer ? data.byteLength - (dataOffset ?? 0) : (data as ArrayBufferView).byteLength - (dataOffset ?? 0))
    if (dataSize > 1024) {
      weights.push({ buffer, offset, size: dataSize })
    }
    // Capture write data during decode token
    if (capturePhase === 'capturing') {
      let bytes: Uint8Array
      if (data instanceof ArrayBuffer) {
        const o = dataOffset ?? 0, s = size ?? (data.byteLength - o)
        bytes = new Uint8Array(data.slice(o, o + s))
      } else if (ArrayBuffer.isView(data)) {
        const o = dataOffset ?? 0, s = size ?? (data.byteLength - o)
        bytes = new Uint8Array(data.buffer.slice(data.byteOffset + o, data.byteOffset + o + s))
      } else {
        bytes = new Uint8Array(0)
      }
      pendingWrites.push({ buffer, offset, data: bytes })
    }
    origWriteBuffer(buffer, offset, data, dataOffset, size)
  }

  // Capture bind group entries
  const origCreateBG = device.createBindGroup.bind(device)
  device.createBindGroup = function(desc: GPUBindGroupDescriptor): GPUBindGroup {
    const bg = origCreateBG(desc)
    const entries: CapturedBGEntry[] = []
    for (const e of desc.entries) {
      const r = e.resource
      if (r && typeof r === 'object' && 'buffer' in r) {
        const br = r as GPUBufferBinding
        entries.push({
          binding: e.binding,
          buffer: br.buffer,
          offset: br.offset ?? 0,
          size: br.size ?? br.buffer.size,
        })
      }
    }
    bgToEntries.set(bg, entries)
    return bg
  }

  // Capture dispatches + writes during first decode token
  const pendingDispatches: CapturedDispatch[] = []
  const pendingWrites: CapturedWrite[] = []

  const origCreateEnc = device.createCommandEncoder.bind(device)
  device.createCommandEncoder = function(desc?: GPUCommandEncoderDescriptor): GPUCommandEncoder {
    const enc = origCreateEnc(desc)
    if (capturePhase !== 'capturing') return enc

    // Capture copyBufferToBuffer
    const origCopy = enc.copyBufferToBuffer.bind(enc)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(enc as any).copyBufferToBuffer = function(...args: unknown[]) {
      if (args.length >= 5) {
        const [s, so, d, d2, sz] = args as [GPUBuffer, number, GPUBuffer, number, number]
        capturedCopy = { src: s, srcOffset: so, dst: d, dstOffset: d2, size: sz }
      }
      return (origCopy as Function)(...args)
    }

    const origBeginPass = enc.beginComputePass.bind(enc)
    enc.beginComputePass = function(pd?: GPUComputePassDescriptor): GPUComputePassEncoder {
      const pass = origBeginPass(pd)
      let currentPipeline: GPUComputePipeline | null = null
      let currentEntries: CapturedBGEntry[] = []

      const origSetPipeline = pass.setPipeline.bind(pass)
      pass.setPipeline = function(p: GPUComputePipeline) {
        currentPipeline = p
        return origSetPipeline(p)
      }

      const origSetBG = pass.setBindGroup.bind(pass)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(pass as any).setBindGroup = function(idx: number, bg: GPUBindGroup, ...rest: unknown[]) {
        const entries = bgToEntries.get(bg)
        if (entries) currentEntries = entries
        return (origSetBG as Function)(idx, bg, ...rest)
      }

      const origDispatch = pass.dispatchWorkgroups.bind(pass)
      pass.dispatchWorkgroups = function(x: number, y?: number, z?: number) {
        if (currentPipeline) {
          const idx = pipelines.findIndex(p => p.pipeline === currentPipeline)
          pendingDispatches.push({
            pipeline: currentPipeline,
            pipelineIndex: idx,
            entries: currentEntries,
            workgroups: [x, y ?? 1, z ?? 1],
          })
        }
        return origDispatch(x, y, z)
      }
      return pass
    }
    return enc
  }

  // Token boundary detection + prevent buffer destruction
  const origCreateBuf = device.createBuffer.bind(device)
  device.createBuffer = function(desc: GPUBufferDescriptor): GPUBuffer {
    const buf = origCreateBuf(desc)

    // Prevent destruction of any buffer after capture starts
    const realDestroy = buf.destroy.bind(buf)
    buf.destroy = function() {
      if (capturePhase === 'capturing' || capturePhase === 'done') return
      return realDestroy()
    }
    if (desc.usage & GPUBufferUsage.MAP_READ) {
      const realMap = buf.mapAsync.bind(buf)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(buf as any).mapAsync = function(...args: Parameters<GPUBuffer['mapAsync']>) {
        // State machine: loading → prefill_done → capturing → done
        if (capturePhase === 'loading') {
          tokenBoundaryCount++
          if (tokenBoundaryCount === 1) {
            capturePhase = 'prefill_done'
          }
        } else if (capturePhase === 'prefill_done') {
          capturePhase = 'capturing'
          pendingDispatches.length = 0
        } else if (capturePhase === 'capturing') {
          dispatches = [...pendingDispatches]
          capturedWrites = [...pendingWrites]
          capturePhase = 'done'
          console.log(`[capture] Captured ${dispatches.length} dispatches, ${capturedWrites.length} writes`)
        }
        return realMap(...args)
      }
    }
    return buf
  }
}
