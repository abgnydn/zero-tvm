/**
 * DISPATCH RECORDER — Capture full dispatch tape for analysis.
 *
 * Records every writeBuffer and dispatch operation for two consecutive
 * decode tokens, then diffs them to determine what changes per-token
 * vs what's constant. This tells us if single-encoder replay is feasible.
 */

interface WriteOp {
  bufferLabel: string
  bufferSize: number
  offset: number
  dataSize: number
  dataHash: number
  data: Uint8Array
}

interface DispatchOp {
  pipelineLabel: string
  workgroups: [number, number, number]
  bindGroupIndices: number[]
}

interface TokenRecord {
  writes: WriteOp[]
  dispatches: DispatchOp[]
}

let enabled = false
let tokenIndex = 0
const tokenRecords: TokenRecord[] = []
let currentRecord: TokenRecord = { writes: [], dispatches: [] }

export function setRecorderEnabled(on: boolean): void {
  enabled = on
  tokenIndex = 0
  tokenRecords.length = 0
  currentRecord = { writes: [], dispatches: [] }
}

/** Fast hash for comparing data blobs */
function hashBytes(data: Uint8Array): number {
  let h = 0x811c9dc5
  for (let i = 0; i < data.length; i++) {
    h ^= data[i]
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

function onTokenBoundary(): void {
  if (!enabled) return
  tokenIndex++
  if (tokenIndex >= 2 && currentRecord.dispatches.length > 0) {
    tokenRecords.push(currentRecord)
    if (tokenRecords.length >= 3) {
      // Got prefill + 2 decode tokens — analyze
      enabled = false
      console.log('[recorder] Captured', tokenRecords.length, 'tokens. Analyzing...')
      analyzeTokens()
    }
  }
  currentRecord = { writes: [], dispatches: [] }
}

function analyzeTokens(): void {
  // Skip token 0 (prefill), compare tokens 1 and 2 (decode)
  if (tokenRecords.length < 3) return
  const tok1 = tokenRecords[1]
  const tok2 = tokenRecords[2]

  console.log(`\n=== DISPATCH TAPE DIFF ===`)
  console.log(`Token 1: ${tok1.writes.length} writes, ${tok1.dispatches.length} dispatches`)
  console.log(`Token 2: ${tok2.writes.length} writes, ${tok2.dispatches.length} dispatches`)

  // Diff writes
  const maxWrites = Math.max(tok1.writes.length, tok2.writes.length)
  let sameWrites = 0
  let diffWrites = 0
  let missingWrites = 0
  const diffDetails: string[] = []

  for (let i = 0; i < maxWrites; i++) {
    const w1 = tok1.writes[i]
    const w2 = tok2.writes[i]
    if (!w1 || !w2) {
      missingWrites++
      continue
    }
    if (w1.dataHash === w2.dataHash && w1.bufferLabel === w2.bufferLabel && w1.offset === w2.offset) {
      sameWrites++
    } else {
      diffWrites++
      if (diffDetails.length < 20) {
        // Show what changed
        let detail = `  write[${i}]: ${w1.bufferLabel}+${w1.offset} (${w1.dataSize}B)`
        if (w1.dataHash !== w2.dataHash) {
          detail += ' DATA_CHANGED'
          // Show hex diff for small buffers
          if (w1.dataSize <= 32) {
            const hex1 = Array.from(w1.data).map(b => b.toString(16).padStart(2, '0')).join(' ')
            const hex2 = Array.from(w2.data).map(b => b.toString(16).padStart(2, '0')).join(' ')
            detail += `\n    tok1: ${hex1}\n    tok2: ${hex2}`
          }
        }
        if (w1.bufferLabel !== w2.bufferLabel) detail += ` BUFFER_CHANGED(${w2.bufferLabel})`
        diffDetails.push(detail)
      }
    }
  }

  console.log(`\nWriteBuffer diff:`)
  console.log(`  Same: ${sameWrites}/${maxWrites} (${(sameWrites / maxWrites * 100).toFixed(0)}%)`)
  console.log(`  Changed: ${diffWrites}`)
  console.log(`  Missing: ${missingWrites}`)
  if (diffDetails.length > 0) {
    console.log(`\nChanged writes:`)
    for (const d of diffDetails) console.log(d)
  }

  // Diff dispatches
  const maxDispatches = Math.max(tok1.dispatches.length, tok2.dispatches.length)
  let sameDispatches = 0
  for (let i = 0; i < maxDispatches; i++) {
    const d1 = tok1.dispatches[i]
    const d2 = tok2.dispatches[i]
    if (d1 && d2 && d1.pipelineLabel === d2.pipelineLabel &&
        d1.workgroups[0] === d2.workgroups[0] &&
        d1.workgroups[1] === d2.workgroups[1] &&
        d1.workgroups[2] === d2.workgroups[2]) {
      sameDispatches++
    }
  }

  console.log(`\nDispatch diff:`)
  console.log(`  Same: ${sameDispatches}/${maxDispatches} (${(sameDispatches / maxDispatches * 100).toFixed(0)}%)`)

  if (sameWrites === maxWrites && sameDispatches === maxDispatches) {
    console.log(`\n*** ALL WRITES AND DISPATCHES ARE IDENTICAL BETWEEN TOKENS ***`)
    console.log(`*** SINGLE-ENCODER REPLAY IS FEASIBLE WITH ZERO PER-TOKEN OVERHEAD ***`)
  } else if (diffWrites <= 5) {
    console.log(`\n*** Only ${diffWrites} writes change per token — replay is feasible ***`)
    console.log(`*** Pre-bake 342 dispatches, update only ${diffWrites} values per token ***`)
  }

  // Expose for devtools
  ;(window as unknown as Record<string, unknown>).__tokenRecords = tokenRecords
}

/** Patch a GPUDevice to record dispatch tape */
export function patchDeviceRecorder(device: GPUDevice): void {
  // Label buffers for tracking
  const bufferLabels = new WeakMap<GPUBuffer, string>()
  let bufferCount = 0

  const prevCreateBuffer = device.createBuffer
  device.createBuffer = function(descriptor: GPUBufferDescriptor) {
    const buffer = prevCreateBuffer.call(device, descriptor)
    const label = descriptor.label || `buf_${bufferCount++}_${descriptor.size}B`
    bufferLabels.set(buffer, label)

    // Token boundary detection
    if (descriptor.usage & GPUBufferUsage.MAP_READ) {
      const origMapAsync = buffer.mapAsync.bind(buffer)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(buffer as any).mapAsync = function(...args: Parameters<GPUBuffer['mapAsync']>) {
        onTokenBoundary()
        return origMapAsync(...args)
      }
    }

    return buffer
  }

  // Record writeBuffer
  const prevWriteBuffer = device.queue.writeBuffer
  device.queue.writeBuffer = function(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: BufferSource | SharedArrayBuffer,
    dataOffset?: number,
    size?: number,
  ) {
    if (enabled && tokenIndex >= 1) {
      let bytes: Uint8Array
      if (data instanceof ArrayBuffer) {
        const off = dataOffset ?? 0
        const sz = size ?? (data.byteLength - off)
        bytes = new Uint8Array(data.slice(off, off + sz))
      } else if (ArrayBuffer.isView(data)) {
        const off = dataOffset ?? 0
        const sz = size ?? (data.byteLength - off)
        bytes = new Uint8Array(data.buffer.slice(data.byteOffset + off, data.byteOffset + off + sz))
      } else {
        bytes = new Uint8Array(0)
      }

      currentRecord.writes.push({
        bufferLabel: bufferLabels.get(buffer) ?? 'unknown',
        bufferSize: buffer.size,
        offset: bufferOffset,
        dataSize: bytes.length,
        dataHash: hashBytes(bytes),
        data: bytes,
      })
    }

    return prevWriteBuffer.call(device.queue, buffer, bufferOffset, data, dataOffset, size)
  }

  // Record dispatches
  const prevCreateEncoder = device.createCommandEncoder
  device.createCommandEncoder = function(desc?: GPUCommandEncoderDescriptor) {
    const encoder = prevCreateEncoder.call(device, desc)
    const prevBeginPass = encoder.beginComputePass

    encoder.beginComputePass = function(passDesc?: GPUComputePassDescriptor) {
      const pass = prevBeginPass.call(encoder, passDesc)

      if (enabled && tokenIndex >= 1) {
        let currentPipelineLabel = 'unknown'
        const currentBindGroups: number[] = []

        const prevSetPipeline = pass.setPipeline.bind(pass)
        pass.setPipeline = function(pipeline: GPUComputePipeline) {
          currentPipelineLabel = pipeline.label || 'unlabeled'
          return prevSetPipeline(pipeline)
        }

        const prevSetBindGroup = pass.setBindGroup.bind(pass)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ;(pass as any).setBindGroup = function(index: number, ...rest: unknown[]) {
          currentBindGroups.push(index)
          return (prevSetBindGroup as Function)(index, ...rest)
        }

        const prevDispatch = pass.dispatchWorkgroups.bind(pass)
        pass.dispatchWorkgroups = function(x: number, y?: number, z?: number) {
          currentRecord.dispatches.push({
            pipelineLabel: currentPipelineLabel,
            workgroups: [x, y ?? 1, z ?? 1],
            bindGroupIndices: [...currentBindGroups],
          })
          return prevDispatch(x, y, z)
        }
      }

      return pass
    }

    return encoder
  }
}
