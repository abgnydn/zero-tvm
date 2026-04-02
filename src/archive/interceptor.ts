/**
 * SHADER INTERCEPTOR — Replaces TVM's matmul shaders at runtime.
 *
 * Hooks into the profiler's existing monkey-patches to:
 * 1. Detect matmul shaders in createShaderModule
 * 2. Replace WGSL source with our tiled version
 * 3. Adjust dispatch workgroup counts for the new tile size
 *
 * Enable with setInterceptEnabled(true) before model load.
 */

import { generateTiledMatmul, detectMatmul, TILE_ROWS } from './tiled-matmul.js'

interface ReplacedShader {
  originalCode: string
  tiledCode: string
  dPacked: number
  entryPoint: string
}

let enabled = false
const replacedShaders = new Map<number, ReplacedShader>()  // shader index → replacement info
const replacedPipelines = new WeakSet<GPUComputePipeline>()

/** Enable/disable shader interception */
export function setInterceptEnabled(on: boolean): void {
  enabled = on
}

/** Get replacement stats */
export function getInterceptStats(): string {
  const lines = [`Shader interception: ${enabled ? 'ON' : 'OFF'}`]
  lines.push(`Replaced shaders: ${replacedShaders.size}`)
  for (const [idx, info] of replacedShaders) {
    const D = info.dPacked * 8
    lines.push(`  shader[${idx}] ${info.entryPoint}: D=${D}, ${TILE_ROWS}x tiling`)
  }
  return lines.join('\n')
}

/**
 * Patch a GPUDevice to intercept and replace TVM matmul shaders.
 * Call this in addition to patchDevice (profiler).
 * Must be called before model load.
 */
export function patchDeviceIntercept(device: GPUDevice): void {
  let shaderIndex = 0

  // === INTERCEPT createShaderModule ===
  const prevCreateShaderModule = device.createShaderModule
  device.createShaderModule = function(desc: GPUShaderModuleDescriptor) {
    const idx = shaderIndex++
    if (!enabled) return prevCreateShaderModule.call(device, desc)

    const code = desc.code
    // Extract entry point from the function declaration
    const fnMatch = code.match(/fn\s+(\w+)\s*\(/)
    const entryPoint = fnMatch?.[1] ?? 'unknown'

    const matmulInfo = detectMatmul(code, entryPoint)
    if (matmulInfo) {
      const tiledCode = generateTiledMatmul(matmulInfo.dPacked, entryPoint)
      replacedShaders.set(idx, {
        originalCode: code,
        tiledCode,
        dPacked: matmulInfo.dPacked,
        entryPoint,
      })
      console.log(`[interceptor] Replacing shader[${idx}] ${entryPoint} (D=${matmulInfo.dPacked * 8}) with tiled version`)
      return prevCreateShaderModule.call(device, { ...desc, code: tiledCode })
    }

    return prevCreateShaderModule.call(device, desc)
  }

  // === TAG REPLACED PIPELINES ===
  function tagIfReplaced(pipeline: GPUComputePipeline, desc: GPUComputePipelineDescriptor): void {
    const module = desc.compute.module as unknown as Record<string, unknown>
    const capturedIdx = module.__capturedIndex as number | undefined
    if (capturedIdx !== undefined && replacedShaders.has(capturedIdx)) {
      replacedPipelines.add(pipeline)
    }
  }

  const prevCreatePipeline = device.createComputePipeline
  device.createComputePipeline = function(desc: GPUComputePipelineDescriptor) {
    const pipeline = prevCreatePipeline.call(device, desc)
    tagIfReplaced(pipeline, desc)
    return pipeline
  }

  const prevCreatePipelineAsync = device.createComputePipelineAsync
  device.createComputePipelineAsync = async function(desc: GPUComputePipelineDescriptor) {
    const pipeline = await prevCreatePipelineAsync.call(device, desc)
    tagIfReplaced(pipeline, desc)
    return pipeline
  }

  // === ADJUST DISPATCH WORKGROUP COUNTS ===
  const prevCreateEncoder = device.createCommandEncoder
  device.createCommandEncoder = function(desc?: GPUCommandEncoderDescriptor) {
    const encoder = prevCreateEncoder.call(device, desc)
    const prevBeginPass = encoder.beginComputePass

    encoder.beginComputePass = function(passDesc?: GPUComputePassDescriptor) {
      const pass = prevBeginPass.call(encoder, passDesc)
      let currentPipelineReplaced = false

      const prevSetPipeline = pass.setPipeline.bind(pass)
      pass.setPipeline = function(pipeline: GPUComputePipeline) {
        currentPipelineReplaced = replacedPipelines.has(pipeline)
        return prevSetPipeline(pipeline)
      }

      const prevDispatch = pass.dispatchWorkgroups.bind(pass)
      pass.dispatchWorkgroups = function(x: number, y?: number, z?: number) {
        if (currentPipelineReplaced) {
          // Tiled shader processes TILE_ROWS outputs per workgroup
          const tiledX = Math.ceil(x / TILE_ROWS)
          return prevDispatch(tiledX, y, z)
        }
        return prevDispatch(x, y, z)
      }

      return pass
    }

    return encoder
  }
}
