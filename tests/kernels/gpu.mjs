// Minimal headless-WebGPU harness over @kmamal/gpu (Dawn native bindings).
//
// Why not Chrome? Chrome's GPU process blocklists WebGPU on software
// rasterizers, so headless WebGPU never initializes on a machine without a
// real GPU. Dawn-native has no such blocklist: it talks to whatever Vulkan
// adapter is present, including Mesa's lavapipe (a CPU software device).
//
// On a real-GPU machine (a dev Mac), the same code uses the real adapter.
// On a Linux CI runner, install lavapipe and point the loader at it:
//
//   apt-get install -y mesa-vulkan-drivers
//   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
//
// lavapipe is slow, so this suite checks CORRECTNESS, never throughput.

import gpu from '@kmamal/gpu'

export const BU = gpu.GPUBufferUsage
export const MM = gpu.GPUMapMode

/** Request a Dawn device, opting into shader-f16 when the adapter has it. */
export async function getDevice() {
  const instance = gpu.create(['enable-dawn-features=allow_unsafe_apis'])
  const adapter = await instance.requestAdapter()
  if (!adapter) throw new Error('no WebGPU adapter (is a Vulkan ICD visible?)')
  const features = new Set(adapter.features)
  const f16 = features.has('shader-f16')
  const device = await adapter.requestDevice({
    requiredFeatures: f16 ? ['shader-f16'] : [],
  })
  let lastError = null
  device.addEventListener?.('uncapturederror', (e) => {
    lastError = e.error?.message ?? String(e.error)
  })
  return { device, info: adapter.info ?? {}, f16, getLastError: () => lastError }
}

/** Create a buffer pre-filled with a typed array. */
export function buffer(device, data, usage) {
  const b = device.createBuffer({ size: Math.max(4, data.byteLength), usage })
  device.queue.writeBuffer(b, 0, data)
  return b
}

/**
 * Bind `buffers` to bindings 0..n-1 (in array order), dispatch the pipeline,
 * then copy `readBytes` out of `buffers[readIndex]` and return the raw bytes.
 */
export async function runCompute(device, pipeline, buffers, workgroups, readIndex, readBytes) {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  })
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1)
  pass.end()
  const readback = device.createBuffer({ size: readBytes, usage: BU.COPY_DST | BU.MAP_READ })
  enc.copyBufferToBuffer(buffers[readIndex], 0, readback, 0, readBytes)
  device.queue.submit([enc.finish()])
  await readback.mapAsync(MM.READ)
  return readback.getMappedRange().slice(0)
}

export function pipelineFor(device, wgsl, entryPoint) {
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: wgsl }), entryPoint },
  })
}
