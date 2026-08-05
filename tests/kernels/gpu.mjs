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
//
// That is not just a lavapipe caveat — it holds on a real GPU too. Measured
// 2026-08-05 on an M2 Max: `queue.onSubmittedWorkDone()` here resolves on a
// fixed ~100 ms tick, so ANY submit costs ~100 ms regardless of what is in it.
// An empty kernel, 1 dispatch or 100, 1 workgroup or 4096 — all 100 ms. A MoE
// block timed 100.7 ms here and 13.4 ms in Chrome for identical work (7.5x).
// Time kernels in a browser; use this harness to decide whether they are right.

import gpu from '@kmamal/gpu'

export const BU = gpu.GPUBufferUsage
export const MM = gpu.GPUMapMode

/** Request a Dawn device, opting into shader-f16 and subgroups when the
 *  adapter has them. `subgroups` is what the engine's `_sg` shader variants
 *  need; lavapipe in CI may lack it, so callers must gate on the flag. */
export async function getDevice() {
  const instance = gpu.create(['enable-dawn-features=allow_unsafe_apis'])
  const adapter = await instance.requestAdapter()
  if (!adapter) throw new Error('no WebGPU adapter (is a Vulkan ICD visible?)')
  const features = new Set(adapter.features)
  const f16 = features.has('shader-f16')
  const subgroups = features.has('subgroups')
  const requiredFeatures = []
  if (f16) requiredFeatures.push('shader-f16')
  if (subgroups) requiredFeatures.push('subgroups')
  // Lift the storage-binding/buffer ceilings to whatever the adapter offers:
  // the default 128MB maxStorageBufferBindingSize is smaller than Qwen3's
  // tied-embedding LM-head weight matrix (151936 × 320 u32 ≈ 195MB).
  // Requesting the adapter's own limits is always valid; tests that need
  // more than the granted limit must check device.limits and SKIP.
  const requiredLimits = {}
  if (adapter.limits) {
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize
    requiredLimits.maxBufferSize = adapter.limits.maxBufferSize
  }
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits })
  let lastError = null
  device.addEventListener?.('uncapturederror', (e) => {
    lastError = e.error?.message ?? String(e.error)
  })
  return { device, info: adapter.info ?? {}, f16, subgroups, getLastError: () => lastError }
}

/** Actual subgroup size, probed on-device: the `_sg` kernels hard-assume 32
 *  lanes (see e.g. qkv_fused_sg.wgsl), and adapters are allowed to pick any
 *  power of two — so we ask the hardware rather than trusting a constant. */
export async function probeSubgroupSize(device) {
  const probeWgsl = `
    enable subgroups;
    @group(0) @binding(0) var<storage, read_write> out : array<u32>;
    @compute @workgroup_size(32, 1, 1)
    fn probe(@builtin(subgroup_size) sg : u32, @builtin(local_invocation_id) tid : vec3<u32>) {
      if (tid.x == 0u) { out[0] = sg; }
    }`
  const pipe = pipelineFor(device, probeWgsl, 'probe')
  const out = device.createBuffer({ size: 4, usage: BU.STORAGE | BU.COPY_SRC })
  const bytes = await runCompute(device, pipe, [out], [1], 0, 4)
  return new Uint32Array(bytes)[0]
}

/** Create a buffer pre-filled with a typed array. */
export function buffer(device, data, usage) {
  const b = device.createBuffer({ size: Math.max(4, data.byteLength), usage })
  device.queue.writeBuffer(b, 0, data)
  return b
}

/**
 * Bind `buffers` to bindings 0..n-1 (in array order), dispatch the pipeline
 * once, then copy out each requested buffer. `reads` is [{ index, bytes }];
 * returns an array of ArrayBuffers in the same order.
 */
export async function runComputeReads(device, pipeline, buffers, workgroups, reads) {
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
  const outs = reads.map(({ bytes }) =>
    device.createBuffer({ size: bytes, usage: BU.COPY_DST | BU.MAP_READ }),
  )
  reads.forEach(({ index, bytes }, i) => enc.copyBufferToBuffer(buffers[index], 0, outs[i], 0, bytes))
  device.queue.submit([enc.finish()])
  await Promise.all(outs.map((b) => b.mapAsync(MM.READ)))
  return outs.map((b) => b.getMappedRange().slice(0))
}

/** Single-buffer convenience wrapper around runComputeReads. */
export async function runCompute(device, pipeline, buffers, workgroups, readIndex, readBytes) {
  const [bytes] = await runComputeReads(device, pipeline, buffers, workgroups, [
    { index: readIndex, bytes: readBytes },
  ])
  return bytes
}

export function pipelineFor(device, wgsl, entryPoint) {
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: wgsl }), entryPoint },
  })
}
