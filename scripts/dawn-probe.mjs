// DAWN-NODE PROBE — go/no-go for a native host (docs/PREFILL_RESEARCH.md L2).
// 1. tick test: is submit-and-wait sane, or the @kmamal 100ms disease?
// 2. features: subgroups / f16 / subgroup-matrix?
// 3. toggles: disable_robustness accepted?
// 4. the real kernel: sg-e1 on gate_up M=256, vs Chrome's 6.65ms (safe) /
//    5.27ms (unsafe).
import { create, globals } from 'webgpu'
Object.assign(globalThis, globals)
// Run from the repo root:  npm i --no-save webgpu && node --experimental-strip-types scripts/dawn-probe.mjs
import { int4MatmulSgE1WGSL } from '../src/compiler/shaders/int4_matmul.gen.ts'

const gpu = create([
  'enable-dawn-features=allow_unsafe_apis,disable_robustness',
])
const adapter = await gpu.requestAdapter()
const feats = [...adapter.features]
console.log('features:', feats.filter(f => /subgroup|f16|matrix/.test(f)).join(', ') || '(none relevant)')
const want = ['shader-f16', 'subgroups', 'chromium-experimental-subgroup-matrix'].filter(f => adapter.features.has(f))
const device = await adapter.requestDevice({ requiredFeatures: want })
console.log('device features granted:', want.join(', '))

// tick test
const t0 = performance.now()
for (let i = 0; i < 5; i++) {
  device.queue.submit([device.createCommandEncoder().finish()])
  await device.queue.onSubmittedWorkDone()
}
const tick = (performance.now() - t0) / 5
console.log(`empty submit-and-wait: ${tick.toFixed(2)} ms  ${tick > 20 ? '— THE TICK DISEASE, stop here' : '— sane'}`)

if (!adapter.features.has('chromium-experimental-subgroup-matrix')) {
  console.log('no subgroup-matrix — native host loses the fast kernel; stop.')
  process.exit(1)
}

// the real kernel: gate_up M=256 (K=2560, N=19456), affine
const M = 256, K = 2560, N = 19456
const KP = K / 8, SPR = K / 64
const pipe = device.createComputePipeline({
  layout: 'auto',
  compute: { module: device.createShaderModule({ code: int4MatmulSgE1WGSL(true) }), entryPoint: 'int4_matmul_sg_e1_affine' },
})
const buf = (bytes, usage) => device.createBuffer({ size: bytes, usage })
const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
const wB = buf(N * KP * 4, ST), scB = buf(N * SPR * 2, ST), biB = buf(N * SPR * 2, ST)
const aB = buf(M * K * 2, ST), outB = buf(M * N * 2, ST)
const podB = buf(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
device.queue.writeBuffer(podB, 0, new Uint32Array([KP, SPR, N, M]))
const bg = device.createBindGroup({
  layout: pipe.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: outB } }, { binding: 1, resource: { buffer: aB } },
    { binding: 2, resource: { buffer: scB } }, { binding: 3, resource: { buffer: wB } },
    { binding: 4, resource: { buffer: podB } }, { binding: 5, resource: { buffer: biB } },
  ],
})
const run = (iters) => {
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipe); pass.setBindGroup(0, bg)
  for (let i = 0; i < iters; i++) pass.dispatchWorkgroups(Math.ceil(N / 64), Math.ceil(M / 32), 1)
  pass.end()
  device.queue.submit([enc.finish()])
}
run(3); await device.queue.onSubmittedWorkDone()
const t1 = performance.now()
run(20); await device.queue.onSubmittedWorkDone()
const ms = (performance.now() - t1) / 20
const gf = (2 * M * N * K / 1e9) / (ms / 1000)
console.log(`sg-e1 gate_up M=256: ${ms.toFixed(2)} ms  (${gf.toFixed(0)} GF)`)
console.log(`Chrome same shape: safe 6.65 ms (3832 GF), unsafe 5.27 ms (4839 GF)`)
