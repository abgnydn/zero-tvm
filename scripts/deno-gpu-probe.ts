// DENO-GPU-PROBE — is a non-browser WebGPU runtime fast enough to host the engine?
//
//   deno run --unstable-webgpu --allow-all scripts/deno-gpu-probe.ts
//
// THE QUESTION. The engine runs in a browser tab, which costs an open window
// and ~3x throughput when Chrome backgrounds it. Moving to a server runtime
// would fix both — IF the runtime's WebGPU submits at a sane rate.
//
// It is not a given. tests/kernels/gpu.mjs already runs this repo's kernels
// headless on Node via @kmamal/gpu (Dawn), and its header records the killer:
// `onSubmittedWorkDone()` resolves on a fixed ~100 ms tick there, so EVERY
// submit costs ~100 ms no matter what is in it — a MoE block measured 100.7 ms
// against 13.4 ms in Chrome for identical work. Decode submits at least once
// per token, so that binding caps a 65 tok/s model at ~10.
//
// Deno's WebGPU is wgpu, not Dawn. Different implementation, so the tick may
// not be there. This measures it before anyone ports anything.
//
// What matters for decode is SUBMIT-AND-WAIT LATENCY, not raw FLOPs: the
// engine's blocking path submits a token's worth of dispatches and waits.
// So the probe times an empty submit, a trivial submit, and a heavy one. If
// all three take the same wall-clock, the runtime is tick-limited and useless
// for decode regardless of how fast the GPU is.

const N = 30

function stats(xs: number[]): string {
  const s = [...xs].sort((a, b) => a - b)
  const med = s[Math.floor(s.length / 2)]
  return `median ${med.toFixed(2)} ms  (min ${s[0].toFixed(2)}, max ${s[s.length - 1].toFixed(2)})`
}

const adapter = await navigator.gpu?.requestAdapter()
if (!adapter) {
  console.log('no WebGPU adapter — run with --unstable-webgpu')
  Deno.exit(1)
}
// shader-f16 is REQUIRED, not optional: nearly every kernel here starts with
// `enable f16;`. A device without it fails 46 of 50 shaders on that line alone,
// which reads as "naga hates this WGSL" rather than "you forgot a feature".
const device = await adapter.requestDevice({
  requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
})
const info = (adapter as unknown as { info?: GPUAdapterInfo }).info
console.log(`adapter: ${info?.vendor ?? '?'} / ${info?.architecture ?? '?'} ${info?.description ?? ''}`)
console.log(`maxBufferSize ${(adapter.limits.maxBufferSize / 2 ** 20).toFixed(0)} MiB, `
  + `maxStorageBufferBindingSize ${(adapter.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`)
console.log(`shader-f16: ${adapter.features.has('shader-f16')}, subgroups: ${adapter.features.has('subgroups')}\n`)

// A trivial compute shader, and a heavy one, over the same buffer. The point is
// the DIFFERENCE between them — if there isn't one, the cost is the submit.
const mk = (iters: number) => device.createComputePipeline({
  layout: 'auto',
  compute: {
    module: device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  var acc = buf[gid.x];
  for (var i = 0u; i < ${iters}u; i++) { acc = acc * 1.0000001 + 0.0000001; }
  buf[gid.x] = acc;
}`,
    }),
    entryPoint: 'main',
  },
})

const SIZE = 1 << 22                       // 4 Mi floats = 16 MiB
const buf = device.createBuffer({ size: SIZE * 4, usage: GPUBufferUsage.STORAGE })
const light = mk(1)
const heavy = mk(4000)
const bg = (p: GPUComputePipeline) => device.createBindGroup({
  layout: p.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: buf } }],
})
const bgL = bg(light)
const bgH = bg(heavy)

async function timeSubmit(kind: 'empty' | 'light' | 'heavy', dispatches: number): Promise<number[]> {
  const out: number[] = []
  for (let i = 0; i < N; i++) {
    const enc = device.createCommandEncoder()
    if (kind !== 'empty') {
      const pass = enc.beginComputePass()
      pass.setPipeline(kind === 'light' ? light : heavy)
      pass.setBindGroup(0, kind === 'light' ? bgL : bgH)
      for (let d = 0; d < dispatches; d++) pass.dispatchWorkgroups(SIZE / 64 / 64)
      pass.end()
    }
    const t0 = performance.now()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
    out.push(performance.now() - t0)
  }
  return out.slice(5)                       // drop warmup
}

console.log('submit-and-wait latency, the thing decode pays once per token:\n')
const e = await timeSubmit('empty', 0)
console.log(`  empty submit          ${stats(e)}`)
const l = await timeSubmit('light', 1)
console.log(`  1 light dispatch      ${stats(l)}`)
const l10 = await timeSubmit('light', 10)
console.log(`  10 light dispatches   ${stats(l10)}`)
const h = await timeSubmit('heavy', 1)
console.log(`  1 heavy dispatch      ${stats(h)}`)

const med = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]

// THE DECISIVE ONE. The engine's generatePipelined keeps several tokens in
// flight precisely to hide submit latency. So the question is not "what does a
// submit-and-wait cost" but "is that cost per SUBMIT or per AWAIT". If K
// submits then one await costs the same as one submit then one await, the
// latency pipelines away and the runtime is fine. If it scales with K, it does
// not and nothing can hide it.
async function timePipelined(k: number): Promise<number[]> {
  const out: number[] = []
  for (let i = 0; i < N; i++) {
    const t0 = performance.now()
    for (let j = 0; j < k; j++) {
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(light)
      pass.setBindGroup(0, bgL)
      pass.dispatchWorkgroups(SIZE / 64 / 64)
      pass.end()
      device.queue.submit([enc.finish()])
    }
    await device.queue.onSubmittedWorkDone()
    out.push((performance.now() - t0) / k)     // per-submit
  }
  return out.slice(5)
}

console.log(`\npipelined — K submits, ONE await, cost divided by K:\n`)
const p1 = await timePipelined(1)
console.log(`  K=1                   ${stats(p1)} per submit`)
const p8 = await timePipelined(8)
console.log(`  K=8                   ${stats(p8)} per submit`)
const p32 = await timePipelined(32)
console.log(`  K=32                  ${stats(p32)} per submit`)

const emptyMed = med(e)
const heavyMed = med(h)
const pipeMed = med(p32)

console.log(`\n  VERDICT`)
if (emptyMed > 20 && heavyMed / emptyMed < 1.5) {
  console.log(`  TICK-LIMITED. An EMPTY submit costs ${emptyMed.toFixed(1)} ms and real work costs`)
  console.log(`  about the same, so wall-clock is set by the runtime's wait, not the GPU.`)
  console.log(`  Decode submits at least once per token: ceiling ~${(1000 / emptyMed).toFixed(0)} tok/s.`)
  console.log(`  This is the same defect @kmamal/gpu has (~100 ms/submit, tests/kernels/gpu.mjs).`)
} else if (emptyMed < 5) {
  console.log(`  USABLE. An empty submit costs ${emptyMed.toFixed(2)} ms, and heavy work costs`)
  console.log(`  ${(heavyMed / emptyMed).toFixed(1)}x that — the GPU is what is being measured.`)
  console.log(`  A decode loop here would not be submit-bound. Porting the engine is worth costing.`)
} else if (pipeMed < emptyMed / 4) {
  console.log(`  PIPELINES AWAY. A lone submit-and-wait costs ${emptyMed.toFixed(1)} ms, but 32 submits`)
  console.log(`  behind one await cost ${pipeMed.toFixed(2)} ms each — ${(emptyMed / pipeMed).toFixed(0)}x less. The cost is per`)
  console.log(`  AWAIT, not per submit, so a pipelined decode loop hides it exactly as`)
  console.log(`  generatePipelined already does in the browser. Porting is worth costing.`)
} else {
  console.log(`  SUBMIT-BOUND. ${emptyMed.toFixed(1)} ms for a lone submit-and-wait, and still`)
  console.log(`  ${pipeMed.toFixed(1)} ms each at K=32 — the cost does NOT pipeline away.`)
  console.log(`  Ceiling ~${(1000 / pipeMed).toFixed(0)} submits/s regardless of GPU speed.`)
}
console.log(`\n  For reference: Chrome on this machine runs the shipped models at 55-70 tok/s,`)
console.log(`  i.e. ~15 ms per token for ALL of a token's dispatches.`)

// ---- the other half: will naga even compile our shaders? -------------------
// wgpu uses naga, Chrome uses Tint. They disagree, and a runtime that cannot
// compile the kernels is a non-starter no matter how fast it submits.
const { withPrelude } = await import('../src/compiler/shader-prelude.ts')
const { PHI3 } = await import('../src/compiler/model-spec.ts')
const dir = new URL('../src/compiler/shaders', import.meta.url).pathname
let ok = 0
const sgFail: string[] = []
const otherFail: string[] = []
for (const f of [...Deno.readDirSync(dir)].filter((e) => e.name.endsWith('.wgsl'))) {
  const src = Deno.readTextFileSync(`${dir}/${f.name}`)
  device.pushErrorScope('validation')
  try { device.createShaderModule({ code: withPrelude(src, PHI3) }) } catch { /* the scope reports it */ }
  const err = await device.popErrorScope()
  if (!err) { ok++; continue }
  ;(/subgroup/i.test(err.message) ? sgFail : otherFail).push(f.name)
}
console.log(`\nWGSL acceptance (naga, with the engine's prelude):`)
console.log(`  ${ok}/${ok + sgFail.length + otherFail.length} compile`)
console.log(`  ${sgFail.length} rejected for subgroups: naga has no `
  + `'enable subgroups' yet, though requestDevice ACCEPTS the feature name`)
console.log(`  ${otherFail.length} rejected for anything else`
  + (otherFail.length ? `: ${otherFail.join(', ')}` : ' — the WGSL is otherwise portable'))
console.log(`\n  What no subgroups costs, from this repo's own notes:`)
console.log(`   - MoE specs cannot run AT ALL (buildDecodeEngine throws; no scalar router)`)
console.log(`   - Qwen3.5 chunked prefill needs sg32 and falls back per-token: -2.98x prefill`)
console.log(`   - attention/FFN drop to scalar variants`)
console.log(`  Prefill is the agentic bottleneck, so that third bullet is the expensive one.`)
