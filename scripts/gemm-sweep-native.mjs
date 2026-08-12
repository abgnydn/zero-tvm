#!/usr/bin/env node
// GEMM-SWEEP-NATIVE — the E3/E5 sweep, headless on dawn.node.
//
// FIRST RUN POST-MORTEM (2026-08-12): the whole run was VOID, and the fault
// was the GATE, not the kernels — a mechanical WGSL diff proved the builder's
// E1 config semantically identical to the shipped kernel, and the "failures"
// traced to the CPU reference bit-decoding FLOAT VALUES (all-zero reference).
// Two gate fixes are in: NaN fails (!(rel<=tol)), and the reference uses the
// raw float values directly.
//
//   npm i --no-save webgpu
//   node --experimental-strip-types scripts/gemm-sweep-native.mjs [--full]
//
// Chrome is out of the loop entirely: each config compiles, gates against a
// CPU reference, and times in ~2s. The builder parameterizes the E1 family:
//
//   sg    subgroups per workgroup (4 or 8)         WG = sg*32 threads
//   tm,tn workgroup tile (M x N); per-subgroup subtile is always 16 x 32
//         (2 A-frags x 4 B-frags = 8 named accumulators — arrays of
//         fragments trip the MSL spill bug, crbug 443794633)
//   tk    K-tile (16 / 32 / 64)
//   sw    B-shared layout: stride-pad (tk+8) or llama.cpp's 8x8-block
//         swizzle (each fragment a dense 64-element block — E5)
//
// Correctness gate first, per config, vs a from-the-formula CPU reference —
// a config that fails prints VOID and never posts a time. Same discipline as
// scripts/gemm-bench.mjs; both catches there were real.

import { installShims } from './native/shims.mjs'
import { toF16, f16Array, f16BitsToF32 } from '../tests/kernels/half.mjs'

const FULL = process.argv.includes('--full')

// ---------------------------------------------------------------- builder

function sweepWGSL({ sg, tm, tn, tk, sw, sk = 1 }) {
  // subtile grid: how the sg subgroups cover the tm x tn tile in 16x32 steps
  const rows = tm / 16, cols = tn / 32
  if (rows * cols !== sg) throw new Error(`sg=${sg} cannot tile ${tm}x${tn}`)
  const WG = sg * 32
  const aStride = tk + 8
  const aElems = tm * aStride
  const bElems = sw === 'swz' ? tn * tk : tn * (tk + 8)
  const bStride = sw === 'swz' ? 0 : tk + 8
  const kWords = tk / 8
  const wordsTotal = (tn * kWords)
  const wordsPer = Math.ceil(wordsTotal / WG)
  const aPer = Math.ceil((tm * tk) / WG)
  const oPer = Math.ceil((tm * tn) / WG)

  // B addressing: pad layout stores row-major [n][k] stride tk+8 and loads
  // col-major fragments; swizzle stores each (n8, k8) 8x8 block contiguously
  // (64 elements, k-major inside), fragment loads are dense stride-8.
  const bStore = sw === 'swz'
    ? `let blk = (row / 8u) * ${tk / 8}u + wcol;   // (n8 block row) x (k8)
        let base = blk * 64u + (row % 8u);            // column inside the 8x8: k-major
        // store 8 k-values for this n at stride 8
        `
    : `let base = row * ${tk + 8}u + wcol * 8u;
        `
  const bStoreElem = sw === 'swz'
    ? (i) => `Bsh[base + ${i}u * 8u] = ${i % 2 === 0 ? 'lo' : 'hi'}[${(i - (i % 2)) / 2}u];`
    : (i) => `Bsh[base + ${i}u] = ${i % 2 === 0 ? 'lo' : 'hi'}[${(i - (i % 2)) / 2}u];`
  // ^ value order: element k sits at nibble k; lo carries k=0,2,4,6, hi k=1,3,5,7.
  //   pad layout: Bsh[base+k] = row-major [n][k], which for the k x n right
  //   operand is COL-major (stride tk+8). swizzle: block element (k, n) sits
  //   at blk*64 + k*8 + n — ROW-major inside the dense 8x8 block, so the
  //   fragment load must say colMajor=false with stride 8 (the first sweep
  //   said true, grading every swz config against a transposed B; it also
  //   stepped nSub by 32 when a fragment covers 8 N).
  const bLoadOff = sw === 'swz'
    ? (nSub) => `(((bBase + ${nSub} * 8u) / 8u) * ${tk / 8}u + k8) * 64u`
    : (nSub) => `bBase * ${tk + 8}u + ${nSub} * 8u * ${tk + 8}u + k8 * 8u`
  const bLoadStride = sw === 'swz' ? 8 : tk + 8
  const bLoadCol = sw === 'swz' ? 'false' : 'true'

  return `
diagnostic(off, chromium.subgroup_matrix_uniformity);
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

${sk > 1
    ? '@group(0) @binding(0) var<storage, read_write> partials : array<f32>; // [sk][M][N]'
    : '@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;'}
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;
@group(0) @binding(5) var<storage, read> biases : array<f16>;
struct PODArgs { K_PACKED: u32, SCALES_PER_ROW: u32, packGridDimX: u32, M_ROWS: u32 }
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> Ash : array<f16, ${aElems}>;
var<workgroup> Bsh : array<f16, ${bElems}>;
var<workgroup> Osh : array<f32, ${tm * tn}>;

@compute @workgroup_size(${WG}, 1, 1)
fn sweep(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
  @builtin(subgroup_size) sgSize : u32
) {
  let K_PACKED = podArgs.K_PACKED;
  let SPR = podArgs.SCALES_PER_ROW;
  let N = podArgs.packGridDimX;
  let M = podArgs.M_ROWS;
  let K = K_PACKED * 8u;
  let nBase = wid.x * ${tn}u;
  let mBase = wid.y * ${tm}u;
  let tid = lid.x;
  let sgi = tid / sgSize;
  let sgRow = sgi / ${cols}u;
  let sgCol = sgi % ${cols}u;

  var c00 : subgroup_matrix_result<f32, 8, 8>; var c01 : subgroup_matrix_result<f32, 8, 8>;
  var c02 : subgroup_matrix_result<f32, 8, 8>; var c03 : subgroup_matrix_result<f32, 8, 8>;
  var c10 : subgroup_matrix_result<f32, 8, 8>; var c11 : subgroup_matrix_result<f32, 8, 8>;
  var c12 : subgroup_matrix_result<f32, 8, 8>; var c13 : subgroup_matrix_result<f32, 8, 8>;

  // split-K (E4): grid z slices K; each slice is a tk-multiple, computed from
  // the RUNTIME K so one pipeline serves every shape. sk=1 degenerates to the
  // whole range.
  let KS = ((K / ${sk}u + ${tk - 1}u) / ${tk}u) * ${tk}u;
  let kLo = wid.z * KS;
  let kHi = min(kLo + KS, K);
  for (var k0 : u32 = kLo; k0 < kHi; k0 = k0 + ${tk}u) {
    for (var i : u32 = 0u; i < ${aPer}u; i = i + 1u) {
      let idx = tid * ${aPer}u + i;
      if (idx < ${tm * tk}u) {
        Ash[(idx / ${tk}u) * ${aStride}u + (idx % ${tk}u)] =
          input_buf[(mBase + idx / ${tk}u) * K + k0 + (idx % ${tk}u)];
      }
    }
    for (var wq : u32 = 0u; wq < ${wordsPer}u; wq = wq + 1u) {
      let widx = tid * ${wordsPer}u + wq;
      if (widx < ${wordsTotal}u) {
        let row = widx / ${kWords}u;
        let wcol = widx % ${kWords}u;
        let n2 = nBase + row;
        ${bStore}if (n2 < N) {
          let wordIdx = (k0 >> 3u) + wcol;
          let p = weights[n2 * K_PACKED + wordIdx];
          let sv = f16(scales[n2 * SPR + (wordIdx >> 3u)]);
          let bv = f16(biases[n2 * SPR + (wordIdx >> 3u)]);
          let lo = vec4<f16>(unpack4xU8(p & 0x0F0F0F0Fu)) * sv + vec4<f16>(bv);
          let hi = vec4<f16>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) * sv + vec4<f16>(bv);
          ${[...Array(8).keys()].map(bStoreElem).join('\n          ')}
        } else {
          ${[...Array(8).keys()].map((i) => (sw === 'swz' ? `Bsh[base + ${i}u * 8u] = 0.0h;` : `Bsh[base + ${i}u] = 0.0h;`)).join('\n          ')}
        }
      }
    }
    workgroupBarrier();

    let bBase = sgCol * 32u;
    for (var k8 : u32 = 0u; k8 < ${tk / 8}u; k8 = k8 + 1u) {
      let aOff = sgRow * 16u * ${aStride}u + k8 * 8u;
      let a0 = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&Ash, aOff, false, ${aStride}u);
      let a1 = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&Ash, aOff + 8u * ${aStride}u, false, ${aStride}u);
      let b0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, ${bLoadOff('0u')}, ${bLoadCol}, ${bLoadStride}u);
      let b1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, ${bLoadOff('1u')}, ${bLoadCol}, ${bLoadStride}u);
      let b2 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, ${bLoadOff('2u')}, ${bLoadCol}, ${bLoadStride}u);
      let b3 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, ${bLoadOff('3u')}, ${bLoadCol}, ${bLoadStride}u);
      c00 = subgroupMatrixMultiplyAccumulate(a0, b0, c00);
      c01 = subgroupMatrixMultiplyAccumulate(a0, b1, c01);
      c02 = subgroupMatrixMultiplyAccumulate(a0, b2, c02);
      c03 = subgroupMatrixMultiplyAccumulate(a0, b3, c03);
      c10 = subgroupMatrixMultiplyAccumulate(a1, b0, c10);
      c11 = subgroupMatrixMultiplyAccumulate(a1, b1, c11);
      c12 = subgroupMatrixMultiplyAccumulate(a1, b2, c12);
      c13 = subgroupMatrixMultiplyAccumulate(a1, b3, c13);
    }
    workgroupBarrier();
  }

  let oBase = sgRow * 16u * ${tn}u + sgCol * 32u;
  subgroupMatrixStore(&Osh, oBase, c00, false, ${tn}u);
  subgroupMatrixStore(&Osh, oBase + 8u, c01, false, ${tn}u);
  subgroupMatrixStore(&Osh, oBase + 16u, c02, false, ${tn}u);
  subgroupMatrixStore(&Osh, oBase + 24u, c03, false, ${tn}u);
  subgroupMatrixStore(&Osh, oBase + 8u * ${tn}u, c10, false, ${tn}u);
  subgroupMatrixStore(&Osh, oBase + 8u * ${tn}u + 8u, c11, false, ${tn}u);
  subgroupMatrixStore(&Osh, oBase + 8u * ${tn}u + 16u, c12, false, ${tn}u);
  subgroupMatrixStore(&Osh, oBase + 8u * ${tn}u + 24u, c13, false, ${tn}u);
  workgroupBarrier();
  for (var i : u32 = 0u; i < ${oPer}u; i = i + 1u) {
    let idx = tid * ${oPer}u + i;
    if (idx < ${tm * tn}u) {
      let m = idx / ${tn}u;
      let n2 = idx % ${tn}u;
      if (mBase + m < M && nBase + n2 < N) {
        ${sk > 1
    // Accumulators zero-init, so a z-slice past K still stores zeros —
    // partials never need a clear pass.
    ? `partials[(wid.z * M + mBase + m) * N + nBase + n2] = Osh[idx];`
    : `output_buf[(mBase + m) * N + nBase + n2] = f16(Osh[idx]);`}
      }
    }
  }
}`
}

// E4's second dispatch: sum the sk f32 partial planes into the f16 output.
function reduceWGSL(sk) {
  return `
enable f16;
@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> partials : array<f32>;
struct PODArgs { K_PACKED: u32, SCALES_PER_ROW: u32, packGridDimX: u32, M_ROWS: u32 }
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn reduce(@builtin(global_invocation_id) gid : vec3<u32>) {
  let MN = podArgs.M_ROWS * podArgs.packGridDimX;
  if (gid.x >= MN) { return; }
  var acc = 0.0;
  for (var z : u32 = 0u; z < ${sk}u; z = z + 1u) {
    acc = acc + partials[z * MN + gid.x];
  }
  output_buf[gid.x] = f16(acc);
}`
}

// ---------------------------------------------------------------- harness

await installShims({ unsafe: true })
const adapter = await navigator.gpu.requestAdapter()
if (!adapter.features.has('chromium-experimental-subgroup-matrix')) {
  console.log('no subgroup-matrix on this adapter'); process.exit(1)
}
const device = await adapter.requestDevice({
  requiredFeatures: ['shader-f16', 'subgroups', 'chromium-experimental-subgroup-matrix'],
  requiredLimits: {
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    // split-K partials at M=512 x N=19456 x sk4 (f32) outgrow the 128 MiB default
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  },
})

const rnd = (() => { let st = 47; return () => (st = (st * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff })()
const mk = (code, entry = 'sweep') => {
  device.pushErrorScope('validation')
  const pipe = device.createComputePipeline({
    layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: entry },
  })
  return device.popErrorScope().then((e) => (e ? { err: e.message.split('\n')[0] } : { pipe }))
}
const buf = (data, usage) => {
  const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(b, 0, data)
  return b
}
const ST = GPUBufferUsage.STORAGE

function setup(M, N, K, CAP) {
  const KP = K / 8, SPR = K / 64
  const w = new Uint32Array(N * KP); for (let i = 0; i < w.length; i++) w[i] = (rnd() * 0xffffffff) >>> 0
  const sc = Array.from({ length: N * SPR }, () => toF16(rnd() * 0.05 + 0.01))
  const bi = Array.from({ length: N * SPR }, () => toF16(rnd() * 0.1 - 0.05))
  const a = Array.from({ length: CAP * K }, () => toF16(rnd() * 2 - 1))
  return {
    M, N, K, KP, SPR, raw: { w, sc, bi, a },
    wB: buf(w, ST), scB: buf(f16Array(sc), ST), biB: buf(f16Array(bi), ST), aB: buf(f16Array(a), ST),
    outB: device.createBuffer({ size: CAP * N * 2, usage: ST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
    podB: buf(new Uint32Array([KP, SPR, N, M]), GPUBufferUsage.UNIFORM),
  }
}
const bind = (pipe, s, first) => device.createBindGroup({
  layout: pipe.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: first ?? s.outB } }, { binding: 1, resource: { buffer: s.aB } },
    { binding: 2, resource: { buffer: s.scB } }, { binding: 3, resource: { buffer: s.wB } },
    { binding: 4, resource: { buffer: s.podB } }, { binding: 5, resource: { buffer: s.biB } },
  ],
})
const reduceBind = (pipe, s, part) => device.createBindGroup({
  layout: pipe.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: s.outB } },
    { binding: 1, resource: { buffer: part } },
    { binding: 2, resource: { buffer: s.podB } },
  ],
})
// One config = one or two dispatches (split-K adds the reduce). WebGPU orders
// dispatches within a pass, so main→reduce chains without a pass break.
function encodeRun(pass, made, cfg, s, part) {
  pass.setPipeline(made.pipe)
  pass.setBindGroup(0, bind(made.pipe, s, part))
  pass.dispatchWorkgroups(Math.ceil(s.N / cfg.tn), Math.ceil(s.M / cfg.tm), cfg.sk ?? 1)
  if (made.reduce) {
    pass.setPipeline(made.reduce)
    pass.setBindGroup(0, reduceBind(made.reduce, s, part))
    pass.dispatchWorkgroups(Math.ceil((s.M * s.N) / 256), 1, 1)
  }
}
const mkPart = (cfg, s) => (cfg.sk ?? 1) > 1
  ? device.createBuffer({ size: cfg.sk * s.M * s.N * 4, usage: ST })
  : null

// gate: small shape vs the formula
async function gate(made, cfg) {
  const s = setup(13, 96, 128, 32)
  device.queue.writeBuffer(s.outB, 0, new Uint16Array(32 * 96))
  const part = mkPart(cfg, s)
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  encodeRun(pass, made, cfg, s, part)
  pass.end()
  const rb = device.createBuffer({ size: 13 * 96 * 2, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  enc.copyBufferToBuffer(s.outB, 0, rb, 0, 13 * 96 * 2)
  device.queue.submit([enc.finish()])
  await rb.mapAsync(GPUMapMode.READ)
  const got = new Uint16Array(rb.getMappedRange().slice(0))
  rb.unmap()
  let maxRel = 0
  for (let m = 0; m < 13; m++) for (let n = 0; n < 96; n++) {
    let acc = 0
    for (let k = 0; k < 128; k++) {
      const nib = (s.raw.w[n * 16 + (k >> 3)] >>> ((k & 7) * 4)) & 15
      const g = k >> 6
      // raw.* hold ROUNDED FLOAT VALUES (toF16 returns the value, not bits) —
      // wrapping them in f16BitsToF32 treated floats as bit patterns, JS
      // coerced them to 0, and the whole first sweep was graded against a
      // zero reference: every kernel "failed" at |got|/1e-2 and the shipped
      // E1's own config read 9.1e+2. Only `got` needs bit-decoding.
      // The kernel rounds the dequantized weight to f16 (lo/hi are vec4<f16>)
      // BEFORE the MMA; a reference that keeps it in f64 disagrees by up to
      // ~1e-1 on unlucky draws, and with per-config random data the old gate
      // flipped configs across the 6e-2 line run to run. Round like the GPU.
      acc += toF16(s.raw.sc[n * 2 + g] * nib + s.raw.bi[n * 2 + g]) * s.raw.a[m * 128 + k]
    }
    const gv = f16BitsToF32(got[m * 96 + n])
    maxRel = Math.max(maxRel, Math.abs(gv - acc) / Math.max(1e-2, Math.abs(acc)))
  }
  part?.destroy()
  return maxRel
}

async function time(made, cfg, s) {
  const part = mkPart(cfg, s)
  const once = (iters) => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    for (let i = 0; i < iters; i++) encodeRun(pass, made, cfg, s, part)
    pass.end()
    device.queue.submit([enc.finish()])
  }
  once(3); await device.queue.onSubmittedWorkDone()
  const t0 = performance.now()
  once(20); await device.queue.onSubmittedWorkDone()
  const ms = (performance.now() - t0) / 20
  part?.destroy()
  return ms
}

// ---------------------------------------------------------------- sweep

const CONFIGS = []
for (const sg of [4, 8]) {
  for (const [tm, tn] of sg === 4 ? [[32, 64], [64, 32], [16, 128]] : [[64, 64], [32, 128], [128, 32]]) {
    if ((tm / 16) * (tn / 32) !== sg) continue
    for (const tk of FULL ? [16, 32, 64] : [32, 64]) {
      for (const sw of ['pad', 'swz']) {
        CONFIGS.push({ sg, tm, tn, tk, sw })
      }
    }
  }
}
// E4: split-K on the E1 shape. Motivation (docs/PREFILL_RESEARCH.md): MLX
// splits K when mTiles*nTiles < ~400; at chunk M=256 our ffn_down/o_proj
// grids are ~320 tiles. sk multiplies occupancy at the cost of an f32
// partials round-trip + a reduce dispatch.
for (const sk of [2, 4]) CONFIGS.push({ sg: 4, tm: 32, tn: 64, tk: 32, sw: 'pad', sk })
const SHAPES = [['gate_up', 2560, 19456], ['ffn_down', 9728, 2560], ['o_proj', 4096, 2560]]
const MS = FULL ? [64, 256, 512] : [64, 256]

console.log(`${CONFIGS.length} configs x ${SHAPES.length} shapes x M ${MS.join('/')} — E1 baseline: 3.55/2.86/2.80 TF at M=64\n`)
const results = []
for (const cfg of CONFIGS) {
  const name = `sg${cfg.sg} ${cfg.tm}x${cfg.tn} k${cfg.tk} ${cfg.sw}${cfg.sk ? ` sk${cfg.sk}` : ''}`
  let made
  try { made = await mk(sweepWGSL(cfg)) } catch (e) { console.log(`${name}: build threw ${e.message.slice(0, 60)}`); continue }
  if (made.err) { console.log(`${name}: ${made.err.slice(0, 90)}`); continue }
  if (cfg.sk) {
    const r = await mk(reduceWGSL(cfg.sk), 'reduce')
    if (r.err) { console.log(`${name}: reduce ${r.err.slice(0, 80)}`); continue }
    made.reduce = r.pipe
  }
  const rel = await gate(made, cfg)
  // NaN must FAIL: NaN > x is false, and the first run let NaN-producing
  // configs through the gate and into the timing table.
  // With the reference rounding dequant to f16 like the kernel, every correct
  // config lands ~5e-4; a layout/transpose bug lands O(1). 5e-3 splits them
  // with 10x headroom each way (the old 6e-2 was noise-driven).
  if (!(rel <= 5e-3)) { console.log(`${name.padEnd(22)} GATE FAIL ${Number.isNaN(rel) ? 'NaN' : rel.toExponential(1)} — VOID`); continue }
  // --gate-only: correctness without timing — valid on any power/thermal state.
  if (process.argv.includes('--gate-only')) { console.log(`${name.padEnd(22)} gate ${rel.toExponential(1)} PASS`); continue }
  const row = { name, cfg, tf: {} }
  for (const M of MS) {
    let sum = 0
    for (const [label, K, N] of SHAPES) {
      const s = setup(M, N, K, M)
      const ms = await time(made, cfg, s)
      const gf = (2 * M * N * K / 1e9) / (ms / 1000)
      row.tf[`${label}@${M}`] = gf
      sum += gf
      for (const b of [s.wB, s.scB, s.biB, s.aB, s.outB, s.podB]) b.destroy()
    }
    row.tf[`mean@${M}`] = sum / SHAPES.length
  }
  results.push(row)
  console.log(`${name.padEnd(22)} gate ${rel.toExponential(1)}  ` + MS.map((M) => `M${M} ${(row.tf[`mean@${M}`] / 1000).toFixed(2)}TF`).join('  '))
}

for (const M of MS) {
  const best = [...results].sort((a, b) => b.tf[`mean@${M}`] - a.tf[`mean@${M}`])[0]
  if (best) console.log(`\nBEST @M=${M}: ${best.name} — ` + SHAPES.map(([l]) => `${l} ${(best.tf[`${l}@${M}`] / 1000).toFixed(2)}TF`).join(', '))
}

