#!/usr/bin/env node
// GEMM-BENCH — the chunk-prefill GEMM kernels in isolation, on real shapes.
//
//   node scripts/gemm-bench.mjs
//
// One Chrome boot, NO model load (synthetic weights — timing does not care),
// correctness-gated before any timing: each kernel must match a CPU reference
// on a small shape first, so a wrong kernel cannot post a fast number.
//
// Exists because timing kernels through the full engine costs 2-3 minutes a
// data point (browser boot + OPFS weights + four engine arms) and the machine
// state drifts across that. Here a variant is one page.evaluate.
//
// Shapes are qwen3mlx's actual chunk GEMMs: gate_up 2560->19456,
// ffn_down 9728->2560, o_proj 4096->2560. M=64 is today's CHUNK_CAP; M=256
// is the cap the tiled kernels make interesting.

import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

await startHarness()
try {
  const page = await newPage('/docs.html')
  const out = await page.evaluate(async () => {
    const G = await import('/src/compiler/shaders/int4_matmul.gen.ts')

    // EXPERIMENTAL: subgroup-matrix GEMM (Metal simdgroup_float8x8 via
    // chromium-experimental-subgroup-matrix). One subgroup (32 threads) owns a
    // 32x32 C tile as 4x4 fragments of 8x8; K-steps of 64 stage A and a
    // DEQUANTIZED W tile in shared f16, then 8 fragment MACs per step.
    // f16 inputs, f32 accumulate — the same arithmetic MLX runs.
    const SGMAT_WGSL = `
enable f16;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;
@group(0) @binding(5) var<storage, read> biases : array<f16>;

struct PODArgs { K_PACKED: u32, SCALES_PER_ROW: u32, packGridDimX: u32, M_ROWS: u32 }
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> Ash : array<f16, 2048>;   // 32 m x 64 k
var<workgroup> Wsh : array<f16, 2048>;   // 32 n x 64 k, dequantized (s*q+b)
var<workgroup> Osh : array<f32, 1024>;   // 32 x 32 staging for guarded writes

@compute @workgroup_size(32, 1, 1)
fn sgmat(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
) {
  let K_PACKED = podArgs.K_PACKED;
  let SPR = podArgs.SCALES_PER_ROW;
  let N = podArgs.packGridDimX;
  let M = podArgs.M_ROWS;
  let K = K_PACKED * 8u;
  let nBase = wid.x * 32u;
  let mBase = wid.y * 32u;
  let tid = lid.x;

  var acc : array<subgroup_matrix_result<f32, 8, 8>, 16>;

  for (var k0 : u32 = 0u; k0 < K; k0 = k0 + 64u) {
    for (var i : u32 = 0u; i < 64u; i = i + 1u) {
      let idx = tid * 64u + i;
      Ash[idx] = input_buf[(mBase + idx / 64u) * K + k0 + (idx % 64u)];
    }
    for (var wq : u32 = 0u; wq < 8u; wq = wq + 1u) {
      let widx = tid * 8u + wq;
      let wrow = widx / 8u;
      let wcol = widx % 8u;
      let row = nBase + wrow;
      let base = wrow * 64u + wcol * 8u;
      if (row < N) {
        let wordIdx = (k0 >> 3u) + wcol;
        let p = weights[row * K_PACKED + wordIdx];
        let sVal = f32(scales[row * SPR + (wordIdx >> 3u)]);
        let bVal = f32(biases[row * SPR + (wordIdx >> 3u)]);
        for (var b2 : u32 = 0u; b2 < 8u; b2 = b2 + 1u) {
          Wsh[base + b2] = f16(sVal * f32((p >> (b2 * 4u)) & 15u) + bVal);
        }
      } else {
        for (var b2 : u32 = 0u; b2 < 8u; b2 = b2 + 1u) { Wsh[base + b2] = 0.0h; }
      }
    }
    workgroupBarrier();

    for (var k8 : u32 = 0u; k8 < 8u; k8 = k8 + 1u) {
      var L : array<subgroup_matrix_left<f16, 8, 8>, 4>;
      var R : array<subgroup_matrix_right<f16, 8, 8>, 4>;
      for (var i : u32 = 0u; i < 4u; i = i + 1u) {
        L[i] = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&Ash, (i * 8u) * 64u + k8 * 8u, false, 64u);
        R[i] = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Wsh, (i * 8u) * 64u + k8 * 8u, true, 64u);
      }
      for (var mi : u32 = 0u; mi < 4u; mi = mi + 1u) {
        for (var ni : u32 = 0u; ni < 4u; ni = ni + 1u) {
          acc[mi * 4u + ni] = subgroupMatrixMultiplyAccumulate(L[mi], R[ni], acc[mi * 4u + ni]);
        }
      }
    }
    workgroupBarrier();
  }

  // Stage results and write guarded (raggedness lives here, not in fragments).
  for (var mi : u32 = 0u; mi < 4u; mi = mi + 1u) {
    for (var ni : u32 = 0u; ni < 4u; ni = ni + 1u) {
      subgroupMatrixStore(&Osh, (mi * 8u) * 32u + ni * 8u, acc[mi * 4u + ni], false, 32u);
    }
  }
  workgroupBarrier();
  for (var i : u32 = 0u; i < 32u; i = i + 1u) {
    let idx = tid * 32u + i;
    let m = idx / 32u;
    let n2 = idx % 32u;
    if (mBase + m < M && nBase + n2 < N) {
      output_buf[(mBase + m) * N + nBase + n2] = f16(Osh[idx]);
    }
  }
}`
    const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    const SGMAT = ad.features.has('chromium-experimental-subgroup-matrix')
    const dev = await ad.requestDevice({
      requiredFeatures: ['shader-f16',
        ...(ad.features.has('subgroups') ? ['subgroups'] : []),
        ...(SGMAT ? ['chromium-experimental-subgroup-matrix'] : [])],
    })
    const lines = []
    const mk = (code, entry) => dev.createComputePipeline({
      layout: 'auto',
      compute: { module: dev.createShaderModule({ code }), entryPoint: entry },
    })
    // enable f16 is already in the generated source; no prelude needed (these
    // kernels read every dim from PODArgs).
    dev.pushErrorScope('validation')
    const KERNELS = [
      ['matvec', mk(G.int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, mDyn: true, affine: true }), 'int4_matmul_batched_dyn_affine'), 'x4'],
      ['tiled-v1', mk(G.int4MatmulTiledMWGSL(true), 'int4_matmul_tiled_m_affine'), 'tile'],
      ['tiled-v2', mk(G.int4MatmulTiledStWGSL(true), 'int4_matmul_tiled_st_affine'), 'tile'],
    ]
    if (SGMAT) {
      try { KERNELS.push(['sgmat', mk(SGMAT_WGSL, 'sgmat'), 'tile']) } catch (e) { lines.push('sgmat create threw: ' + e.message) }
    }
    const compileErr = await dev.popErrorScope()
    if (compileErr) lines.push('sgmat validation: ' + compileErr.message.split('\n').slice(0, 4).join(' | '))
    const buf = (data, usage) => {
      const b = dev.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST })
      dev.queue.writeBuffer(b, 0, data)
      return b
    }
    const ST = GPUBufferUsage.STORAGE
    const f16bits = (x) => {
      const f32 = new Float32Array([x]); const u = new Uint32Array(f32.buffer)[0]
      const s = (u >> 16) & 0x8000; let e = ((u >> 23) & 0xff) - 112; let m = (u >> 13) & 0x3ff
      if (e <= 0) return s; if (e >= 31) return s | 0x7bff
      return s | (e << 10) | m
    }
    const rnd = (() => { let st = 41; return () => (st = (st * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff })()

    const setup = (M, N, K, CAP) => {
      const KP = K / 8, SPR = K / 64
      const w = new Uint32Array(N * KP); for (let i = 0; i < w.length; i++) w[i] = (rnd() * 0xffffffff) >>> 0
      const sc = new Uint16Array(N * SPR); for (let i = 0; i < sc.length; i++) sc[i] = f16bits(rnd() * 0.05 + 0.01)
      const bi = new Uint16Array(N * SPR); for (let i = 0; i < bi.length; i++) bi[i] = f16bits(rnd() * 0.1 - 0.05)
      const a = new Uint16Array(CAP * K); for (let i = 0; i < a.length; i++) a[i] = f16bits(rnd() * 2 - 1)
      return {
        M, N, K, KP, SPR,
        wB: buf(w, ST), scB: buf(sc, ST), biB: buf(bi, ST), aB: buf(a, ST),
        outB: dev.createBuffer({ size: CAP * N * 2, usage: ST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
        podB: buf(new Uint32Array([KP, SPR, N, M]), GPUBufferUsage.UNIFORM),
        raw: { w, sc, bi, a },
      }
    }
    const bind = (pipe, s2) => dev.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: s2.outB } },
        { binding: 1, resource: { buffer: s2.aB } },
        { binding: 2, resource: { buffer: s2.scB } },
        { binding: 3, resource: { buffer: s2.wB } },
        { binding: 4, resource: { buffer: s2.podB } },
        { binding: 5, resource: { buffer: s2.biB } },
      ],
    })
    const grids = (kind, M, N) => kind === 'x4' ? [Math.ceil(N / 4), 1] : [Math.ceil(N / 32), Math.ceil(M / 32)]

    // ---- correctness gate: small shape vs CPU, per kernel -----------------
    {
      const s2 = setup(13, 96, 128, 32)
      const h2f = (h) => { const s3 = (h & 0x8000) << 16; const e = (h >> 10) & 31; const m2 = h & 1023
        if (e === 0) return m2 === 0 ? 0 : (s3 ? -1 : 1) * m2 * 2 ** -24
        if (e === 31) return NaN
        const f = new Uint32Array([s3 | ((e + 112) << 23) | (m2 << 13)]); return new Float32Array(f.buffer)[0] }
      const ref = []
      for (let m = 0; m < 13; m++) for (let n = 0; n < 96; n++) {
        let acc = 0
        for (let k = 0; k < 128; k++) {
          const nib = (s2.raw.w[n * 16 + (k >> 3)] >>> ((k & 7) * 4)) & 15
          const g = k >> 6
          acc += (h2f(s2.raw.sc[n * 2 + g]) * nib + h2f(s2.raw.bi[n * 2 + g])) * h2f(s2.raw.a[m * 128 + k])
        }
        ref.push(acc)
      }
      for (const [name, pipe, kind] of KERNELS) {
        dev.queue.writeBuffer(s2.outB, 0, new Uint16Array(32 * 96))
        const bgx = bind(pipe, s2)
        const enc = dev.createCommandEncoder()
        const pass = enc.beginComputePass()
        pass.setPipeline(pipe); pass.setBindGroup(0, bgx)
        const [gx, gy] = grids(kind, 13, 96)
        pass.dispatchWorkgroups(gx, gy, 1); pass.end()
        const rb = dev.createBuffer({ size: 13 * 96 * 2, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
        enc.copyBufferToBuffer(s2.outB, 0, rb, 0, 13 * 96 * 2)
        dev.queue.submit([enc.finish()])
        await rb.mapAsync(GPUMapMode.READ)
        const got = new Uint16Array(rb.getMappedRange().slice(0))
        rb.unmap()
        let maxRel = 0
        for (let i = 0; i < ref.length; i++) {
          const g = h2f(got[i])
          maxRel = Math.max(maxRel, Math.abs(g - ref[i]) / Math.max(1e-2, Math.abs(ref[i])))
        }
        lines.push(`gate  ${name.padEnd(9)} max rel ${maxRel.toExponential(1)} ${maxRel < 2e-2 ? 'OK' : 'WRONG — its timings below are void'}`)
      }
    }

    // ---- timing -----------------------------------------------------------
    const SHAPES = [
      ['gate_up', 2560, 19456],
      ['ffn_down', 9728, 2560],
      ['o_proj', 4096, 2560],
    ]
    const time = async (pipe, kind, s2) => {
      const bgx = bind(pipe, s2)
      const once = (iters) => {
        const enc = dev.createCommandEncoder()
        const pass = enc.beginComputePass()
        pass.setPipeline(pipe); pass.setBindGroup(0, bgx)
        const [gx, gy] = grids(kind, s2.M, s2.N)
        for (let i = 0; i < iters; i++) pass.dispatchWorkgroups(gx, gy, 1)
        pass.end()
        dev.queue.submit([enc.finish()])
      }
      once(3); await dev.queue.onSubmittedWorkDone()          // warm
      const t0 = performance.now()
      once(20); await dev.queue.onSubmittedWorkDone()
      return (performance.now() - t0) / 20
    }
    for (const M of [64, 256]) {
      for (const [label, K, N] of SHAPES) {
        const s2 = setup(M, N, K, M)
        const row = [`M=${String(M).padEnd(3)} ${label.padEnd(8)} K=${K} N=${N}`]
        for (const [name, pipe, kind] of KERNELS) {
          const ms = await time(pipe, kind, s2)
          const gf = (2 * M * N * K / 1e9) / (ms / 1000)
          row.push(`${name} ${ms.toFixed(2)}ms (${gf.toFixed(0)} GF)`)
        }
        lines.push(row.join('  |  '))
        for (const b of [s2.wB, s2.scB, s2.biB, s2.aB, s2.outB, s2.podB]) b.destroy()
      }
    }
    return lines
  })
  for (const l of out) console.log(l)
} finally {
  await stopHarness()
}
