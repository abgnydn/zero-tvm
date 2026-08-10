/**
 * MLX REPACK — does src/zero-tvm/mlx-weights.ts turn the real checkpoint into
 * the bytes the kernels expect?
 *
 * The reference bundles under .weights-local/kernel-refs are built by
 * scripts/make-kernel-ref.py, whose layout is already validated against
 * mlx_lm's own modules (real-weights.mjs). But Python cannot run in a browser,
 * so the repacking — concatenating projections, appending the shared expert,
 * converting bf16 — has to exist a second time in TypeScript, and a second
 * implementation is a second chance to get it wrong.
 *
 * So this compares the TS repacker's output to the Python bundle BYTE FOR BYTE,
 * straight off the safetensors shards. No GPU, no tolerance: either the two
 * agree exactly or one of them is wrong.
 *
 *   node tests/kernels/mlx-repack.mjs
 */
import { existsSync, openSync, readSync, closeSync, readFileSync } from 'node:fs'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseSafetensorsHeader, planLayer, planGlobal, planBytes, buildBuffer, bf16ToF16, bf16ToF32 }
  from '../../src/zero-tvm/mlx-weights.ts'
import { QWEN36_35B_A3B, DEEPSEEK_V2_LITE } from '../../src/compiler/model-spec.ts'
import { openMlxCheckpoint, planModel, planKey, buildPlan, modelBytes }
  from '../../src/zero-tvm/weight-loader-mlx.ts'
// The test side's own f16, deliberately: decoding the loader's output with the
// loader's own decoder would only prove it is self-consistent.
import { f16BitsToF32, toF16 } from './half.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const CKPT = join(ROOT, '.weights-local/Qwen3.6-35B-A3B-MLX-4bit')
const REFS = join(ROOT, '.weights-local/kernel-refs')

/** Read shard headers once; return a record -> reader over the whole checkpoint. */
function openCheckpoint(dir) {
  const map = JSON.parse(readFileSync(join(dir, 'model.safetensors.index.json'), 'utf8')).weight_map
  const shards = {}
  for (const file of new Set(Object.values(map))) {
    const fd = openSync(join(dir, file), 'r')
    const len = Buffer.alloc(8)
    readSync(fd, len, 0, 8, 0)
    const headerLen = Number(len.readBigUInt64LE(0))
    const head = Buffer.alloc(8 + headerLen)
    readSync(fd, head, 0, head.length, 0)
    shards[file] = { fd, ...parseSafetensorsHeader(head.buffer.slice(head.byteOffset, head.byteOffset + head.length)) }
  }
  const read = (name) => {
    const file = map[name]
    if (!file) throw new Error(`record not in checkpoint: ${name}`)
    const s = shards[file]
    const info = s.tensors[name]
    const buf = Buffer.alloc(info.end - info.begin)
    readSync(s.fd, buf, 0, buf.length, s.dataStart + info.begin)
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.length)
  }
  const info = (name) => {
    const file = map[name]
    if (!file) throw new Error(`record not in checkpoint: ${name}`)
    return shards[file].tensors[name]
  }
  return { read, info, close: () => Object.values(shards).forEach((s) => closeSync(s.fd)) }
}

const same = (a, b) => {
  if (a.length !== b.length) return `length ${a.length} vs ${b.length}`
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return `first differs at byte ${i}: ${a[i]} vs ${b[i]}`
  return null
}

const results = []
const check = (name, detail, pass) => results.push({ name, detail, pass })

// ── bf16 conversion, against numpy's answers ────────────────────────────────
// Bit patterns are DERIVED from the values, not hand-written: a hand-written
// hex constant is one more thing that can be wrong in the same direction as
// the code it is testing.
{
  const toBf16 = (v) => {          // round-to-nearest-even f32 -> bf16
    const f = new Float32Array([v]), u = new Uint32Array(f.buffer)
    const x = u[0], lo = x & 0xffff
    let hi = x >>> 16
    if (lo > 0x8000 || (lo === 0x8000 && (hi & 1))) hi++
    return hi & 0xffff
  }
  const vals = [
    1, -2, 65280,        // exact in both
    1e38,                // overflows f16
    3e-5, 1e-6, 1e-7,    // f16 SUBNORMAL band (6.0e-8 .. 6.1e-5)
    1e-40, 0, -0,        // below f16 entirely / zeros
  ]
  const bf = new Uint16Array(vals.map(toBf16))
  const f32 = bf16ToF32(bf)
  const bad32 = vals.findIndex((v, i) => {
    const exact = new Float32Array([v]); const u = new Uint32Array(exact.buffer)
    return f32[i] !== new Float32Array(new Uint32Array([bf[i] << 16]).buffer)[0] || (v === 0 && u[0] !== 0 && false)
  })
  const { out, overflow } = bf16ToF16(bf)
  // Independent decode: the f16 bits must read back as the bf16 value, within
  // f16's own resolution — and the subnormals must NOT be zero.
  const dec = (h) => {
    const s2 = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff
    if (e === 0) return s2 * m * 2 ** -24
    if (e === 31) return s2 * (m ? NaN : Infinity)
    return s2 * (1 + m / 1024) * 2 ** (e - 15)
  }
  // Tolerance is HALF AN ULP, not relative. f16 subnormals are spaced a fixed
  // 5.96e-8 apart, so 1e-7 correctly becomes 1.19e-7 — a 19% relative error
  // that is nonetheless the nearest representable value. A relative tolerance
  // here fails the conversion for being right.
  const ULP = (v) => {
    const a = Math.abs(v)
    if (a < 6.103515625e-5) return 2 ** -24                 // subnormal spacing
    return 2 ** (Math.floor(Math.log2(a)) - 10)
  }
  const bad16 = []
  for (let i = 0; i < vals.length; i++) {
    const want = f32[i]
    const got = dec(out[i])
    if (i === 3) { if (got !== 65504) bad16.push(`1e38 -> ${got}, expected clamp to 65504`); continue }
    if (Math.abs(want) < 2 ** -25) { if (got !== 0) bad16.push(`${want} -> ${got}, expected 0`); continue }
    if (Math.abs(got - want) > ULP(want) / 2 + Number.EPSILON) {
      bad16.push(`${want} -> ${got} (more than half an ulp, ${ULP(want)})`)
    }
  }
  for (const b of bad16) console.error(`      ${b}`)
  check('bf16 convert', `f32 ${bad32 < 0 ? 'exact' : `WRONG at ${bad32}`}, `
    + `f16 ${bad16.length ? `${bad16.length} WRONG` : 'exact incl. subnormals'}, `
    + `${overflow} overflow clamped`, bad32 < 0 && bad16.length === 0 && overflow === 1)
}

// ── DeepSeek-V2 layer 0: the MLA loader prep, against the numpy bundle ───────
// The whole verification of the loader step, and it needs neither a GPU nor the
// 9 GB checkpoint. The kernel-ref bundle is a FLAT PER-TENSOR DIRECTORY and
// buildBuffer takes readRecord/dtypeOf as callbacks, so the real loader runs
// over it unmodified — this is not a reimplementation of the prep.
//
// kv_b is checked BYTE-EXACTLY, zero tolerance: make-dsv2-layer-ref.py rounds
// through float16 before widening for the dump, so every value in wk_t.bin /
// wv.bin is exactly f16-representable and any difference at all is a real one.
// It is also the ONLY check that can catch a wrong transpose — an untransposed
// Wk is still a well-formed [heads, kvLora, nope]-sized matrix that mla_proj
// runs to completion on and that every tolerance-based check downstream
// accepts.
{
  const dir = join(REFS, 'dsv2layer0')
  if (!existsSync(join(dir, 'meta.json'))) {
    check('dsv2 loader prep', `no bundle at ${dir} — run scripts/make-dsv2-layer-ref.py`, null)
  } else {
    const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8'))
    const S = DEEPSEEK_V2_LITE
    const { heads: HEADS, nope: NOPE, rope: R, v: V, kv_lora: KVL, d: D, query_at: QI } = meta
    const bad = []
    const rec = (name) => {
      const r = meta.tensors[name]
      if (!r) throw new Error(`record not in bundle: ${name}`)
      return r
    }
    // Copies, not views: readFileSync returns a POOLED Buffer for small files,
    // so its byteOffset is not necessarily a multiple of 4 and a Float32Array
    // view over it can throw.
    const bytesOf = (file) => {
      const b = readFileSync(join(dir, file))
      return new Uint8Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length))
    }
    const readRecord = (name) => bytesOf(rec(name).file)
    const dtypeOf = (name) => rec(name).dtype
    const refF32 = (name) => new Float32Array(bytesOf(`${name}.bin`).buffer)
    const plans = Object.fromEntries(planLayer(S, 0).map((p) => [p.name, p]))
    const built = (name) => buildBuffer(plans[name], readRecord, dtypeOf).data
    const relErr = (got, ref) => {
      let scale = 0
      for (let i = 0; i < ref.length; i++) scale = Math.max(scale, Math.abs(ref[i]))
      let m = 0
      for (let i = 0; i < ref.length; i++) m = Math.max(m, Math.abs(got[i] - ref[i]) / scale)
      return m
    }

    // ── kv_b, byte-exact ────────────────────────────────────────────────────
    const kvb = built('mla_kvb')
    const got = new Uint16Array(kvb.buffer, kvb.byteOffset, kvb.byteLength / 2)
    const wkT = refF32('wk_t')     // [heads, kvLora, nope] — Wk ALREADY transposed
    const wV = refF32('wv')        // [heads, vHeadDim, kvLora]
    let diffs = 0
    let firstDiff = ''
    for (let i = 0; i < got.length; i++) {
      const want = i < wkT.length ? wkT[i] : wV[i - wkT.length]
      const have = f16BitsToF32(got[i])
      if (have !== want) {
        diffs++
        if (!firstDiff) {
          firstDiff = i < wkT.length
            ? `K^T[h=${(i / (KVL * NOPE)) | 0}][${((i / NOPE) | 0) % KVL}][${i % NOPE}] ${have} vs ${want}`
            : `V[${i - wkT.length}] ${have} vs ${want}`
        }
      }
    }
    const wantBytes = HEADS * (NOPE + V) * KVL * 2
    if (kvb.byteLength !== wantBytes) bad.push(`mla_kvb is ${kvb.byteLength} B, expected ${wantBytes} B`)
    if (diffs) bad.push(`mla_kvb: ${diffs} of ${got.length} values differ (first ${firstDiff})`)
    const pb = planBytes(plans.mla_kvb, (n) => rec(n).bytes)
    if (pb !== 4194304) bad.push(`planBytes(mla_kvb) = ${pb}, expected 4194304 — residency would be under-reported`)

    // ── the permutation, check (a): whole rows, all three components ────────
    // Byte-level so it also covers scales and biases: permuting the weights and
    // leaving the scales behind is a real failure mode, and dequantizing first
    // would hide which of the three moved.
    const deint = (j) => (j < R / 2 ? 2 * j : 2 * (j - R / 2) + 1)
    const qSrc = (j) => { const i = j % (NOPE + R); return i < NOPE ? j : j - i + NOPE + deint(i - NOPE) }
    const kvaSrc = (j) => (j < KVL ? j : KVL + deint(j - KVL))
    const rowsMatch = (name, rows, src) => {
      const b2 = built(name)
      const raw = readRecord(plans[name].parts[0].record)
      if (b2.byteLength !== raw.byteLength) { bad.push(`${name}: ${b2.byteLength} B built vs ${raw.byteLength} B source`); return }
      const rb = b2.byteLength / rows
      for (let j = 0; j < rows; j++) {
        const s = src(j)
        for (let o = 0; o < rb; o++) {
          if (b2[j * rb + o] !== raw[s * rb + o]) {
            bad.push(`${name}: row ${j} is not source row ${s} (differs at byte ${o} of ${rb})`)
            return
          }
        }
      }
    }
    for (const c of ['w', 's', 'b']) {
      rowsMatch(`mla_q_${c}`, S.mlaQProjRows, qSrc)
      rowsMatch(`mla_kva_${c}`, S.mlaKvaRows, kvaSrc)
    }

    // ── the permutation, check (b): through the bundle's own evidence ───────
    // (a) is circular — it checks the loader against the same map the loader
    // used, and passes just as happily on the INVERSE map. ref_qperm /
    // ref_kvaperm are the only independent evidence in the repo: the reference
    // script refuses to write them unless permuting the rows reproduces
    // DeepSeek's interleaved rotation to 1e-6.
    const h1 = refF32('ref_h1').subarray(QI * D, (QI + 1) * D)
    const affineMatvec = (w, s, b, rows, K) => {
      const wpr = K / 8, gpr = K / 64
      const wd = new DataView(w.buffer, w.byteOffset, w.byteLength)
      const sd = new DataView(s.buffer, s.byteOffset, s.byteLength)
      const bd = new DataView(b.buffer, b.byteOffset, b.byteLength)
      const out = new Float32Array(rows)
      for (let r = 0; r < rows; r++) {
        let acc = 0
        for (let g = 0; g < gpr; g++) {
          const sc = f16BitsToF32(sd.getUint16((r * gpr + g) * 2, true))
          const bi = f16BitsToF32(bd.getUint16((r * gpr + g) * 2, true))
          for (let c = g * 64; c < (g + 1) * 64; c++) {
            const word = wd.getUint32((r * wpr + (c >> 3)) * 4, true)
            acc += toF16(((word >>> ((c & 7) * 4)) & 15) * sc + bi) * h1[c]
          }
        }
        out[r] = acc
      }
      return out
    }
    const errQ = relErr(affineMatvec(built('mla_q_w'), built('mla_q_s'), built('mla_q_b'), S.mlaQProjRows, D),
                        refF32('ref_qperm'))
    const errKva = relErr(affineMatvec(built('mla_kva_w'), built('mla_kva_s'), built('mla_kva_b'), S.mlaKvaRows, D),
                          refF32('ref_kvaperm'))
    if (!(errQ < 1e-3)) bad.push(`permuted q_proj vs ref_qperm: ${errQ.toExponential(2)}`)
    if (!(errKva < 1e-3)) bad.push(`permuted kv_a_proj vs ref_kvaperm: ${errKva.toExponential(2)}`)

    // ── kv_a_layernorm: an F16 record reaching the kernels unchanged ────────
    // It is planned 'bf16->f16' ("the kernels read f16"), NOT 'raw', and only
    // buildBuffer's dtype guard turns that into a pass-through for this
    // checkpoint. Planning it 'raw' produces the same bytes today and silently
    // stops converting on any sibling that ships bf16.
    const gamma = built('mla_kva_norm')
    const gammaSrc = readRecord(plans.mla_kva_norm.parts[0].record)
    if (gamma.byteLength !== KVL * 2) bad.push(`mla_kva_norm is ${gamma.byteLength} B, expected ${KVL * 2} B`)
    else if (same(gamma, gammaSrc)) bad.push(`mla_kva_norm: ${same(gamma, gammaSrc)}`)

    // ── the OPFS key: a content hash for op-bearing plans ONLY ──────────────
    const kvbKey = planKey(plans.mla_kvb, 0)
    if (!/^l0\.mla_kvb\.[0-9a-f]{4}$/.test(kvbKey)) bad.push(`planKey(mla_kvb) = ${kvbKey}, expected an op hash suffix`)
    if (planKey(plans.norm1, 0) !== 'l0.norm1') {
      bad.push(`planKey(norm1) = ${planKey(plans.norm1, 0)} — every cached buffer on disk would be invalidated`)
    }

    for (const b of bad) console.error(`      ${b}`)
    check('dsv2 loader prep',
      `kv_b ${diffs === 0 ? 'BYTE-EXACT' : `${diffs} VALUES DIFFER`} vs wk_t‖wv `
      + `(${(kvb.byteLength / 1048576).toFixed(0)} MiB f16 from ${(1179648 / 1048576).toFixed(2)} MiB of int4, `
      + `planBytes ${pb}); pe rows permuted in weight+scales+biases; `
      + `q ${errQ.toExponential(1)} / kv_a ${errKva.toExponential(1)} vs ref_qperm/ref_kvaperm at token ${QI}; `
      + `key ${kvbKey}`,
      bad.length === 0)
  }
}

if (!existsSync(CKPT)) {
  console.log(`SKIP  mlx repack — no checkpoint at ${CKPT}`)
  console.log('      huggingface-cli download lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit '
    + `--local-dir ${CKPT}`)
} else {
  const ck = openCheckpoint(CKPT)
  const S = QWEN36_35B_A3B

  // ── layer 0 (GDN) and layer 3 (attention) against the validated bundles ────
  const cases = [
    { layer: 0, bundle: 'qwen36gdn', map: {
      gdn_proj_w: 'qkv_w_u32|z_w_u32|a_w_u32|b_w_u32',
      gdn_proj_s: 'qkv_s_f16|z_s_f16|a_s_f16|b_s_f16',
      gdn_proj_b: 'qkv_b_f16|z_b_f16|a_b_f16|b_b_f16',
      gdn_out_w: 'out_w_u32', gdn_out_s: 'out_s_f16', gdn_out_b: 'out_b_f16',
      gdn_conv: 'conv1d_f16', gdn_norm: 'gnorm_gamma_f16',
      gdn_a_log: 'A_log_f32', gdn_dt_bias: 'dt_bias_f32',
      norm1: 'norm1_gamma_f16',
    } },
    { layer: 3, bundle: 'qwen36attn', map: {
      c_attn_w: 'q_w_u32|k_w_u32|v_w_u32',
      c_attn_s: 'q_s_f16|k_s_f16|v_s_f16',
      c_attn_b: 'q_b_f16|k_b_f16|v_b_f16',
      o_proj_w: 'o_w_u32', o_proj_s: 'o_s_f16', o_proj_b: 'o_b_f16',
      q_norm: 'q_norm_f16', k_norm: 'k_norm_f16',
      norm1: 'norm1_gamma_f16',
    } },
    { layer: 0, bundle: 'qwen36moe_block', map: {
      moe_gate_proj_w: 'exp_gate_proj_w_u32', moe_gate_proj_s: 'exp_gate_proj_s_f16',
      moe_gate_proj_b: 'exp_gate_proj_b_f16',
      moe_up_proj_w: 'exp_up_proj_w_u32', moe_up_proj_s: 'exp_up_proj_s_f16',
      moe_up_proj_b: 'exp_up_proj_b_f16',
      moe_down_proj_w: 'exp_down_proj_w_u32', moe_down_proj_s: 'exp_down_proj_s_f16',
      moe_down_proj_b: 'exp_down_proj_b_f16',
      router_w: 'router_w_u32', router_s: 'router_s_f16', router_b: 'router_b_f16',
    } },
  ]

  for (const c of cases) {
    const dir = join(REFS, c.bundle)
    if (!existsSync(dir)) { check(`repack L${c.layer} (${c.bundle})`, 'no bundle — run make-kernel-ref.py', null); continue }
    const plans = Object.fromEntries(planLayer(S, c.layer).map((p) => [p.name, p]))
    const bad = []
    let totalOverflow = 0
    let checked = 0
    for (const [planName, refNames] of Object.entries(c.map)) {
      const plan = plans[planName]
      if (!plan) { bad.push(`${planName}: not in plan`); continue }
      const { data, overflow } = buildBuffer(plan, ck.read)
      totalOverflow += overflow
      // The bundle stores each part as its own file, so concatenate the same way.
      const parts = refNames.split('|').map((n) => readFileSync(join(dir, `${n}.bin`)))
      const ref = new Uint8Array(parts.reduce((a, b) => a + b.length, 0))
      let o = 0
      for (const part of parts) { ref.set(part, o); o += part.length }
      const diff = same(data, ref)
      if (diff) bad.push(`${planName}: ${diff}`)
      checked++
    }
    for (const b of bad) console.error(`      ${b}`)
    check(`repack L${c.layer} (${c.bundle})`,
      `${checked} buffers byte-identical to the mlx-validated bundle`
      + (totalOverflow ? `, ${totalOverflow} bf16 OVERFLOWS` : ''),
      bad.length === 0 && totalOverflow === 0)
  }

  // ── planGlobal: embed / lm_head / final_norm ──────────────────────────────
  {
    const dir = join(REFS, 'qwen36embed')
    const plans = Object.fromEntries(planGlobal(S).map((p) => [p.name, p]))
    const bad = []
    if (existsSync(dir)) {
      for (const [planName, refName] of Object.entries({
        embed_w: 'weights_u32', embed_s: 'scales_f16', embed_b: 'bias_f16',
      })) {
        const { data } = buildBuffer(plans[planName], ck.read)
        const diff = same(data, new Uint8Array(readFileSync(join(dir, `${refName}.bin`))))
        if (diff) bad.push(`${planName}: ${diff}`)
      }
    }
    // lm_head and final_norm have no bundle; assert they RESOLVE, which is the
    // thing that is easy to get wrong — lm_head sits one prefix level above the
    // decoder stack (language_model.lm_head, not language_model.model.lm_head).
    for (const n of ['lm_head_w', 'lm_head_s', 'lm_head_b', 'final_norm']) {
      try { buildBuffer(plans[n], ck.read) } catch (e) { bad.push(`${n}: ${e.message}`) }
    }
    for (const b of bad) console.error(`      ${b}`)
    check('planGlobal', `embed byte-identical, lm_head + final_norm resolve`, bad.length === 0)
  }

  // ── whole-model budget, straight from the plan ────────────────────────────
  // The residency question answered by the thing that will actually allocate,
  // rather than by an estimate: every plan for every layer, plus globals.
  {
    const src = (name) => { const s2 = ck.info(name); return s2.end - s2.begin }
    let total = 0, biggest = { name: '', bytes: 0 }, count = 0
    const tally = (plans) => {
      for (const p of plans) {
        const b = planBytes(p, src)
        total += b; count++
        if (b > biggest.bytes) biggest = { name: p.name, bytes: b }
      }
    }
    for (let L = 0; L < S.layers; L++) tally(planLayer(S, L))
    tally(planGlobal(S))
    // KV lives only on attention layers; the GDN recurrent state only on the rest.
    const attnLayers = S.layerKinds.filter((k) => k === 'attn').length
    const kv = attnLayers * S.maxPages * S.kvPageStride * 2
    const gdnState = (S.layers - attnLayers) * S.gdnVHeads * S.gdnStatePerHead * 4
    const GB = (n) => (n / 1e9).toFixed(2)
    const LIMIT = 134217728   // WebGPU's default maxStorageBufferBindingSize
    const overLimit = biggest.bytes > LIMIT
    check('model budget',
      `${count} weight buffers ${GB(total)} GB + KV ${GB(kv)} GB + GDN state ${GB(gdnState)} GB `
      + `= ${GB(total + kv + gdnState)} GB resident; largest buffer ${biggest.name} `
      + `${(biggest.bytes / 1048576).toFixed(1)} MiB`
      + (overLimit ? ` — OVER the 128 MiB WebGPU default, needs a raised requiredLimit` : ''),
      total + kv + gdnState < 21e9)
  }

  // ── loader replay: the exact ranges the browser will request ─────────────
  // The repack tests above hand buildBuffer whole records off local fs. The
  // browser cannot do that — a 5.3 GB shard is not one ArrayBuffer — so it
  // reads BYTE RANGES through weight-loader-mlx. Replaying that path over the
  // same files, against the same Python bundles, covers the I/O planning and
  // not just the layout.
  {
    const opened = openSync(join(CKPT, 'model.safetensors.index.json'), 'r')
    closeSync(opened)
    const fds = {}
    const src = {
      whole: async (file) => new Uint8Array(readFileSync(join(CKPT, file))),
      range: async (file, begin, end) => {
        fds[file] ??= openSync(join(CKPT, file), 'r')
        const buf = Buffer.alloc(end - begin)
        readSync(fds[file], buf, 0, buf.length, begin)
        return new Uint8Array(buf.buffer, buf.byteOffset, buf.length)
      },
    }
    const bad = []
    // Count header reads SEPARATELY from plan reads — the interesting number is
    // what it costs to learn the layout of 20 GB, and mixing the two hid it.
    let headerBytes = 0
    const headerSrc = { ...src, range: async (f, b, e) => { headerBytes += e - b; return src.range(f, b, e) } }
    const { locate, shards } = await openMlxCheckpoint(headerSrc)
    let planBytesRead = 0
    const counting = { ...src, range: async (f, b, e) => { planBytesRead += e - b; return src.range(f, b, e) } }

    // Same three bundles, now via ranges instead of whole records.
    for (const [layer, bundle, map] of [
      [0, 'qwen36gdn', { gdn_proj_w: 'qkv_w_u32|z_w_u32|a_w_u32|b_w_u32', gdn_a_log: 'A_log_f32' }],
      [3, 'qwen36attn', { c_attn_w: 'q_w_u32|k_w_u32|v_w_u32', q_norm: 'q_norm_f16' }],
      [0, 'qwen36moe_block', { moe_gate_proj_w: 'exp_gate_proj_w_u32', router_s: 'router_s_f16' }],
    ]) {
      const dir = join(REFS, bundle)
      if (!existsSync(dir)) continue
      const plans = Object.fromEntries(planLayer(S, layer).map((p) => [p.name, p]))
      for (const [planName, refNames] of Object.entries(map)) {
        const data = await buildPlan(plans[planName], locate, counting)
        const parts = refNames.split('|').map((n) => readFileSync(join(dir, `${n}.bin`)))
        const ref = new Uint8Array(parts.reduce((a, b) => a + b.length, 0))
        let o = 0
        for (const part of parts) { ref.set(part, o); o += part.length }
        const diff = same(data, ref)
        if (diff) bad.push(`L${layer} ${planName}: ${diff}`)
      }
    }

    // Every plan must RESOLVE — a typo'd record name is the whole failure mode
    // of a naming-driven loader, and it costs one index lookup to rule out.
    const all = planModel(S)
    const keys = new Set()
    for (const { plan, layer } of all) {
      const k = planKey(plan, layer)
      if (keys.has(k)) bad.push(`duplicate plan key ${k}`)
      keys.add(k)
      for (const part of plan.parts) {
        try { locate(part.record) } catch (e) { bad.push(e.message) }
      }
    }
    const total = modelBytes(S, locate)
    for (const b of bad.slice(0, 8)) console.error(`      ${b}`)
    check('loader replay',
      `${all.length} plans over ${shards.length} shards resolve, byte-identical via ranges; `
      + `${(headerBytes / 1024).toFixed(0)} KB of headers maps all ${(total / 1e9).toFixed(2)} GB `
      + `(${(planBytesRead / 1e6).toFixed(0)} MB read for the 6 compared plans)`,
      bad.length === 0)
    Object.values(fds).forEach(closeSync)
  }

  ck.close()
}

let failed = 0
for (const r of results) {
  const tag = r.pass === null ? 'SKIP' : r.pass ? 'PASS' : 'FAIL'
  if (r.pass === false) failed++
  console.log(`${tag}  ${r.name.padEnd(28)} ${r.detail}`)
}
console.log(`\n${failed === 0 ? 'mlx repack correct' : `${failed} repack check(s) FAILED`}`)
process.exit(failed === 0 ? 0 : 1)
