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
import { parseSafetensorsHeader, planLayer, buildBuffer, bf16ToF16, bf16ToF32 }
  from '../../src/zero-tvm/mlx-weights.ts'
import { QWEN36_35B_A3B } from '../../src/compiler/model-spec.ts'

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
  return { read, close: () => Object.values(shards).forEach((s) => closeSync(s.fd)) }
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
