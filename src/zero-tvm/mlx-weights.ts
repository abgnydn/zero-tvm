/**
 * MLX SAFETENSORS — read a HuggingFace safetensors checkpoint quantised
 * MLX-affine, and repack it into the buffers the kernels bind.
 *
 * This is the format side of the Qwen3.6 port. Every existing spec loads the
 * MLC shard format (symmetric q4f16_1, group 32, no bias, one record per
 * fused tensor); MLX ships `weight` / `scales` / `biases` triples at group 64
 * with the projections UNFUSED, so the loader has to do three things the MLC
 * path never did:
 *
 *   1. Concatenate. c_attn is q_proj ++ k_proj ++ v_proj; the GDN input
 *      projection is in_proj_qkv ++ z ++ a ++ b. Both are row concatenations
 *      of row-major tensors, so they are byte concatenations — no transpose.
 *   2. Stack the shared expert. It becomes index E of every stacked expert
 *      tensor and its gate becomes row E of the router, which is again a byte
 *      append ([E,N,K] ++ [N,K] is [E+1,N,K]) and is what lets one dispatch
 *      cover all K+1 slots (see moe_router_topk.wgsl).
 *   3. Convert bf16. Scales, biases, norms and conv weights ship bfloat16;
 *      the kernels read f16 (and f32 for A_log/dt_bias). bf16 and f16 are
 *      DIFFERENT layouts — 8 exponent bits against 5 — so this is a
 *      conversion, not a reinterpret, and it can overflow. `bf16ToF16` counts
 *      overflows rather than quietly writing Infinity.
 *
 * Kept dependency-free and erasable-TS so tests/kernels/*.mjs can import it
 * under Node type stripping, the same constraint model-spec.ts carries.
 */

import type { ModelSpec } from '../compiler/model-spec.ts'

// ============================================================
// safetensors container
// ============================================================

export interface TensorInfo {
  dtype: string
  shape: number[]
  /** Byte range WITHIN the shard's data section (i.e. after the header). */
  begin: number
  end: number
}

export interface SafetensorsHeader {
  tensors: Record<string, TensorInfo>
  /** Absolute byte offset where the data section starts in the file. */
  dataStart: number
}

/**
 * Parse a safetensors header. Pass at least the first 8 bytes to learn the
 * header length, then the first `8 + len` bytes to parse it — a checkpoint
 * shard is gigabytes and the header is kilobytes, so a ranged read is the
 * point.
 */
export function parseSafetensorsHeader(head: ArrayBuffer): SafetensorsHeader {
  if (head.byteLength < 8) throw new Error('safetensors: need at least 8 bytes for the header length')
  const view = new DataView(head)
  const lo = view.getUint32(0, true)
  const hi = view.getUint32(4, true)
  if (hi !== 0) throw new Error('safetensors: header longer than 4 GB')
  if (head.byteLength < 8 + lo) {
    throw new Error(`safetensors: header is ${lo} bytes, only ${head.byteLength - 8} supplied`)
  }
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(head, 8, lo)))
  const tensors: Record<string, TensorInfo> = {}
  for (const [name, meta] of Object.entries(json as Record<string, unknown>)) {
    if (name === '__metadata__') continue
    const m = meta as { dtype: string; shape: number[]; data_offsets: [number, number] }
    tensors[name] = { dtype: m.dtype, shape: m.shape, begin: m.data_offsets[0], end: m.data_offsets[1] }
  }
  return { tensors, dataStart: 8 + lo }
}

// ============================================================
// bfloat16
// ============================================================

/**
 * bf16 → f16, via the f32 bit pattern bf16 already is.
 *
 * bf16 is [s|e:8|m:7] biased 127 and f16 is [s|e:5|m:10] biased 15, so the
 * exponent rebiases by 112. f16 has MORE mantissa bits, so a NORMAL value
 * converts exactly — but the ranges do not nest, and both ends matter:
 *
 *   - bf16 reaches 3e38 where f16 stops at 65504. Overflows are counted and
 *     clamped to f16 max rather than written as Infinity: an inf scale turns
 *     into NaN logits several thousand dispatches later with nothing pointing
 *     back here. Callers should treat a nonzero count as a hard error.
 *   - f16 SUBNORMALS run from 6.0e-8 to 6.1e-5, and quantization scales land
 *     in that band routinely. Flushing them to zero silently zeroes weights;
 *     it cost this module its first byte-for-byte comparison against numpy.
 *     Subnormals are rounded to nearest even, as numpy's astype does.
 *
 * bf16 below ~1e-38 (exp field 0) is far under f16's smallest subnormal, so
 * that really is zero to the kernel and is flushed without counting.
 */
export function bf16ToF16(src: Uint16Array): { out: Uint16Array; overflow: number } {
  const out = new Uint16Array(src.length)
  let overflow = 0
  for (let i = 0; i < src.length; i++) {
    const b = src[i]
    const sign = b & 0x8000
    const rawExp = (b >> 7) & 0xff
    // bf16 IS the top half of an f32, so its f32 mantissa is the 7 bits << 16.
    const mant0 = (b & 0x7f) << 16
    if (rawExp === 0xff) { out[i] = sign | 0x7c00 | (mant0 ? 0x200 : 0); continue }
    const exp = rawExp - 112
    if (exp >= 0x1f) { out[i] = sign | 0x7bff; overflow++; continue }
    if (exp <= 0) {
      if (exp < -10) { out[i] = sign; continue }        // includes rawExp == 0
      const mant = mant0 | 0x800000
      const sh = 14 - exp
      let h = mant >>> sh
      const rem = mant & ((1 << sh) - 1)
      const half = 1 << (sh - 1)
      if (rem > half || (rem === half && (h & 1))) h++   // round to nearest even
      out[i] = sign | h
      continue
    }
    // Normal: the low 13 bits of the f32 mantissa are zero here, so no rounding.
    out[i] = sign | (exp << 10) | (mant0 >>> 13)
  }
  return { out, overflow }
}

/** bf16 → f32: bf16 IS the top half of an f32, so this is a shift. */
export function bf16ToF32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length)
  const u32 = new Uint32Array(out.buffer)
  for (let i = 0; i < src.length; i++) u32[i] = src[i] << 16
  return out
}

// ============================================================
// layer plan
// ============================================================

/** How a source record's bytes reach the GPU. */
export type Convert = 'raw' | 'bf16->f16' | 'bf16->f32'

export interface BufferPart {
  record: string      // safetensors tensor name
  convert: Convert
}

/** One GPU buffer, built by concatenating `parts` in order. */
export interface BufferPlan {
  /** Stable id the engine binds by. */
  name: string
  parts: BufferPart[]
}

/**
 * Every buffer one Qwen3.6 decoder layer needs, and which records build it.
 *
 * `prefix` is everything before `layers.N.` — `spec.mlxPrefix + 'model.'` by
 * default: 'model.' for a text-only checkpoint, 'language_model.model.' for
 * Qwen3.6, whose repo is multimodal and nests the text tower under a
 * vision-capable root.
 *
 * Attention and GDN layers are different shapes, so the plan branches on
 * `spec.layerKinds[L]` exactly as the engine does.
 */
export function planLayer(spec: ModelSpec, L: number, prefix?: string): BufferPlan[] {
  const p = `${prefix ?? `${spec.mlxPrefix ?? ''}model.`}layers.${L}`
  const q = (base: string) => [
    { record: `${base}.weight`, convert: 'raw' as Convert },
    { record: `${base}.scales`, convert: 'bf16->f16' as Convert },
    { record: `${base}.biases`, convert: 'bf16->f16' as Convert },
  ]
  /** One quantized buffer trio per logical tensor: weights, scales, biases —
   *  each the concatenation of the same component of every part. */
  const trio = (name: string, bases: string[]): BufferPlan[] => {
    const parts = bases.map(q)
    return [
      { name: `${name}_w`, parts: parts.map((t) => t[0]) },
      { name: `${name}_s`, parts: parts.map((t) => t[1]) },
      { name: `${name}_b`, parts: parts.map((t) => t[2]) },
    ]
  }
  const plans: BufferPlan[] = [
    { name: 'norm1', parts: [{ record: `${p}.input_layernorm.weight`, convert: 'bf16->f16' }] },
    { name: 'norm2', parts: [{ record: `${p}.post_attention_layernorm.weight`, convert: 'bf16->f16' }] },
  ]

  if (spec.layerKinds[L] === 'attn') {
    // c_attn: q_proj already emits per-head [Q|gate], which is the interleave
    // gated_qkv_split expects, so K and V simply follow it.
    plans.push(...trio('c_attn', [`${p}.self_attn.q_proj`, `${p}.self_attn.k_proj`, `${p}.self_attn.v_proj`]))
    plans.push(...trio('o_proj', [`${p}.self_attn.o_proj`]))
    if (spec.qkNorm) {
      plans.push({ name: 'q_norm', parts: [{ record: `${p}.self_attn.q_norm.weight`, convert: 'bf16->f16' }] })
      plans.push({ name: 'k_norm', parts: [{ record: `${p}.self_attn.k_norm.weight`, convert: 'bf16->f16' }] })
    }
  } else {
    const g = `${p}.linear_attn`
    // One K=d matmul of gdnProjRows rows; the downstream kernels read z and
    // [a|b] out of the result at fixed offsets.
    plans.push(...trio('gdn_proj', [`${g}.in_proj_qkv`, `${g}.in_proj_z`, `${g}.in_proj_a`, `${g}.in_proj_b`]))
    plans.push(...trio('gdn_out', [`${g}.out_proj`]))
    plans.push({ name: 'gdn_conv', parts: [{ record: `${g}.conv1d.weight`, convert: 'bf16->f16' }] })
    plans.push({ name: 'gdn_norm', parts: [{ record: `${g}.norm.weight`, convert: 'bf16->f16' }] })
    // f32: HF notes that -exp(A_log) overflows to -inf in fp16.
    plans.push({ name: 'gdn_a_log', parts: [{ record: `${g}.A_log`, convert: 'bf16->f32' }] })
    plans.push({ name: 'gdn_dt_bias', parts: [{ record: `${g}.dt_bias`, convert: 'bf16->f32' }] })
  }

  if (spec.moe) {
    // When the checkpoint has a shared expert it is appended to every stacked
    // tensor, and its gate to the router, so nothing downstream treats it as a
    // special case. Without one (Mixtral / Qwen3-MoE style) the stacks simply
    // end at the routed experts — there is no index E to append, and
    // moe_router_topk stops emitting the slot that would have read it.
    const shared = spec.moe.sharedExpert ?? true
    for (const proj of ['gate_proj', 'up_proj', 'down_proj']) {
      plans.push(...trio(`moe_${proj}`, shared
        ? [`${p}.mlp.switch_mlp.${proj}`, `${p}.mlp.shared_expert.${proj}`]
        : [`${p}.mlp.switch_mlp.${proj}`]))
    }
    plans.push(...trio('router', shared
      ? [`${p}.mlp.gate`, `${p}.mlp.shared_expert_gate`]
      : [`${p}.mlp.gate`]))
  } else {
    plans.push(...trio('ffn', [`${p}.mlp.gate_proj`, `${p}.mlp.up_proj`]))
    plans.push(...trio('ffn_down', [`${p}.mlp.down_proj`]))
  }
  return plans
}

/**
 * The buffers that are NOT per-layer: the embedding table, the untied lm_head,
 * and the final norm.
 *
 * `prefix` stops one level higher than planLayer's (default `spec.mlxPrefix`).
 * The text tower lives under `<prefix>model.` but lm_head is `<prefix>lm_head`
 * — a level up, because it is not part of the decoder stack. Getting that
 * wrong is a missing record, so it fails loudly; it is called out because it
 * looks like a typo.
 *
 * Tied-embeddings checkpoints ship NO lm_head records at all (mlx_lm reuses
 * the quantized embedding matrix), so the lm_head trio is skipped and the
 * loader aliases the embedding buffers instead — see assembleMlx.
 *
 * Bias record names come from `spec.paramNaming.biasFor` rather than a suffix
 * spelled here a second time — that field exists precisely so one place knows
 * how MLX names companions.
 */
export function planGlobal(
  spec: ModelSpec,
  prefix?: string,
  /** Pipeline stage bounds. The embedding belongs to the stage that starts the
   *  model and the lm_head / final norm to the one that ends it; a middle
   *  stage loads neither, which is most of what makes a stage small. */
  range?: { start: number; end: number },
): BufferPlan[] {
  const pre = prefix ?? spec.mlxPrefix ?? ''
  const first = (range?.start ?? 0) === 0
  const last = (range?.end ?? spec.layers) === spec.layers
  const bias = spec.paramNaming.biasFor
  if (!bias) throw new Error(`${spec.id}: planGlobal needs paramNaming.biasFor (affine checkpoints only)`)
  const q = (name: string, base: string): BufferPlan[] => [
    { name: `${name}_w`, parts: [{ record: `${base}.weight`, convert: 'raw' }] },
    { name: `${name}_s`, parts: [{ record: `${base}.scales`, convert: 'bf16->f16' }] },
    { name: `${name}_b`, parts: [{ record: bias(`${base}.weight`), convert: 'bf16->f16' }] },
  ]
  return [
    // A tied lm_head IS the embedding table, so the last stage needs it even
    // when it does not start the model.
    ...(first || (last && spec.tiedEmbeddings) ? q('embed', `${pre}model.embed_tokens`) : []),
    ...(spec.tiedEmbeddings || !last ? [] : q('lm_head', `${pre}lm_head`)),  // NOT under model. — see above
    ...(last ? [{ name: 'final_norm', parts: [{ record: `${pre}model.norm.weight`, convert: 'bf16->f16' as Convert }] }] : []),
  ]
}

/** Bytes a plan produces on the GPU, given each record's on-disk byte length.
 *
 *  NOT the sum of the source ranges: `bf16->f32` doubles. A_log and dt_bias are
 *  64 B in the file and 128 B on the GPU, and a half-length A_log buffer poisons
 *  every GDN decay gate without crashing anything.
 */
export function planBytes(plan: BufferPlan, sourceBytes: (record: string) => number): number {
  let n = 0
  for (const part of plan.parts) {
    const b = sourceBytes(part.record)
    n += part.convert === 'bf16->f32' ? b * 2 : b
  }
  return n
}

/**
 * Build one plan's bytes from already-read record data.
 *
 * `readRecord` returns the raw little-endian bytes of a tensor. Kept as a
 * callback so the same repacking runs over a local file in tests and over
 * ranged fetches in the browser — the format logic is the part worth sharing,
 * the I/O is not.
 *
 * `dtypeOf` (safetensors header dtype per record) makes the conversions
 * dtype-AWARE: the plans' 'bf16->f16' means "the kernels read f16", not "the
 * file is bf16" — Qwen MLX checkpoints ship scales/norms as BF16, but
 * Llama-3.2's ships them F16 already, and running the bf16 shift over f16
 * bits collapses every scale to ±0 (f16's exponent lands in bf16's ~1e-21
 * range): zero logits, no error anywhere. Absent = assume BF16 (the historical
 * behavior, kept for callers without header access).
 */
export function buildBuffer(
  plan: BufferPlan,
  readRecord: (name: string) => Uint8Array,
  dtypeOf?: (name: string) => string,
): { data: Uint8Array<ArrayBuffer>; overflow: number } {
  const pieces: Uint8Array[] = []
  let overflow = 0
  for (const part of plan.parts) {
    const raw = readRecord(part.record)
    const dtype = dtypeOf?.(part.record) ?? 'BF16'
    if (part.convert === 'raw') {
      pieces.push(raw)
    } else if (part.convert === 'bf16->f16' && dtype === 'F16') {
      pieces.push(raw)   // already what the kernels read — pass through
    } else if (dtype !== 'BF16') {
      // f32-target plans (A_log/dt_bias) double their byte count in
      // planBytes; a non-BF16 source would need that audit too. No shipped
      // checkpoint hits this — fail loudly rather than resize silently.
      throw new Error(`${plan.name}: ${part.record} is ${dtype}, expected BF16 for ${part.convert}`)
    } else {
      const src = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2)
      if (part.convert === 'bf16->f16') {
        const r = bf16ToF16(src)
        overflow += r.overflow
        pieces.push(new Uint8Array(r.out.buffer))
      } else {
        pieces.push(new Uint8Array(bf16ToF32(src).buffer))
      }
    }
  }
  const total = pieces.reduce((a, b) => a + b.byteLength, 0)
  const data = new Uint8Array(new ArrayBuffer(total))
  let o = 0
  for (const piece of pieces) { data.set(piece, o); o += piece.byteLength }
  // A clamped scale is not a rounding error, it is a wrong weight, and it
  // surfaces as NaN logits thousands of dispatches downstream. This checkpoint
  // produces none; if a future one does, the loader should stop, not warn.
  if (overflow > 0) {
    throw new Error(
      `${plan.name}: ${overflow} bf16 value(s) exceed f16 range and were clamped — `
      + 'these weights cannot be represented at f16 and the model would run wrong',
    )
  }
  return { data, overflow }
}
