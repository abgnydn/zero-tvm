// CPU reference for the Qwen3.5 GatedDeltaNet (GDN) block and the gated
// attention block — plain JS, formula-by-formula transcriptions of the ground
// truth, cited per function:
//
//   [HF]  huggingface/transformers src/transformers/models/qwen3_5/
//         modeling_qwen3_5.py (Qwen3_5GatedDeltaNet, l2norm,
//         torch_recurrent_gated_delta_rule, Qwen3_5RMSNormGated,
//         Qwen3_5Attention, apply_rotary_pos_emb)
//   [MLC] mlc-ai/mlc-llm python/mlc_llm/model/qwen35/qwen35_model.py
//         (create_gated_delta_net_func, _causal_conv1d_with_state,
//         _compute_gate_beta, Qwen35Attention) — the implementation the
//         mlc-ai/Qwen3.5-4B-q4f16_1-MLC weight cache is built for.
//
// Where HF and MLC differ only in factoring (e.g. the 1/sqrt(dk) scale on q
// vs on the readout) the math is identical; comments note both.
//
// Math runs in f64; toF16 marks the points where the GPU pipeline stores an
// f16 buffer, so chain tests compare like against like.

import { toF16 } from './half.mjs'

export const silu = (x) => x / (1 + Math.exp(-x))
export const sigmoid = (x) => 1 / (1 + Math.exp(-x))
// [MLC] _compute_gate_beta._softplus: x if x > 20 else log(1+exp(x))
export const softplus = (x) => (x > 20 ? x : Math.log(1 + Math.exp(x)))

// ── int4 q4f16_1 dequant matvec ──────────────────────────────────────────────
// Same convention as the kernel suites' makeDot: nibble n → (n - 7) · scale,
// scale per 32-element group. Output f16-rounded (the matmul kernels write
// f16). Returns out[N].
export function int4Matvec(weights, scales, input, K, N) {
  const KP = K / 8
  const SPR = K / 32
  const out = new Array(N)
  for (let row = 0; row < N; row++) {
    let acc = 0
    for (let w = 0; w < KP; w++) {
      const packed = weights[row * KP + w]
      const scale = scales[row * SPR + (w >> 2)]
      const base = w * 8
      for (let n = 0; n < 8; n++) acc += input[base + n] * (((packed >>> (4 * n)) & 15) - 7) * scale
    }
    out[row] = toF16(acc)
  }
  return out
}

// ── GDN causal conv + SiLU ───────────────────────────────────────────────────
// [HF] causal_conv1d_update / causal_conv1d_fn with activation="silu":
//   out[t][c] = silu( Σ_{j=0}^{K-1} w[c,j] · x[t-(K-1)+j][c] ),  x[<0] = 0
// [MLC] _causal_conv1d_with_state then op.silu — conv over RAW in_proj_qkv
// outputs (the state stores pre-activation values), SiLU strictly after.
// `xs` is the full raw history (array of per-token arrays, length C each);
// returns the conv output for token t, f16-rounded.
export function refConvStep(xs, t, w, C, K) {
  const out = new Array(C)
  for (let c = 0; c < C; c++) {
    let acc = 0
    for (let j = 0; j < K; j++) {
      const ti = t - (K - 1) + j
      if (ti >= 0) acc += w[c * K + j] * xs[ti][c]
    }
    out[c] = toF16(silu(acc))
  }
  return out
}

// Expected ring-buffer content after processing token t (kernel contract in
// gdn_conv.wgsl): slot s ∈ [0, K-1) holds the raw x[t'] with the largest
// t' ≤ t satisfying t' ≡ s (mod K-1); slots never written stay 0.
export function expectedConvRing(xs, t, C, K) {
  const ring = K - 1
  const state = new Array(ring * C).fill(0)
  for (let ti = Math.max(0, t - ring + 1); ti <= t; ti++) {
    const slot = ti % ring
    for (let c = 0; c < C; c++) state[slot * C + c] = xs[ti][c]
  }
  return state
}

// ── decay + beta gates ───────────────────────────────────────────────────────
// [HF] Qwen3_5GatedDeltaNet.forward:
//   beta = sigmoid(b);  g = -exp(A_log) · softplus(a + dt_bias)   (f32 — the
//   HF comment warns -exp(A_log) overflows in fp16), recurrence uses exp(g).
// [MLC] _compute_gate_beta emits exp(g) directly:
//   gate = exp(-exp(A_log) · softplus(alpha + dt_bias))
// Returns { gExp, beta } (f64 — the GPU kernel keeps these f32).
export function refGates(aRaw, bRaw, aLog, dtBias) {
  const H = aRaw.length
  const gExp = new Array(H)
  const beta = new Array(H)
  for (let h = 0; h < H; h++) {
    gExp[h] = Math.exp(-Math.exp(aLog[h]) * softplus(aRaw[h] + dtBias[h]))
    beta[h] = sigmoid(bRaw[h])
  }
  return { gExp, beta }
}

// ── gated delta rule (recurrent form) ────────────────────────────────────────
// [HF] torch_recurrent_gated_delta_rule with use_qk_l2norm_in_kernel=True:
//   qn = l2norm(q) / sqrt(dk);  kn = l2norm(k)
//   with l2norm(x) = x · rsqrt(Σx² + 1e-6)   ← eps on the SUM (l2norm())
//   per token:  S      = S · exp(g)                       (per-v-head scalar)
//               kv_mem = Σ_dk S[dk,:] · kn[dk]
//               delta  = (v − kv_mem) · beta
//               S      = S + kn ⊗ delta
//               out    = Σ_dk S[dk,:] · qn[dk]            (UPDATED S)
// [MLC] create_gated_delta_net_func: identical ops; the 1/sqrt(dk) scale is
// applied to the readout instead of q (equivalent — readout is linear in q).
//
// GVA pairing: v-head h reads q/k from k-head floor(h / (vHeads/kHeads)) —
// [HF] repeat_interleave(num_v_heads // num_k_heads, dim=2);
// [MLC] kh = h_idx // heads_per_group. `pairing` is injectable so the test
// suite can prove a wrong pairing FAILS.
//
// qs/ks: per-token arrays of length kHeads·headK; vs: length vHeads·headV;
// state: Float64Array [vHeads · headK · headV], layout S[h][dk][dv], MUTATED
// in place (carryover across calls = carryover across dispatches). Returns
// out[T][vHeads·headV] (f64 — the GPU keeps the readout f32).
export function refRecur(qs, ks, vs, gExps, betas, state, dims, pairing) {
  const { kHeads, vHeads, headK, headV } = dims
  const gva = vHeads / kHeads
  const pair = pairing ?? ((h) => Math.floor(h / gva))
  const scale = 1 / Math.sqrt(headK)
  const T = qs.length
  const outs = []
  for (let t = 0; t < T; t++) {
    const out = new Array(vHeads * headV)
    for (let h = 0; h < vHeads; h++) {
      const kh = pair(h)
      let sk = 0
      let sq = 0
      for (let d = 0; d < headK; d++) {
        sk += ks[t][kh * headK + d] * ks[t][kh * headK + d]
        sq += qs[t][kh * headK + d] * qs[t][kh * headK + d]
      }
      const ik = 1 / Math.sqrt(sk + 1e-6)
      const iq = (1 / Math.sqrt(sq + 1e-6)) * scale
      const gt = gExps[t][h]
      const bt = betas[t][h]
      const sBase = h * headK * headV
      for (let dv = 0; dv < headV; dv++) {
        // decay, then kv_mem against the DECAYED state (HF order)
        let kv = 0
        for (let dk = 0; dk < headK; dk++) {
          state[sBase + dk * headV + dv] *= gt
          kv += state[sBase + dk * headV + dv] * ks[t][kh * headK + dk] * ik
        }
        const delta = (vs[t][h * headV + dv] - kv) * bt
        // rank-1 update, readout against the UPDATED state
        let o = 0
        for (let dk = 0; dk < headK; dk++) {
          state[sBase + dk * headV + dv] += ks[t][kh * headK + dk] * ik * delta
          o += state[sBase + dk * headV + dv] * qs[t][kh * headK + dk]
        }
        out[h * headV + dv] = o * iq
      }
    }
    outs.push(out)
  }
  return outs
}

// ── gated RMSNorm + z-gate ───────────────────────────────────────────────────
// [HF] Qwen3_5RMSNormGated ("Norm before gate"):
//   y = x · rsqrt(mean(x²) + eps) · weight;  out = y · silu(z)   (f32)
// per v-head over headV dims, gamma shared across heads; NO +1 offset on
// gamma ([MLC loader] _is_rmsnorm_weight excludes linear_attn.norm).
// x: f64 array [vHeads·headV] (recurrence output); returns f16-rounded array.
export function refGatedNorm(x, gamma, z, eps, headV) {
  const out = new Array(x.length)
  for (let base = 0; base < x.length; base += headV) {
    let ss = 0
    for (let d = 0; d < headV; d++) ss += x[base + d] * x[base + d]
    const rinv = 1 / Math.sqrt(ss / headV + eps)
    for (let d = 0; d < headV; d++) {
      out[base + d] = toF16(x[base + d] * rinv * gamma[d] * silu(z[base + d]))
    }
  }
  return out
}

// ── full GDN block ───────────────────────────────────────────────────────────
// From the post-input_layernorm hidden state to the out_proj output (the
// residual add stays in the engine). [HF] Qwen3_5GatedDeltaNet.forward:
//   qkv_raw = in_proj_qkv(x); z = in_proj_z(x); a = in_proj_a(x); b = in_proj_b(x)
//   conv+silu → split [key_dim, key_dim, value_dim] → l2norm q,k → gates →
//   recurrence → RMSNormGated(·, z) → out_proj
// `weights` carries int4 records {qkvW,qkvS,zW,zS,aW,aS,bW,bS,outW,outS},
// convW (f16 [C·K]), gamma (f16 [headV]), aLog/dtBias (f32 [vHeads]).
// `xs` is the token stream (arrays of length d); `state` an object
// { history: [], recur: Float64Array } mutated across calls.
// Returns per-token block outputs [T][d] (f16-rounded).
export function refGdnBlock(xs, weights, spec, state) {
  const d = spec.d
  const { gdnKDim, gdnVDim, gdnQkvDim, gdnConvK } = spec
  const dims = {
    kHeads: spec.gdnKHeads, vHeads: spec.gdnVHeads,
    headK: spec.gdnHeadK, headV: spec.gdnHeadV,
  }
  const outs = []
  for (const x of xs) {
    const t = state.history.length
    // Projections off the SAME normed hidden state (a/b/z are parallel to
    // qkv, not post-conv — [HF] forward order).
    const qkvRaw = int4Matvec(weights.qkvW, weights.qkvS, x, d, gdnQkvDim)
    const z = int4Matvec(weights.zW, weights.zS, x, d, gdnVDim)
    const aRaw = int4Matvec(weights.aW, weights.aS, x, d, dims.vHeads)
    const bRaw = int4Matvec(weights.bW, weights.bS, x, d, dims.vHeads)
    state.history.push(qkvRaw)
    const conv = refConvStep(state.history, t, weights.convW, gdnQkvDim, gdnConvK)
    const { gExp, beta } = refGates(aRaw, bRaw, weights.aLog, weights.dtBias)
    const q = conv.slice(0, gdnKDim)
    const k = conv.slice(gdnKDim, 2 * gdnKDim)
    const v = conv.slice(2 * gdnKDim)
    const [recurOut] = refRecur([q], [k], [v], [gExp], [beta], state.recur, dims)
    const gated = refGatedNorm(recurOut, weights.gamma, z, spec.rmsEps, dims.headV)
    outs.push(int4Matvec(weights.outW, weights.outS, gated, gdnVDim, d))
  }
  return outs
}

// ── gated attention pieces ───────────────────────────────────────────────────

// c_attn output → { qkv (engine [Q|K|V] layout), gate } — [MLC loader]
// c_attn = concat(q_proj, k_proj, v_proj) with [HF] q_proj per-head
// interleave [Q_dims | gate_dims] (view(…, -1, head_dim·2).chunk(2)).
export function refGatedSplit(cAttn, spec) {
  const { qDim, kvDim, headDim } = spec
  const qkv = new Array(qDim + 2 * kvDim)
  const gate = new Array(qDim)
  for (let h = 0; h < spec.heads; h++) {
    for (let r = 0; r < headDim; r++) {
      qkv[h * headDim + r] = cAttn[h * 2 * headDim + r]
      gate[h * headDim + r] = cAttn[h * 2 * headDim + headDim + r]
    }
  }
  for (let i = 0; i < 2 * kvDim; i++) qkv[qDim + i] = cAttn[2 * qDim + i]
  return { qkv, gate }
}

// Per-head RMSNorm (q_norm/k_norm) — standard x·rsqrt(mean+eps)·gamma; the
// MLC cache pre-bakes HF's +1.0 offset into gamma ([MLC loader]). In place
// over one head slice starting at `base`.
export function refHeadRmsNorm(vec, base, gamma, headDim, eps) {
  let ss = 0
  for (let d = 0; d < headDim; d++) ss += vec[base + d] * vec[base + d]
  const rinv = 1 / Math.sqrt(ss / headDim + eps)
  for (let d = 0; d < headDim; d++) vec[base + d] = toF16(vec[base + d] * rinv * gamma[d])
}

// Partial RoPE on one head slice, in place — [HF] apply_rotary_pos_emb (GLM
// non-interleaved): the first rotaryDim dims rotate in pairs (j, j+half) with
//   freq(j) = pos / theta^(2j / rotaryDim),  j ∈ [0, rotaryDim/2)
// (inv_freq from compute_default_rope_parameters with dim = rotaryDim);
// dims ≥ rotaryDim pass through untouched.
export function refPartialRope(vec, base, pos, rotaryDim, theta) {
  const half = rotaryDim / 2
  for (let j = 0; j < half; j++) {
    const freq = pos / Math.pow(theta, (2 * j) / rotaryDim)
    const c = Math.cos(freq)
    const s = Math.sin(freq)
    const lo = vec[base + j]
    const hi = vec[base + j + half]
    vec[base + j] = toF16(c * lo - s * hi)
    vec[base + j + half] = toF16(c * hi + s * lo)
  }
}

// Full gated-attention block: post-input_layernorm hidden → o_proj output.
// [HF] Qwen3_5Attention.forward / [MLC] Qwen35Attention.forward:
//   c_attn → split Q/gate/K/V → q_norm/k_norm (per head, BEFORE RoPE) →
//   partial RoPE on q,k → append to KV cache → softmax attention
//   (scale headDim^-0.5, GQA h → floor(h/gqaGroup)) → ·sigmoid(gate) → o_proj
// `weights`: {cAttnW,cAttnS,oW,oS,qGamma,kGamma}; `cache`: {k: [], v: []}
// (arrays of per-token kvDim rows, mutated). Returns [d] f16-rounded.
export function refGatedAttnBlock(x, pos, weights, spec, cache) {
  const { d, headDim, qDim, kvDim, heads, kvHeads, gqaGroup, rotaryDim, ropeTheta, rmsEps } = spec
  const cAttn = int4Matvec(weights.cAttnW, weights.cAttnS, x, d, spec.cAttnDim)
  const { qkv, gate } = refGatedSplit(cAttn, spec)
  for (let h = 0; h < heads; h++) refHeadRmsNorm(qkv, h * headDim, weights.qGamma, headDim, rmsEps)
  for (let h = 0; h < kvHeads; h++) refHeadRmsNorm(qkv, qDim + h * headDim, weights.kGamma, headDim, rmsEps)
  for (let h = 0; h < heads; h++) refPartialRope(qkv, h * headDim, pos, rotaryDim, ropeTheta)
  for (let h = 0; h < kvHeads; h++) refPartialRope(qkv, qDim + h * headDim, pos, rotaryDim, ropeTheta)
  cache.k.push(qkv.slice(qDim, qDim + kvDim))
  cache.v.push(qkv.slice(qDim + kvDim))
  const T = cache.k.length
  const smScale = 1 / Math.sqrt(headDim)
  const attnOut = new Array(qDim)
  for (let h = 0; h < heads; h++) {
    const kh = Math.floor(h / gqaGroup)
    const scores = new Array(T)
    for (let t = 0; t < T; t++) {
      let dot = 0
      for (let dd = 0; dd < headDim; dd++) dot += qkv[h * headDim + dd] * cache.k[t][kh * headDim + dd]
      scores[t] = dot * smScale
    }
    const m = Math.max(...scores)
    const exps = scores.map((s) => Math.exp(s - m))
    const sum = exps.reduce((a, b) => a + b, 0)
    for (let dd = 0; dd < headDim; dd++) {
      let o = 0
      for (let t = 0; t < T; t++) o += (exps[t] / sum) * cache.v[t][kh * headDim + dd]
      // f16 store (attention kernel output), then the sigmoid gate in f32.
      attnOut[h * headDim + dd] = toF16(toF16(o) * sigmoid(gate[h * headDim + dd]))
    }
  }
  return int4Matvec(weights.oW, weights.oS, attnOut, qDim, d)
}
