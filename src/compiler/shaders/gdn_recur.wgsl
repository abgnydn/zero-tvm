// GDN_RECUR — Qwen3.5 gated delta rule, sequential per-token recurrence.
//
// Ground truth (both transcribed exactly, they agree):
//   HF modeling_qwen3_5.torch_recurrent_gated_delta_rule (decode path,
//   use_qk_l2norm_in_kernel=True):
//     qn = l2norm(q) * 1/sqrt(dk);  kn = l2norm(k)          (eps 1e-6 on the
//     l2norm SUM: x * rsqrt(sum(x²) + eps), modeling_qwen3_5.l2norm)
//     per token:  S      = S * exp(g)                        (per-v-head decay)
//                 kv_mem = Σ_dk S[dk,:] · kn[dk]
//                 delta  = (v - kv_mem) * beta
//                 S      = S + kn ⊗ delta
//                 out    = Σ_dk S[dk,:] · qn[dk]             (UPDATED S)
//   MLC qwen35_model.create_gated_delta_net_func: identical, with the
//   1/sqrt(dk) scale applied to the output instead of q (equivalent — the
//   readout is linear in q). exp(g) and beta arrive precomputed (gdn_gates).
//
// GVA head pairing: v-head h reads q/k from k-head h / GDN_GVA_GROUP —
// HF: query.repeat_interleave(num_v_heads // num_k_heads, dim=2);
// MLC TIR: kh = h_idx // heads_per_group. (NOT h % GDN_K_HEADS — that is the
// GGUF tile order some ports use; the MLC weight cache is repeat-interleave.)
//
// State: f32 (the accuracy-critical accumulator), one [GDN_HEAD_K,GDN_HEAD_V]
// matrix per v-head, laid out S[h][dk][dv] = state[h*GDN_STATE_PER_HEAD +
// dk*GDN_HEAD_V + dv]. 32×128×128 f32 ≈ 2MB per layer. Updated IN PLACE
// (each thread owns column dv of its head — no cross-thread aliasing), so a
// seq_len=8 dispatch and 8 seq_len=1 dispatches produce bit-identical f32
// results (state round-trips through the buffer exactly).
//
// One workgroup per v-head (grid GDN_V_HEADS), GDN_HEAD_V threads; thread dv
// keeps state column S[:,dv] in registers across the token loop. q/k of the
// shared k-head are staged in workgroup memory; every thread reduces the
// l2-norm sums serially (redundant but barrier-free, 128 shared reads).
//
// Inputs q/k/v are consecutive slices of the conv output (rows
// [0,GDN_K_DIM) = Q, [GDN_K_DIM,2·GDN_K_DIM) = K, rest = V — HF split
// [key_dim, key_dim, value_dim]), so one buffer serves all three.
// seq_len > 1 loops tokens inside the dispatch (prefill v1: sequential scan,
// O(seq) — the chunked scan is future work).
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_buf : array<f32>;  // seq * GDN_V_DIM
@group(0) @binding(1) var<storage, read> conv_out : array<f16>;       // seq * GDN_QKV_DIM
@group(0) @binding(2) var<storage, read> gates : array<f32>;          // seq * 2*GDN_V_HEADS ([exp(g)|beta] per token)
@group(0) @binding(3) var<storage, read_write> state : array<f32>;    // GDN_V_HEADS * GDN_STATE_PER_HEAD

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32   // GDN_V_HEADS
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> qsh : array<f32, GDN_HEAD_K>;
var<workgroup> ksh : array<f32, GDN_HEAD_K>;

@compute @workgroup_size(GDN_HEAD_V, 1, 1)
fn gdn_recur(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let h : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);   // v-head
  if (u32(h) >= podArgs.packGridDimX) { return; }           // uniform per workgroup

  let dv : i32 = i32(threadIdx.x);                          // state column
  let kh : i32 = h / GDN_GVA_GROUP;                         // shared q/k head (repeat-interleave)
  let sbase : i32 = h * GDN_STATE_PER_HEAD + dv;            // S[h][dk][dv], dk stride GDN_HEAD_V
  let scale : f32 = 1.0 / sqrt(f32(GDN_HEAD_K));

  // Load this thread's state column into registers.
  var s : array<f32, GDN_HEAD_K>;
  for (var dk : i32 = 0; dk < GDN_HEAD_K; dk = dk + 1) {
    s[dk] = state[sbase + dk * GDN_HEAD_V];
  }

  for (var t : i32 = 0; t < podArgs.seq_len; t = t + 1) {
    // Stage this token's raw q/k (shared k-head) in workgroup memory.
    let qb : i32 = t * GDN_QKV_DIM + kh * GDN_HEAD_K;
    let kb : i32 = qb + GDN_K_DIM;
    for (var i : i32 = dv; i < GDN_HEAD_K; i = i + GDN_HEAD_V) {
      qsh[i] = f32(conv_out[qb + i]);
      ksh[i] = f32(conv_out[kb + i]);
    }
    workgroupBarrier();

    // l2-norm factors (HF l2norm: rsqrt(sum + 1e-6) — eps on the SUM, and a
    // fixed 1e-6 regardless of RMS_EPS). Scale folded into the q factor.
    var sk : f32 = 0.0;
    var sq : f32 = 0.0;
    for (var dk : i32 = 0; dk < GDN_HEAD_K; dk = dk + 1) {
      sk = sk + ksh[dk] * ksh[dk];
      sq = sq + qsh[dk] * qsh[dk];
    }
    let ik : f32 = inverseSqrt(sk + 1e-6);
    let iq : f32 = inverseSqrt(sq + 1e-6) * scale;

    let gt : f32 = gates[t * 2 * GDN_V_HEADS + h];                 // exp(g)
    let bt : f32 = gates[t * 2 * GDN_V_HEADS + GDN_V_HEADS + h];   // beta
    let vv : f32 = f32(conv_out[t * GDN_QKV_DIM + 2 * GDN_K_DIM + h * GDN_HEAD_V + dv]);

    // Decay, then kv_mem against the DECAYED state (HF order).
    var kv : f32 = 0.0;
    for (var dk : i32 = 0; dk < GDN_HEAD_K; dk = dk + 1) {
      s[dk] = s[dk] * gt;
      kv = kv + s[dk] * ksh[dk] * ik;
    }
    let delta : f32 = (vv - kv) * bt;

    // Rank-1 update, readout against the UPDATED state.
    var o : f32 = 0.0;
    for (var dk : i32 = 0; dk < GDN_HEAD_K; dk = dk + 1) {
      s[dk] = s[dk] + ksh[dk] * ik * delta;
      o = o + s[dk] * qsh[dk];
    }
    out_buf[t * GDN_V_DIM + h * GDN_HEAD_V + dv] = o * iq;

    workgroupBarrier();   // qsh/ksh reused next token
  }

  // Persist the state column.
  for (var dk : i32 = 0; dk < GDN_HEAD_K; dk = dk + 1) {
    state[sbase + dk * GDN_HEAD_V] = s[dk];
  }
}
