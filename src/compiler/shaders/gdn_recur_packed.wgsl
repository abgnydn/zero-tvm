// GDN_RECUR_PACKED — gdn_recur.wgsl for a chunk that holds SEVERAL independent
// continuations of one resident prefix (the packed-branch prefill, see
// attention_prefill_seg.wgsl for the attention half).
//
// The gated delta rule is a recurrence: token t's output depends on the state
// after token t-1. In a packed chunk token t-1 may belong to ANOTHER branch,
// so at every branch start this kernel RELOADS the state column from `state`
// — which holds the prefix's state, because a packed chunk never writes it —
// and runs the branch's tokens forward from there. Per branch that is exactly
// the arithmetic gdn_recur performs on `prefix + branch` after the prefix, so
// the outputs are bit-exact against running the branches one at a time from
// the same state (tests/kernels pins that, and that nothing persists).
//
// `state` is READ-ONLY here (the persist at the end of gdn_recur is what a
// packed chunk must not do). seg[t] is the branch id of chunk token t; a start
// is t == 0 or seg[t] != seg[t-1].
//
// Same bindings 0..3 and same PODArgs as gdn_recur.wgsl, plus seg at 4 and the
// uniform at 5. Model-shape constants from src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_buf : array<f32>;  // seq * GDN_V_DIM
@group(0) @binding(1) var<storage, read> conv_out : array<f16>;       // seq * GDN_QKV_DIM
@group(0) @binding(2) var<storage, read> gates : array<f32>;          // seq * 2*GDN_V_HEADS
@group(0) @binding(3) var<storage, read> state : array<f32>;          // the PREFIX's state, never written
@group(0) @binding(4) var<storage, read> seg : array<i32>;            // seq: branch id per token

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32   // GDN_V_HEADS
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> qsh : array<f32, GDN_HEAD_K>;
var<workgroup> ksh : array<f32, GDN_HEAD_K>;

@compute @workgroup_size(GDN_HEAD_V, 1, 1)
fn gdn_recur_packed(
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

  var s : array<f32, GDN_HEAD_K>;

  for (var t : i32 = 0; t < podArgs.seq_len; t = t + 1) {
    // A branch start: this token continues the PREFIX, not the previous token.
    if (t == 0 || seg[t] != seg[t - 1]) {
      for (var dk : i32 = 0; dk < GDN_HEAD_K; dk = dk + 1) {
        s[dk] = state[sbase + dk * GDN_HEAD_V];
      }
    }

    let qb : i32 = t * GDN_QKV_DIM + kh * GDN_HEAD_K;
    let kb : i32 = qb + GDN_K_DIM;
    for (var i : i32 = dv; i < GDN_HEAD_K; i = i + GDN_HEAD_V) {
      qsh[i] = f32(conv_out[qb + i]);
      ksh[i] = f32(conv_out[kb + i]);
    }
    workgroupBarrier();

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

    var kv : f32 = 0.0;
    for (var dk : i32 = 0; dk < GDN_HEAD_K; dk = dk + 1) {
      s[dk] = s[dk] * gt;
      kv = kv + s[dk] * ksh[dk] * ik;
    }
    let delta : f32 = (vv - kv) * bt;

    var o : f32 = 0.0;
    for (var dk : i32 = 0; dk < GDN_HEAD_K; dk = dk + 1) {
      s[dk] = s[dk] + ksh[dk] * ik * delta;
      o = o + s[dk] * qsh[dk];
    }
    out_buf[t * GDN_V_DIM + h * GDN_HEAD_V + dv] = o * iq;

    workgroupBarrier();   // qsh/ksh reused next token
  }
  // No persist: the prefix's state stays the prefix's state.
}
