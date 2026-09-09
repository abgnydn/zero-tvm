// GDN_CONV_SEQ — chunked-prefill sibling of gdn_conv.wgsl: the same depthwise
// causal Conv1d (width GDN_CONV_K) + SiLU over a CHUNK of seq_len tokens in
// one dispatch, plus a separate ring-commit entry point.
//
// Same ground truth as gdn_conv.wgsl (HF causal_conv1d_update / MLC
// _causal_conv1d_with_state): conv over RAW in_proj qkv outputs, SiLU strictly
// after; f32 accumulation, identical tap order — so given identical inputs the
// chunked output is BIT-EXACT vs per-token gdn_conv (pinned in
// tests/kernels/compile-qwen35.mjs).
//
// Paired with gdn_conv_commit.wgsl, dispatched back-to-back:
//
//   gdn_conv_seq     (this file) reads the ring state for taps that precede
//                    the chunk (x[< base_pos]) and the batched raw projections
//                    for taps inside it; writes conv_out[t * GDN_QKV_DIM + c].
//                    The ring is READ-ONLY here: tokens 0..RING-1 of the chunk
//                    read slots that tokens seq_len-RING.. would overwrite,
//                    and workgroups are unordered within a dispatch — so the
//                    rotate is split out (into its own file so each entry's
//                    'auto' bind-group layout carries only what it uses).
//   gdn_conv_commit  copies the last min(RING, seq_len) raw projections into
//                    their ring slots (slot = absolute pos % RING — the same
//                    invariant gdn_conv maintains per token), leaving the ring
//                    exactly as if the chunk had been processed token-by-token.
//
// qkv_raw is the BATCHED fused-projection output: token t's qkv region starts
// at t * qkv_stride (engine: qkv_stride = gdnProjRows; only channels
// < GDN_QKV_DIM are read). x[pos < 0] taps resolve to ring reads, which are
// zero after the position-0 clearGdnState — same behavior as gdn_conv.
//
// Grid: flat over channels — seq_len * (GDN_QKV_DIM/256).
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> conv_out : array<f16>;   // seq * GDN_QKV_DIM
@group(0) @binding(1) var<storage, read> qkv_raw : array<f16>;          // seq * qkv_stride
@group(0) @binding(2) var<storage, read_write> conv_state : array<f16>; // RING * GDN_QKV_DIM
@group(0) @binding(3) var<storage, read> conv_w : array<f16>;           // GDN_QKV_DIM * GDN_CONV_K

struct PODArgs {
  base_pos: i32,      // absolute position of the chunk's first token
  seq_len: i32,       // tokens in this chunk
  qkv_stride: i32,    // f16 elements between consecutive tokens in qkv_raw
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const RING = GDN_CONV_K - 1;   // history slots (kernel width minus current token)

@compute @workgroup_size(256, 1, 1)
fn gdn_conv_seq(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg) >= podArgs.packGridDimX) { return; }

  let flat : i32 = wg * 256 + i32(threadIdx.x);
  let t : i32 = flat / GDN_QKV_DIM;       // token index within the chunk
  let c : i32 = flat % GDN_QKV_DIM;       // channel
  if (t >= podArgs.seq_len) { return; }

  let pos : i32 = podArgs.base_pos + t;

  // Taps j = 0..K-1 are x[pos - (K-1) + j]; the last tap is the current token.
  var acc : f32 = 0.0;
  for (var j : i32 = 0; j < GDN_CONV_K; j = j + 1) {
    let src_t : i32 = t - RING + j;       // chunk-relative source token
    var x : f32;
    if (src_t >= 0) {
      x = f32(qkv_raw[src_t * podArgs.qkv_stride + c]);
    } else {
      // Before the chunk: the ring slot for absolute position pos-RING+j
      // (zeroed state ⇒ 0, matching gdn_conv's x[<0] = 0 semantics).
      let slot : i32 = (pos + j) % RING;  // ≡ (pos - RING + j) mod RING
      x = f32(conv_state[slot * GDN_QKV_DIM + c]);
    }
    acc = acc + f32(conv_w[c * GDN_CONV_K + j]) * x;
  }

  // silu(x) = x * sigmoid(x), in f32 — identical to gdn_conv.
  conv_out[t * GDN_QKV_DIM + c] = f16(acc / (1.0 + exp(-acc)));
}
