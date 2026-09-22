// GDN_CONV_SEQ_PACKED — gdn_conv_seq.wgsl for a chunk of SEVERAL independent
// continuations of one resident prefix (packed-branch prefill).
//
// The causal conv's taps reach RING = GDN_CONV_K-1 tokens back. In a packed
// chunk those tokens may belong to another branch, so a tap that falls before
// the token's OWN branch start reads the prefix's ring instead — the same ring
// slot a per-branch gdn_conv_seq would read for that absolute position,
// because every branch starts right after the prefix at position state_len.
// Within a branch the taps are the branch's own raw projections, as before.
//
// seg_start[t] is the chunk-relative index of token t's branch start; the
// token's absolute position is state_len + (t - seg_start[t]). The ring is
// read-only here and there is NO commit for a packed chunk: the ring stays the
// prefix's ring, which is what the next packed chunk (or the next call) needs.
//
// Bit-exact vs per-branch gdn_conv_seq over `prefix + branch` (tests/kernels).
// Model-shape constants from src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> conv_out : array<f16>;   // seq * GDN_QKV_DIM
@group(0) @binding(1) var<storage, read> qkv_raw : array<f16>;          // seq * qkv_stride
@group(0) @binding(2) var<storage, read> conv_state : array<f16>;       // RING * GDN_QKV_DIM — the prefix's ring
@group(0) @binding(3) var<storage, read> conv_w : array<f16>;           // GDN_QKV_DIM * GDN_CONV_K
@group(0) @binding(4) var<storage, read> seg_start : array<i32>;        // seq: chunk index of the token's branch start

struct PODArgs {
  state_len: i32,     // prefix length: every branch's first token sits at this position
  seq_len: i32,       // tokens in this chunk
  qkv_stride: i32,    // f16 elements between consecutive tokens in qkv_raw
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

const RING = GDN_CONV_K - 1;

@compute @workgroup_size(256, 1, 1)
fn gdn_conv_seq_packed(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg) >= podArgs.packGridDimX) { return; }

  let flat : i32 = wg * 256 + i32(threadIdx.x);
  let t : i32 = flat / GDN_QKV_DIM;
  let c : i32 = flat % GDN_QKV_DIM;
  if (t >= podArgs.seq_len) { return; }

  let start : i32 = seg_start[t];
  let pos : i32 = podArgs.state_len + (t - start);   // absolute position of token t

  var acc : f32 = 0.0;
  for (var j : i32 = 0; j < GDN_CONV_K; j = j + 1) {
    let src_t : i32 = t - RING + j;
    var x : f32;
    if (src_t >= start) {
      x = f32(qkv_raw[src_t * podArgs.qkv_stride + c]);
    } else {
      // Before this branch: the prefix's ring slot for absolute position
      // pos-RING+j, exactly as gdn_conv_seq resolves a tap before its chunk.
      let slot : i32 = (pos + j) % RING;
      x = f32(conv_state[slot * GDN_QKV_DIM + c]);
    }
    acc = acc + f32(conv_w[c * GDN_CONV_K + j]) * x;
  }

  conv_out[t * GDN_QKV_DIM + c] = f16(acc / (1.0 + exp(-acc)));
}
