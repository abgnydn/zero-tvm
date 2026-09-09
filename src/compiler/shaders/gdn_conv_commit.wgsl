// GDN_CONV_COMMIT — ring-rotate half of the chunked GDN conv (see
// gdn_conv_seq.wgsl): after gdn_conv_seq has read the ring for the chunk's
// leading tokens, this pass copies the last min(RING, seq_len) RAW projections
// into their ring slots (slot = absolute pos % RING — the invariant gdn_conv
// maintains per token), leaving the ring bit-identical to a token-by-token
// run. Split into its own file so its 'auto' bind-group layout carries only
// the bindings this entry uses.
//
// Grid: RING * (GDN_QKV_DIM/256) workgroups (podArgs.packGridDimX).
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> conv_state : array<f16>; // RING * GDN_QKV_DIM
@group(0) @binding(1) var<storage, read> qkv_raw : array<f16>;          // seq * qkv_stride

struct PODArgs {
  base_pos: i32,      // absolute position of the chunk's first token
  seq_len: i32,       // tokens in this chunk
  qkv_stride: i32,    // f16 elements between consecutive tokens in qkv_raw
  packGridDimX: u32
}
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

const RING = GDN_CONV_K - 1;

@compute @workgroup_size(256, 1, 1)
fn gdn_conv_commit(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg) >= podArgs.packGridDimX) { return; }

  let flat : i32 = wg * 256 + i32(threadIdx.x);
  let j : i32 = flat / GDN_QKV_DIM;       // 0..RING-1: which of the last RING tokens
  let c : i32 = flat % GDN_QKV_DIM;
  if (j >= RING) { return; }

  let t : i32 = podArgs.seq_len - RING + j;   // chunk-relative token index
  if (t < 0) { return; }                       // short chunk: older slots stay valid

  let pos : i32 = podArgs.base_pos + t;
  conv_state[(pos % RING) * GDN_QKV_DIM + c] = qkv_raw[t * podArgs.qkv_stride + c];
}
