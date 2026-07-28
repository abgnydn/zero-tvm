// ROPE — Rotary Position Embedding + QKV split.
//
// PREFILL-ONLY: the decode path fuses this into qkv_fused.wgsl (QKV matmul +
// RoPE + KV append in one dispatch); this kernel is kept for the sequential
// prefill path and for parity testing.
//
// Input: QKV projection output [QKV_DIM f16] = Q[Q_DIM] + K[KV_DIM] + V[KV_DIM]
// Output: Q buffer [Q_DIM], K buffer [KV_DIM], V buffer [KV_DIM]
//
// GQA-aware: the K/V groups hold KV_HEADS heads each (== HEADS for MHA, so
// the Phi-3 layout is unchanged). Q and K get RoPE applied (sin/cos rotation
// of pairs). V is just copied.
//
// Rotation pairs are at distance HALF_HEAD_DIM within each head.
// theta = position / (ROPE_THETA ^ (2i/HEAD_DIM))
//
// Model-shape constants (Q_DIM, KV_DIM, QKV_DIM, HEAD_DIM, HALF_HEAD_DIM,
// ROPE_THETA) are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> q_out : array<f16>;
@group(0) @binding(1) var<storage, read_write> k_out : array<f16>;
@group(0) @binding(2) var<storage, read_write> v_out : array<f16>;
@group(0) @binding(3) var<storage, read> qkv : array<f16>;
@group(0) @binding(4) var<storage, read> position_map : array<i32>;

struct PODArgs {
  apply_rope: i32,
  position_map_elem_offset: i32,
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn rope_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let global_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(global_id) >= podArgs.packGridDimX) { return; }

  // Decompose: global_id covers seq_len * (QKV_DIM / 256) workgroups.
  // Each element is one f16 in the QKV tensor.
  let tid : i32 = i32(threadIdx.x);
  let flat : i32 = global_id * 256 + tid;

  let seq_idx : i32 = flat / QKV_DIM;
  let within : i32 = flat % QKV_DIM;

  // within layout: Q [0, Q_DIM) — HEADS heads; K [Q_DIM, Q_DIM+KV_DIM) and
  // V [Q_DIM+KV_DIM, QKV_DIM) — KV_HEADS heads each (unequal groups for GQA).
  let dim_idx : i32 = within % HEAD_DIM;

  let qkv_val : f16 = qkv[flat];

  if (within < Q_DIM) {
    // Q head — apply RoPE
    var out_val : f16 = qkv_val;
    if (podArgs.apply_rope != 0) {
      let pos : f32 = f32(position_map[seq_idx + podArgs.position_map_elem_offset]);
      let freq : f32 = pos / pow(ROPE_THETA, f32((dim_idx % HALF_HEAD_DIM) * 2) / f32(HEAD_DIM));
      let cos_f : f32 = cos(freq);
      let sin_f : f32 = sin(freq);

      var pair_val : f16;
      if (dim_idx < HALF_HEAD_DIM) {
        pair_val = qkv[seq_idx * QKV_DIM + within + HALF_HEAD_DIM] * -1.0h;
      } else {
        pair_val = qkv[seq_idx * QKV_DIM + within - HALF_HEAD_DIM];
      }
      out_val = f16(cos_f * f32(qkv_val) + sin_f * f32(pair_val));
    }
    q_out[seq_idx * Q_DIM + within] = out_val;
  } else if (within < Q_DIM + KV_DIM) {
    // K head — apply RoPE
    let k_within : i32 = within - Q_DIM;
    var out_val : f16 = qkv_val;
    if (podArgs.apply_rope != 0) {
      let pos : f32 = f32(position_map[seq_idx + podArgs.position_map_elem_offset]);
      let freq : f32 = pos / pow(ROPE_THETA, f32((dim_idx % HALF_HEAD_DIM) * 2) / f32(HEAD_DIM));
      let cos_f : f32 = cos(freq);
      let sin_f : f32 = sin(freq);

      var pair_val : f16;
      if (dim_idx < HALF_HEAD_DIM) {
        pair_val = qkv[seq_idx * QKV_DIM + within + HALF_HEAD_DIM] * -1.0h;
      } else {
        pair_val = qkv[seq_idx * QKV_DIM + within - HALF_HEAD_DIM];
      }
      out_val = f16(cos_f * f32(qkv_val) + sin_f * f32(pair_val));
    }
    k_out[seq_idx * KV_DIM + k_within] = out_val;
  } else {
    // V head — just copy
    let v_within : i32 = within - Q_DIM - KV_DIM;
    v_out[seq_idx * KV_DIM + v_within] = qkv_val;
  }
}
