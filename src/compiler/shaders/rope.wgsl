// ROPE — Rotary Position Embedding + QKV split.
//
// Input: QKV projection output [9216 f16] = Q[3072] + K[3072] + V[3072]
// Output: Q buffer [3072], K buffer [3072], V buffer [3072]
//
// Q and K get RoPE applied (sin/cos rotation of pairs).
// V is just copied.
//
// head_dim=96, so rotation pairs are at distance 48.
// theta = position / (10000 ^ (2i/96))

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

  // Decompose: global_id covers seq_len * 36 workgroups
  // 36 = 32 Q heads * 96/256 + ... = (32+32+32) heads * 96 / 256
  // Simpler: each element is one f16 in the QKV tensor
  let tid : i32 = i32(threadIdx.x);
  let flat : i32 = global_id * 256 + tid;

  let seq_idx : i32 = flat / 9216;
  let within : i32 = flat % 9216;

  let head_idx : i32 = within / 96;
  let dim_idx : i32 = within % 96;

  let qkv_val : f16 = qkv[flat];

  if (head_idx < 32) {
    // Q head — apply RoPE
    var out_val : f16 = qkv_val;
    if (podArgs.apply_rope != 0) {
      let pos : f32 = f32(position_map[seq_idx + podArgs.position_map_elem_offset]);
      let freq : f32 = pos / pow(10000.0, f32(((dim_idx % 48) * 2)) / 96.0);
      let cos_f : f32 = cos(freq);
      let sin_f : f32 = sin(freq);

      var pair_val : f16;
      if (dim_idx < 48) {
        pair_val = qkv[seq_idx * 9216 + head_idx * 96 + dim_idx + 48] * -1.0h;
      } else {
        pair_val = qkv[seq_idx * 9216 + head_idx * 96 + dim_idx - 48];
      }
      out_val = f16(cos_f * f32(qkv_val) + sin_f * f32(pair_val));
    }
    q_out[seq_idx * 3072 + head_idx * 96 + dim_idx] = out_val;
  } else if (head_idx < 64) {
    // K head — apply RoPE
    let k_head : i32 = head_idx - 32;
    var out_val : f16 = qkv_val;
    if (podArgs.apply_rope != 0) {
      let pos : f32 = f32(position_map[seq_idx + podArgs.position_map_elem_offset]);
      let freq : f32 = pos / pow(10000.0, f32(((dim_idx % 48) * 2)) / 96.0);
      let cos_f : f32 = cos(freq);
      let sin_f : f32 = sin(freq);

      var pair_val : f16;
      if (dim_idx < 48) {
        pair_val = qkv[seq_idx * 9216 + head_idx * 96 + dim_idx + 48] * -1.0h;
      } else {
        pair_val = qkv[seq_idx * 9216 + head_idx * 96 + dim_idx - 48];
      }
      out_val = f16(cos_f * f32(qkv_val) + sin_f * f32(pair_val));
    }
    k_out[seq_idx * 3072 + k_head * 96 + dim_idx] = out_val;
  } else {
    // V head — just copy
    let v_head : i32 = head_idx - 64;
    v_out[seq_idx * 3072 + v_head * 96 + dim_idx] = qkv_val;
  }
}
