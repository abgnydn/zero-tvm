// EMBEDDING — int4 dequant + token lookup.
// output[seq * D + i] = dequant(embd_weight[token_id, i])
// Same int4 format: (nibble - 7) * scale, group_size=32
//
// Matches TVM's fused_dequantize_take1_kernel.
//
// Model-shape constants (D, D_PACKED, D_SCALES, ...) are injected by
// src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_ids : array<i32>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn embedding(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let global_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(global_id) >= podArgs.packGridDimX) { return; }

  let flat : i32 = global_id * 256 + i32(threadIdx.x);
  // flat covers seq_len * D elements
  // Each workgroup block covers 256 contiguous output elements
  // (D / 256 blocks per token)

  let token_idx : i32 = flat / D;
  if (token_idx >= podArgs.seq_len) { return; }

  let dim : i32 = flat % D;
  let token_id : i32 = input_ids[token_idx];

  // Dequantize: 8 nibbles per u32, group_size=32
  let packed_idx : i32 = token_id * D_PACKED + (dim / 8);
  let nibble_idx : u32 = u32(dim % 8);
  let packed : u32 = weights[packed_idx];
  let scale : f16 = scales[token_id * D_SCALES + (dim / 32)]; // D_SCALES = D/32 scales per row

  let nibble : u32 = (packed >> (nibble_idx * 4u)) & 15u;
  output_buf[flat] = (f16(nibble) - 7.0h) * scale;
}
