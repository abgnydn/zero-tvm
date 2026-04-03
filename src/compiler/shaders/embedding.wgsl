// EMBEDDING — int4 dequant + token lookup.
// output[seq * 3072 + i] = dequant(embd_weight[token_id, i])
// Same int4 format: (nibble - 7) * scale, group_size=32
//
// Matches TVM's fused_dequantize_take1_kernel.

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
  // flat covers seq_len * 3072 elements
  // Each workgroup block covers 256 contiguous output elements
  // 3072 / 256 = 12 blocks per token

  let token_idx : i32 = flat / 3072;
  if (token_idx >= podArgs.seq_len) { return; }

  let dim : i32 = flat % 3072;
  let token_id : i32 = input_ids[token_idx];

  // Dequantize: 8 nibbles per u32, group_size=32
  let packed_idx : i32 = token_id * 384 + (dim / 8);  // 384 = 3072/8
  let nibble_idx : u32 = u32(dim % 8);
  let packed : u32 = weights[packed_idx];
  let scale : f16 = scales[token_id * 96 + (dim / 32)]; // 96 = 3072/32

  let nibble : u32 = (packed >> (nibble_idx * 4u)) & 15u;
  output_buf[flat] = (f16(nibble) - 7.0h) * scale;
}
