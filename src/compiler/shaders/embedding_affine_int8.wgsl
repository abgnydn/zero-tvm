// EMBEDDING_AFFINE_INT8 — int8 dequant + token lookup, MLX affine layout.
//
// The 8-bit twin of embedding_affine.wgsl: output[seq * D + i] =
// scale * byte + bias over groups of 64. Same bindings, same uniform, same
// grid — only the unpack differs (one byte per value, not one nibble), so the
// engine swaps pipelines, not plumbing.
//
// Model-shape constants (D, D_PACKED, ...) are injected by
// src/compiler/shader-prelude.ts. D_PACKED is bits-aware (D/4 here, not
// D/8), which is what makes the same indexing code correct for both widths.

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_ids : array<i32>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;
@group(0) @binding(5) var<storage, read> biases : array<f16>;   // MLX bias, verbatim

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const EMB_GROUPS = D / 64;   // affine groups per embedding row

@compute @workgroup_size(256, 1, 1)
fn embedding_affine_int8(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let global_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(global_id) >= podArgs.packGridDimX) { return; }

  let flat : i32 = global_id * 256 + i32(threadIdx.x);
  let token_idx : i32 = flat / D;
  if (token_idx >= podArgs.seq_len) { return; }

  let dim : i32 = flat % D;
  let token_id : i32 = input_ids[token_idx];

  // 4 bytes per u32; one (scale, bias) pair per 64 values.
  let packed : u32 = weights[token_id * D_PACKED + (dim / 4)];
  let g : i32 = token_id * EMB_GROUPS + (dim / 64);
  let byte : u32 = (packed >> (u32(dim % 4) * 8u)) & 255u;

  output_buf[flat] = f16(byte) * scales[g] + biases[g];
}
