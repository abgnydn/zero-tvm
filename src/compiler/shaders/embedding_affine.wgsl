// EMBEDDING_AFFINE — int4 dequant + token lookup, MLX affine layout.
// output[seq * D + i] = scale * nibble + bias, group_size=64
//
// The affine sibling of embedding.wgsl, which is hard-symmetric:
// `(nibble - 7) * scale` over groups of 32, with no bias binding at all.
// Binding an MLX embedding table to that kernel is IN BOUNDS and ERROR-FREE —
// the buffers are the right size, the scale index just lands in the wrong
// group and every value is off by the bias — so it produces plausible token
// vectors at the very top of the forward pass and nothing downstream points
// back here. That is why this is a separate kernel rather than a flag.
//
// Separate FILE rather than a prelude branch: every other variant in this repo
// is selected by file/entry name, and the two layouts share no arithmetic.
//
// Model-shape constants (D, D_PACKED, ...) are injected by
// src/compiler/shader-prelude.ts. Note D_SCALES is D/32 and is NOT used here —
// affine groups are 64, so the row stride is D/64.

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
fn embedding_affine(
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

  // 8 nibbles per u32; one (scale, bias) pair per 64 values.
  let packed : u32 = weights[token_id * D_PACKED + (dim / 8)];
  let g : i32 = token_id * EMB_GROUPS + (dim / 64);
  let nibble : u32 = (packed >> (u32(dim % 8) * 4u)) & 15u;

  output_buf[flat] = f16(nibble) * scales[g] + biases[g];
}
