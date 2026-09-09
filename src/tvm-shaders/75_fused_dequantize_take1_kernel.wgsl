// [75] fused_dequantize_take1_kernel (shader #77, 28 lines)
//----------------------------------------
// Function: fused_dequantize_take1_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> T_take : array<f16>;
@group(0) @binding(1) var<storage, read> input_ids : array<i32>;
@group(0) @binding(2) var<storage, read> transformer_embd_q_scale : array<f16>;
@group(0) @binding(3) var<storage, read> transformer_embd_q_weight : array<u32>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn fused_dequantize_take1_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  T_take[((v__1 * 256i) + i32(threadIdx.x))] = ((f16(((transformer_embd_q_weight[(((input_ids[(v__1 / 12i)] * 384i) + ((v__1 % 12i) * 32i)) + (i32(threadIdx.x)>>3u))]>>u32(((i32(threadIdx.x) & 7i) * 4i))) & 15u)) - 7.000000e+00h) * transformer_embd_q_scale[(((input_ids[(v__1 / 12i)] * 96i) + ((v__1 % 12i) * 8i)) + (i32(threadIdx.x)>>5u))]);
}
