// [21] apply_logit_bias_inplace_kernel (shader #21, 30 lines)
//----------------------------------------
// Function: apply_logit_bias_inplace_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> logit_bias : array<f32>;
@group(0) @binding(1) var<storage, read_write> logits : array<f32>;
@group(0) @binding(2) var<storage, read> pos2seq_id : array<i32>;
@group(0) @binding(3) var<storage, read> token_ids : array<i32>;

struct PODArgs {
  batch_size: i32,
  num_token: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn apply_logit_bias_inplace_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < podArgs.num_token) {
    logits[((pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))] * podArgs.vocab_size) + token_ids[((v__1 * 256i) + i32(threadIdx.x))])] = (logits[((pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))] * podArgs.vocab_size) + token_ids[((v__1 * 256i) + i32(threadIdx.x))])] + logit_bias[((v__1 * 256i) + i32(threadIdx.x))]);
  }
}
