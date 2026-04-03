// [78] apply_penalty_inplace_kernel (shader #78, 40 lines)
//----------------------------------------
// Function: apply_penalty_inplace_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> logits : array<f32>;
@group(0) @binding(1) var<storage, read> penalties : array<f32>;
@group(0) @binding(2) var<storage, read> pos2seq_id : array<i32>;
@group(0) @binding(3) var<storage, read> seq_ids : array<i32>;
@group(0) @binding(4) var<storage, read> token_cnt : array<i32>;
@group(0) @binding(5) var<storage, read> token_ids : array<i32>;

struct PODArgs {
  batch_size: i32,
  num_seq: i32,
  num_token: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn apply_penalty_inplace_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < podArgs.num_token) {
    logits[((seq_ids[pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))]] * podArgs.vocab_size) + token_ids[((v__1 * 256i) + i32(threadIdx.x))])] = (logits[((seq_ids[pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))]] * podArgs.vocab_size) + token_ids[((v__1 * 256i) + i32(threadIdx.x))])] - fma(f32(token_cnt[((v__1 * 256i) + i32(threadIdx.x))]), penalties[((pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))] * 3i) + 1i)], penalties[(pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))] * 3i)]));
    var condval : f32;
    if ((logits[((seq_ids[pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))]] * podArgs.vocab_size) + token_ids[((v__1 * 256i) + i32(threadIdx.x))])] < 0.000000e+00f)) {
      condval = (logits[((seq_ids[pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))]] * podArgs.vocab_size) + token_ids[((v__1 * 256i) + i32(threadIdx.x))])] * penalties[((pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))] * 3i) + 2i)]);
} else {
      condval = (logits[((seq_ids[pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))]] * podArgs.vocab_size) + token_ids[((v__1 * 256i) + i32(threadIdx.x))])] / penalties[((pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))] * 3i) + 2i)]);
}
    logits[((seq_ids[pos2seq_id[((v__1 * 256i) + i32(threadIdx.x))]] * podArgs.vocab_size) + token_ids[((v__1 * 256i) + i32(threadIdx.x))])] = condval;
  }
}
