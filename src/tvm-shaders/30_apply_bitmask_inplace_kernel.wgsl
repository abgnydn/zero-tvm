// [30] apply_bitmask_inplace_kernel (shader #30, 35 lines)
//----------------------------------------
// Function: apply_bitmask_inplace_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> bitmask : array<i32>;
@group(0) @binding(1) var<storage, read_write> logits : array<f32>;
@group(0) @binding(2) var<storage, read> seq_ids : array<i32>;

struct PODArgs {
  batch_size: i32,
  num_seq: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn apply_bitmask_inplace_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.num_seq * podArgs.vocab_size)) {
    var condval : f32;
    if ((((bitmask[(((((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size)>>5u) + (seq_ids[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.num_seq)) / podArgs.vocab_size)] * ((podArgs.vocab_size + 31i)>>5u)))]>>u32(((((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size) & 31i))) & 1i) == 1i)) {
      condval = logits[((seq_ids[(((v__1 * 256i) + i32(threadIdx.x)) / podArgs.vocab_size)] * podArgs.vocab_size) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size))];
} else {
      condval = -3.402823e+38f;
}
    logits[((seq_ids[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.num_seq)) / podArgs.vocab_size)] * podArgs.vocab_size) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size))] = condval;
  }
}
