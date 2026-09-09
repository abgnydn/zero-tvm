// [62] take_sorted_probs_kernel (shader #62, 30 lines)
//----------------------------------------
// Function: take_sorted_probs_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> lv : array<i32>;
@group(0) @binding(1) var<storage, read> probs : array<f32>;
@group(0) @binding(2) var<storage, read_write> take_sorted_probs : array<f32>;

struct PODArgs {
  batch_size: i32,
  batch_size_1: i32,
  vocab_size: i32,
  vocab_size_1: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn take_sorted_probs_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.batch_size_1 * podArgs.vocab_size)) {
    take_sorted_probs[((v__1 * 256i) + i32(threadIdx.x))] = probs[((((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch_size_1)) / podArgs.vocab_size) * podArgs.vocab_size_1) + lv[((((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch_size_1)) / podArgs.vocab_size) * podArgs.vocab_size_1) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size))])];
  }
}
