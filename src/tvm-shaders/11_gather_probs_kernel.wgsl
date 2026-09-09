// [11] gather_probs_kernel (shader #11, 29 lines)
//----------------------------------------
// Function: gather_probs_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> dst : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read> src : array<f32>;

struct PODArgs {
  batch_size: i32,
  m: i32,
  n: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn gather_probs_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.batch_size * podArgs.n)) {
    dst[((v__1 * 256i) + i32(threadIdx.x))] = src[((indices[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.n * podArgs.batch_size)) / podArgs.n)] * podArgs.n) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.n))];
  }
}
