// [10] full_kernel (shader #10, 26 lines)
//----------------------------------------
// Function: full_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> result : array<i32>;

struct PODArgs {
  batch_size: i32,
  value: i32,
  packGridDimX: u32
}
@group(0) @binding(1) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn full_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < podArgs.batch_size) {
    result[((v__1 * 256i) + i32(threadIdx.x))] = podArgs.value;
  }
}
