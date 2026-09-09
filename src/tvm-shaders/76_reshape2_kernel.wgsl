// [76] reshape2_kernel (shader #81, 26 lines)
//----------------------------------------
// Function: reshape2_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> T_reshape : array<f16>;
@group(0) @binding(1) var<storage, read> lv486 : array<f16>;

struct PODArgs {
  batch_size: i32,
  packGridDimX: u32
}
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn reshape2_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  T_reshape[((((v__1 / 12i) * 3072i) + (((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))] = lv486[((((v__1 / 12i) * 3072i) + (((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))];
}
