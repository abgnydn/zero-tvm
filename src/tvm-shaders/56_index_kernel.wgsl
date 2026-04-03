// [56] index_kernel (shader #56, 26 lines)
//----------------------------------------
// Function: index_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> index : array<f16>;
@group(0) @binding(1) var<storage, read> rms_norm64 : array<f16>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn index_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  index[((v__1 * 256i) + i32(threadIdx.x))] = rms_norm64[((((podArgs.seq_len * 3072i) + (v__1 * 256i)) + i32(threadIdx.x)) - 3072i)];
}
