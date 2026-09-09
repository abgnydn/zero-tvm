// [53] gpu_2d_continuous_cumsum1_kernel_3 (shader #53, 35 lines)
//----------------------------------------
// Function: gpu_2d_continuous_cumsum1_kernel_3
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> Out : array<f32>;
@group(0) @binding(1) var<storage, read> Tmp : array<f32>;

struct PODArgs {
  m: i32,
  n: i32,
  packGridDimX: u32
}
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(32, 4, 1)
fn gpu_2d_continuous_cumsum1_kernel_3(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var i : i32 = 0; i < 4i; i++) {
    if (((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i * 32i)) + i32(threadIdx.x)) < podArgs.n) {
      var condval : f32;
      if ((0i < v__1)) {
        condval = Tmp[(((i32(blockIdx.y) * podArgs.n) + v__1) - 1i)];
} else {
        condval = 0.000000e+00f;
}
      Out[(((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i * 32i)) + (i32(blockIdx.y) * podArgs.n)) + i32(threadIdx.x))] = (Out[(((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i * 32i)) + (i32(blockIdx.y) * podArgs.n)) + i32(threadIdx.x))] + condval);
    }
  }
}
