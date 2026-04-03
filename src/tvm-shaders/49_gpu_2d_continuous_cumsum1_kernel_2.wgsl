// [49] gpu_2d_continuous_cumsum1_kernel_2 (shader #49, 38 lines)
//----------------------------------------
// Function: gpu_2d_continuous_cumsum1_kernel_2
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> Tmp : array<f32>;

struct PODArgs {
  cse_v1: i32,
  i: i32,
  m: i32,
  n: i32,
  packGridDimX: u32
}
@group(0) @binding(1) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(32, 4, 1)
fn gpu_2d_continuous_cumsum1_kernel_2(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var i_1 : i32 = 0; i_1 < 4i; i_1++) {
    let rmod : i32 = (((podArgs.n + (1i<<u32(((((i32(ceil(log2(f32(podArgs.n)))) / 9i) * 9i) - podArgs.cse_v1) - 9i)))) - 1i) % (1i<<u32(((((i32(ceil(log2(f32(podArgs.n)))) / 9i) * 9i) - podArgs.cse_v1) - 9i))));
    let rdiv : i32 = (((podArgs.n + (1i<<u32(((((i32(ceil(log2(f32(podArgs.n)))) / 9i) * 9i) - podArgs.cse_v1) - 9i)))) - 1i) / (1i<<u32(((((i32(ceil(log2(f32(podArgs.n)))) / 9i) * 9i) - podArgs.cse_v1) - 9i))));
    if (((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i_1 * 32i)) + i32(threadIdx.x)) < select((rdiv - 1i), rdiv, ((((1i<<u32(((((i32(ceil(log2(f32(podArgs.n)))) / 9i) * 9i) - podArgs.cse_v1) - 9i))) >= 0i) && (rmod >= 0i)) || (((1i<<u32(((((i32(ceil(log2(f32(podArgs.n)))) / 9i) * 9i) - podArgs.cse_v1) - 9i))) < 0i) && (rmod <= 0i))))) {
      var condval : f32;
      if ((0i < v__1)) {
        condval = Tmp[((((i32(blockIdx.y) * podArgs.n) + ((((i32(ceil(log2(f32(podArgs.n)))) / 9i) - podArgs.i) - 1i) * ((podArgs.n + 511i)>>9u))) + v__1) - 1i)];
} else {
        condval = 0.000000e+00f;
}
      Tmp[((((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i_1 * 32i)) + (i32(blockIdx.y) * podArgs.n)) + ((((i32(ceil(log2(f32(podArgs.n)))) / 9i) - podArgs.i) - 2i) * ((podArgs.n + 511i)>>9u))) + i32(threadIdx.x))] = (Tmp[((((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i_1 * 32i)) + (i32(blockIdx.y) * podArgs.n)) + ((((i32(ceil(log2(f32(podArgs.n)))) / 9i) - podArgs.i) - 2i) * ((podArgs.n + 511i)>>9u))) + i32(threadIdx.x))] + condval);
    }
  }
}
