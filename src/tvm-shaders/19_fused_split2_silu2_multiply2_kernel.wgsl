// [19] fused_split2_silu2_multiply2_kernel (shader #19, 25 lines)
//----------------------------------------
// Function: fused_split2_silu2_multiply2_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> T_multiply : array<f16>;
@group(0) @binding(1) var<storage, read> lv389 : array<f16>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn fused_split2_silu2_multiply2_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  T_multiply[((v__1 * 256i) + i32(threadIdx.x))] = (lv389[(((v__1 * 256i) + i32(threadIdx.x)) + 8192i)] * (lv389[((v__1 * 256i) + i32(threadIdx.x))] * (1.000000e+00h / (1.000000e+00h + exp((0.000000e+00h - lv389[((v__1 * 256i) + i32(threadIdx.x))]))))));
}
