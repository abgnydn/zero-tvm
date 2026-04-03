// [5] take_kernel (shader #5, 28 lines)
//----------------------------------------
// Function: take_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> T_take : array<f16>;
@group(0) @binding(1) var<storage, read> logit_positions : array<i32>;
@group(0) @binding(2) var<storage, read> rms_norm194 : array<f16>;

struct PODArgs {
  batch_size: i32,
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn take_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  T_take[((v__1 * 256i) + i32(threadIdx.x))] = rms_norm194[(((logit_positions[(v__1 / 12i)] * 3072i) + ((v__1 % 12i) * 256i)) + i32(threadIdx.x))];
}
