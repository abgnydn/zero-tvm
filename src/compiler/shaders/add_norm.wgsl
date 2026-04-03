// ADD + RMSNORM — residual connection + normalize.
// residual[i] = A[i] + B[i]          (store to residual buffer)
// output[i] = rmsnorm(residual) * gamma[i]
//
// Fuses TVM's fuse_add_norm_decode_kernel.
// 256 threads, each handles 12 elements of D=3072.

enable f16;

@group(0) @binding(0) var<storage, read> A : array<f16>;
@group(0) @binding(1) var<storage, read> B : array<f16>;
@group(0) @binding(2) var<storage, read> gamma : array<f16>;
@group(0) @binding(3) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(4) var<storage, read_write> residual : array<f16>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn add_norm(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(batch) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);
  let base : i32 = batch * 3072;

  // Phase 1: add residual and compute local sum of squares
  var local_vals : array<f16, 12>;
  var sum_sq : f32 = 0.0;

  for (var i : i32 = 0; i < 12; i = i + 1) {
    let idx : i32 = tid + i * 256;
    let val : f16 = A[base + idx] + B[base + idx];
    local_vals[i] = val;
    residual[base + idx] = val;
    sum_sq = sum_sq + f32(val) * f32(val);
  }

  // Tree reduce
  red_buf[tid] = sum_sq;
  workgroupBarrier();
  if (tid < 128) { red_buf[tid] = red_buf[tid] + red_buf[tid + 128]; } workgroupBarrier();
  if (tid < 64) { red_buf[tid] = red_buf[tid] + red_buf[tid + 64]; } workgroupBarrier();
  if (tid < 32) { red_buf[tid] = red_buf[tid] + red_buf[tid + 32]; } workgroupBarrier();
  if (tid < 16) { red_buf[tid] = red_buf[tid] + red_buf[tid + 16]; } workgroupBarrier();
  if (tid < 8) { red_buf[tid] = red_buf[tid] + red_buf[tid + 8]; } workgroupBarrier();
  if (tid < 4) { red_buf[tid] = red_buf[tid] + red_buf[tid + 4]; } workgroupBarrier();
  if (tid < 2) { red_buf[tid] = red_buf[tid] + red_buf[tid + 2]; } workgroupBarrier();
  if (tid < 1) { red_buf[tid] = red_buf[tid] + red_buf[tid + 1]; } workgroupBarrier();

  let rms_inv : f32 = 1.0 / sqrt(red_buf[0] / 3072.0 + 1e-5);

  // Phase 2: normalize
  for (var i : i32 = 0; i < 12; i = i + 1) {
    let idx : i32 = tid + i * 256;
    output_buf[base + idx] = f16(f32(local_vals[i]) * rms_inv * f32(gamma[idx]));
  }
}
