// RMSNORM — normalize hidden state.
// output[i] = (input[i] / rms) * gamma[i]
// where rms = sqrt(mean(input^2) + eps)
//
// All accumulation in f32 (matches TVM's rms_norm2_kernel).
// D=3072, 64 threads, each handles 48 elements.

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> gamma : array<f16>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn rms_norm(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(batch) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);
  let base : i32 = batch * 3072;

  // Phase 1: compute sum of squares in f32
  var sum_sq : f32 = 0.0;
  for (var i : i32 = 0; i < 48; i = i + 1) {
    let idx : i32 = tid * 48 + i;
    let val : f32 = f32(input_buf[base + idx]);
    sum_sq = sum_sq + val * val;
  }

  // Tree reduce
  red_buf[tid] = sum_sq;
  workgroupBarrier();
  if (tid < 32) { red_buf[tid] = red_buf[tid] + red_buf[tid + 32]; }
  workgroupBarrier();
  if (tid < 16) { red_buf[tid] = red_buf[tid] + red_buf[tid + 16]; }
  workgroupBarrier();
  if (tid < 8) { red_buf[tid] = red_buf[tid] + red_buf[tid + 8]; }
  workgroupBarrier();
  if (tid < 4) { red_buf[tid] = red_buf[tid] + red_buf[tid + 4]; }
  workgroupBarrier();
  if (tid < 2) { red_buf[tid] = red_buf[tid] + red_buf[tid + 2]; }
  workgroupBarrier();
  if (tid < 1) { red_buf[tid] = red_buf[tid] + red_buf[tid + 1]; }
  workgroupBarrier();

  let rms_inv : f32 = 1.0 / sqrt(red_buf[0] / 3072.0 + 1e-5);

  // Phase 2: normalize and scale
  for (var i : i32 = 0; i < 48; i = i + 1) {
    let idx : i32 = tid * 48 + i;
    output_buf[base + idx] = f16(f32(input_buf[base + idx]) * rms_inv * f32(gamma[idx]));
  }
}
