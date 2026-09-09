// ADD3 + RMSNORM — three-way residual merge + normalize (?fuseprologue=1).
//
// residual[i] = (A[i] + B[i]) + C[i]   (store to residual buffer)
// output[i]  = rmsnorm(residual) * gamma[i]
//
// Companion to fused_ffn[_tiled_sg]_prologue.wgsl: when the FFN-entry
// add_norm (addNorm1) is folded into the FFN prologue, its residual sum
// (residual + oproj_delta) never materializes in a buffer — so the layer
// tail must merge THREE terms: the incoming residual (A), the o-proj delta
// (B), and the FFN-down delta (C). The additions are sequential f16
// (round(round(A+B)+C)), matching the two-step add_norm ping-pong exactly.
//
// Same structure as add_norm.wgsl otherwise: 256 threads, D/256 elements
// each, one workgroup per token.
//
// Measured 2026-07-25 (M2 Max): -13.7% end-to-end — falsified on Apple
// (the per-WG RMSNorm recompute costs more than the saved dispatch
// bubbles); kept for A/B on other GPUs (see BENCH.md "Measured 2026-07-25").
//
// Model-shape constants (D, ...) are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read> A : array<f16>;
@group(0) @binding(1) var<storage, read> B : array<f16>;
@group(0) @binding(2) var<storage, read> C : array<f16>;
@group(0) @binding(3) var<storage, read> gamma : array<f16>;
@group(0) @binding(4) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(5) var<storage, read_write> residual : array<f16>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn add3_norm(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(batch) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);
  let base : i32 = batch * D;

  // Phase 1: three-way add (sequential f16 rounding — see header) and
  // local sum of squares.
  var local_vals : array<f16, D / 256>;
  var sum_sq : f32 = 0.0;

  for (var i : i32 = 0; i < D / 256; i = i + 1) {
    let idx : i32 = tid + i * 256;
    let val : f16 = (A[base + idx] + B[base + idx]) + C[base + idx];
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

  let rms_inv : f32 = 1.0 / sqrt(red_buf[0] / f32(D) + RMS_EPS);

  // Phase 2: normalize
  for (var i : i32 = 0; i < D / 256; i = i + 1) {
    let idx : i32 = tid + i * 256;
    output_buf[base + idx] = f16(f32(local_vals[i]) * rms_inv * f32(gamma[idx]));
  }
}
