// GDN_NORM_OUT — Qwen3.5 GatedDeltaNet per-head gated RMSNorm + z-gate.
//
// Ground truth:
//   HF modeling_qwen3_5.Qwen3_5RMSNormGated ("Norm before gate"):
//     y   = x * rsqrt(mean(x²) + eps) * weight     (x in f32)
//     out = y * silu(z)                            (z in f32)
//   applied per v-head over GDN_HEAD_V dims (norm.weight is [GDN_HEAD_V],
//   shared across heads); z is the in_proj_z projection reshaped per head.
//   MLC qwen35_model: out = RMSNorm(out_recurrent) * silu(z) — identical.
//   NOTE: unlike the plain Qwen3_5RMSNorm layers, this gamma has NO +1.0
//   offset (qwen35_loader._is_rmsnorm_weight excludes linear_attn.norm).
//
// Input is the f32 recurrence output (gdn_recur); output is f16, ready to be
// the int4 out_proj matmul input. One 32-thread workgroup per (token, v-head)
// like qk_norm.wgsl; each thread owns EPT = GDN_HEAD_V/32 dims; the reduction
// runs in f32.
//
// Grid: seq_len * GDN_V_HEADS workgroups (podArgs.packGridDimX).
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_buf : array<f16>;  // seq * GDN_V_DIM
@group(0) @binding(1) var<storage, read> recur_out : array<f32>;      // seq * GDN_V_DIM
@group(0) @binding(2) var<storage, read> gamma : array<f16>;          // GDN_HEAD_V
@group(0) @binding(3) var<storage, read> z_gate : array<f16>;         // seq * z_stride

struct PODArgs {
  seq_len: i32,
  z_stride: i32,      // f16 elements between consecutive tokens' z rows
                      // (GDN_V_DIM when tightly packed — the decode region
                      // view; gdnProjRows for chunked prefill's batched
                      // fused-projection buffer)
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const GDN_EPT = GDN_HEAD_V / 32;   // dims per thread

var<workgroup> red_buf : array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn gdn_norm_out(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg_id) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);
  let seq_idx : i32 = wg_id / GDN_V_HEADS;
  let head : i32 = wg_id - seq_idx * GDN_V_HEADS;
  if (seq_idx >= podArgs.seq_len) { return; }

  let base : i32 = seq_idx * GDN_V_DIM + head * GDN_HEAD_V;
  let z_base : i32 = seq_idx * podArgs.z_stride + head * GDN_HEAD_V;

  // Phase 1: this thread's dims + local sum of squares.
  var vals : array<f32, GDN_EPT>;
  var sum_sq : f32 = 0.0;
  for (var e : i32 = 0; e < GDN_EPT; e = e + 1) {
    let v : f32 = recur_out[base + tid * GDN_EPT + e];
    vals[e] = v;
    sum_sq = sum_sq + v * v;
  }

  // Tree reduce across 32 threads.
  red_buf[tid] = sum_sq;
  workgroupBarrier();
  if (tid < 16) { red_buf[tid] = red_buf[tid] + red_buf[tid + 16]; }
  workgroupBarrier();
  if (tid < 8)  { red_buf[tid] = red_buf[tid] + red_buf[tid + 8]; }
  workgroupBarrier();
  if (tid < 4)  { red_buf[tid] = red_buf[tid] + red_buf[tid + 4]; }
  workgroupBarrier();
  if (tid < 2)  { red_buf[tid] = red_buf[tid] + red_buf[tid + 2]; }
  workgroupBarrier();
  if (tid < 1)  { red_buf[tid] = red_buf[tid] + red_buf[tid + 1]; }
  workgroupBarrier();

  let rms_inv : f32 = 1.0 / sqrt(red_buf[0] / f32(GDN_HEAD_V) + RMS_EPS);

  // Phase 2: normalize · gamma · silu(z), write f16.
  for (var e : i32 = 0; e < GDN_EPT; e = e + 1) {
    let dim : i32 = tid * GDN_EPT + e;
    let z : f32 = f32(z_gate[z_base + dim]);
    let gated : f32 = vals[e] * rms_inv * f32(gamma[dim]) * (z / (1.0 + exp(-z)));
    out_buf[base + dim] = f16(gated);
  }
}
