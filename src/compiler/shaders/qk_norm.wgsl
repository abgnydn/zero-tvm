// QK_NORM — Qwen3 per-head RMSNorm of Q and K, IN PLACE, before RoPE.
//
//   q_head := q_head / sqrt(mean(q_head²) + RMS_EPS) * q_gamma   (per Q head)
//   k_head := k_head / sqrt(mean(k_head²) + RMS_EPS) * k_gamma   (per K head)
//   V untouched.
//
// Operates on the raw QKV projection output (layout [Q_DIM | KV_DIM | KV_DIM]
// per token) so the existing GQA-aware rope/kv_append (or a scratch-mode
// append) run unchanged afterwards: matmul → qk_norm → rope → append.
// Phi-3 (qkNorm=false) never dispatches this kernel — its pipeline is
// compiled for both specs but only the Qwen3 path schedules it.
//
// One 32-thread workgroup per (token, head); heads are ordered Q then K:
//   head_idx ∈ [0, HEADS)             → Q head (gamma = q_gamma)
//   head_idx ∈ [HEADS, HEADS+KV_HEADS) → K head (gamma = k_gamma)
// Grid: seq_len * (HEADS + KV_HEADS) workgroups. Each thread owns
// EPT = HEAD_DIM/32 dims; per-head reduction runs in f32 like rms_norm.wgsl.
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> qkv : array<f16>;
@group(0) @binding(1) var<storage, read> q_gamma : array<f16>;  // HEAD_DIM
@group(0) @binding(2) var<storage, read> k_gamma : array<f16>;  // HEAD_DIM

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

const EPT = HEAD_DIM / 32;          // dims per thread
const NORM_HEADS = HEADS + KV_HEADS; // normed heads per token (Q then K)

var<workgroup> red_buf : array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn qk_norm(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg_id) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);

  let seq_idx : i32 = wg_id / NORM_HEADS;
  let head_idx : i32 = wg_id - seq_idx * NORM_HEADS;
  if (seq_idx >= podArgs.seq_len) { return; }

  // Q heads live at [0, Q_DIM); K heads at [Q_DIM, Q_DIM + KV_DIM).
  var base : i32;
  if (head_idx < HEADS) {
    base = seq_idx * QKV_DIM + head_idx * HEAD_DIM;
  } else {
    base = seq_idx * QKV_DIM + Q_DIM + (head_idx - HEADS) * HEAD_DIM;
  }

  // Phase 1: this thread's dims + local sum of squares (f32).
  var vals : array<f32, EPT>;
  var sum_sq : f32 = 0.0;
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    let v : f32 = f32(qkv[base + tid * EPT + e]);
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

  let rms_inv : f32 = 1.0 / sqrt(red_buf[0] / f32(HEAD_DIM) + RMS_EPS);

  // Phase 2: normalize + gamma, write back in place (each thread only
  // rewrites its own dims — no cross-thread hazard).
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    let dim : i32 = tid * EPT + e;
    var g : f32;
    if (head_idx < HEADS) {
      g = f32(q_gamma[dim]);
    } else {
      g = f32(k_gamma[dim]);
    }
    qkv[base + dim] = f16(vals[e] * rms_inv * g);
  }
}
