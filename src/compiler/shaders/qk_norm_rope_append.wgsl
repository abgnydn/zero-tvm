// QK_NORM_ROPE_APPEND — fused per-head Q/K RMSNorm + RoPE + paged KV append.
//
// The qkNorm decode path (Qwen3) used to run three dispatches after the QKV
// matmul: qk_norm (in place) → rope (split + rotate) → kv_append (paged
// write). The blocker for folding all of it into qkv_fused is that the norm
// needs the WHOLE head vector before any output can be produced, while
// qkv_fused computes one RoPE pair per workgroup. This kernel keeps the
// matmul separate and fuses everything after it instead:
//
//   int4_matmul (qkv scratch) → qk_norm_rope_append   (2 dispatches, was 4)
//
// One 32-thread workgroup per (token, head); heads ordered Q, K, V:
//   head_idx ∈ [0, HEADS)                   → Q: norm(q_gamma) + RoPE → q_out
//   head_idx ∈ [HEADS, HEADS+KV_HEADS)      → K: norm(k_gamma) + RoPE → pages
//   head_idx ∈ [.., HEADS+2*KV_HEADS)       → V: raw copy → pages V region
// Grid: seq_len * (HEADS + 2*KV_HEADS) workgroups.
//
// The norm reduction runs per head in f32 (same math as qk_norm.wgsl); the
// normalized head vector is staged in workgroup memory so every thread can
// read its RoPE pair partner (±HALF_ROTARY) regardless of which thread owns
// it. Partial-RoPE aware like rope.wgsl: dims >= ROTARY_DIM pass through
// (full-rotation specs have ROTARY_DIM == HEAD_DIM, so Qwen3 rotates all
// dims). Page layout matches kv_append.wgsl exactly.
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> q_out : array<f16>;    // seq × Q_DIM
@group(0) @binding(1) var<storage, read_write> pages : array<f16>;    // paged KV cache
@group(0) @binding(2) var<storage, read> qkv : array<f16>;            // seq × QKV_DIM (matmul out)
@group(0) @binding(3) var<storage, read> q_gamma : array<f16>;        // HEAD_DIM
@group(0) @binding(4) var<storage, read> k_gamma : array<f16>;        // HEAD_DIM
@group(0) @binding(5) var<storage, read> position_map : array<i32>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const EPT = HEAD_DIM / 32;                 // dims per thread
const FUSE_HEADS = HEADS + 2 * KV_HEADS;   // WGs per token (Q + K + V heads)

var<workgroup> red_buf : array<f32, 32>;
var<workgroup> norm_sh : array<f32, HEAD_DIM>;  // normalized head (RoPE pair exchange)

@compute @workgroup_size(32, 1, 1)
fn qk_norm_rope_append(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg_id) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);
  let seq_idx : i32 = wg_id / FUSE_HEADS;
  let head_idx : i32 = wg_id - seq_idx * FUSE_HEADS;
  if (seq_idx >= podArgs.seq_len) { return; }

  let position : i32 = position_map[seq_idx];
  let page_no : i32 = position / PAGE_SIZE;
  let slot : i32 = position - page_no * PAGE_SIZE;

  // V heads: no norm, no RoPE — straight copy into the V page region
  // (uniform per-WG branch: every barrier below is on the Q/K side only).
  if (head_idx >= HEADS + KV_HEADS) {
    let vh : i32 = head_idx - HEADS - KV_HEADS;
    let src : i32 = seq_idx * QKV_DIM + Q_DIM + KV_DIM + vh * HEAD_DIM;
    let dst : i32 = page_no * KV_PAGE_STRIDE + vh * HEAD_PAGE_STRIDE + slot * HEAD_DIM
                    + V_PAGE_OFFSET;
    for (var e : i32 = 0; e < EPT; e = e + 1) {
      let dim : i32 = tid * EPT + e;
      pages[dst + dim] = qkv[src + dim];
    }
    return;
  }

  // Q heads at [0, Q_DIM); K heads at [Q_DIM, Q_DIM + KV_DIM).
  let is_q : bool = head_idx < HEADS;
  var base : i32;
  if (is_q) {
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

  // Tree reduce across 32 threads (same as qk_norm.wgsl / rms_norm.wgsl).
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

  // Phase 2: normalize + gamma into shared memory so RoPE can read the pair
  // partner at ±HALF_ROTARY without caring which thread owns it.
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    let dim : i32 = tid * EPT + e;
    var g : f32;
    if (is_q) { g = f32(q_gamma[dim]); } else { g = f32(k_gamma[dim]); }
    norm_sh[dim] = vals[e] * rms_inv * g;
  }
  workgroupBarrier();

  // Phase 3: RoPE on the rotary slice (dims >= ROTARY_DIM pass through) and
  // write to the destination — q_out for Q heads, paged cache for K heads.
  let posf : f32 = f32(position);
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    let dim : i32 = tid * EPT + e;
    var out_val : f32 = norm_sh[dim];
    if (dim < ROTARY_DIM) {
      let freq : f32 = posf / pow(ROPE_THETA, f32((dim % HALF_ROTARY) * 2) / f32(ROTARY_DIM));
      var pair : f32;
      if (dim < HALF_ROTARY) {
        pair = -norm_sh[dim + HALF_ROTARY];
      } else {
        pair = norm_sh[dim - HALF_ROTARY];
      }
      out_val = cos(freq) * out_val + sin(freq) * pair;
    }
    if (is_q) {
      q_out[seq_idx * Q_DIM + head_idx * HEAD_DIM + dim] = f16(out_val);
    } else {
      let kh : i32 = head_idx - HEADS;
      let k_base : i32 = page_no * KV_PAGE_STRIDE + kh * HEAD_PAGE_STRIDE + slot * HEAD_DIM;
      pages[k_base + dim] = f16(out_val);
    }
  }
}
