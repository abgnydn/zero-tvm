// SPLIT-K ATTENTION COMBINE — merges attention_splitk[_sg] partials.
//
// One 32-thread workgroup per head (dispatch (1, HEADS, 1)). Each partial
// carries the unnormalized online-softmax state (m_p, d_p, o_p[HEAD_DIM])
// for its KV slice; the exact merge is
//
//   M  = max over non-empty p of m_p
//   D  = Σ_p d_p · exp(m_p − M)
//   out = ( Σ_p o_p · exp(m_p − M) ) / D
//
// which equals the single-pass online softmax up to fp rounding. Partials
// with d_p == 0 (empty partitions when kv_len covers fewer pages than
// num_splits) are skipped — they carry no probability mass and their
// m = -50000 sentinel must not enter the max.
//
// m_p/d_p are per-partition scalars, so every thread redundantly walks the
// (tiny) partition list; each thread then owns 3 output dims (32 × 3 = 96).
// No cross-thread reduction is needed.
//
// Measured 2026-07-25 (M2 Max): ~+3% end-to-end at short context; stays
// opt-in until a long-context A/B (see BENCH.md "Measured 2026-07-25").
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read> partials : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf : array<f16>;

struct PODArgs { num_splits: u32 }
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

const PARTIAL_STRIDE = HEAD_DIM + 2;   // per-(head, partition) f32 stride: m, d, o[HEAD_DIM]

@compute @workgroup_size(32, 1, 1)
fn attention_combine(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);
  let n : i32 = i32(podArgs.num_splits);

  // Pass 1: global max over non-empty partitions.
  var gm : f32 = -50000.0;
  for (var p : i32 = 0; p < n; p = p + 1) {
    let base : i32 = (head * n + p) * PARTIAL_STRIDE;
    if (partials[base + 1] > 0.0) { gm = max(gm, partials[base]); }
  }

  // Pass 2: rescale and accumulate denominators + this thread's 3 dims.
  var gd : f32 = 0.0;
  var o0 : f32 = 0.0;
  var o1 : f32 = 0.0;
  var o2 : f32 = 0.0;
  for (var p : i32 = 0; p < n; p = p + 1) {
    let base : i32 = (head * n + p) * PARTIAL_STRIDE;
    let d_p : f32 = partials[base + 1];
    if (d_p == 0.0) { continue; }   // empty partition — no mass, skip
    let w : f32 = exp(partials[base] - gm);
    gd = gd + d_p * w;
    o0 = o0 + partials[base + 2 + tid * 3] * w;
    o1 = o1 + partials[base + 2 + tid * 3 + 1] * w;
    o2 = o2 + partials[base + 2 + tid * 3 + 2] * w;
  }

  // Same "write only when there is mass" behavior as attention.wgsl.
  if (gd > 0.0) {
    let inv_d : f32 = 1.0 / gd;
    output_buf[head * HEAD_DIM + tid * 3] = f16(o0 * inv_d);
    output_buf[head * HEAD_DIM + tid * 3 + 1] = f16(o1 * inv_d);
    output_buf[head * HEAD_DIM + tid * 3 + 2] = f16(o2 * inv_d);
  }
}
