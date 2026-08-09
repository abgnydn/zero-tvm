// MLA_COMBINE — stage 2: softmax the scores, then take the weighted sum of the
// LATENT cache. Output is one 512-wide latent context per head, which the
// caller re-expands to that head's 128 values through kv_b_proj's V half:
//
//   out_h = W_v_h ( sum_t p_t c_t )     rather than     sum_t p_t (W_v_h c_t)
//
// Same value, and the left form touches the up-projection once per token
// instead of once per cached position — the output side of the same trick
// mla_scores plays on the query side.
//
// Softmax is written back over the scores buffer in place. The alternative is
// recomputing exp() inside the accumulation loop, where it would run once per
// (position, latent element) — L/64 times more often than it needs to.
//
// Grid: one workgroup per head.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_lat : array<f32>;   // [heads, L]
@group(0) @binding(1) var<storage, read_write> scores : array<f32>;    // [heads, T] in, probabilities out
@group(0) @binding(2) var<storage, read> cache_c : array<f16>;         // [T, L]

struct PODArgs {
  L: u32,   // kv_lora_rank
  T: u32    // cached positions
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

var<workgroup> red : array<f32, 64>;
var<workgroup> shMax : f32;
var<workgroup> shSum : f32;

@compute @workgroup_size(64, 1, 1)
fn mla_combine(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let h : u32 = blockIdx.x;
  let tid : u32 = threadIdx.x;
  let base : u32 = h * podArgs.T;

  // ── running max, so exp() never sees a positive argument ──
  var m : f32 = -3.0e38;
  for (var t : u32 = tid; t < podArgs.T; t = t + 64u) { m = max(m, scores[base + t]); }
  red[tid] = m;
  workgroupBarrier();
  for (var st : u32 = 32u; st > 0u; st = st >> 1u) {
    if (tid < st) { red[tid] = max(red[tid], red[tid + st]); }
    workgroupBarrier();
  }
  if (tid == 0u) { shMax = red[0]; }
  workgroupBarrier();
  let peak : f32 = shMax;

  // ── exponentiate in place, and total ──
  var su : f32 = 0.0;
  for (var t : u32 = tid; t < podArgs.T; t = t + 64u) {
    let e : f32 = exp(scores[base + t] - peak);
    scores[base + t] = e;
    su = su + e;
  }
  // scores lives in the STORAGE address space and this is the one place in the
  // repo where invocations hand data to each other through it: lane tid writes
  // exp() for its own positions above, and every lane reads ALL positions in
  // the accumulation below. workgroupBarrier() orders the WORKGROUP address
  // space only — that is the entire distinction between it and storageBarrier()
  // — so it supplies the execution sync but not the visibility. Tint lowers it
  // to threadgroup_barrier(mem_threadgroup) on Metal and
  // GroupMemoryBarrierWithGroupSync() on D3D12, neither of which orders device
  // memory. It happens to pass on Metal today; that is luck about the backend,
  // not correctness.
  storageBarrier();
  red[tid] = su;
  workgroupBarrier();
  for (var st : u32 = 32u; st > 0u; st = st >> 1u) {
    if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
    workgroupBarrier();
  }
  if (tid == 0u) { shSum = red[0]; }
  workgroupBarrier();
  let inv : f32 = 1.0 / shSum;

  // ── weighted sum of the cache. Each thread owns a stride of the latent, so
  // the inner walk over positions reads one column and the workgroup together
  // reads whole rows. No reduction here: every output element belongs to
  // exactly one thread.
  for (var i : u32 = tid; i < podArgs.L; i = i + 64u) {
    var acc : f32 = 0.0;
    for (var t : u32 = 0u; t < podArgs.T; t = t + 1u) {
      acc = acc + scores[base + t] * f32(cache_c[t * podArgs.L + i]);
    }
    out_lat[h * podArgs.L + i] = acc * inv;
  }
}
