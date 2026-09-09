// MLA_SCORES — stage 1 of multi-head latent attention: one logit per (head,
// cached position), scored against the LATENT cache rather than per-head K.
//
// Ordinary attention stores K and V for every head: DeepSeek-V2-Lite would
// cache 16 heads x 128 x 2 = 4096 values a token. MLA stores one 512-wide
// latent plus ONE 64-wide RoPE key shared by every head — 576 values, 7x less,
// which is the entire reason to want it in a browser where context is bounded
// by the cache, not by the weights.
//
// The trick that makes scoring against it cheap is algebraic, not numerical:
//
//   q_nope . K_nope = q_nope . (W_k c) = (W_k^T q_nope) . c
//
// so the query is pushed into latent space ONCE per token (a small per-head
// matmul against kv_b_proj's K half, done before this kernel) and then dotted
// straight against the cache. K is never materialised. The 64-wide RoPE part
// is a separate dot against the shared key and simply adds:
//
//   score = ( q_lat . c_t  +  q_pe . k_pe_t ) * scale
//
// Grid: (position, head). One workgroup per score, 64 threads walking the
// 512 + 64 elements and reducing in f32 — the accumulation is over 576 terms,
// which is well past what f16 can add without drifting.

enable f16;

@group(0) @binding(0) var<storage, read_write> scores : array<f32>;    // [heads, T]
@group(0) @binding(1) var<storage, read> q_lat : array<f16>;           // [heads, L]
@group(0) @binding(2) var<storage, read> q_pe : array<f16>;            // [heads, R]
@group(0) @binding(3) var<storage, read> cache_c : array<f16>;         // [T, L]
@group(0) @binding(4) var<storage, read> cache_kpe : array<f16>;       // [T, R]

struct PODArgs {
  L: u32,       // kv_lora_rank — the latent width
  R: u32,       // qk_rope_head_dim — the shared key's width
  T: u32,       // cached positions
  scale: f32    // 1/sqrt(qk_nope + qk_rope); note the DENOMINATOR is the full
                // head width, not the latent width — MLA scores stand in for a
                // dot product that never happens at that width.
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> red : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn mla_scores(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let t : u32 = blockIdx.x;
  let h : u32 = blockIdx.y;
  if (t >= podArgs.T) { return; }
  let tid : u32 = threadIdx.x;

  var acc : f32 = 0.0;
  // Latent half. Threads take strided elements so a wavefront's loads land on
  // consecutive addresses of both the query and the cache row.
  for (var i : u32 = tid; i < podArgs.L; i = i + 64u) {
    acc = acc + f32(q_lat[h * podArgs.L + i]) * f32(cache_c[t * podArgs.L + i]);
  }
  // RoPE half. cache_kpe has no head index — every head reads the same row.
  for (var i : u32 = tid; i < podArgs.R; i = i + 64u) {
    acc = acc + f32(q_pe[h * podArgs.R + i]) * f32(cache_kpe[t * podArgs.R + i]);
  }

  red[tid] = acc;
  workgroupBarrier();
  for (var st : u32 = 32u; st > 0u; st = st >> 1u) {
    if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
    workgroupBarrier();
  }
  if (tid == 0u) { scores[h * podArgs.T + t] = red[0] * podArgs.scale; }
}
