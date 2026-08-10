// MLA_KV_WRITE — turn kv_a_proj's one output row into this token's cache entry.
//
//   in  : kva  [L + R]     (kv_a_proj_with_mqa output, one token)
//   out : cache_c   [.., L] at row `position`  — RMSNorm(kva[0..L], gamma)
//         cache_kpe [.., R] at row `position`  — RoPE(kva[L..L+R])
//
// This is the whole of MLA's cache write: 576 values a token where MHA would
// store heads * (nope + v) = 4096. The latent is normalised (kv_a_layernorm)
// and the 64-wide RoPE key is not — the norm covers the latent half ONLY, and
// normalising the pe half as well is a change no test outside the reference
// bundle would flag.
//
// No existing kernel does this:
//   - rms_norm.wgsl hardwires the row to the prelude's D with a `D / 64`
//     blocked loop. Pointed at a 512-wide latent under a spec whose D is 2048
//     it reads 2048 elements; WGSL clamps the out-of-range storage reads
//     rather than faulting, so it returns a finite, wrong norm.
//   - rope.wgsl rotates the FIRST ROTARY_DIM of each head and expects a head
//     axis; k_pe is one un-headed row and its rotated slice is the LAST R.
//   - kv_append.wgsl addresses
//     `page_no*KV_PAGE_STRIDE + head*HEAD_PAGE_STRIDE + slot*HEAD_DIM + dim`
//     and writes a V region. MLA's cache has neither a head axis nor a V.
//
// As in mla_q_split, the rotation is plain half-split RoPE because the loader
// de-interleaved kv_a_proj's pe ROWS at load; fed an unpermuted kv_a_proj this
// rotates the wrong pairs and produces fluent nonsense.
//
// ADDRESSING: `position * L` and `position * R`, flat. That is the paged
// layout with the page table collapsed — `(t/pageSize)*(pageSize*L) +
// (t%pageSize)*L + i` IS `t*L + i` for the identity table the engine writes
// once and never permutes. Real paging (eviction, block reuse, a shared prefix
// pool) would have to teach this kernel and mla_scores/mla_combine the table;
// none of them read one today.
//
// Grid: (1, 1, 1). One 64-lane workgroup — the whole reduction is 512 values,
// and a single token's cache row is not worth splitting further.

enable f16;

@group(0) @binding(0) var<storage, read> kva : array<f16>;               // [L + R]
@group(0) @binding(1) var<storage, read> gamma : array<f16>;             // [L] kv_a_layernorm
@group(0) @binding(2) var<storage, read> inv_freq : array<f32>;          // R/2 entries
@group(0) @binding(3) var<storage, read> position_map : array<i32>;
@group(0) @binding(4) var<storage, read_write> cache_c : array<f16>;     // [maxContext, L]
@group(0) @binding(5) var<storage, read_write> cache_kpe : array<f16>;   // [maxContext, R]

struct PODArgs {
  L: u32,   // kv_lora_rank — the latent width, and the norm's row length
  R: u32    // qk_rope_head_dim — the shared key's width; pairs sit R/2 apart
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

var<workgroup> red : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn mla_kv_write(@builtin(local_invocation_id) threadIdx : vec3<u32>) {
  let tid : u32 = threadIdx.x;

  // ── sum of squares over the latent half, f32 (the accumulation is 512 terms) ──
  var sum_sq : f32 = 0.0;
  for (var i : u32 = tid; i < podArgs.L; i = i + 64u) {
    let v : f32 = f32(kva[i]);
    sum_sq = sum_sq + v * v;
  }
  red[tid] = sum_sq;
  workgroupBarrier();
  // The barrier sits at LOOP level, outside the `if` — `st` is a plain local
  // counting down from 32, so every invocation runs the same seven iterations
  // and reaches every barrier. This kernel has no early return at all, and in
  // particular none predicated on `position_map`: a storage load is not a
  // uniform value, so branching on one before a barrier is the classic way to
  // put workgroupBarrier() in non-uniform control flow. `position` is read
  // below and used only as an address.
  for (var st : u32 = 32u; st > 0u; st = st >> 1u) {
    if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
    workgroupBarrier();
  }
  // mean over L, not over L + R: the reference norms the latent slice alone.
  let rms_inv : f32 = 1.0 / sqrt(red[0] / f32(podArgs.L) + RMS_EPS);

  let position : u32 = u32(position_map[0]);

  for (var i : u32 = tid; i < podArgs.L; i = i + 64u) {
    cache_c[position * podArgs.L + i] = f16(f32(kva[i]) * rms_inv * f32(gamma[i]));
  }

  // ── the shared RoPE key: rotated, NOT normed, written un-headed ──
  let posf : f32 = f32(position);
  let half : u32 = podArgs.R >> 1u;
  for (var j : u32 = tid; j < podArgs.R; j = j + 64u) {
    let freq : f32 = posf * inv_freq[j % half];
    var pair : f32;
    if (j < half) {
      pair = -f32(kva[podArgs.L + j + half]);
    } else {
      pair = f32(kva[podArgs.L + j - half]);
    }
    cache_kpe[position * podArgs.R + j] =
      f16(cos(freq) * f32(kva[podArgs.L + j]) + sin(freq) * pair);
  }
}
