// MLA_Q_SPLIT — cut q_proj's output into the two halves MLA scores separately,
// and rotate the second one.
//
//   in  : q       [heads, nope + rope]   (q_proj output, one token)
//   out : q_nope  [heads, nope]          (verbatim — feeds mla_proj's K half)
//         q_pe    [heads, rope]          (RoPE'd — dotted against the shared key)
//
// The copy of q_nope is not redundant: mla_proj addresses its input as
// `h * K` for a contiguous [heads, K] block, and here head h's nope values
// start at `h * (nope + rope)`. Binding the projection output directly would
// read the right number of f16s from the wrong place for every head but the
// first — well-formed, in-bounds, wrong model.
//
// THE ROTATION IS ORDINARY HALF-SPLIT RoPE, and that is a property of the
// WEIGHTS, not of this kernel. DeepSeek's apply_rotary_pos_emb de-interleaves
// before rotate_half, i.e. q_pe is STORED as [a0,b0,a1,b1,…] where we pair
// (j, j + rope/2). The loader applies that de-interleave to q_proj's pe ROWS
// once at load — a permutation of a projection's output commutes with
// permuting the matrix's rows — so by the time the values reach this kernel
// the interleave is gone. If this kernel is ever fed an UNPERMUTED q_proj it
// rotates the wrong pairs: no error, fluent output, a quietly broken model.
//
// No existing kernel does this:
//   - rope.wgsl computes `dim_idx = within % HEAD_DIM` and rotates
//     `dim_idx < ROTARY_DIM`, i.e. the FIRST 64 of each head. MLA's pe is the
//     LAST 64 of 192. It also splits a Q‖K‖V concatenation MLA does not have.
//   - qk_norm_rope_append.wgsl normalises first and writes into a paged K/V
//     cache with a head axis, neither of which applies here.
//
// No barriers, deliberately: every RoPE pair partner is read from the INPUT
// buffer, which this kernel never writes, so there is nothing to stage in
// workgroup memory and nothing to synchronise. qk_norm_rope_append needs the
// staging only because it rotates values it has just normalised.
//
// Grid: (heads, 1, 1) — grid x IS the head index, one 64-lane workgroup each,
// and it is exact, so there is no partial workgroup to guard against.

enable f16;

@group(0) @binding(0) var<storage, read> q_in : array<f16>;          // [heads, N + R]
@group(0) @binding(1) var<storage, read> inv_freq : array<f32>;      // R/2 entries
@group(0) @binding(2) var<storage, read> position_map : array<i32>;
@group(0) @binding(3) var<storage, read_write> q_nope : array<f16>;  // [heads, N]
@group(0) @binding(4) var<storage, read_write> q_pe : array<f16>;    // [heads, R]

struct PODArgs {
  N: u32,   // qk_nope_head_dim — the un-rotated head slice
  R: u32    // qk_rope_head_dim — the rotated slice; pairs sit R/2 apart
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(64, 1, 1)
fn mla_q_split(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let h : u32 = blockIdx.x;
  let tid : u32 = threadIdx.x;
  let base : u32 = h * (podArgs.N + podArgs.R);

  // Strided rather than blocked (tid * N/64 + i) so nothing here assumes the
  // head widths divide 64 — the mla_* family stays dim-agnostic because every
  // dim arrives in the uniform, not in the prelude.
  for (var i : u32 = tid; i < podArgs.N; i = i + 64u) {
    q_nope[h * podArgs.N + i] = q_in[base + i];
  }

  let posf : f32 = f32(position_map[0]);
  let half : u32 = podArgs.R >> 1u;
  for (var j : u32 = tid; j < podArgs.R; j = j + 64u) {
    // inv_freq holds R/2 entries and the angle repeats across the split, so
    // both members of a pair share one frequency — j % half, not j.
    let freq : f32 = posf * inv_freq[j % half];
    var pair : f32;
    if (j < half) {
      pair = -f32(q_in[base + podArgs.N + j + half]);
    } else {
      pair = f32(q_in[base + podArgs.N + j - half]);
    }
    // f32 trig and mix, f16 store — the same shape as rope.wgsl. cos/sin at
    // f16 argument would quantise the angle itself, which grows with position.
    q_pe[h * podArgs.R + j] =
      f16(cos(freq) * f32(q_in[base + podArgs.N + j]) + sin(freq) * pair);
  }
}
