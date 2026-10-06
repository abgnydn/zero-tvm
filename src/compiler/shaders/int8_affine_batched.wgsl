// INT8_AFFINE_BATCHED — batched GEMM over MLX-style 8-bit affine weights.
//
// The chunked-prefill twin of int8_affine_matvec: M token rows share one
// dispatch, out[b, r] = Σ x[b, i]·w[r, i] with w = scale*q + bias over groups
// of 64, q an unsigned byte, 4 per u32 word. Same factored inner loop as the
// GEMV (s·Σxq + b·Σx per group); the batch row rides grid y.
//
// Uniforms and bindings mirror int4_matmul_batched_dyn_affine on purpose —
// [out, inp, scales, wts, uni, biases] with uni = {K_PACKED, SCALES_PER_ROW,
// N, M_ROWS} — so the engine binds it through the same dynBg. K_PACKED here
// is K/4 u32 words (not K/8: a byte per value, not a nibble).
//
// Grid: one workgroup per (output row, batch row); rows past M_ROWS are
// guarded, never written, so a ragged final chunk leaves no garbage for the
// epilogue to read.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_buf : array<f16>;  // [CAP, N]
@group(0) @binding(1) var<storage, read> x : array<f16>;              // [CAP, K]
@group(0) @binding(2) var<storage, read> scales : array<f16>;         // [N, K/64]
@group(0) @binding(3) var<storage, read> w : array<u32>;              // [N, K/4]
@group(0) @binding(4) var<uniform> podArgs : PODArgs;
@group(0) @binding(5) var<storage, read> biases : array<f16>;         // [N, K/64]

struct PODArgs {
  K_PACKED : u32,   // u32 words per weight row (K/4)
  SCALES_PER_ROW : u32,  // K/64
  N : u32,          // output rows
  M_ROWS : u32,     // batch rows actually present (<= CAP)
}

var<workgroup> part : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn int8_affine_batched(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row : u32 = blockIdx.z * gridDim.x + blockIdx.x;
  let b : u32 = blockIdx.y;
  if (row >= podArgs.N || b >= podArgs.M_ROWS) { return; }
  let tid : u32 = threadIdx.x;
  let GPR : u32 = podArgs.K_PACKED / 16u;   // groups per row (16 u32 words each)
  let WPR : u32 = podArgs.K_PACKED;         // u32 words per row

  var acc : f32 = 0.0;
  // One group per thread per pass; a group is 64 values = 16 u32 words, so the
  // whole group lies inside one (scale, bias) pair and needs no cross-lane work.
  for (var g : u32 = tid; g < GPR; g = g + 64u) {
    let s : f32 = f32(scales[row * GPR + g]);
    let b_ : f32 = f32(biases[row * GPR + g]);
    var dot : f32 = 0.0;
    var xs : f32 = 0.0;
    for (var wi : u32 = 0u; wi < 16u; wi = wi + 1u) {
      let word : u32 = w[row * WPR + g * 16u + wi];
      let base : u32 = b * podArgs.K_PACKED * 4u + g * 64u + wi * 4u;
      for (var n : u32 = 0u; n < 4u; n = n + 1u) {
        let v : f32 = f32(x[base + n]);
        dot = dot + v * f32((word >> (8u * n)) & 255u);
        xs = xs + v;
      }
    }
    acc = acc + s * dot + b_ * xs;
  }
  part[tid] = acc;
  workgroupBarrier();
  for (var st : u32 = 32u; st > 0u; st = st >> 1u) {
    if (tid < st) { part[tid] = part[tid] + part[tid + st]; }
    workgroupBarrier();
  }
  if (tid == 0u) { out_buf[b * podArgs.N + row] = f16(part[0]); }
}
