// FUSED_FFN_TILED_SG WITH ADD+RMSNORM PROLOGUE (?fuseprologue=1 experiment).
//
// fused_ffn_tiled_sg.wgsl (the shipping default FFN on sg32 hardware) with
// the FFN-entry add_norm (addNorm1) folded into the shared-memory load:
// instead of snapshotting a pre-normed hidden vector, each workgroup reads
// the residual and the o-proj delta, computes residual+delta and its RMSNorm
// into shared memory itself, then runs the identical 4-row gate/up tile.
//
// The tradeoff under test: 2048 workgroups each redundantly recompute the
// norm — in exchange the addNorm1 dispatch, its dependency-chain bubble, and
// the hidden1 buffer round-trip disappear.
//
// The sum-of-squares reduction is one subgroupAdd (32 threads = one
// subgroup, same sg_size == 32 assumption as the base kernel).
//
// Bindings (same order as fused_ffn_prologue.wgsl):
//   @binding(0): output   @binding(1): residual  @binding(2): delta
//   @binding(3): gamma    @binding(4): scales    @binding(5): weights
//   @binding(6): podArgs {packGridDimX}
//
// NOTE: this kernel does NOT write the residual sum anywhere — in
// ?fuseprologue=1 mode add3_norm.wgsl reconstructs it at the layer tail.
//
// Measured 2026-07-25 (M2 Max): -13.7% end-to-end — falsified on Apple
// (the per-WG RMSNorm recompute costs more than the saved dispatch
// bubbles); kept for A/B on other GPUs (see BENCH.md "Measured 2026-07-25").
//
// Model-shape constants (D, D_PACKED, D_SCALES, FFN_DIM) are injected by
// src/compiler/shader-prelude.ts.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> residual_in : array<f16>;
@group(0) @binding(2) var<storage, read> delta : array<f16>;
@group(0) @binding(3) var<storage, read> gamma : array<f16>;
@group(0) @binding(4) var<storage, read> scales : array<f16>;
@group(0) @binding(5) var<storage, read> weights : array<u32>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG : u32 = 4u;

var<workgroup> shared_input : array<f16, D>;

@compute @workgroup_size(32, 1, 1)
fn fused_ffn_tiled_sg_prologue(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let wg_idx : u32 = blockIdx.z * gridDim.x + blockIdx.x;
  let row_base : u32 = wg_idx * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let tid : u32 = threadIdx.x;

  // Prologue: residual add into shared memory + sum of squares.
  // f16 addition matches add_norm.wgsl's `A[idx] + B[idx]`.
  var sum_sq : f32 = 0.0;
  for (var i : u32 = 0u; i < D / 32u; i = i + 1u) {
    let idx : u32 = tid + i * 32u;
    let val : f16 = residual_in[idx] + delta[idx];
    shared_input[idx] = val;
    sum_sq = sum_sq + f32(val) * f32(val);
  }
  // One subgroup — subgroupAdd returns the workgroup-wide sum on all lanes.
  let total_sq : f32 = subgroupAdd(sum_sq);
  let rms_inv : f32 = 1.0 / sqrt(total_sq / f32(D) + 1e-5);

  // Normalize in place: each thread rewrites the elements it loaded,
  // rounding to f16 — exactly what the hidden1 store used to do.
  for (var i : u32 = 0u; i < D / 32u; i = i + 1u) {
    let idx : u32 = tid + i * 32u;
    shared_input[idx] = f16(f32(shared_input[idx]) * rms_inv * f32(gamma[idx]));
  }
  workgroupBarrier();

  // From here on: identical to fused_ffn_tiled_sg.wgsl.
  var gacc0 : f32 = 0.0; var gacc1 : f32 = 0.0; var gacc2 : f32 = 0.0; var gacc3 : f32 = 0.0;
  var uacc0 : f32 = 0.0; var uacc1 : f32 = 0.0; var uacc2 : f32 = 0.0; var uacc3 : f32 = 0.0;

  let gr0 = row_base;      let gr1 = row_base + 1u;
  let gr2 = row_base + 2u; let gr3 = row_base + 3u;
  let ur0 = gr0 + FFN_DIM; let ur1 = gr1 + FFN_DIM;
  let ur2 = gr2 + FFN_DIM; let ur3 = gr3 + FFN_DIM;

  for (var chunk : u32 = 0u; chunk < D_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base     : u32 = w_offset * 8u;
    let sc_idx   : u32 = w_offset >> 2u;

    let i0 = f32(shared_input[base     ]);
    let i1 = f32(shared_input[base + 1u]);
    let i2 = f32(shared_input[base + 2u]);
    let i3 = f32(shared_input[base + 3u]);
    let i4 = f32(shared_input[base + 4u]);
    let i5 = f32(shared_input[base + 5u]);
    let i6 = f32(shared_input[base + 6u]);
    let i7 = f32(shared_input[base + 7u]);

    // Gate row 0
    {
      let p = weights[gr0 * D_PACKED + w_offset];
      let s = f32(scales[gr0 * D_SCALES + sc_idx]);
      gacc0 = gacc0 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
    // Gate row 1
    {
      let p = weights[gr1 * D_PACKED + w_offset];
      let s = f32(scales[gr1 * D_SCALES + sc_idx]);
      gacc1 = gacc1 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
    // Gate row 2
    {
      let p = weights[gr2 * D_PACKED + w_offset];
      let s = f32(scales[gr2 * D_SCALES + sc_idx]);
      gacc2 = gacc2 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
    // Gate row 3
    {
      let p = weights[gr3 * D_PACKED + w_offset];
      let s = f32(scales[gr3 * D_SCALES + sc_idx]);
      gacc3 = gacc3 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
    // Up row 0
    {
      let p = weights[ur0 * D_PACKED + w_offset];
      let s = f32(scales[ur0 * D_SCALES + sc_idx]);
      uacc0 = uacc0 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
    // Up row 1
    {
      let p = weights[ur1 * D_PACKED + w_offset];
      let s = f32(scales[ur1 * D_SCALES + sc_idx]);
      uacc1 = uacc1 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
    // Up row 2
    {
      let p = weights[ur2 * D_PACKED + w_offset];
      let s = f32(scales[ur2 * D_SCALES + sc_idx]);
      uacc2 = uacc2 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
    // Up row 3
    {
      let p = weights[ur3 * D_PACKED + w_offset];
      let s = f32(scales[ur3 * D_SCALES + sc_idx]);
      uacc3 = uacc3 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
  }

  let g0 = subgroupAdd(gacc0); let g1 = subgroupAdd(gacc1);
  let g2 = subgroupAdd(gacc2); let g3 = subgroupAdd(gacc3);
  let u0 = subgroupAdd(uacc0); let u1 = subgroupAdd(uacc1);
  let u2 = subgroupAdd(uacc2); let u3 = subgroupAdd(uacc3);

  if (tid == 0u) {
    let silu0 = g0 * (1.0 / (1.0 + exp(-g0)));
    let silu1 = g1 * (1.0 / (1.0 + exp(-g1)));
    let silu2 = g2 * (1.0 / (1.0 + exp(-g2)));
    let silu3 = g3 * (1.0 / (1.0 + exp(-g3)));
    output_buf[gr0] = f16(u0 * silu0);
    output_buf[gr1] = f16(u1 * silu1);
    output_buf[gr2] = f16(u2 * silu2);
    output_buf[gr3] = f16(u3 * silu3);
  }
}
