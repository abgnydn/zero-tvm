// FUSED FFN WITH ADD+RMSNORM PROLOGUE (?fuseprologue=1 experiment).
//
// fused_ffn.wgsl with the FFN-entry add_norm (addNorm1) folded into phase 1:
// instead of reading a pre-normed hidden vector, each workgroup reads the
// residual and the o-proj delta, computes residual+delta and its RMSNorm
// into shared memory itself, then runs the identical gate/up dot products.
//
// The tradeoff under test: every workgroup redundantly recomputes the norm
// (a ~3K-element reduction) — in exchange the addNorm1 dispatch, its
// dependency-chain bubble, and the hidden1 buffer round-trip disappear.
//
// Numerics match the sequential add_norm → fused_ffn path: the residual sum
// is computed in f16 (as add_norm does), the normed value is rounded to f16
// before the dot products (as the hidden1 buffer store did), and the sum of
// squares / dots accumulate in f32.
//
// Bindings:
//   @binding(0): output   array<f16> (read_write) — SiLU result, FFN_DIM elements
//   @binding(1): residual array<f16> (read)       — running residual (pre-attention)
//   @binding(2): delta    array<f16> (read)       — o-proj output to be added
//   @binding(3): gamma    array<f16> (read)       — RMSNorm gain (normGamma2)
//   @binding(4): scales   array<f16> (read)
//   @binding(5): weights  array<u32> (read)
//   @binding(6): podArgs  uniform    — {packGridDimX}
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

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> residual_in : array<f16>;
@group(0) @binding(2) var<storage, read> delta : array<f16>;
@group(0) @binding(3) var<storage, read> gamma : array<f16>;
@group(0) @binding(4) var<storage, read> scales : array<f16>;
@group(0) @binding(5) var<storage, read> weights : array<u32>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

var<workgroup> shared_input : array<f16, D>;
var<workgroup> red_gate : array<f32, 64>;
var<workgroup> red_up : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn fused_ffn_prologue(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x >= podArgs.packGridDimX) { return; }
  let output_idx : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  let tid : u32 = threadIdx.x;

  // Phase 1a: residual add into shared memory + local sum of squares.
  // f16 addition matches add_norm.wgsl's `A[idx] + B[idx]`.
  var sum_sq : f32 = 0.0;
  for (var i : u32 = 0u; i < D / 64u; i = i + 1u) {
    let idx : u32 = tid + i * 64u;
    let val : f16 = residual_in[idx] + delta[idx];
    shared_input[idx] = val;
    sum_sq = sum_sq + f32(val) * f32(val);
  }

  // Phase 1b: 64→1 tree reduction of the sum of squares (reuses red_gate).
  red_gate[tid] = sum_sq;
  workgroupBarrier();
  if (tid < 32u) { red_gate[tid] = red_gate[tid] + red_gate[tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { red_gate[tid] = red_gate[tid] + red_gate[tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { red_gate[tid] = red_gate[tid] + red_gate[tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { red_gate[tid] = red_gate[tid] + red_gate[tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { red_gate[tid] = red_gate[tid] + red_gate[tid + 2u]; }
  workgroupBarrier();
  if (tid < 1u) { red_gate[tid] = red_gate[tid] + red_gate[tid + 1u]; }
  workgroupBarrier();

  let rms_inv : f32 = 1.0 / sqrt(red_gate[0] / f32(D) + 1e-5);

  // Phase 1c: normalize in place. Each thread rewrites the elements it
  // loaded, rounding to f16 — exactly what the hidden1 store used to do.
  for (var i : u32 = 0u; i < D / 64u; i = i + 1u) {
    let idx : u32 = tid + i * 64u;
    shared_input[idx] = f16(f32(shared_input[idx]) * rms_inv * f32(gamma[idx]));
  }
  workgroupBarrier();

  // Phase 2: dual dot product — gate (row i) and up (row i+FFN_DIM).
  // Identical to fused_ffn.wgsl from here on.
  let gate_row : i32 = output_idx;
  let up_row : i32 = output_idx + FFN_DIM;

  var gate_acc : f32 = 0.0;
  var up_acc : f32 = 0.0;

  for (var chunk : i32 = 0i; chunk < D_PACKED / 64; chunk = chunk + 1i) {
    let w_offset : i32 = i32(threadIdx.x) + chunk * 64i;

    let gate_packed : u32 = weights[gate_row * D_PACKED + w_offset];
    let gate_scale : f32 = f32(scales[gate_row * D_SCALES + (w_offset >> 2u)]);

    let up_packed : u32 = weights[up_row * D_PACKED + w_offset];
    let up_scale : f32 = f32(scales[up_row * D_SCALES + (w_offset >> 2u)]);

    let base : i32 = w_offset * 8i;

    gate_acc = fma(gate_scale,
        f32(shared_input[base])     * (f32((gate_packed >>  0u) & 15u) - 7.0)
      + f32(shared_input[base + 1]) * (f32((gate_packed >>  4u) & 15u) - 7.0)
      + f32(shared_input[base + 2]) * (f32((gate_packed >>  8u) & 15u) - 7.0)
      + f32(shared_input[base + 3]) * (f32((gate_packed >> 12u) & 15u) - 7.0)
      + f32(shared_input[base + 4]) * (f32((gate_packed >> 16u) & 15u) - 7.0)
      + f32(shared_input[base + 5]) * (f32((gate_packed >> 20u) & 15u) - 7.0)
      + f32(shared_input[base + 6]) * (f32((gate_packed >> 24u) & 15u) - 7.0)
      + f32(shared_input[base + 7]) * (f32((gate_packed >> 28u) & 15u) - 7.0),
      gate_acc);

    up_acc = fma(up_scale,
        f32(shared_input[base])     * (f32((up_packed >>  0u) & 15u) - 7.0)
      + f32(shared_input[base + 1]) * (f32((up_packed >>  4u) & 15u) - 7.0)
      + f32(shared_input[base + 2]) * (f32((up_packed >>  8u) & 15u) - 7.0)
      + f32(shared_input[base + 3]) * (f32((up_packed >> 12u) & 15u) - 7.0)
      + f32(shared_input[base + 4]) * (f32((up_packed >> 16u) & 15u) - 7.0)
      + f32(shared_input[base + 5]) * (f32((up_packed >> 20u) & 15u) - 7.0)
      + f32(shared_input[base + 6]) * (f32((up_packed >> 24u) & 15u) - 7.0)
      + f32(shared_input[base + 7]) * (f32((up_packed >> 28u) & 15u) - 7.0),
      up_acc);
  }

  // Phase 3: tree reduction (64 → 1) for both gate and up.
  red_gate[threadIdx.x] = gate_acc;
  red_up[threadIdx.x] = up_acc;
  workgroupBarrier();

  if (threadIdx.x < 32u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 32u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 32u]; }
  workgroupBarrier();
  if (threadIdx.x < 16u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 16u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 16u]; }
  workgroupBarrier();
  if (threadIdx.x < 8u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 8u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 8u]; }
  workgroupBarrier();
  if (threadIdx.x < 4u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 4u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 4u]; }
  workgroupBarrier();
  if (threadIdx.x < 2u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 2u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 2u]; }
  workgroupBarrier();
  if (threadIdx.x < 1u) { red_gate[threadIdx.x] = red_gate[threadIdx.x] + red_gate[threadIdx.x + 1u]; red_up[threadIdx.x] = red_up[threadIdx.x] + red_up[threadIdx.x + 1u]; }
  workgroupBarrier();

  // Phase 4: SiLU(gate) * up — all f32, only the final store rounds to f16.
  if (threadIdx.x == 0u) {
    let gate_val : f32 = red_gate[0];
    let up_val : f32 = red_up[0];
    let silu_gate : f32 = gate_val * (1.0 / (1.0 + exp(-gate_val)));
    output_buf[output_idx] = f16(up_val * silu_gate);
  }
}
