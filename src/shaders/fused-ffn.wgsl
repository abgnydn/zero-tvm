// FUSED FFN: gate+up int4 matmul + SiLU in ONE dispatch.
//
// Replaces 2 TVM dispatches:
//   fused_dequantize3_NT_matmul12_kernel (16384 workgroups, 64 threads)
//   fused_split2_silu2_multiply2_kernel  (32 workgroups, 256 threads)
// With: 8192 workgroups, 64 threads each
//
// Weight layout (confirmed from TVM source):
//   Rows 0..8191    = gate weights
//   Rows 8192..16383 = up weights
//   output[i] = SiLU(gate[i]) * up[i]
//
// CRITICAL: input and output are the SAME buffer (BUF#730 in decode).
// We cache the 3072 f16 input in shared memory to avoid the race condition.
//
// Bindings (match TVM's matmul pattern):
//   @binding(0): output  array<f16>  (read_write) — SiLU result, 8192 elements
//   @binding(1): input   array<f16>  (read)       — normed hidden state, 3072 elements
//   @binding(2): scales  array<f16>  (read)       — weight scales
//   @binding(3): weights array<u32>  (read)       — int4 packed weights
//   @binding(4): podArgs uniform     — {packGridDimX}

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

// Cache input (3072 f16 = 6KB) + reduction buffers for gate and up
var<workgroup> shared_input : array<f16, 3072>;
var<workgroup> red_gate : array<f16, 64>;
var<workgroup> red_up : array<f16, 64>;

@compute @workgroup_size(64, 1, 1)
fn fused_ffn_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let output_idx : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);

  // Phase 1: Cooperatively load input into shared memory (48 elements per thread)
  for (var i : u32 = 0u; i < 48u; i = i + 1u) {
    let idx : u32 = threadIdx.x * 48u + i;
    if (idx < 3072u) {
      shared_input[idx] = input_buf[idx];
    }
  }
  workgroupBarrier();

  // Phase 2: Dual dot product — gate (row i) and up (row i+8192)
  let gate_row : i32 = output_idx;
  let up_row : i32 = output_idx + 8192i;
  let D_PACKED : i32 = 384i;       // 3072 / 8
  let SCALES_PER_ROW : i32 = 96i;  // 3072 / 32

  var gate_acc : f16 = 0.000000e+00h;
  var up_acc : f16 = 0.000000e+00h;

  // 6 chunks × 64 threads × 8 nibbles = 3072 elements (matches TVM's unrolled structure)
  for (var chunk : i32 = 0i; chunk < 6i; chunk = chunk + 1i) {
    let w_offset : i32 = i32(threadIdx.x) + chunk * 64i;

    let gate_packed : u32 = weights[gate_row * D_PACKED + w_offset];
    let gate_scale : f16 = scales[gate_row * SCALES_PER_ROW + (w_offset >> 2u)];

    let up_packed : u32 = weights[up_row * D_PACKED + w_offset];
    let up_scale : f16 = scales[up_row * SCALES_PER_ROW + (w_offset >> 2u)];

    let base : i32 = w_offset * 8i;

    // Unpack 8 int4 values, accumulate both gate and up using shared_input
    gate_acc = fma(shared_input[base],     (f16(((gate_packed >>  0u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 1], (f16(((gate_packed >>  4u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 2], (f16(((gate_packed >>  8u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 3], (f16(((gate_packed >> 12u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 4], (f16(((gate_packed >> 16u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 5], (f16(((gate_packed >> 20u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 6], (f16(((gate_packed >> 24u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);
    gate_acc = fma(shared_input[base + 7], (f16(((gate_packed >> 28u) & 15u)) - 7.000000e+00h) * gate_scale, gate_acc);

    up_acc = fma(shared_input[base],     (f16(((up_packed >>  0u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 1], (f16(((up_packed >>  4u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 2], (f16(((up_packed >>  8u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 3], (f16(((up_packed >> 12u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 4], (f16(((up_packed >> 16u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 5], (f16(((up_packed >> 20u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 6], (f16(((up_packed >> 24u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
    up_acc = fma(shared_input[base + 7], (f16(((up_packed >> 28u) & 15u)) - 7.000000e+00h) * up_scale, up_acc);
  }

  // Phase 3: Tree reduction (64 → 1) for both gate and up
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

  // Phase 4: SiLU(gate) * up — matches TVM's formula exactly
  if (threadIdx.x == 0u) {
    let gate_val : f32 = f32(red_gate[0]);
    let up_val : f16 = red_up[0];
    let silu_gate : f16 = f16(gate_val * (1.0 / (1.0 + exp(-gate_val))));
    output_buf[output_idx] = up_val * silu_gate;
  }
}
