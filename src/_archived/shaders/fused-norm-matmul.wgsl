// FUSED RMSNORM + INT4 MATMUL
//
// Replaces: fuse_add_norm_decode (12 workgroups) + fused_dequantize_NT_matmul (N workgroups)
// With: ONE dispatch that normalizes the input vector, then does the matmul
//
// This eliminates the intermediate normalized buffer — the normalization
// happens inside each matmul workgroup using shared memory.
//
// Strategy:
// 1. All 64 threads cooperatively compute the RMS norm of the input
// 2. Each thread then does its portion of the dot product using the normed input
// 3. Tree reduction produces the final output value
//
// Bindings:
//   @binding(0): output     array<f16>  (read_write) — matmul output
//   @binding(1): input      array<f16>  (read)       — unnormed hidden state
//   @binding(2): gamma      array<f16>  (read)       — RMSNorm weight
//   @binding(3): scales     array<f16>  (read)       — matmul weight scales
//   @binding(4): weights    array<u32>  (read)       — matmul weights (int4)
//   @binding(5): podArgs    uniform                  — {packGridDimX, D}

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> gamma : array<f16>;
@group(0) @binding(3) var<storage, read> scales : array<f16>;
@group(0) @binding(4) var<storage, read> weights : array<u32>;

struct PODArgs {
  packGridDimX: u32,
  D: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> shared_input : array<f16, 3072>;  // normed input cached in shared mem
var<workgroup> red_buf : array<f16, 64>;

@compute @workgroup_size(64, 1, 1)
fn fused_norm_matmul_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let row : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  let D : u32 = podArgs.D;

  // Phase 1: Cooperative RMSNorm — all threads compute sum of squares
  // Each thread handles D/64 elements
  var sum_sq : f32 = 0.0;
  let chunk : u32 = D / 64u;
  for (var i : u32 = 0u; i < chunk; i = i + 1u) {
    let idx : u32 = threadIdx.x * chunk + i;
    if (idx < D) {
      let val : f32 = f32(input_buf[idx]);
      sum_sq = sum_sq + val * val;
    }
  }

  // Reduce sum_sq across threads
  red_buf[threadIdx.x] = f16(sum_sq);
  workgroupBarrier();
  if (threadIdx.x < 32u) { red_buf[threadIdx.x] = f16(f32(red_buf[threadIdx.x]) + f32(red_buf[threadIdx.x + 32u])); }
  workgroupBarrier();
  if (threadIdx.x < 16u) { red_buf[threadIdx.x] = f16(f32(red_buf[threadIdx.x]) + f32(red_buf[threadIdx.x + 16u])); }
  workgroupBarrier();
  if (threadIdx.x < 8u) { red_buf[threadIdx.x] = f16(f32(red_buf[threadIdx.x]) + f32(red_buf[threadIdx.x + 8u])); }
  workgroupBarrier();
  if (threadIdx.x < 4u) { red_buf[threadIdx.x] = f16(f32(red_buf[threadIdx.x]) + f32(red_buf[threadIdx.x + 4u])); }
  workgroupBarrier();
  if (threadIdx.x < 2u) { red_buf[threadIdx.x] = f16(f32(red_buf[threadIdx.x]) + f32(red_buf[threadIdx.x + 2u])); }
  workgroupBarrier();
  if (threadIdx.x < 1u) { red_buf[threadIdx.x] = f16(f32(red_buf[threadIdx.x]) + f32(red_buf[threadIdx.x + 1u])); }
  workgroupBarrier();

  let rms : f32 = 1.0 / sqrt(f32(red_buf[0]) / f32(D) + 1e-5);

  // Phase 2: Normalize input into shared memory (cooperative)
  for (var i : u32 = 0u; i < chunk; i = i + 1u) {
    let idx : u32 = threadIdx.x * chunk + i;
    if (idx < D) {
      shared_input[idx] = f16(f32(input_buf[idx]) * rms) * gamma[idx];
    }
  }
  workgroupBarrier();

  // Phase 3: Int4 matmul using normed input from shared memory
  let D_PACKED : i32 = i32(D / 8u);
  let SCALES_PER_ROW : i32 = i32(D / 32u);
  var acc : f16 = 0.000000e+00h;

  let chunks_per_thread : i32 = D_PACKED / 64i;
  for (var c : i32 = 0i; c < chunks_per_thread; c = c + 1i) {
    let w_offset : i32 = i32(threadIdx.x) + c * 64i;
    let packed : u32 = weights[row * D_PACKED + w_offset];
    let s : f16 = scales[row * SCALES_PER_ROW + (w_offset >> 2u)];
    let base : i32 = w_offset * 8i;

    acc = fma(shared_input[base],     (f16(((packed >>  0u) & 15u)) - 7.000000e+00h) * s, acc);
    acc = fma(shared_input[base + 1], (f16(((packed >>  4u) & 15u)) - 7.000000e+00h) * s, acc);
    acc = fma(shared_input[base + 2], (f16(((packed >>  8u) & 15u)) - 7.000000e+00h) * s, acc);
    acc = fma(shared_input[base + 3], (f16(((packed >> 12u) & 15u)) - 7.000000e+00h) * s, acc);
    acc = fma(shared_input[base + 4], (f16(((packed >> 16u) & 15u)) - 7.000000e+00h) * s, acc);
    acc = fma(shared_input[base + 5], (f16(((packed >> 20u) & 15u)) - 7.000000e+00h) * s, acc);
    acc = fma(shared_input[base + 6], (f16(((packed >> 24u) & 15u)) - 7.000000e+00h) * s, acc);
    acc = fma(shared_input[base + 7], (f16(((packed >> 28u) & 15u)) - 7.000000e+00h) * s, acc);
  }

  // Tree reduction
  red_buf[threadIdx.x] = acc;
  workgroupBarrier();
  if (threadIdx.x < 32u) { red_buf[threadIdx.x] = red_buf[threadIdx.x] + red_buf[threadIdx.x + 32u]; }
  workgroupBarrier();
  if (threadIdx.x < 16u) { red_buf[threadIdx.x] = red_buf[threadIdx.x] + red_buf[threadIdx.x + 16u]; }
  workgroupBarrier();
  if (threadIdx.x < 8u) { red_buf[threadIdx.x] = red_buf[threadIdx.x] + red_buf[threadIdx.x + 8u]; }
  workgroupBarrier();
  if (threadIdx.x < 4u) { red_buf[threadIdx.x] = red_buf[threadIdx.x] + red_buf[threadIdx.x + 4u]; }
  workgroupBarrier();
  if (threadIdx.x < 2u) { red_buf[threadIdx.x] = red_buf[threadIdx.x] + red_buf[threadIdx.x + 2u]; }
  workgroupBarrier();
  if (threadIdx.x < 1u) { red_buf[threadIdx.x] = red_buf[threadIdx.x] + red_buf[threadIdx.x + 1u]; }
  workgroupBarrier();

  if (threadIdx.x == 0u) {
    output_buf[row] = red_buf[0];
  }
}
