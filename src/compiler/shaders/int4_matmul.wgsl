// GENERIC INT4 DEQUANT MATMUL — replaces 31 TVM shader variants.
//
// Computes: output[row] = dot(input[0..K-1], dequant(weights[row, 0..K-1]))
//   where dequant(packed) = (nibble - 7) * scale
//
// One workgroup = one output element (row).
// 64 threads cooperatively compute the dot product, then tree-reduce.
//
// Handles ANY dimension via uniforms:
//   K_PACKED  = K / 8  (number of u32 words per weight row)
//   SCALES_PER_ROW = K / 32 (number of scale values per weight row)
//   CHUNKS = K_PACKED / 64 (iterations per thread)
//
// Accumulates in f32 to avoid TVM's f16 precision loss.
//
// Bindings (match TVM convention):
//   @binding(0): output     array<f16>  (read_write) — matmul result
//   @binding(1): input      array<f16>  (read)       — input vector
//   @binding(2): scales     array<f16>  (read)       — per-group scales
//   @binding(3): weights    array<u32>  (read)       — int4 packed weights
//   @binding(4): podArgs    uniform     — dimensions

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  K_PACKED: u32,        // K / 8 (e.g. 384 for K=3072, 1024 for K=8192)
  SCALES_PER_ROW: u32,  // K / 32 (e.g. 96 for K=3072, 256 for K=8192)
  packGridDimX: u32     // number of output elements
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn int4_matmul(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(row) >= podArgs.packGridDimX) { return; }

  let K_PACKED : i32 = i32(podArgs.K_PACKED);
  let SCALES_PER_ROW : i32 = i32(podArgs.SCALES_PER_ROW);
  let tid : i32 = i32(threadIdx.x);

  // Each thread accumulates its portion of the dot product in f32
  var acc : f32 = 0.0;

  // Process K_PACKED / 64 chunks per thread
  for (var chunk : i32 = 0; chunk < K_PACKED / 64; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 64;
    let packed : u32 = weights[row * K_PACKED + w_offset];
    let scale : f32 = f32(scales[row * SCALES_PER_ROW + (w_offset >> 2)]);
    let base : i32 = w_offset * 8;

    // Unpack 8 int4 values and accumulate
    acc = acc + f32(input_buf[base])     * (f32(((packed >>  0u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 1]) * (f32(((packed >>  4u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 2]) * (f32(((packed >>  8u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 3]) * (f32(((packed >> 12u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 4]) * (f32(((packed >> 16u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 5]) * (f32(((packed >> 20u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 6]) * (f32(((packed >> 24u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 7]) * (f32(((packed >> 28u) & 15u)) - 7.0) * scale;
  }

  // Tree reduction in f32 (fixes TVM's f16 precision loss)
  red_buf[tid] = acc;
  workgroupBarrier();

  if (tid < 32) { red_buf[tid] = red_buf[tid] + red_buf[tid + 32]; }
  workgroupBarrier();
  if (tid < 16) { red_buf[tid] = red_buf[tid] + red_buf[tid + 16]; }
  workgroupBarrier();
  if (tid < 8) { red_buf[tid] = red_buf[tid] + red_buf[tid + 8]; }
  workgroupBarrier();
  if (tid < 4) { red_buf[tid] = red_buf[tid] + red_buf[tid + 4]; }
  workgroupBarrier();
  if (tid < 2) { red_buf[tid] = red_buf[tid] + red_buf[tid + 2]; }
  workgroupBarrier();
  if (tid < 1) { red_buf[tid] = red_buf[tid] + red_buf[tid + 1]; }
  workgroupBarrier();

  if (tid == 0) {
    output_buf[row] = f16(red_buf[0]);
  }
}
