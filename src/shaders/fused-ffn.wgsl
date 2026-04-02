// FUSED FFN: gate+up matmul + SiLU activation in ONE dispatch.
//
// Replaces: fused_dequantize3_NT_matmul12 (16384 workgroups) + fused_split2_silu2_multiply2 (32 workgroups)
// With: fused_ffn (8192 workgroups) — each workgroup computes gate[i] AND up[i], then outputs SiLU(gate[i]) * up[i]
//
// The gate and up weight rows are interleaved in the weight matrix:
//   Row 0 = gate[0], Row 1 = up[0], Row 2 = gate[1], Row 3 = up[1], ...
//   OR: Row 0..8191 = gate, Row 8192..16383 = up
//
// We need to compute two dot products per output element:
//   gate[i] = dot(input, weights[i])
//   up[i] = dot(input, weights[i + 8192])
//   output[i] = SiLU(gate[i]) * up[i]
//
// where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Bindings match TVM's matmul pattern + SiLU output:
//   @binding(0): output (8192 f16, NOT 16384 — SiLU reduces by half)
//   @binding(1): input (hidden state, f16)
//   @binding(2): scales (f16)
//   @binding(3): weights (u32, int4 packed)
//   @binding(4): podArgs uniform

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf_gate : array<f16, 64>;
var<workgroup> red_buf_up : array<f16, 64>;

@compute @workgroup_size(64, 1, 1)
fn fused_ffn_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let output_idx : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);

  // Row indices for gate and up in the weight matrix
  let gate_row : i32 = output_idx;
  let up_row : i32 = output_idx + 8192i;

  // D_PACKED = 3072 / 8 = 384 (number of packed u32s per weight row)
  let D_PACKED : i32 = 384i;
  let SCALES_PER_ROW : i32 = 96i;  // 3072 / 32

  // Accumulate gate and up dot products in parallel
  var gate_acc : f16 = 0.000000e+00h;
  var up_acc : f16 = 0.000000e+00h;

  // Each thread processes D_PACKED/64 = 6 weight chunks
  for (var chunk : i32 = 0i; chunk < 6i; chunk = chunk + 1i) {
    let w_offset : i32 = i32(threadIdx.x) + chunk * 64i;

    // Load gate weight
    let gate_packed : u32 = weights[gate_row * D_PACKED + w_offset];
    let gate_scale : f16 = scales[gate_row * SCALES_PER_ROW + (w_offset >> 2u)];

    // Load up weight
    let up_packed : u32 = weights[up_row * D_PACKED + w_offset];
    let up_scale : f16 = scales[up_row * SCALES_PER_ROW + (w_offset >> 2u)];

    let base : i32 = w_offset * 8i;

    // Unpack 8 int4 values and accumulate both gate and up
    for (var nibble : u32 = 0u; nibble < 8u; nibble = nibble + 1u) {
      let shift : u32 = nibble * 4u;
      let inp : f16 = input_buf[u32(base) + nibble];

      let gate_w : f16 = (f16(((gate_packed >> shift) & 15u)) - 7.000000e+00h) * gate_scale;
      let up_w : f16 = (f16(((up_packed >> shift) & 15u)) - 7.000000e+00h) * up_scale;

      gate_acc = fma(inp, gate_w, gate_acc);
      up_acc = fma(inp, up_w, up_acc);
    }
  }

  // Tree reduction for gate
  red_buf_gate[threadIdx.x] = gate_acc;
  red_buf_up[threadIdx.x] = up_acc;
  workgroupBarrier();

  if (threadIdx.x < 32u) {
    red_buf_gate[threadIdx.x] = red_buf_gate[threadIdx.x] + red_buf_gate[threadIdx.x + 32u];
    red_buf_up[threadIdx.x] = red_buf_up[threadIdx.x] + red_buf_up[threadIdx.x + 32u];
  }
  workgroupBarrier();
  if (threadIdx.x < 16u) {
    red_buf_gate[threadIdx.x] = red_buf_gate[threadIdx.x] + red_buf_gate[threadIdx.x + 16u];
    red_buf_up[threadIdx.x] = red_buf_up[threadIdx.x] + red_buf_up[threadIdx.x + 16u];
  }
  workgroupBarrier();
  if (threadIdx.x < 8u) {
    red_buf_gate[threadIdx.x] = red_buf_gate[threadIdx.x] + red_buf_gate[threadIdx.x + 8u];
    red_buf_up[threadIdx.x] = red_buf_up[threadIdx.x] + red_buf_up[threadIdx.x + 8u];
  }
  workgroupBarrier();
  if (threadIdx.x < 4u) {
    red_buf_gate[threadIdx.x] = red_buf_gate[threadIdx.x] + red_buf_gate[threadIdx.x + 4u];
    red_buf_up[threadIdx.x] = red_buf_up[threadIdx.x] + red_buf_up[threadIdx.x + 4u];
  }
  workgroupBarrier();
  if (threadIdx.x < 2u) {
    red_buf_gate[threadIdx.x] = red_buf_gate[threadIdx.x] + red_buf_gate[threadIdx.x + 2u];
    red_buf_up[threadIdx.x] = red_buf_up[threadIdx.x] + red_buf_up[threadIdx.x + 2u];
  }
  workgroupBarrier();
  if (threadIdx.x < 1u) {
    red_buf_gate[threadIdx.x] = red_buf_gate[threadIdx.x] + red_buf_gate[threadIdx.x + 1u];
    red_buf_up[threadIdx.x] = red_buf_up[threadIdx.x] + red_buf_up[threadIdx.x + 1u];
  }
  workgroupBarrier();

  // Thread 0: apply SiLU and write output
  if (threadIdx.x == 0u) {
    let gate_val : f32 = f32(red_buf_gate[0]);
    let up_val : f16 = red_buf_up[0];

    // SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
    let sigmoid_gate : f32 = 1.0 / (1.0 + exp(-gate_val));
    let silu_gate : f16 = f16(gate_val * sigmoid_gate);

    output_buf[output_idx] = silu_gate * up_val;
  }
}
