// INT4 DEQUANT MATMUL — batched (M=4) variant of int4_matmul_tiled.wgsl.
//
// Shape: input [M, K] × weights [N, K] → output [M, N]    with M fixed at 4.
// Each WG computes a TILE_M × ROWS_PER_WG = 4 × 4 = 16 output cells:
//   - Loads ROWS_PER_WG weight rows (and their scales) ONCE per K-chunk
//     and reuses them across all TILE_M input rows — this is the point.
//     Running total-weight-bytes for M=4 forwards becomes 1×W instead of
//     4×W, which is the enabler for prompt-lookup spec decoding on
//     memory-bandwidth-bound hardware.
//   - Loads TILE_M × 8 f32 inputs per chunk and reuses across all 4 weight
//     rows — same input-amortization argument as the non-batched shader.
//
// Output layout: output_buf[m * N + n] for batch row m, output column n.
//                Column N is podArgs.packGridDimX (matches existing kernels).
// Input layout:  input_buf[m * K + k] — the engine strides M copies of
//                the activation vector contiguously.
//
// Bindings match int4_matmul_tiled.wgsl exactly; sizes of input_buf and
// output_buf are M× larger (the engine owns buffer sizing).
//
// Requires sg_size = 32; gated in chat.ts.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read>       input_buf  : array<f16>;
@group(0) @binding(2) var<storage, read>       scales     : array<f16>;
@group(0) @binding(3) var<storage, read>       weights    : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32,   // N (number of output cols)
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG : u32 = 4u;
const TILE_M      : u32 = 4u;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_batched_m4(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED       : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let K              : u32 = K_PACKED * 8u;          // elements per input row
  let tid : u32 = threadIdx.x;

  // 16 f32 accumulators per thread: 4 batch rows × 4 output rows.
  var a00 : f32 = 0.0; var a01 : f32 = 0.0; var a02 : f32 = 0.0; var a03 : f32 = 0.0;
  var a10 : f32 = 0.0; var a11 : f32 = 0.0; var a12 : f32 = 0.0; var a13 : f32 = 0.0;
  var a20 : f32 = 0.0; var a21 : f32 = 0.0; var a22 : f32 = 0.0; var a23 : f32 = 0.0;
  var a30 : f32 = 0.0; var a31 : f32 = 0.0; var a32 : f32 = 0.0; var a33 : f32 = 0.0;

  let r0 = row_base;      let r1 = row_base + 1u;
  let r2 = row_base + 2u; let r3 = row_base + 3u;

  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base     : u32 = w_offset * 8u;
    let sc_idx   : u32 = w_offset >> 2u;

    // Load 8 f32 inputs for EACH of the 4 batch rows (32 inputs per thread).
    let b0_0 = f32(input_buf[0u * K + base     ]);
    let b0_1 = f32(input_buf[0u * K + base + 1u]);
    let b0_2 = f32(input_buf[0u * K + base + 2u]);
    let b0_3 = f32(input_buf[0u * K + base + 3u]);
    let b0_4 = f32(input_buf[0u * K + base + 4u]);
    let b0_5 = f32(input_buf[0u * K + base + 5u]);
    let b0_6 = f32(input_buf[0u * K + base + 6u]);
    let b0_7 = f32(input_buf[0u * K + base + 7u]);
    let b1_0 = f32(input_buf[1u * K + base     ]);
    let b1_1 = f32(input_buf[1u * K + base + 1u]);
    let b1_2 = f32(input_buf[1u * K + base + 2u]);
    let b1_3 = f32(input_buf[1u * K + base + 3u]);
    let b1_4 = f32(input_buf[1u * K + base + 4u]);
    let b1_5 = f32(input_buf[1u * K + base + 5u]);
    let b1_6 = f32(input_buf[1u * K + base + 6u]);
    let b1_7 = f32(input_buf[1u * K + base + 7u]);
    let b2_0 = f32(input_buf[2u * K + base     ]);
    let b2_1 = f32(input_buf[2u * K + base + 1u]);
    let b2_2 = f32(input_buf[2u * K + base + 2u]);
    let b2_3 = f32(input_buf[2u * K + base + 3u]);
    let b2_4 = f32(input_buf[2u * K + base + 4u]);
    let b2_5 = f32(input_buf[2u * K + base + 5u]);
    let b2_6 = f32(input_buf[2u * K + base + 6u]);
    let b2_7 = f32(input_buf[2u * K + base + 7u]);
    let b3_0 = f32(input_buf[3u * K + base     ]);
    let b3_1 = f32(input_buf[3u * K + base + 1u]);
    let b3_2 = f32(input_buf[3u * K + base + 2u]);
    let b3_3 = f32(input_buf[3u * K + base + 3u]);
    let b3_4 = f32(input_buf[3u * K + base + 4u]);
    let b3_5 = f32(input_buf[3u * K + base + 5u]);
    let b3_6 = f32(input_buf[3u * K + base + 6u]);
    let b3_7 = f32(input_buf[3u * K + base + 7u]);

    // For each of the 4 output rows: load (packed, scale) ONCE, dequantize
    // 8 weights, and multiply-accumulate into 4 batch rows. 4 × 4 = 16 FMAs
    // per row per chunk iteration.
    {
      let p = weights[r0 * K_PACKED + w_offset];
      let s = f32(scales[r0 * SCALES_PER_ROW + sc_idx]);
      let w0 = (f32((p >>  0u) & 15u) - 7.0) * s;
      let w1 = (f32((p >>  4u) & 15u) - 7.0) * s;
      let w2 = (f32((p >>  8u) & 15u) - 7.0) * s;
      let w3 = (f32((p >> 12u) & 15u) - 7.0) * s;
      let w4 = (f32((p >> 16u) & 15u) - 7.0) * s;
      let w5 = (f32((p >> 20u) & 15u) - 7.0) * s;
      let w6 = (f32((p >> 24u) & 15u) - 7.0) * s;
      let w7 = (f32((p >> 28u) & 15u) - 7.0) * s;
      a00 = a00 + b0_0*w0 + b0_1*w1 + b0_2*w2 + b0_3*w3 + b0_4*w4 + b0_5*w5 + b0_6*w6 + b0_7*w7;
      a10 = a10 + b1_0*w0 + b1_1*w1 + b1_2*w2 + b1_3*w3 + b1_4*w4 + b1_5*w5 + b1_6*w6 + b1_7*w7;
      a20 = a20 + b2_0*w0 + b2_1*w1 + b2_2*w2 + b2_3*w3 + b2_4*w4 + b2_5*w5 + b2_6*w6 + b2_7*w7;
      a30 = a30 + b3_0*w0 + b3_1*w1 + b3_2*w2 + b3_3*w3 + b3_4*w4 + b3_5*w5 + b3_6*w6 + b3_7*w7;
    }
    {
      let p = weights[r1 * K_PACKED + w_offset];
      let s = f32(scales[r1 * SCALES_PER_ROW + sc_idx]);
      let w0 = (f32((p >>  0u) & 15u) - 7.0) * s;
      let w1 = (f32((p >>  4u) & 15u) - 7.0) * s;
      let w2 = (f32((p >>  8u) & 15u) - 7.0) * s;
      let w3 = (f32((p >> 12u) & 15u) - 7.0) * s;
      let w4 = (f32((p >> 16u) & 15u) - 7.0) * s;
      let w5 = (f32((p >> 20u) & 15u) - 7.0) * s;
      let w6 = (f32((p >> 24u) & 15u) - 7.0) * s;
      let w7 = (f32((p >> 28u) & 15u) - 7.0) * s;
      a01 = a01 + b0_0*w0 + b0_1*w1 + b0_2*w2 + b0_3*w3 + b0_4*w4 + b0_5*w5 + b0_6*w6 + b0_7*w7;
      a11 = a11 + b1_0*w0 + b1_1*w1 + b1_2*w2 + b1_3*w3 + b1_4*w4 + b1_5*w5 + b1_6*w6 + b1_7*w7;
      a21 = a21 + b2_0*w0 + b2_1*w1 + b2_2*w2 + b2_3*w3 + b2_4*w4 + b2_5*w5 + b2_6*w6 + b2_7*w7;
      a31 = a31 + b3_0*w0 + b3_1*w1 + b3_2*w2 + b3_3*w3 + b3_4*w4 + b3_5*w5 + b3_6*w6 + b3_7*w7;
    }
    {
      let p = weights[r2 * K_PACKED + w_offset];
      let s = f32(scales[r2 * SCALES_PER_ROW + sc_idx]);
      let w0 = (f32((p >>  0u) & 15u) - 7.0) * s;
      let w1 = (f32((p >>  4u) & 15u) - 7.0) * s;
      let w2 = (f32((p >>  8u) & 15u) - 7.0) * s;
      let w3 = (f32((p >> 12u) & 15u) - 7.0) * s;
      let w4 = (f32((p >> 16u) & 15u) - 7.0) * s;
      let w5 = (f32((p >> 20u) & 15u) - 7.0) * s;
      let w6 = (f32((p >> 24u) & 15u) - 7.0) * s;
      let w7 = (f32((p >> 28u) & 15u) - 7.0) * s;
      a02 = a02 + b0_0*w0 + b0_1*w1 + b0_2*w2 + b0_3*w3 + b0_4*w4 + b0_5*w5 + b0_6*w6 + b0_7*w7;
      a12 = a12 + b1_0*w0 + b1_1*w1 + b1_2*w2 + b1_3*w3 + b1_4*w4 + b1_5*w5 + b1_6*w6 + b1_7*w7;
      a22 = a22 + b2_0*w0 + b2_1*w1 + b2_2*w2 + b2_3*w3 + b2_4*w4 + b2_5*w5 + b2_6*w6 + b2_7*w7;
      a32 = a32 + b3_0*w0 + b3_1*w1 + b3_2*w2 + b3_3*w3 + b3_4*w4 + b3_5*w5 + b3_6*w6 + b3_7*w7;
    }
    {
      let p = weights[r3 * K_PACKED + w_offset];
      let s = f32(scales[r3 * SCALES_PER_ROW + sc_idx]);
      let w0 = (f32((p >>  0u) & 15u) - 7.0) * s;
      let w1 = (f32((p >>  4u) & 15u) - 7.0) * s;
      let w2 = (f32((p >>  8u) & 15u) - 7.0) * s;
      let w3 = (f32((p >> 12u) & 15u) - 7.0) * s;
      let w4 = (f32((p >> 16u) & 15u) - 7.0) * s;
      let w5 = (f32((p >> 20u) & 15u) - 7.0) * s;
      let w6 = (f32((p >> 24u) & 15u) - 7.0) * s;
      let w7 = (f32((p >> 28u) & 15u) - 7.0) * s;
      a03 = a03 + b0_0*w0 + b0_1*w1 + b0_2*w2 + b0_3*w3 + b0_4*w4 + b0_5*w5 + b0_6*w6 + b0_7*w7;
      a13 = a13 + b1_0*w0 + b1_1*w1 + b1_2*w2 + b1_3*w3 + b1_4*w4 + b1_5*w5 + b1_6*w6 + b1_7*w7;
      a23 = a23 + b2_0*w0 + b2_1*w1 + b2_2*w2 + b2_3*w3 + b2_4*w4 + b2_5*w5 + b2_6*w6 + b2_7*w7;
      a33 = a33 + b3_0*w0 + b3_1*w1 + b3_2*w2 + b3_3*w3 + b3_4*w4 + b3_5*w5 + b3_6*w6 + b3_7*w7;
    }
  }

  let s00 = subgroupAdd(a00); let s01 = subgroupAdd(a01);
  let s02 = subgroupAdd(a02); let s03 = subgroupAdd(a03);
  let s10 = subgroupAdd(a10); let s11 = subgroupAdd(a11);
  let s12 = subgroupAdd(a12); let s13 = subgroupAdd(a13);
  let s20 = subgroupAdd(a20); let s21 = subgroupAdd(a21);
  let s22 = subgroupAdd(a22); let s23 = subgroupAdd(a23);
  let s30 = subgroupAdd(a30); let s31 = subgroupAdd(a31);
  let s32 = subgroupAdd(a32); let s33 = subgroupAdd(a33);

  if (tid == 0u) {
    let N = podArgs.packGridDimX;
    output_buf[0u * N + r0] = f16(s00); output_buf[0u * N + r1] = f16(s01);
    output_buf[0u * N + r2] = f16(s02); output_buf[0u * N + r3] = f16(s03);
    output_buf[1u * N + r0] = f16(s10); output_buf[1u * N + r1] = f16(s11);
    output_buf[1u * N + r2] = f16(s12); output_buf[1u * N + r3] = f16(s13);
    output_buf[2u * N + r0] = f16(s20); output_buf[2u * N + r1] = f16(s21);
    output_buf[2u * N + r2] = f16(s22); output_buf[2u * N + r3] = f16(s23);
    output_buf[3u * N + r0] = f16(s30); output_buf[3u * N + r1] = f16(s31);
    output_buf[3u * N + r2] = f16(s32); output_buf[3u * N + r3] = f16(s33);
  }
}
