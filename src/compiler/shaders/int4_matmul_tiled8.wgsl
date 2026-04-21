// INT4 DEQUANT MATMUL (tiled-8 subgroup variant) — 8 output rows per
// workgroup. Same binding layout as int4_matmul.wgsl / int4_matmul_tiled.wgsl.
//
// Why 8 instead of 4: for ffnDown (K=8192, N=3072) the 4-row tile still reads
// ~12.5 MB of input vector per call (one read per WG × 768 WGs × 16 KB). With
// 8 rows per WG, only 384 WGs fire and input DRAM traffic halves to ~6.3 MB —
// for this matmul that's ~24% of total weight+input bytes.
//
// For LM head (K=3072, N=32064), 8-row tiling saves ~25 MB of input reads out
// of 105 MB total bytes per dispatch (~24%).
//
// Each thread holds 8 f32 accumulators; with 32-thread subgroup that's well
// inside the M-series GPU register budget.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read>       input_buf  : array<f16>;
@group(0) @binding(2) var<storage, read>       scales     : array<f16>;
@group(0) @binding(3) var<storage, read>       weights    : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG : u32 = 8u;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_tiled8(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED       : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let tid : u32 = threadIdx.x;

  var a0 : f32 = 0.0; var a1 : f32 = 0.0; var a2 : f32 = 0.0; var a3 : f32 = 0.0;
  var a4 : f32 = 0.0; var a5 : f32 = 0.0; var a6 : f32 = 0.0; var a7 : f32 = 0.0;

  let r0 = row_base;      let r1 = row_base + 1u; let r2 = row_base + 2u; let r3 = row_base + 3u;
  let r4 = row_base + 4u; let r5 = row_base + 5u; let r6 = row_base + 6u; let r7 = row_base + 7u;

  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base     : u32 = w_offset * 8u;
    let sc_idx   : u32 = w_offset >> 2u;

    let i0 = f32(input_buf[base]);
    let i1 = f32(input_buf[base + 1u]);
    let i2 = f32(input_buf[base + 2u]);
    let i3 = f32(input_buf[base + 3u]);
    let i4 = f32(input_buf[base + 4u]);
    let i5 = f32(input_buf[base + 5u]);
    let i6 = f32(input_buf[base + 6u]);
    let i7 = f32(input_buf[base + 7u]);

    // 8 rows unrolled. Factor the dequantise+multiply into a per-row lambda-ish
    // block; the compiler inlines it.
    { let p = weights[r0 * K_PACKED + w_offset]; let s = f32(scales[r0 * SCALES_PER_ROW + sc_idx]);
      a0 = a0
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r1 * K_PACKED + w_offset]; let s = f32(scales[r1 * SCALES_PER_ROW + sc_idx]);
      a1 = a1
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r2 * K_PACKED + w_offset]; let s = f32(scales[r2 * SCALES_PER_ROW + sc_idx]);
      a2 = a2
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r3 * K_PACKED + w_offset]; let s = f32(scales[r3 * SCALES_PER_ROW + sc_idx]);
      a3 = a3
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r4 * K_PACKED + w_offset]; let s = f32(scales[r4 * SCALES_PER_ROW + sc_idx]);
      a4 = a4
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r5 * K_PACKED + w_offset]; let s = f32(scales[r5 * SCALES_PER_ROW + sc_idx]);
      a5 = a5
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r6 * K_PACKED + w_offset]; let s = f32(scales[r6 * SCALES_PER_ROW + sc_idx]);
      a6 = a6
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[r7 * K_PACKED + w_offset]; let s = f32(scales[r7 * SCALES_PER_ROW + sc_idx]);
      a7 = a7
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
  }

  let s0 = subgroupAdd(a0); let s1 = subgroupAdd(a1); let s2 = subgroupAdd(a2); let s3 = subgroupAdd(a3);
  let s4 = subgroupAdd(a4); let s5 = subgroupAdd(a5); let s6 = subgroupAdd(a6); let s7 = subgroupAdd(a7);

  if (tid == 0u) {
    output_buf[r0] = f16(s0); output_buf[r1] = f16(s1);
    output_buf[r2] = f16(s2); output_buf[r3] = f16(s3);
    output_buf[r4] = f16(s4); output_buf[r5] = f16(s5);
    output_buf[r6] = f16(s6); output_buf[r7] = f16(s7);
  }
}
