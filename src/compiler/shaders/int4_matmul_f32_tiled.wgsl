// INT4 DEQUANT MATMUL WITH F32 OUTPUT (tiled subgroup variant).
// Same design as int4_matmul_tiled.wgsl but writes f32 logits for the LM head.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f32>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  K_PACKED: u32,
  SCALES_PER_ROW: u32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const ROWS_PER_WG : u32 = 4u;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_f32_tiled(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * ROWS_PER_WG;
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let tid : u32 = threadIdx.x;

  var acc0 : f32 = 0.0;
  var acc1 : f32 = 0.0;
  var acc2 : f32 = 0.0;
  var acc3 : f32 = 0.0;

  let r0 = row_base;
  let r1 = row_base + 1u;
  let r2 = row_base + 2u;
  let r3 = row_base + 3u;

  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base : u32 = w_offset * 8u;

    let i0 = f32(input_buf[base]);
    let i1 = f32(input_buf[base + 1u]);
    let i2 = f32(input_buf[base + 2u]);
    let i3 = f32(input_buf[base + 3u]);
    let i4 = f32(input_buf[base + 4u]);
    let i5 = f32(input_buf[base + 5u]);
    let i6 = f32(input_buf[base + 6u]);
    let i7 = f32(input_buf[base + 7u]);

    let scale_idx : u32 = w_offset >> 2u;

    {
      let packed = weights[r0 * K_PACKED + w_offset];
      let s = f32(scales[r0 * SCALES_PER_ROW + scale_idx]);
      acc0 = acc0
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
    {
      let packed = weights[r1 * K_PACKED + w_offset];
      let s = f32(scales[r1 * SCALES_PER_ROW + scale_idx]);
      acc1 = acc1
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
    {
      let packed = weights[r2 * K_PACKED + w_offset];
      let s = f32(scales[r2 * SCALES_PER_ROW + scale_idx]);
      acc2 = acc2
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
    {
      let packed = weights[r3 * K_PACKED + w_offset];
      let s = f32(scales[r3 * SCALES_PER_ROW + scale_idx]);
      acc3 = acc3
        + i0 * (f32((packed >>  0u) & 15u) - 7.0) * s
        + i1 * (f32((packed >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((packed >>  8u) & 15u) - 7.0) * s
        + i3 * (f32((packed >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((packed >> 16u) & 15u) - 7.0) * s
        + i5 * (f32((packed >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((packed >> 24u) & 15u) - 7.0) * s
        + i7 * (f32((packed >> 28u) & 15u) - 7.0) * s;
    }
  }

  let sum0 = subgroupAdd(acc0);
  let sum1 = subgroupAdd(acc1);
  let sum2 = subgroupAdd(acc2);
  let sum3 = subgroupAdd(acc3);

  if (tid == 0u) {
    output_buf[r0] = sum0;
    output_buf[r1] = sum1;
    output_buf[r2] = sum2;
    output_buf[r3] = sum3;
  }
}
