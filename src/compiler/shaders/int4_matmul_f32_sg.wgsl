// INT4 DEQUANT MATMUL WITH F32 OUTPUT (subgroup variant) — for the LM head.
// Same structure as int4_matmul_sg.wgsl but writes f32 logits.

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

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_f32_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(row) >= podArgs.packGridDimX) { return; }

  let K_PACKED : i32 = i32(podArgs.K_PACKED);
  let SCALES_PER_ROW : i32 = i32(podArgs.SCALES_PER_ROW);
  let tid : i32 = i32(threadIdx.x);

  var acc : f32 = 0.0;

  for (var chunk : i32 = 0; chunk < K_PACKED / 32; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 32;
    let packed : u32 = weights[row * K_PACKED + w_offset];
    let scale : f32 = f32(scales[row * SCALES_PER_ROW + (w_offset >> 2)]);
    let base : i32 = w_offset * 8;

    acc = acc + f32(input_buf[base])     * (f32(((packed >>  0u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 1]) * (f32(((packed >>  4u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 2]) * (f32(((packed >>  8u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 3]) * (f32(((packed >> 12u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 4]) * (f32(((packed >> 16u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 5]) * (f32(((packed >> 20u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 6]) * (f32(((packed >> 24u) & 15u)) - 7.0) * scale;
    acc = acc + f32(input_buf[base + 7]) * (f32(((packed >> 28u) & 15u)) - 7.0) * scale;
  }

  let sum : f32 = subgroupAdd(acc);

  if (tid == 0) {
    output_buf[row] = sum;
  }
}
