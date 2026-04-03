// INT4 MATMUL WITH F32 OUTPUT — for LM head (logits).
// Same as int4_matmul but writes f32 instead of f16.
// TVM's NT_matmul14_cast2 does this — the sampling pipeline needs f32 logits.

enable f16;

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

var<workgroup> red_buf : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn int4_matmul_f32(
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

  for (var chunk : i32 = 0; chunk < K_PACKED / 64; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 64;
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
    output_buf[row] = red_buf[0];
  }
}
