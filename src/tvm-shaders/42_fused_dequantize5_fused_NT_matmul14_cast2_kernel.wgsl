// [42] fused_dequantize5_fused_NT_matmul14_cast2_kernel (shader #42, 118 lines)
//----------------------------------------
// Function: fused_dequantize5_fused_NT_matmul14_cast2_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> compute : array<f32>;
@group(0) @binding(1) var<storage, read> lm_head_q_scale2 : array<f16>;
@group(0) @binding(2) var<storage, read> lm_head_q_weight2 : array<u32>;
@group(0) @binding(3) var<storage, read> rms_norm129 : array<f16>;

struct PODArgs {
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f16, 64>;
@compute @workgroup_size(64, 1, 1)
fn fused_dequantize5_fused_NT_matmul14_cast2_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var NT_matmul_rf_local : array<f16, 1>;
  var lm_head_q_weight2_local : array<u32, 1>;
  var NT_matmul_rf_local_1 : array<f16, 1>;
  NT_matmul_rf_local[0i] = 0.000000e+00h;
  lm_head_q_weight2_local[0i] = lm_head_q_weight2[((v__1 * 384i) + i32(threadIdx.x))];
  NT_matmul_rf_local[0i] = fma(rms_norm129[(i32(threadIdx.x) * 8i)], ((f16(((lm_head_q_weight2_local[0i]>>0u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[((v__1 * 96i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1i)], ((f16(((lm_head_q_weight2_local[0i]>>4u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[((v__1 * 96i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2i)], ((f16(((lm_head_q_weight2_local[0i]>>8u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[((v__1 * 96i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 3i)], ((f16(((lm_head_q_weight2_local[0i]>>12u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[((v__1 * 96i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 4i)], ((f16(((lm_head_q_weight2_local[0i]>>16u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[((v__1 * 96i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 5i)], ((f16(((lm_head_q_weight2_local[0i]>>20u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[((v__1 * 96i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 6i)], ((f16(((lm_head_q_weight2_local[0i]>>24u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[((v__1 * 96i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 7i)], ((f16(((lm_head_q_weight2_local[0i]>>28u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[((v__1 * 96i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  lm_head_q_weight2_local[0i] = lm_head_q_weight2[(((v__1 * 384i) + i32(threadIdx.x)) + 64i)];
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 512i)], ((f16(((lm_head_q_weight2_local[0i]>>0u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 513i)], ((f16(((lm_head_q_weight2_local[0i]>>4u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 514i)], ((f16(((lm_head_q_weight2_local[0i]>>8u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 515i)], ((f16(((lm_head_q_weight2_local[0i]>>12u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 516i)], ((f16(((lm_head_q_weight2_local[0i]>>16u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 517i)], ((f16(((lm_head_q_weight2_local[0i]>>20u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 518i)], ((f16(((lm_head_q_weight2_local[0i]>>24u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 519i)], ((f16(((lm_head_q_weight2_local[0i]>>28u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  lm_head_q_weight2_local[0i] = lm_head_q_weight2[(((v__1 * 384i) + i32(threadIdx.x)) + 128i)];
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1024i)], ((f16(((lm_head_q_weight2_local[0i]>>0u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1025i)], ((f16(((lm_head_q_weight2_local[0i]>>4u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1026i)], ((f16(((lm_head_q_weight2_local[0i]>>8u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1027i)], ((f16(((lm_head_q_weight2_local[0i]>>12u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1028i)], ((f16(((lm_head_q_weight2_local[0i]>>16u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1029i)], ((f16(((lm_head_q_weight2_local[0i]>>20u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1030i)], ((f16(((lm_head_q_weight2_local[0i]>>24u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1031i)], ((f16(((lm_head_q_weight2_local[0i]>>28u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  lm_head_q_weight2_local[0i] = lm_head_q_weight2[(((v__1 * 384i) + i32(threadIdx.x)) + 192i)];
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1536i)], ((f16(((lm_head_q_weight2_local[0i]>>0u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1537i)], ((f16(((lm_head_q_weight2_local[0i]>>4u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1538i)], ((f16(((lm_head_q_weight2_local[0i]>>8u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1539i)], ((f16(((lm_head_q_weight2_local[0i]>>12u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1540i)], ((f16(((lm_head_q_weight2_local[0i]>>16u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1541i)], ((f16(((lm_head_q_weight2_local[0i]>>20u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1542i)], ((f16(((lm_head_q_weight2_local[0i]>>24u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 1543i)], ((f16(((lm_head_q_weight2_local[0i]>>28u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  lm_head_q_weight2_local[0i] = lm_head_q_weight2[(((v__1 * 384i) + i32(threadIdx.x)) + 256i)];
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2048i)], ((f16(((lm_head_q_weight2_local[0i]>>0u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 64i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2049i)], ((f16(((lm_head_q_weight2_local[0i]>>4u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 64i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2050i)], ((f16(((lm_head_q_weight2_local[0i]>>8u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 64i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2051i)], ((f16(((lm_head_q_weight2_local[0i]>>12u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 64i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2052i)], ((f16(((lm_head_q_weight2_local[0i]>>16u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 64i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2053i)], ((f16(((lm_head_q_weight2_local[0i]>>20u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 64i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2054i)], ((f16(((lm_head_q_weight2_local[0i]>>24u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 64i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2055i)], ((f16(((lm_head_q_weight2_local[0i]>>28u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 64i)]), NT_matmul_rf_local[0i]);
  lm_head_q_weight2_local[0i] = lm_head_q_weight2[(((v__1 * 384i) + i32(threadIdx.x)) + 320i)];
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2560i)], ((f16(((lm_head_q_weight2_local[0i]>>0u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 80i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2561i)], ((f16(((lm_head_q_weight2_local[0i]>>4u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 80i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2562i)], ((f16(((lm_head_q_weight2_local[0i]>>8u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 80i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2563i)], ((f16(((lm_head_q_weight2_local[0i]>>12u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 80i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2564i)], ((f16(((lm_head_q_weight2_local[0i]>>16u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 80i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2565i)], ((f16(((lm_head_q_weight2_local[0i]>>20u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 80i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2566i)], ((f16(((lm_head_q_weight2_local[0i]>>24u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 80i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm129[((i32(threadIdx.x) * 8i) + 2567i)], ((f16(((lm_head_q_weight2_local[0i]>>28u) & 15u)) - 7.000000e+00h) * lm_head_q_scale2[(((v__1 * 96i) + (i32(threadIdx.x)>>2u)) + 80i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local_1[0i] = 0.000000e+00h;
  NT_matmul_rf_local_1[0i] = (NT_matmul_rf_local_1[0i] + NT_matmul_rf_local[0i]);
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = NT_matmul_rf_local_1[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 32i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 32i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 8i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 4i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 4i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 2i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 1i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    compute[v__1] = f32(red_buf0[0i]);
  }
}
