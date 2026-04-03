// [28] fused_dequantize3_NT_matmul2_kernel_1 (shader #28, 182 lines)
//----------------------------------------
// Function: fused_dequantize3_NT_matmul2_kernel_1
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> NT_matmul : array<f16>;
@group(0) @binding(1) var<storage, read> rms_norm196 : array<f16>;
@group(0) @binding(2) var<storage, read> transformer_h_0_mlp_gate_up_proj_q_scale4 : array<f16>;
@group(0) @binding(3) var<storage, read> transformer_h_0_mlp_gate_up_proj_q_weight4 : array<u32>;

struct PODArgs {
  batch_size: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f16, 64>;
@compute @workgroup_size(64, 1, 1)
fn fused_dequantize3_NT_matmul2_kernel_1(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  var NT_matmul_intermediate_pad_rf_local : array<f16, 8>;
  var dequantize_local : array<f16, 16>;
  var NT_matmul_intermediate_pad_rf_local_1 : array<f16, 8>;
  var NT_matmul_intermediate_pad_local : array<f16, 8>;
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var ax0_1_init : i32 = 0; ax0_1_init < 4i; ax0_1_init++) {
    for (var ax1_fused_2_init : i32 = 0; ax1_fused_2_init < 2i; ax1_fused_2_init++) {
      NT_matmul_intermediate_pad_rf_local[((ax0_1_init * 2i) + ax1_fused_2_init)] = 0.000000e+00h;
    }
  }
  for (var ax2_fused_0 : i32 = 0; ax2_fused_0 < 6i; ax2_fused_0++) {
    dequantize_local[0i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[(((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>0u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[(((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[1i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[(((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>4u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[(((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[2i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[(((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>8u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[(((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[3i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[(((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>12u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[(((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[4i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[(((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>16u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[(((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[5i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[(((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>20u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[(((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[6i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[(((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>24u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[(((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[7i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[(((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>28u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[(((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[8i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[((((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 384i)]>>0u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[((((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 96i)]);
    dequantize_local[9i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[((((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 384i)]>>4u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[((((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 96i)]);
    dequantize_local[10i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[((((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 384i)]>>8u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[((((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 96i)]);
    dequantize_local[11i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[((((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 384i)]>>12u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[((((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 96i)]);
    dequantize_local[12i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[((((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 384i)]>>16u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[((((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 96i)]);
    dequantize_local[13i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[((((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 384i)]>>20u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[((((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 96i)]);
    dequantize_local[14i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[((((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 384i)]>>24u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[((((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 96i)]);
    dequantize_local[15i] = ((f16(((transformer_h_0_mlp_gate_up_proj_q_weight4[((((v__1 * 768i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 384i)]>>28u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_gate_up_proj_q_scale4[((((v__1 * 192i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 96i)]);
    if ((i32(blockIdx.y) * 4i) < podArgs.batch_size) {
      NT_matmul_intermediate_pad_rf_local[0i] = fma(rms_norm196[(((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i))], dequantize_local[0i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 1i)], dequantize_local[1i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 2i)], dequantize_local[2i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3i)], dequantize_local[3i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 4i)], dequantize_local[4i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 5i)], dequantize_local[5i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6i)], dequantize_local[6i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 7i)], dequantize_local[7i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(rms_norm196[(((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i))], dequantize_local[8i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 1i)], dequantize_local[9i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 2i)], dequantize_local[10i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3i)], dequantize_local[11i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 4i)], dequantize_local[12i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 5i)], dequantize_local[13i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6i)], dequantize_local[14i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 7i)], dequantize_local[15i], NT_matmul_intermediate_pad_rf_local[1i]);
    }
    if (((i32(blockIdx.y) * 4i) + 1i) < podArgs.batch_size) {
      NT_matmul_intermediate_pad_rf_local[2i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3072i)], dequantize_local[0i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3073i)], dequantize_local[1i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3074i)], dequantize_local[2i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3075i)], dequantize_local[3i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3076i)], dequantize_local[4i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3077i)], dequantize_local[5i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3078i)], dequantize_local[6i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3079i)], dequantize_local[7i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3072i)], dequantize_local[8i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3073i)], dequantize_local[9i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3074i)], dequantize_local[10i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3075i)], dequantize_local[11i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3076i)], dequantize_local[12i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3077i)], dequantize_local[13i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3078i)], dequantize_local[14i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3079i)], dequantize_local[15i], NT_matmul_intermediate_pad_rf_local[3i]);
    }
    if (((i32(blockIdx.y) * 4i) + 2i) < podArgs.batch_size) {
      NT_matmul_intermediate_pad_rf_local[4i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6144i)], dequantize_local[0i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6145i)], dequantize_local[1i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6146i)], dequantize_local[2i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6147i)], dequantize_local[3i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6148i)], dequantize_local[4i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6149i)], dequantize_local[5i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6150i)], dequantize_local[6i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6151i)], dequantize_local[7i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6144i)], dequantize_local[8i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6145i)], dequantize_local[9i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6146i)], dequantize_local[10i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6147i)], dequantize_local[11i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6148i)], dequantize_local[12i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6149i)], dequantize_local[13i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6150i)], dequantize_local[14i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6151i)], dequantize_local[15i], NT_matmul_intermediate_pad_rf_local[5i]);
    }
    if (((i32(blockIdx.y) * 4i) + 3i) < podArgs.batch_size) {
      NT_matmul_intermediate_pad_rf_local[6i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9216i)], dequantize_local[0i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9217i)], dequantize_local[1i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9218i)], dequantize_local[2i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9219i)], dequantize_local[3i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9220i)], dequantize_local[4i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9221i)], dequantize_local[5i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9222i)], dequantize_local[6i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9223i)], dequantize_local[7i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9216i)], dequantize_local[8i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9217i)], dequantize_local[9i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9218i)], dequantize_local[10i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9219i)], dequantize_local[11i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9220i)], dequantize_local[12i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9221i)], dequantize_local[13i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9222i)], dequantize_local[14i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(rms_norm196[((((i32(blockIdx.y) * 12288i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 9223i)], dequantize_local[15i], NT_matmul_intermediate_pad_rf_local[7i]);
    }
  }
  NT_matmul_intermediate_pad_rf_local_1[0i + 0] = vec2<f16>(0.000000e+00h, 0.000000e+00h)[0];
  NT_matmul_intermediate_pad_rf_local_1[0i + 1] = vec2<f16>(0.000000e+00h, 0.000000e+00h)[1];
  NT_matmul_intermediate_pad_rf_local_1[0i + 0] = (vec2<f16>(NT_matmul_intermediate_pad_rf_local_1[0i + 0], NT_matmul_intermediate_pad_rf_local_1[0i + 1]) + vec2<f16>(NT_matmul_intermediate_pad_rf_local[0i + 0], NT_matmul_intermediate_pad_rf_local[0i + 1]))[0];
  NT_matmul_intermediate_pad_rf_local_1[0i + 1] = (vec2<f16>(NT_matmul_intermediate_pad_rf_local_1[0i + 0], NT_matmul_intermediate_pad_rf_local_1[0i + 1]) + vec2<f16>(NT_matmul_intermediate_pad_rf_local[0i + 0], NT_matmul_intermediate_pad_rf_local[0i + 1]))[1];
  NT_matmul_intermediate_pad_rf_local_1[2i + 0] = vec2<f16>(0.000000e+00h, 0.000000e+00h)[0];
  NT_matmul_intermediate_pad_rf_local_1[2i + 1] = vec2<f16>(0.000000e+00h, 0.000000e+00h)[1];
  NT_matmul_intermediate_pad_rf_local_1[2i + 0] = (vec2<f16>(NT_matmul_intermediate_pad_rf_local_1[2i + 0], NT_matmul_intermediate_pad_rf_local_1[2i + 1]) + vec2<f16>(NT_matmul_intermediate_pad_rf_local[2i + 0], NT_matmul_intermediate_pad_rf_local[2i + 1]))[0];
  NT_matmul_intermediate_pad_rf_local_1[2i + 1] = (vec2<f16>(NT_matmul_intermediate_pad_rf_local_1[2i + 0], NT_matmul_intermediate_pad_rf_local_1[2i + 1]) + vec2<f16>(NT_matmul_intermediate_pad_rf_local[2i + 0], NT_matmul_intermediate_pad_rf_local[2i + 1]))[1];
  NT_matmul_intermediate_pad_rf_local_1[4i + 0] = vec2<f16>(0.000000e+00h, 0.000000e+00h)[0];
  NT_matmul_intermediate_pad_rf_local_1[4i + 1] = vec2<f16>(0.000000e+00h, 0.000000e+00h)[1];
  NT_matmul_intermediate_pad_rf_local_1[4i + 0] = (vec2<f16>(NT_matmul_intermediate_pad_rf_local_1[4i + 0], NT_matmul_intermediate_pad_rf_local_1[4i + 1]) + vec2<f16>(NT_matmul_intermediate_pad_rf_local[4i + 0], NT_matmul_intermediate_pad_rf_local[4i + 1]))[0];
  NT_matmul_intermediate_pad_rf_local_1[4i + 1] = (vec2<f16>(NT_matmul_intermediate_pad_rf_local_1[4i + 0], NT_matmul_intermediate_pad_rf_local_1[4i + 1]) + vec2<f16>(NT_matmul_intermediate_pad_rf_local[4i + 0], NT_matmul_intermediate_pad_rf_local[4i + 1]))[1];
  NT_matmul_intermediate_pad_rf_local_1[6i + 0] = vec2<f16>(0.000000e+00h, 0.000000e+00h)[0];
  NT_matmul_intermediate_pad_rf_local_1[6i + 1] = vec2<f16>(0.000000e+00h, 0.000000e+00h)[1];
  NT_matmul_intermediate_pad_rf_local_1[6i + 0] = (vec2<f16>(NT_matmul_intermediate_pad_rf_local_1[6i + 0], NT_matmul_intermediate_pad_rf_local_1[6i + 1]) + vec2<f16>(NT_matmul_intermediate_pad_rf_local[6i + 0], NT_matmul_intermediate_pad_rf_local[6i + 1]))[0];
  NT_matmul_intermediate_pad_rf_local_1[6i + 1] = (vec2<f16>(NT_matmul_intermediate_pad_rf_local_1[6i + 0], NT_matmul_intermediate_pad_rf_local_1[6i + 1]) + vec2<f16>(NT_matmul_intermediate_pad_rf_local[6i + 0], NT_matmul_intermediate_pad_rf_local[6i + 1]))[1];
  for (var ax2_fused_2 : i32 = 0; ax2_fused_2 < 2i; ax2_fused_2++) {
    for (var ax1 : i32 = 0; ax1 < 4i; ax1++) {
      workgroupBarrier();
      red_buf0[i32(threadIdx.x)] = NT_matmul_intermediate_pad_rf_local_1[((ax1 * 2i) + ax2_fused_2)];
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
      NT_matmul_intermediate_pad_local[((ax1 * 2i) + ax2_fused_2)] = red_buf0[0i];
    }
  }
  for (var ax0 : i32 = 0; ax0 < 4i; ax0++) {
    for (var ax1_fused_2 : i32 = 0; ax1_fused_2 < 2i; ax1_fused_2++) {
      if (((i32(threadIdx.x) == 0i) && (((i32(blockIdx.y) - ((podArgs.batch_size + 3i)>>2u)) < 0i) || (((i32(blockIdx.y) * 4i) + ax0) == 0i))) && (((i32(blockIdx.y) * 4i) + ax0) < podArgs.batch_size)) {
        NT_matmul[((((i32(blockIdx.y) * 65536i) + (ax0 * 16384i)) + (v__1 * 2i)) + ax1_fused_2)] = NT_matmul_intermediate_pad_local[((ax0 * 2i) + ax1_fused_2)];
      }
    }
  }
}
