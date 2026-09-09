// [34] fused_dequantize4_NT_matmul8_kernel_1 (shader #34, 182 lines)
//----------------------------------------
// Function: fused_dequantize4_NT_matmul8_kernel_1
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> NT_matmul : array<f16>;
@group(0) @binding(1) var<storage, read> lv33 : array<f16>;
@group(0) @binding(2) var<storage, read> transformer_h_0_mlp_down_proj_q_scale3 : array<f16>;
@group(0) @binding(3) var<storage, read> transformer_h_0_mlp_down_proj_q_weight3 : array<u32>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f16, 64>;
@compute @workgroup_size(64, 1, 1)
fn fused_dequantize4_NT_matmul8_kernel_1(
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
  for (var ax2_fused_0 : i32 = 0; ax2_fused_0 < 16i; ax2_fused_0++) {
    dequantize_local[0i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[(((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>0u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[(((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[1i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[(((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>4u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[(((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[2i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[(((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>8u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[(((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[3i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[(((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>12u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[(((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[4i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[(((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>16u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[(((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[5i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[(((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>20u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[(((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[6i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[(((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>24u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[(((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[7i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[(((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x))]>>28u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[(((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]);
    dequantize_local[8i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[((((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 1024i)]>>0u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[((((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 256i)]);
    dequantize_local[9i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[((((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 1024i)]>>4u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[((((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 256i)]);
    dequantize_local[10i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[((((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 1024i)]>>8u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[((((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 256i)]);
    dequantize_local[11i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[((((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 1024i)]>>12u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[((((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 256i)]);
    dequantize_local[12i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[((((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 1024i)]>>16u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[((((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 256i)]);
    dequantize_local[13i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[((((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 1024i)]>>20u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[((((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 256i)]);
    dequantize_local[14i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[((((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 1024i)]>>24u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[((((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 256i)]);
    dequantize_local[15i] = ((f16(((transformer_h_0_mlp_down_proj_q_weight3[((((v__1 * 2048i) + (ax2_fused_0 * 64i)) + i32(threadIdx.x)) + 1024i)]>>28u) & 15u)) - 7.000000e+00h) * transformer_h_0_mlp_down_proj_q_scale3[((((v__1 * 512i) + (ax2_fused_0 * 16i)) + (i32(threadIdx.x)>>2u)) + 256i)]);
    if ((i32(blockIdx.y) * 4i) < podArgs.seq_len) {
      NT_matmul_intermediate_pad_rf_local[0i] = fma(lv33[(((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i))], dequantize_local[0i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 1i)], dequantize_local[1i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 2i)], dequantize_local[2i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3i)], dequantize_local[3i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 4i)], dequantize_local[4i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 5i)], dequantize_local[5i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6i)], dequantize_local[6i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[0i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 7i)], dequantize_local[7i], NT_matmul_intermediate_pad_rf_local[0i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(lv33[(((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i))], dequantize_local[8i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 1i)], dequantize_local[9i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 2i)], dequantize_local[10i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 3i)], dequantize_local[11i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 4i)], dequantize_local[12i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 5i)], dequantize_local[13i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 6i)], dequantize_local[14i], NT_matmul_intermediate_pad_rf_local[1i]);
      NT_matmul_intermediate_pad_rf_local[1i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 7i)], dequantize_local[15i], NT_matmul_intermediate_pad_rf_local[1i]);
    }
    if (((i32(blockIdx.y) * 4i) + 1i) < podArgs.seq_len) {
      NT_matmul_intermediate_pad_rf_local[2i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8192i)], dequantize_local[0i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8193i)], dequantize_local[1i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8194i)], dequantize_local[2i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8195i)], dequantize_local[3i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8196i)], dequantize_local[4i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8197i)], dequantize_local[5i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8198i)], dequantize_local[6i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[2i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8199i)], dequantize_local[7i], NT_matmul_intermediate_pad_rf_local[2i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8192i)], dequantize_local[8i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8193i)], dequantize_local[9i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8194i)], dequantize_local[10i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8195i)], dequantize_local[11i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8196i)], dequantize_local[12i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8197i)], dequantize_local[13i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8198i)], dequantize_local[14i], NT_matmul_intermediate_pad_rf_local[3i]);
      NT_matmul_intermediate_pad_rf_local[3i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 8199i)], dequantize_local[15i], NT_matmul_intermediate_pad_rf_local[3i]);
    }
    if (((i32(blockIdx.y) * 4i) + 2i) < podArgs.seq_len) {
      NT_matmul_intermediate_pad_rf_local[4i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16384i)], dequantize_local[0i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16385i)], dequantize_local[1i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16386i)], dequantize_local[2i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16387i)], dequantize_local[3i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16388i)], dequantize_local[4i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16389i)], dequantize_local[5i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16390i)], dequantize_local[6i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[4i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16391i)], dequantize_local[7i], NT_matmul_intermediate_pad_rf_local[4i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16384i)], dequantize_local[8i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16385i)], dequantize_local[9i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16386i)], dequantize_local[10i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16387i)], dequantize_local[11i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16388i)], dequantize_local[12i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16389i)], dequantize_local[13i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16390i)], dequantize_local[14i], NT_matmul_intermediate_pad_rf_local[5i]);
      NT_matmul_intermediate_pad_rf_local[5i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 16391i)], dequantize_local[15i], NT_matmul_intermediate_pad_rf_local[5i]);
    }
    if (((i32(blockIdx.y) * 4i) + 3i) < podArgs.seq_len) {
      NT_matmul_intermediate_pad_rf_local[6i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24576i)], dequantize_local[0i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24577i)], dequantize_local[1i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24578i)], dequantize_local[2i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24579i)], dequantize_local[3i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24580i)], dequantize_local[4i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24581i)], dequantize_local[5i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24582i)], dequantize_local[6i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[6i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24583i)], dequantize_local[7i], NT_matmul_intermediate_pad_rf_local[6i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24576i)], dequantize_local[8i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24577i)], dequantize_local[9i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24578i)], dequantize_local[10i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24579i)], dequantize_local[11i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24580i)], dequantize_local[12i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24581i)], dequantize_local[13i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24582i)], dequantize_local[14i], NT_matmul_intermediate_pad_rf_local[7i]);
      NT_matmul_intermediate_pad_rf_local[7i] = fma(lv33[((((i32(blockIdx.y) * 32768i) + (ax2_fused_0 * 512i)) + (i32(threadIdx.x) * 8i)) + 24583i)], dequantize_local[15i], NT_matmul_intermediate_pad_rf_local[7i]);
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
      if (((i32(threadIdx.x) == 0i) && (((i32(blockIdx.y) - ((podArgs.seq_len + 3i)>>2u)) < 0i) || (((i32(blockIdx.y) * 4i) + ax0) == 0i))) && (((i32(blockIdx.y) * 4i) + ax0) < podArgs.seq_len)) {
        NT_matmul[((((i32(blockIdx.y) * 12288i) + (ax0 * 3072i)) + (v__1 * 2i)) + ax1_fused_2)] = NT_matmul_intermediate_pad_local[((ax0 * 2i) + ax1_fused_2)];
      }
    }
  }
}
