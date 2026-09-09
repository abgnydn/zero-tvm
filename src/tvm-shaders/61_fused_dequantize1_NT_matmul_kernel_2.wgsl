// [61] fused_dequantize1_NT_matmul_kernel_2 (shader #61, 243 lines)
//----------------------------------------
// Function: fused_dequantize1_NT_matmul_kernel_2
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> NT_matmul : array<f16>;
@group(0) @binding(1) var<storage, read> rms_norm195 : array<f16>;
@group(0) @binding(2) var<storage, read> transformer_h_0_mixer_qkv_proj_q_scale4 : array<f16>;
@group(0) @binding(3) var<storage, read> transformer_h_0_mixer_qkv_proj_q_weight4 : array<u32>;

struct PODArgs {
  batch_size: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> rms_norm195_reindex_pad_shared : array<f16, 256>;
var<workgroup> dequantize_reindex_shared : array<f16, 256>;
@compute @workgroup_size(8, 8, 1)
fn fused_dequantize1_NT_matmul_kernel_2(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  var NT_matmul_intermediate_reindex_pad_local : array<f16, 16>;
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var var_1 : i32 = 0; var_1 < 1i; var_1++) {
    NT_matmul_intermediate_reindex_pad_local[0i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[1i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[2i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[3i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[4i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[5i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[6i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[7i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[8i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[9i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[10i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[11i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[12i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[13i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[14i] = 0.000000e+00h;
    NT_matmul_intermediate_reindex_pad_local[15i] = 0.000000e+00h;
    for (var ax3_0 : i32 = 0; ax3_0 < 384i; ax3_0++) {
      workgroupBarrier();
      if ((((v__1 * 32i) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>1u)) < podArgs.batch_size) {
        rms_norm195_reindex_pad_shared[((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i))] = rms_norm195[(((((v__1 * 98304i) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>1u) * 3072i)) + (ax3_0 * 8i)) + ((i32(threadIdx.x) & 1i) * 4i))];
        rms_norm195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 1i)] = rms_norm195[((((((v__1 * 98304i) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>1u) * 3072i)) + (ax3_0 * 8i)) + ((i32(threadIdx.x) & 1i) * 4i)) + 1i)];
        rms_norm195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 2i)] = rms_norm195[((((((v__1 * 98304i) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>1u) * 3072i)) + (ax3_0 * 8i)) + ((i32(threadIdx.x) & 1i) * 4i)) + 2i)];
        rms_norm195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 3i)] = rms_norm195[((((((v__1 * 98304i) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>1u) * 3072i)) + (ax3_0 * 8i)) + ((i32(threadIdx.x) & 1i) * 4i)) + 3i)];
      } else {
        rms_norm195_reindex_pad_shared[((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i))] = 0.000000e+00h;
        rms_norm195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 1i)] = 0.000000e+00h;
        rms_norm195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 2i)] = 0.000000e+00h;
        rms_norm195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 3i)] = 0.000000e+00h;
      }
      dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i))] = ((f16(((transformer_h_0_mixer_qkv_proj_q_weight4[((((i32(blockIdx.y) * 12288i) + (i32(threadIdx.y) * 1536i)) + ((i32(threadIdx.x)>>1u) * 384i)) + ax3_0)]>>u32(((i32(threadIdx.x) & 1i) * 16i))) & 15u)) - 7.000000e+00h) * transformer_h_0_mixer_qkv_proj_q_scale4[((((i32(blockIdx.y) * 3072i) + (i32(threadIdx.y) * 384i)) + ((i32(threadIdx.x)>>1u) * 96i)) + (ax3_0>>2u))]);
      dequantize_reindex_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 1i)] = ((f16(((transformer_h_0_mixer_qkv_proj_q_weight4[((((i32(blockIdx.y) * 12288i) + (i32(threadIdx.y) * 1536i)) + ((i32(threadIdx.x)>>1u) * 384i)) + ax3_0)]>>u32((((i32(threadIdx.x) & 1i) * 16i) + 4i))) & 15u)) - 7.000000e+00h) * transformer_h_0_mixer_qkv_proj_q_scale4[((((i32(blockIdx.y) * 3072i) + (i32(threadIdx.y) * 384i)) + ((i32(threadIdx.x)>>1u) * 96i)) + (ax3_0>>2u))]);
      dequantize_reindex_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 2i)] = ((f16(((transformer_h_0_mixer_qkv_proj_q_weight4[((((i32(blockIdx.y) * 12288i) + (i32(threadIdx.y) * 1536i)) + ((i32(threadIdx.x)>>1u) * 384i)) + ax3_0)]>>u32((((i32(threadIdx.x) & 1i) * 16i) + 8i))) & 15u)) - 7.000000e+00h) * transformer_h_0_mixer_qkv_proj_q_scale4[((((i32(blockIdx.y) * 3072i) + (i32(threadIdx.y) * 384i)) + ((i32(threadIdx.x)>>1u) * 96i)) + (ax3_0>>2u))]);
      dequantize_reindex_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 3i)] = ((f16(((transformer_h_0_mixer_qkv_proj_q_weight4[((((i32(blockIdx.y) * 12288i) + (i32(threadIdx.y) * 1536i)) + ((i32(threadIdx.x)>>1u) * 384i)) + ax3_0)]>>u32((((i32(threadIdx.x) & 1i) * 16i) + 12i))) & 15u)) - 7.000000e+00h) * transformer_h_0_mixer_qkv_proj_q_scale4[((((i32(blockIdx.y) * 3072i) + (i32(threadIdx.y) * 384i)) + ((i32(threadIdx.x)>>1u) * 96i)) + (ax3_0>>2u))]);
      workgroupBarrier();
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(rms_norm195_reindex_pad_shared[(i32(threadIdx.x) * 32i)], dequantize_reindex_shared[(i32(threadIdx.y) * 32i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(rms_norm195_reindex_pad_shared[(i32(threadIdx.x) * 32i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 8i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(rms_norm195_reindex_pad_shared[(i32(threadIdx.x) * 32i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 16i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(rms_norm195_reindex_pad_shared[(i32(threadIdx.x) * 32i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 24i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 8i)], dequantize_reindex_shared[(i32(threadIdx.y) * 32i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 8i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 8i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 8i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 16i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 8i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 24i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 16i)], dequantize_reindex_shared[(i32(threadIdx.y) * 32i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 16i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 8i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 16i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 16i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 16i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 24i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 24i)], dequantize_reindex_shared[(i32(threadIdx.y) * 32i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 24i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 8i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 24i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 16i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 24i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 24i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 1i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 1i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 1i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 9i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 1i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 17i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 1i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 25i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 9i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 1i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 9i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 9i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 9i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 17i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 9i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 25i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 17i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 1i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 17i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 9i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 17i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 17i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 17i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 25i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 25i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 1i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 25i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 9i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 25i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 17i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 25i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 25i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 2i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 2i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 2i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 10i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 2i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 18i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 2i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 26i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 10i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 2i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 10i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 10i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 10i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 18i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 10i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 26i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 18i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 2i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 18i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 10i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 18i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 18i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 18i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 26i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 26i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 2i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 26i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 10i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 26i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 18i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 26i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 26i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 3i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 3i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 3i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 11i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 3i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 19i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 3i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 27i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 11i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 3i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 11i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 11i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 11i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 19i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 11i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 27i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 19i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 3i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 19i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 11i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 19i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 19i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 19i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 27i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 27i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 3i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 27i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 11i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 27i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 19i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 27i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 27i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 4i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 4i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 4i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 12i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 4i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 20i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 4i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 28i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 12i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 4i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 12i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 12i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 12i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 20i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 12i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 28i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 20i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 4i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 20i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 12i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 20i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 20i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 20i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 28i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 28i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 4i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 28i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 12i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 28i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 20i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 28i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 28i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 5i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 5i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 5i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 13i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 5i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 21i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 5i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 29i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 13i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 5i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 13i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 13i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 13i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 21i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 13i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 29i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 21i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 5i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 21i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 13i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 21i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 21i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 21i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 29i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 29i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 5i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 29i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 13i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 29i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 21i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 29i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 29i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 6i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 6i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 6i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 14i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 6i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 22i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 6i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 30i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 14i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 6i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 14i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 14i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 14i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 22i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 14i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 30i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 22i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 6i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 22i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 14i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 22i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 22i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 22i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 30i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 30i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 6i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 30i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 14i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 30i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 22i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 30i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 30i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 7i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 7i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 7i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 15i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 7i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 23i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 7i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 31i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 15i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 7i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 15i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 15i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 15i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 23i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 15i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 31i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 23i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 7i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 23i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 15i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 23i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 23i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 23i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 31i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 31i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 7i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 31i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 15i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 31i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 23i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(rms_norm195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 31i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 31i)], NT_matmul_intermediate_reindex_pad_local[15i]);
    }
    if (((v__1 * 32i) + (i32(threadIdx.x) * 4i)) < podArgs.batch_size) {
      NT_matmul[((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i))] = NT_matmul_intermediate_reindex_pad_local[0i];
    }
    if (((v__1 * 32i) + (i32(threadIdx.x) * 4i)) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1i)] = NT_matmul_intermediate_reindex_pad_local[1i];
    }
    if (((v__1 * 32i) + (i32(threadIdx.x) * 4i)) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2i)] = NT_matmul_intermediate_reindex_pad_local[2i];
    }
    if (((v__1 * 32i) + (i32(threadIdx.x) * 4i)) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 3i)] = NT_matmul_intermediate_reindex_pad_local[3i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 9216i)] = NT_matmul_intermediate_reindex_pad_local[4i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 9217i)] = NT_matmul_intermediate_reindex_pad_local[5i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 9218i)] = NT_matmul_intermediate_reindex_pad_local[6i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 9219i)] = NT_matmul_intermediate_reindex_pad_local[7i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 18432i)] = NT_matmul_intermediate_reindex_pad_local[8i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 18433i)] = NT_matmul_intermediate_reindex_pad_local[9i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 18434i)] = NT_matmul_intermediate_reindex_pad_local[10i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 18435i)] = NT_matmul_intermediate_reindex_pad_local[11i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 27648i)] = NT_matmul_intermediate_reindex_pad_local[12i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 27649i)] = NT_matmul_intermediate_reindex_pad_local[13i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 27650i)] = NT_matmul_intermediate_reindex_pad_local[14i];
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i) < podArgs.batch_size) {
      NT_matmul[(((((v__1 * 294912i) + (i32(threadIdx.x) * 36864i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 27651i)] = NT_matmul_intermediate_reindex_pad_local[15i];
    }
  }
}
