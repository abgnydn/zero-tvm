// [20] rms_norm1_kernel (shader #20, 157 lines)
//----------------------------------------
// Function: rms_norm1_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> T_cast : array<f16>;
@group(0) @binding(1) var<storage, read> input_embeds : array<f16>;
@group(0) @binding(2) var<storage, read> transformer_h_0_ln_weight3 : array<f16>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f32, 64>;
var<workgroup> T_multiply_red_shared : array<f32, 1>;
@compute @workgroup_size(64, 1, 1)
fn rms_norm1_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var T_multiply_red_rf_local : array<f32, 1>;
  T_multiply_red_rf_local[0i] = 0.000000e+00f;
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[((v__1 * 3072i) + i32(threadIdx.x))]), f32(input_embeds[((v__1 * 3072i) + i32(threadIdx.x))]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 64i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 64i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 128i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 128i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 192i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 192i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 256i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 256i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 320i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 320i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 384i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 384i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 448i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 448i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 512i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 512i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 576i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 576i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 640i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 640i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 704i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 704i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 768i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 768i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 832i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 832i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 896i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 896i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 960i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 960i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1024i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1024i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1088i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1088i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1152i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1152i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1216i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1216i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1280i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1280i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1344i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1344i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1408i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1408i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1472i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1472i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1536i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1536i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1600i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1600i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1664i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1664i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1728i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1728i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1792i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1792i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1856i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1856i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1920i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1920i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1984i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1984i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2048i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2048i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2112i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2112i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2176i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2176i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2240i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2240i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2304i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2304i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2368i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2368i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2432i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2432i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2496i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2496i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2560i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2560i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2624i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2624i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2688i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2688i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2752i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2752i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2816i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2816i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2880i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2880i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2944i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2944i)]), T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 3008i)]), f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 3008i)]), T_multiply_red_rf_local[0i]);
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = T_multiply_red_rf_local[0i];
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
    T_multiply_red_shared[0i] = red_buf0[0i];
  }
  workgroupBarrier();
  T_cast[((v__1 * 3072i) + i32(threadIdx.x))] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[((v__1 * 3072i) + i32(threadIdx.x))])) * f32(transformer_h_0_ln_weight3[i32(threadIdx.x)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 64i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 64i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 64i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 128i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 128i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 128i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 192i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 192i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 192i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 256i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 256i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 256i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 320i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 320i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 320i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 384i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 384i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 384i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 448i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 448i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 448i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 512i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 512i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 512i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 576i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 576i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 576i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 640i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 640i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 640i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 704i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 704i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 704i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 768i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 768i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 768i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 832i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 832i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 832i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 896i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 896i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 896i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 960i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 960i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 960i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1024i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1024i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1024i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1088i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1088i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1088i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1152i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1152i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1152i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1216i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1216i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1216i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1280i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1280i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1280i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1344i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1344i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1344i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1408i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1408i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1408i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1472i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1472i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1472i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1536i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1536i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1536i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1600i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1600i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1600i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1664i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1664i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1664i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1728i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1728i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1728i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1792i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1792i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1792i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1856i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1856i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1856i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1920i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1920i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1920i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 1984i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 1984i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 1984i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2048i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2048i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2048i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2112i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2112i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2112i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2176i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2176i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2176i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2240i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2240i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2240i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2304i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2304i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2304i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2368i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2368i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2368i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2432i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2432i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2432i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2496i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2496i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2496i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2560i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2560i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2560i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2624i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2624i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2624i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2688i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2688i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2688i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2752i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2752i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2752i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2816i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2816i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2816i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2880i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2880i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2880i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 2944i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 2944i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 2944i)])));
  T_cast[(((v__1 * 3072i) + i32(threadIdx.x)) + 3008i)] = f16((((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 3.072000e+03f) + 1.000000e-05f))) * f32(input_embeds[(((v__1 * 3072i) + i32(threadIdx.x)) + 3008i)])) * f32(transformer_h_0_ln_weight3[(i32(threadIdx.x) + 3008i)])));
}
