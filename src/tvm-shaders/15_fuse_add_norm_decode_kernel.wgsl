// [15] fuse_add_norm_decode_kernel (shader #15, 120 lines)
//----------------------------------------
// Function: fuse_add_norm_decode_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> A : array<f16>;
@group(0) @binding(1) var<storage, read> B : array<f16>;
@group(0) @binding(2) var<storage, read> C : array<f16>;
@group(0) @binding(3) var<storage, read_write> O : array<f16>;
@group(0) @binding(4) var<storage, read_write> add : array<f16>;

struct PODArgs {
  batch_size: i32,
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f32, 256>;
var<workgroup> sum_shared : array<f32, 1>;
@compute @workgroup_size(256, 1, 1)
fn fuse_add_norm_decode_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var add_local : array<f16, 12>;
  var sum_local : array<f32, 1>;
  add_local[0i] = (A[((v__1 * 3072i) + i32(threadIdx.x))] + B[((v__1 * 3072i) + i32(threadIdx.x))]);
  add[((v__1 * 3072i) + i32(threadIdx.x))] = add_local[0i];
  add_local[1i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 256i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 256i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 256i)] = add_local[1i];
  add_local[2i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 512i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 512i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 512i)] = add_local[2i];
  add_local[3i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 768i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 768i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 768i)] = add_local[3i];
  add_local[4i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 1024i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 1024i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 1024i)] = add_local[4i];
  add_local[5i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 1280i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 1280i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 1280i)] = add_local[5i];
  add_local[6i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 1536i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 1536i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 1536i)] = add_local[6i];
  add_local[7i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 1792i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 1792i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 1792i)] = add_local[7i];
  add_local[8i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 2048i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 2048i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 2048i)] = add_local[8i];
  add_local[9i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 2304i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 2304i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 2304i)] = add_local[9i];
  add_local[10i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 2560i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 2560i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 2560i)] = add_local[10i];
  add_local[11i] = (A[(((v__1 * 3072i) + i32(threadIdx.x)) + 2816i)] + B[(((v__1 * 3072i) + i32(threadIdx.x)) + 2816i)]);
  add[(((v__1 * 3072i) + i32(threadIdx.x)) + 2816i)] = add_local[11i];
  sum_local[0i] = 0.000000e+00f;
  sum_local[0i] = fma(f32(add_local[0i]), f32(add_local[0i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[1i]), f32(add_local[1i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[2i]), f32(add_local[2i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[3i]), f32(add_local[3i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[4i]), f32(add_local[4i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[5i]), f32(add_local[5i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[6i]), f32(add_local[6i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[7i]), f32(add_local[7i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[8i]), f32(add_local[8i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[9i]), f32(add_local[9i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[10i]), f32(add_local[10i]), sum_local[0i]);
  sum_local[0i] = fma(f32(add_local[11i]), f32(add_local[11i]), sum_local[0i]);
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = sum_local[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 128i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 128i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 64i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 64i)]);
  }
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
    sum_shared[0i] = red_buf0[0i];
  }
  workgroupBarrier();
  O[((v__1 * 3072i) + i32(threadIdx.x))] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[0i])) * f32(C[i32(threadIdx.x)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 256i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[1i])) * f32(C[(i32(threadIdx.x) + 256i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 512i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[2i])) * f32(C[(i32(threadIdx.x) + 512i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 768i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[3i])) * f32(C[(i32(threadIdx.x) + 768i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 1024i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[4i])) * f32(C[(i32(threadIdx.x) + 1024i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 1280i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[5i])) * f32(C[(i32(threadIdx.x) + 1280i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 1536i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[6i])) * f32(C[(i32(threadIdx.x) + 1536i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 1792i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[7i])) * f32(C[(i32(threadIdx.x) + 1792i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 2048i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[8i])) * f32(C[(i32(threadIdx.x) + 2048i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 2304i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[9i])) * f32(C[(i32(threadIdx.x) + 2304i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 2560i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[10i])) * f32(C[(i32(threadIdx.x) + 2560i)])));
  O[(((v__1 * 3072i) + i32(threadIdx.x)) + 2816i)] = f16((((1.000000e+00f / sqrt(fma(sum_shared[0i], 3.255208e-04f, 1.000000e-05f))) * f32(add_local[11i])) * f32(C[(i32(threadIdx.x) + 2816i)])));
}
