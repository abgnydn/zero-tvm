// [17] softmax_with_chunked_sum_kernel (shader #17, 250 lines)
//----------------------------------------
// Function: softmax_with_chunked_sum_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> chunked_max : array<f32>;
@group(0) @binding(2) var<storage, read> chunked_sum : array<f32>;
@group(0) @binding(3) var<storage, read_write> softmax : array<f32>;
@group(0) @binding(4) var<storage, read> temperature : array<f32>;

struct PODArgs {
  batch_size: i32,
  num_chunks: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f32, 256>;
var<workgroup> temp_max_shared : array<f32, 1>;
var<workgroup> red_buf0_1 : array<f32, 256>;
var<workgroup> temp_sum_shared : array<f32, 1>;
@compute @workgroup_size(32, 8, 1)
fn softmax_with_chunked_sum_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var in_thread_temp_max_shared : array<f32, 1>;
  var in_thread_temp_sum_shared : array<f32, 1>;
  in_thread_temp_max_shared[0i] = -3.402823e+38f;
  for (var ax0_0 : i32 = 0; ax0_0 < ((podArgs.num_chunks + 31i)>>5u); ax0_0++) {
    if (((ax0_0 * 32i) + i32(threadIdx.x)) < podArgs.num_chunks) {
      in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], chunked_max[(((ax0_0 * 32i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.num_chunks)) + i32(threadIdx.x))]);
    }
  }
  workgroupBarrier();
  red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = in_thread_temp_max_shared[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i) {
    red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = max(red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))], red_buf0[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i) {
    red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = max(red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))], red_buf0[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 8i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 4i) {
    red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = max(red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))], red_buf0[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 4i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = max(red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))], red_buf0[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 2i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i) {
    red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = max(red_buf0[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))], red_buf0[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 1i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    temp_max_shared[0i] = red_buf0[(i32(threadIdx.y) * 32i)];
  }
  in_thread_temp_sum_shared[0i] = 0.000000e+00f;
  workgroupBarrier();
  for (var ax0_0_1 : i32 = 0; ax0_0_1 < ((podArgs.num_chunks + 31i)>>5u); ax0_0_1++) {
    if (((ax0_0_1 * 32i) + i32(threadIdx.x)) < podArgs.num_chunks) {
      var condval : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval = exp(((chunked_sum[(((ax0_0_1 * 32i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.num_chunks)) + i32(threadIdx.x))] + chunked_max[(((ax0_0_1 * 32i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.num_chunks)) + i32(threadIdx.x))]) - temp_max_shared[0i]));
} else {
        condval = (f32((chunked_max[(((ax0_0_1 * 32i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.num_chunks)) + i32(threadIdx.x))] == temp_max_shared[0i])) * chunked_sum[(((ax0_0_1 * 32i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.num_chunks)) + i32(threadIdx.x))]);
}
      in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval);
    }
  }
  workgroupBarrier();
  red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = in_thread_temp_sum_shared[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i) {
    red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = (red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] + red_buf0_1[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i) {
    red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = (red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] + red_buf0_1[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 8i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 4i) {
    red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = (red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] + red_buf0_1[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 4i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = (red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] + red_buf0_1[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 2i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i) {
    red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = (red_buf0_1[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] + red_buf0_1[(((i32(threadIdx.y) * 32i) + i32(threadIdx.x)) + 1i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    temp_sum_shared[0i] = red_buf0_1[(i32(threadIdx.y) * 32i)];
  }
  workgroupBarrier();
  if (((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) < podArgs.vocab_size) {
    var condval_1 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_1 = exp(((A[(((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x))] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_1 = (f32((A[(((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x))] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[(((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x))] = condval_1;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 256i) < podArgs.vocab_size) {
    var condval_2 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_2 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 256i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_2 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 256i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 256i)] = condval_2;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 512i) < podArgs.vocab_size) {
    var condval_3 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_3 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 512i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_3 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 512i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 512i)] = condval_3;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 768i) < podArgs.vocab_size) {
    var condval_4 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_4 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 768i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_4 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 768i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 768i)] = condval_4;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 1024i) < podArgs.vocab_size) {
    var condval_5 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_5 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1024i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_5 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1024i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1024i)] = condval_5;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 1280i) < podArgs.vocab_size) {
    var condval_6 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_6 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1280i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_6 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1280i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1280i)] = condval_6;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 1536i) < podArgs.vocab_size) {
    var condval_7 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_7 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1536i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_7 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1536i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1536i)] = condval_7;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 1792i) < podArgs.vocab_size) {
    var condval_8 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_8 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1792i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_8 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1792i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1792i)] = condval_8;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 2048i) < podArgs.vocab_size) {
    var condval_9 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_9 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2048i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_9 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2048i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2048i)] = condval_9;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 2304i) < podArgs.vocab_size) {
    var condval_10 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_10 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2304i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_10 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2304i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2304i)] = condval_10;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 2560i) < podArgs.vocab_size) {
    var condval_11 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_11 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2560i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_11 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2560i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2560i)] = condval_11;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 2816i) < podArgs.vocab_size) {
    var condval_12 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_12 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2816i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_12 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2816i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2816i)] = condval_12;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 3072i) < podArgs.vocab_size) {
    var condval_13 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_13 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3072i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_13 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3072i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3072i)] = condval_13;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 3328i) < podArgs.vocab_size) {
    var condval_14 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_14 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3328i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_14 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3328i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3328i)] = condval_14;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 3584i) < podArgs.vocab_size) {
    var condval_15 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_15 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3584i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_15 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3584i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3584i)] = condval_15;
  }
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + i32(threadIdx.x)) + 3840i) < podArgs.vocab_size) {
    var condval_16 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_16 = exp(((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3840i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]) - (log(temp_sum_shared[0i]) + temp_max_shared[0i])));
} else {
      condval_16 = (f32((A[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3840i)] == temp_max_shared[0i])) / temp_sum_shared[0i]);
}
    softmax[((((((v__1 % podArgs.num_chunks) * 4096i) + (i32(threadIdx.y) * 32i)) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3840i)] = condval_16;
  }
}
