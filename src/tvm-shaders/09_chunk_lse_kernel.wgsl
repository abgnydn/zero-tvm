// [9] chunk_lse_kernel (shader #9, 2539 lines)
//----------------------------------------
// Function: chunk_lse_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read_write> chunked_max : array<f32>;
@group(0) @binding(2) var<storage, read_write> chunked_sum : array<f32>;
@group(0) @binding(3) var<storage, read> temperature : array<f32>;

struct PODArgs {
  batch_size: i32,
  num_chunks: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f32, 64>;
var<workgroup> temp_max_shared : array<f32, 1>;
var<workgroup> red_buf0_1 : array<f32, 64>;
var<workgroup> temp_sum_shared : array<f32, 1>;
@compute @workgroup_size(64, 1, 1)
fn chunk_lse_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var in_thread_temp_max_shared : array<f32, 1>;
  var in_thread_temp_sum_shared : array<f32, 1>;
  in_thread_temp_max_shared[0i] = -3.402823e+38f;
  var condval : f32;
  if (((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) < podArgs.vocab_size)) {
    var condval_1 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_1 = (A[((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x))] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_1 = A[((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x))];
}
    condval = condval_1;
} else {
    condval = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval);
  var condval_2 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 64i) < podArgs.vocab_size)) {
    var condval_3 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_3 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 64i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_3 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 64i)];
}
    condval_2 = condval_3;
} else {
    condval_2 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_2);
  var condval_4 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 128i) < podArgs.vocab_size)) {
    var condval_5 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_5 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 128i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_5 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 128i)];
}
    condval_4 = condval_5;
} else {
    condval_4 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_4);
  var condval_6 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 192i) < podArgs.vocab_size)) {
    var condval_7 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_7 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 192i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_7 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 192i)];
}
    condval_6 = condval_7;
} else {
    condval_6 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_6);
  var condval_8 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 256i) < podArgs.vocab_size)) {
    var condval_9 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_9 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 256i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_9 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 256i)];
}
    condval_8 = condval_9;
} else {
    condval_8 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_8);
  var condval_10 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 320i) < podArgs.vocab_size)) {
    var condval_11 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_11 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 320i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_11 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 320i)];
}
    condval_10 = condval_11;
} else {
    condval_10 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_10);
  var condval_12 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 384i) < podArgs.vocab_size)) {
    var condval_13 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_13 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 384i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_13 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 384i)];
}
    condval_12 = condval_13;
} else {
    condval_12 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_12);
  var condval_14 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 448i) < podArgs.vocab_size)) {
    var condval_15 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_15 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 448i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_15 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 448i)];
}
    condval_14 = condval_15;
} else {
    condval_14 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_14);
  var condval_16 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 512i) < podArgs.vocab_size)) {
    var condval_17 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_17 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 512i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_17 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 512i)];
}
    condval_16 = condval_17;
} else {
    condval_16 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_16);
  var condval_18 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 576i) < podArgs.vocab_size)) {
    var condval_19 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_19 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 576i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_19 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 576i)];
}
    condval_18 = condval_19;
} else {
    condval_18 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_18);
  var condval_20 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 640i) < podArgs.vocab_size)) {
    var condval_21 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_21 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 640i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_21 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 640i)];
}
    condval_20 = condval_21;
} else {
    condval_20 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_20);
  var condval_22 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 704i) < podArgs.vocab_size)) {
    var condval_23 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_23 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 704i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_23 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 704i)];
}
    condval_22 = condval_23;
} else {
    condval_22 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_22);
  var condval_24 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 768i) < podArgs.vocab_size)) {
    var condval_25 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_25 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 768i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_25 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 768i)];
}
    condval_24 = condval_25;
} else {
    condval_24 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_24);
  var condval_26 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 832i) < podArgs.vocab_size)) {
    var condval_27 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_27 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 832i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_27 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 832i)];
}
    condval_26 = condval_27;
} else {
    condval_26 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_26);
  var condval_28 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 896i) < podArgs.vocab_size)) {
    var condval_29 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_29 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 896i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_29 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 896i)];
}
    condval_28 = condval_29;
} else {
    condval_28 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_28);
  var condval_30 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 960i) < podArgs.vocab_size)) {
    var condval_31 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_31 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 960i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_31 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 960i)];
}
    condval_30 = condval_31;
} else {
    condval_30 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_30);
  var condval_32 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1024i) < podArgs.vocab_size)) {
    var condval_33 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_33 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1024i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_33 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1024i)];
}
    condval_32 = condval_33;
} else {
    condval_32 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_32);
  var condval_34 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1088i) < podArgs.vocab_size)) {
    var condval_35 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_35 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1088i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_35 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1088i)];
}
    condval_34 = condval_35;
} else {
    condval_34 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_34);
  var condval_36 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1152i) < podArgs.vocab_size)) {
    var condval_37 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_37 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1152i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_37 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1152i)];
}
    condval_36 = condval_37;
} else {
    condval_36 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_36);
  var condval_38 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1216i) < podArgs.vocab_size)) {
    var condval_39 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_39 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1216i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_39 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1216i)];
}
    condval_38 = condval_39;
} else {
    condval_38 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_38);
  var condval_40 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1280i) < podArgs.vocab_size)) {
    var condval_41 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_41 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1280i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_41 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1280i)];
}
    condval_40 = condval_41;
} else {
    condval_40 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_40);
  var condval_42 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1344i) < podArgs.vocab_size)) {
    var condval_43 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_43 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1344i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_43 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1344i)];
}
    condval_42 = condval_43;
} else {
    condval_42 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_42);
  var condval_44 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1408i) < podArgs.vocab_size)) {
    var condval_45 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_45 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1408i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_45 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1408i)];
}
    condval_44 = condval_45;
} else {
    condval_44 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_44);
  var condval_46 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1472i) < podArgs.vocab_size)) {
    var condval_47 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_47 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1472i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_47 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1472i)];
}
    condval_46 = condval_47;
} else {
    condval_46 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_46);
  var condval_48 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1536i) < podArgs.vocab_size)) {
    var condval_49 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_49 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1536i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_49 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1536i)];
}
    condval_48 = condval_49;
} else {
    condval_48 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_48);
  var condval_50 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1600i) < podArgs.vocab_size)) {
    var condval_51 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_51 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1600i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_51 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1600i)];
}
    condval_50 = condval_51;
} else {
    condval_50 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_50);
  var condval_52 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1664i) < podArgs.vocab_size)) {
    var condval_53 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_53 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1664i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_53 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1664i)];
}
    condval_52 = condval_53;
} else {
    condval_52 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_52);
  var condval_54 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1728i) < podArgs.vocab_size)) {
    var condval_55 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_55 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1728i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_55 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1728i)];
}
    condval_54 = condval_55;
} else {
    condval_54 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_54);
  var condval_56 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1792i) < podArgs.vocab_size)) {
    var condval_57 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_57 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1792i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_57 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1792i)];
}
    condval_56 = condval_57;
} else {
    condval_56 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_56);
  var condval_58 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1856i) < podArgs.vocab_size)) {
    var condval_59 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_59 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1856i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_59 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1856i)];
}
    condval_58 = condval_59;
} else {
    condval_58 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_58);
  var condval_60 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1920i) < podArgs.vocab_size)) {
    var condval_61 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_61 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1920i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_61 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1920i)];
}
    condval_60 = condval_61;
} else {
    condval_60 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_60);
  var condval_62 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1984i) < podArgs.vocab_size)) {
    var condval_63 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_63 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1984i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_63 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1984i)];
}
    condval_62 = condval_63;
} else {
    condval_62 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_62);
  var condval_64 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2048i) < podArgs.vocab_size)) {
    var condval_65 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_65 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2048i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_65 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2048i)];
}
    condval_64 = condval_65;
} else {
    condval_64 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_64);
  var condval_66 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2112i) < podArgs.vocab_size)) {
    var condval_67 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_67 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2112i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_67 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2112i)];
}
    condval_66 = condval_67;
} else {
    condval_66 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_66);
  var condval_68 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2176i) < podArgs.vocab_size)) {
    var condval_69 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_69 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2176i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_69 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2176i)];
}
    condval_68 = condval_69;
} else {
    condval_68 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_68);
  var condval_70 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2240i) < podArgs.vocab_size)) {
    var condval_71 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_71 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2240i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_71 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2240i)];
}
    condval_70 = condval_71;
} else {
    condval_70 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_70);
  var condval_72 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2304i) < podArgs.vocab_size)) {
    var condval_73 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_73 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2304i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_73 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2304i)];
}
    condval_72 = condval_73;
} else {
    condval_72 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_72);
  var condval_74 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2368i) < podArgs.vocab_size)) {
    var condval_75 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_75 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2368i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_75 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2368i)];
}
    condval_74 = condval_75;
} else {
    condval_74 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_74);
  var condval_76 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2432i) < podArgs.vocab_size)) {
    var condval_77 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_77 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2432i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_77 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2432i)];
}
    condval_76 = condval_77;
} else {
    condval_76 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_76);
  var condval_78 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2496i) < podArgs.vocab_size)) {
    var condval_79 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_79 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2496i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_79 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2496i)];
}
    condval_78 = condval_79;
} else {
    condval_78 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_78);
  var condval_80 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2560i) < podArgs.vocab_size)) {
    var condval_81 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_81 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2560i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_81 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2560i)];
}
    condval_80 = condval_81;
} else {
    condval_80 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_80);
  var condval_82 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2624i) < podArgs.vocab_size)) {
    var condval_83 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_83 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2624i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_83 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2624i)];
}
    condval_82 = condval_83;
} else {
    condval_82 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_82);
  var condval_84 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2688i) < podArgs.vocab_size)) {
    var condval_85 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_85 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2688i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_85 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2688i)];
}
    condval_84 = condval_85;
} else {
    condval_84 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_84);
  var condval_86 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2752i) < podArgs.vocab_size)) {
    var condval_87 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_87 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2752i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_87 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2752i)];
}
    condval_86 = condval_87;
} else {
    condval_86 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_86);
  var condval_88 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2816i) < podArgs.vocab_size)) {
    var condval_89 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_89 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2816i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_89 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2816i)];
}
    condval_88 = condval_89;
} else {
    condval_88 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_88);
  var condval_90 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2880i) < podArgs.vocab_size)) {
    var condval_91 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_91 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2880i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_91 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2880i)];
}
    condval_90 = condval_91;
} else {
    condval_90 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_90);
  var condval_92 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2944i) < podArgs.vocab_size)) {
    var condval_93 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_93 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2944i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_93 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2944i)];
}
    condval_92 = condval_93;
} else {
    condval_92 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_92);
  var condval_94 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3008i) < podArgs.vocab_size)) {
    var condval_95 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_95 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3008i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_95 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3008i)];
}
    condval_94 = condval_95;
} else {
    condval_94 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_94);
  var condval_96 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3072i) < podArgs.vocab_size)) {
    var condval_97 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_97 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3072i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_97 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3072i)];
}
    condval_96 = condval_97;
} else {
    condval_96 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_96);
  var condval_98 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3136i) < podArgs.vocab_size)) {
    var condval_99 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_99 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3136i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_99 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3136i)];
}
    condval_98 = condval_99;
} else {
    condval_98 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_98);
  var condval_100 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3200i) < podArgs.vocab_size)) {
    var condval_101 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_101 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3200i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_101 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3200i)];
}
    condval_100 = condval_101;
} else {
    condval_100 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_100);
  var condval_102 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3264i) < podArgs.vocab_size)) {
    var condval_103 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_103 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3264i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_103 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3264i)];
}
    condval_102 = condval_103;
} else {
    condval_102 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_102);
  var condval_104 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3328i) < podArgs.vocab_size)) {
    var condval_105 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_105 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3328i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_105 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3328i)];
}
    condval_104 = condval_105;
} else {
    condval_104 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_104);
  var condval_106 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3392i) < podArgs.vocab_size)) {
    var condval_107 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_107 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3392i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_107 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3392i)];
}
    condval_106 = condval_107;
} else {
    condval_106 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_106);
  var condval_108 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3456i) < podArgs.vocab_size)) {
    var condval_109 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_109 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3456i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_109 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3456i)];
}
    condval_108 = condval_109;
} else {
    condval_108 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_108);
  var condval_110 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3520i) < podArgs.vocab_size)) {
    var condval_111 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_111 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3520i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_111 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3520i)];
}
    condval_110 = condval_111;
} else {
    condval_110 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_110);
  var condval_112 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3584i) < podArgs.vocab_size)) {
    var condval_113 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_113 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3584i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_113 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3584i)];
}
    condval_112 = condval_113;
} else {
    condval_112 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_112);
  var condval_114 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3648i) < podArgs.vocab_size)) {
    var condval_115 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_115 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3648i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_115 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3648i)];
}
    condval_114 = condval_115;
} else {
    condval_114 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_114);
  var condval_116 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3712i) < podArgs.vocab_size)) {
    var condval_117 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_117 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3712i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_117 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3712i)];
}
    condval_116 = condval_117;
} else {
    condval_116 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_116);
  var condval_118 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3776i) < podArgs.vocab_size)) {
    var condval_119 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_119 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3776i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_119 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3776i)];
}
    condval_118 = condval_119;
} else {
    condval_118 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_118);
  var condval_120 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3840i) < podArgs.vocab_size)) {
    var condval_121 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_121 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3840i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_121 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3840i)];
}
    condval_120 = condval_121;
} else {
    condval_120 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_120);
  var condval_122 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3904i) < podArgs.vocab_size)) {
    var condval_123 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_123 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3904i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_123 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3904i)];
}
    condval_122 = condval_123;
} else {
    condval_122 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_122);
  var condval_124 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3968i) < podArgs.vocab_size)) {
    var condval_125 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_125 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3968i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_125 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3968i)];
}
    condval_124 = condval_125;
} else {
    condval_124 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_124);
  var condval_126 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 4032i) < podArgs.vocab_size)) {
    var condval_127 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_127 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 4032i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
      condval_127 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 4032i)];
}
    condval_126 = condval_127;
} else {
    condval_126 = -3.402823e+38f;
}
  in_thread_temp_max_shared[0i] = max(in_thread_temp_max_shared[0i], condval_126);
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = in_thread_temp_max_shared[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 32i) {
    red_buf0[i32(threadIdx.x)] = max(red_buf0[i32(threadIdx.x)], red_buf0[(i32(threadIdx.x) + 32i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i) {
    red_buf0[i32(threadIdx.x)] = max(red_buf0[i32(threadIdx.x)], red_buf0[(i32(threadIdx.x) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i) {
    red_buf0[i32(threadIdx.x)] = max(red_buf0[i32(threadIdx.x)], red_buf0[(i32(threadIdx.x) + 8i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 4i) {
    red_buf0[i32(threadIdx.x)] = max(red_buf0[i32(threadIdx.x)], red_buf0[(i32(threadIdx.x) + 4i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0[i32(threadIdx.x)] = max(red_buf0[i32(threadIdx.x)], red_buf0[(i32(threadIdx.x) + 2i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i) {
    red_buf0[i32(threadIdx.x)] = max(red_buf0[i32(threadIdx.x)], red_buf0[(i32(threadIdx.x) + 1i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    temp_max_shared[0i] = red_buf0[0i];
  }
  in_thread_temp_sum_shared[0i] = 0.000000e+00f;
  workgroupBarrier();
  var condval_128 : f32;
  if (((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) < podArgs.vocab_size)) {
    var condval_129 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_130 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_130 = (A[((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x))] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_130 = A[((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x))];
}
      condval_129 = exp((condval_130 - temp_max_shared[0i]));
} else {
      var condval_131 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_131 = (A[((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x))] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_131 = A[((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x))];
}
      condval_129 = f32((condval_131 == temp_max_shared[0i]));
}
    condval_128 = condval_129;
} else {
    condval_128 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_128);
  var condval_132 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 64i) < podArgs.vocab_size)) {
    var condval_133 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_134 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_134 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 64i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_134 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 64i)];
}
      condval_133 = exp((condval_134 - temp_max_shared[0i]));
} else {
      var condval_135 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_135 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 64i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_135 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 64i)];
}
      condval_133 = f32((condval_135 == temp_max_shared[0i]));
}
    condval_132 = condval_133;
} else {
    condval_132 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_132);
  var condval_136 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 128i) < podArgs.vocab_size)) {
    var condval_137 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_138 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_138 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 128i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_138 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 128i)];
}
      condval_137 = exp((condval_138 - temp_max_shared[0i]));
} else {
      var condval_139 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_139 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 128i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_139 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 128i)];
}
      condval_137 = f32((condval_139 == temp_max_shared[0i]));
}
    condval_136 = condval_137;
} else {
    condval_136 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_136);
  var condval_140 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 192i) < podArgs.vocab_size)) {
    var condval_141 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_142 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_142 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 192i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_142 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 192i)];
}
      condval_141 = exp((condval_142 - temp_max_shared[0i]));
} else {
      var condval_143 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_143 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 192i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_143 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 192i)];
}
      condval_141 = f32((condval_143 == temp_max_shared[0i]));
}
    condval_140 = condval_141;
} else {
    condval_140 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_140);
  var condval_144 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 256i) < podArgs.vocab_size)) {
    var condval_145 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_146 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_146 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 256i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_146 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 256i)];
}
      condval_145 = exp((condval_146 - temp_max_shared[0i]));
} else {
      var condval_147 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_147 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 256i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_147 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 256i)];
}
      condval_145 = f32((condval_147 == temp_max_shared[0i]));
}
    condval_144 = condval_145;
} else {
    condval_144 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_144);
  var condval_148 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 320i) < podArgs.vocab_size)) {
    var condval_149 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_150 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_150 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 320i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_150 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 320i)];
}
      condval_149 = exp((condval_150 - temp_max_shared[0i]));
} else {
      var condval_151 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_151 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 320i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_151 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 320i)];
}
      condval_149 = f32((condval_151 == temp_max_shared[0i]));
}
    condval_148 = condval_149;
} else {
    condval_148 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_148);
  var condval_152 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 384i) < podArgs.vocab_size)) {
    var condval_153 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_154 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_154 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 384i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_154 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 384i)];
}
      condval_153 = exp((condval_154 - temp_max_shared[0i]));
} else {
      var condval_155 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_155 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 384i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_155 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 384i)];
}
      condval_153 = f32((condval_155 == temp_max_shared[0i]));
}
    condval_152 = condval_153;
} else {
    condval_152 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_152);
  var condval_156 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 448i) < podArgs.vocab_size)) {
    var condval_157 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_158 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_158 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 448i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_158 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 448i)];
}
      condval_157 = exp((condval_158 - temp_max_shared[0i]));
} else {
      var condval_159 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_159 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 448i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_159 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 448i)];
}
      condval_157 = f32((condval_159 == temp_max_shared[0i]));
}
    condval_156 = condval_157;
} else {
    condval_156 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_156);
  var condval_160 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 512i) < podArgs.vocab_size)) {
    var condval_161 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_162 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_162 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 512i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_162 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 512i)];
}
      condval_161 = exp((condval_162 - temp_max_shared[0i]));
} else {
      var condval_163 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_163 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 512i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_163 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 512i)];
}
      condval_161 = f32((condval_163 == temp_max_shared[0i]));
}
    condval_160 = condval_161;
} else {
    condval_160 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_160);
  var condval_164 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 576i) < podArgs.vocab_size)) {
    var condval_165 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_166 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_166 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 576i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_166 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 576i)];
}
      condval_165 = exp((condval_166 - temp_max_shared[0i]));
} else {
      var condval_167 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_167 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 576i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_167 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 576i)];
}
      condval_165 = f32((condval_167 == temp_max_shared[0i]));
}
    condval_164 = condval_165;
} else {
    condval_164 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_164);
  var condval_168 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 640i) < podArgs.vocab_size)) {
    var condval_169 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_170 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_170 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 640i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_170 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 640i)];
}
      condval_169 = exp((condval_170 - temp_max_shared[0i]));
} else {
      var condval_171 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_171 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 640i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_171 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 640i)];
}
      condval_169 = f32((condval_171 == temp_max_shared[0i]));
}
    condval_168 = condval_169;
} else {
    condval_168 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_168);
  var condval_172 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 704i) < podArgs.vocab_size)) {
    var condval_173 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_174 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_174 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 704i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_174 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 704i)];
}
      condval_173 = exp((condval_174 - temp_max_shared[0i]));
} else {
      var condval_175 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_175 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 704i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_175 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 704i)];
}
      condval_173 = f32((condval_175 == temp_max_shared[0i]));
}
    condval_172 = condval_173;
} else {
    condval_172 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_172);
  var condval_176 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 768i) < podArgs.vocab_size)) {
    var condval_177 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_178 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_178 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 768i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_178 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 768i)];
}
      condval_177 = exp((condval_178 - temp_max_shared[0i]));
} else {
      var condval_179 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_179 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 768i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_179 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 768i)];
}
      condval_177 = f32((condval_179 == temp_max_shared[0i]));
}
    condval_176 = condval_177;
} else {
    condval_176 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_176);
  var condval_180 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 832i) < podArgs.vocab_size)) {
    var condval_181 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_182 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_182 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 832i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_182 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 832i)];
}
      condval_181 = exp((condval_182 - temp_max_shared[0i]));
} else {
      var condval_183 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_183 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 832i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_183 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 832i)];
}
      condval_181 = f32((condval_183 == temp_max_shared[0i]));
}
    condval_180 = condval_181;
} else {
    condval_180 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_180);
  var condval_184 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 896i) < podArgs.vocab_size)) {
    var condval_185 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_186 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_186 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 896i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_186 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 896i)];
}
      condval_185 = exp((condval_186 - temp_max_shared[0i]));
} else {
      var condval_187 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_187 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 896i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_187 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 896i)];
}
      condval_185 = f32((condval_187 == temp_max_shared[0i]));
}
    condval_184 = condval_185;
} else {
    condval_184 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_184);
  var condval_188 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 960i) < podArgs.vocab_size)) {
    var condval_189 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_190 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_190 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 960i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_190 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 960i)];
}
      condval_189 = exp((condval_190 - temp_max_shared[0i]));
} else {
      var condval_191 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_191 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 960i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_191 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 960i)];
}
      condval_189 = f32((condval_191 == temp_max_shared[0i]));
}
    condval_188 = condval_189;
} else {
    condval_188 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_188);
  var condval_192 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1024i) < podArgs.vocab_size)) {
    var condval_193 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_194 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_194 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1024i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_194 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1024i)];
}
      condval_193 = exp((condval_194 - temp_max_shared[0i]));
} else {
      var condval_195 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_195 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1024i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_195 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1024i)];
}
      condval_193 = f32((condval_195 == temp_max_shared[0i]));
}
    condval_192 = condval_193;
} else {
    condval_192 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_192);
  var condval_196 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1088i) < podArgs.vocab_size)) {
    var condval_197 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_198 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_198 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1088i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_198 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1088i)];
}
      condval_197 = exp((condval_198 - temp_max_shared[0i]));
} else {
      var condval_199 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_199 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1088i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_199 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1088i)];
}
      condval_197 = f32((condval_199 == temp_max_shared[0i]));
}
    condval_196 = condval_197;
} else {
    condval_196 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_196);
  var condval_200 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1152i) < podArgs.vocab_size)) {
    var condval_201 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_202 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_202 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1152i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_202 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1152i)];
}
      condval_201 = exp((condval_202 - temp_max_shared[0i]));
} else {
      var condval_203 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_203 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1152i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_203 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1152i)];
}
      condval_201 = f32((condval_203 == temp_max_shared[0i]));
}
    condval_200 = condval_201;
} else {
    condval_200 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_200);
  var condval_204 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1216i) < podArgs.vocab_size)) {
    var condval_205 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_206 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_206 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1216i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_206 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1216i)];
}
      condval_205 = exp((condval_206 - temp_max_shared[0i]));
} else {
      var condval_207 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_207 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1216i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_207 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1216i)];
}
      condval_205 = f32((condval_207 == temp_max_shared[0i]));
}
    condval_204 = condval_205;
} else {
    condval_204 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_204);
  var condval_208 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1280i) < podArgs.vocab_size)) {
    var condval_209 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_210 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_210 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1280i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_210 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1280i)];
}
      condval_209 = exp((condval_210 - temp_max_shared[0i]));
} else {
      var condval_211 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_211 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1280i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_211 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1280i)];
}
      condval_209 = f32((condval_211 == temp_max_shared[0i]));
}
    condval_208 = condval_209;
} else {
    condval_208 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_208);
  var condval_212 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1344i) < podArgs.vocab_size)) {
    var condval_213 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_214 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_214 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1344i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_214 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1344i)];
}
      condval_213 = exp((condval_214 - temp_max_shared[0i]));
} else {
      var condval_215 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_215 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1344i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_215 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1344i)];
}
      condval_213 = f32((condval_215 == temp_max_shared[0i]));
}
    condval_212 = condval_213;
} else {
    condval_212 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_212);
  var condval_216 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1408i) < podArgs.vocab_size)) {
    var condval_217 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_218 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_218 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1408i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_218 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1408i)];
}
      condval_217 = exp((condval_218 - temp_max_shared[0i]));
} else {
      var condval_219 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_219 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1408i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_219 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1408i)];
}
      condval_217 = f32((condval_219 == temp_max_shared[0i]));
}
    condval_216 = condval_217;
} else {
    condval_216 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_216);
  var condval_220 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1472i) < podArgs.vocab_size)) {
    var condval_221 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_222 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_222 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1472i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_222 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1472i)];
}
      condval_221 = exp((condval_222 - temp_max_shared[0i]));
} else {
      var condval_223 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_223 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1472i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_223 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1472i)];
}
      condval_221 = f32((condval_223 == temp_max_shared[0i]));
}
    condval_220 = condval_221;
} else {
    condval_220 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_220);
  var condval_224 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1536i) < podArgs.vocab_size)) {
    var condval_225 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_226 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_226 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1536i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_226 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1536i)];
}
      condval_225 = exp((condval_226 - temp_max_shared[0i]));
} else {
      var condval_227 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_227 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1536i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_227 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1536i)];
}
      condval_225 = f32((condval_227 == temp_max_shared[0i]));
}
    condval_224 = condval_225;
} else {
    condval_224 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_224);
  var condval_228 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1600i) < podArgs.vocab_size)) {
    var condval_229 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_230 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_230 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1600i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_230 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1600i)];
}
      condval_229 = exp((condval_230 - temp_max_shared[0i]));
} else {
      var condval_231 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_231 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1600i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_231 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1600i)];
}
      condval_229 = f32((condval_231 == temp_max_shared[0i]));
}
    condval_228 = condval_229;
} else {
    condval_228 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_228);
  var condval_232 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1664i) < podArgs.vocab_size)) {
    var condval_233 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_234 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_234 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1664i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_234 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1664i)];
}
      condval_233 = exp((condval_234 - temp_max_shared[0i]));
} else {
      var condval_235 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_235 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1664i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_235 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1664i)];
}
      condval_233 = f32((condval_235 == temp_max_shared[0i]));
}
    condval_232 = condval_233;
} else {
    condval_232 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_232);
  var condval_236 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1728i) < podArgs.vocab_size)) {
    var condval_237 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_238 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_238 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1728i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_238 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1728i)];
}
      condval_237 = exp((condval_238 - temp_max_shared[0i]));
} else {
      var condval_239 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_239 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1728i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_239 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1728i)];
}
      condval_237 = f32((condval_239 == temp_max_shared[0i]));
}
    condval_236 = condval_237;
} else {
    condval_236 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_236);
  var condval_240 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1792i) < podArgs.vocab_size)) {
    var condval_241 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_242 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_242 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1792i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_242 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1792i)];
}
      condval_241 = exp((condval_242 - temp_max_shared[0i]));
} else {
      var condval_243 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_243 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1792i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_243 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1792i)];
}
      condval_241 = f32((condval_243 == temp_max_shared[0i]));
}
    condval_240 = condval_241;
} else {
    condval_240 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_240);
  var condval_244 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1856i) < podArgs.vocab_size)) {
    var condval_245 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_246 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_246 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1856i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_246 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1856i)];
}
      condval_245 = exp((condval_246 - temp_max_shared[0i]));
} else {
      var condval_247 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_247 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1856i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_247 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1856i)];
}
      condval_245 = f32((condval_247 == temp_max_shared[0i]));
}
    condval_244 = condval_245;
} else {
    condval_244 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_244);
  var condval_248 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1920i) < podArgs.vocab_size)) {
    var condval_249 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_250 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_250 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1920i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_250 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1920i)];
}
      condval_249 = exp((condval_250 - temp_max_shared[0i]));
} else {
      var condval_251 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_251 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1920i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_251 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1920i)];
}
      condval_249 = f32((condval_251 == temp_max_shared[0i]));
}
    condval_248 = condval_249;
} else {
    condval_248 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_248);
  var condval_252 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 1984i) < podArgs.vocab_size)) {
    var condval_253 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_254 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_254 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1984i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_254 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1984i)];
}
      condval_253 = exp((condval_254 - temp_max_shared[0i]));
} else {
      var condval_255 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_255 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1984i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_255 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 1984i)];
}
      condval_253 = f32((condval_255 == temp_max_shared[0i]));
}
    condval_252 = condval_253;
} else {
    condval_252 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_252);
  var condval_256 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2048i) < podArgs.vocab_size)) {
    var condval_257 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_258 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_258 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2048i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_258 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2048i)];
}
      condval_257 = exp((condval_258 - temp_max_shared[0i]));
} else {
      var condval_259 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_259 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2048i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_259 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2048i)];
}
      condval_257 = f32((condval_259 == temp_max_shared[0i]));
}
    condval_256 = condval_257;
} else {
    condval_256 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_256);
  var condval_260 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2112i) < podArgs.vocab_size)) {
    var condval_261 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_262 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_262 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2112i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_262 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2112i)];
}
      condval_261 = exp((condval_262 - temp_max_shared[0i]));
} else {
      var condval_263 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_263 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2112i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_263 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2112i)];
}
      condval_261 = f32((condval_263 == temp_max_shared[0i]));
}
    condval_260 = condval_261;
} else {
    condval_260 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_260);
  var condval_264 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2176i) < podArgs.vocab_size)) {
    var condval_265 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_266 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_266 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2176i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_266 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2176i)];
}
      condval_265 = exp((condval_266 - temp_max_shared[0i]));
} else {
      var condval_267 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_267 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2176i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_267 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2176i)];
}
      condval_265 = f32((condval_267 == temp_max_shared[0i]));
}
    condval_264 = condval_265;
} else {
    condval_264 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_264);
  var condval_268 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2240i) < podArgs.vocab_size)) {
    var condval_269 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_270 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_270 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2240i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_270 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2240i)];
}
      condval_269 = exp((condval_270 - temp_max_shared[0i]));
} else {
      var condval_271 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_271 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2240i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_271 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2240i)];
}
      condval_269 = f32((condval_271 == temp_max_shared[0i]));
}
    condval_268 = condval_269;
} else {
    condval_268 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_268);
  var condval_272 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2304i) < podArgs.vocab_size)) {
    var condval_273 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_274 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_274 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2304i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_274 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2304i)];
}
      condval_273 = exp((condval_274 - temp_max_shared[0i]));
} else {
      var condval_275 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_275 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2304i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_275 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2304i)];
}
      condval_273 = f32((condval_275 == temp_max_shared[0i]));
}
    condval_272 = condval_273;
} else {
    condval_272 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_272);
  var condval_276 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2368i) < podArgs.vocab_size)) {
    var condval_277 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_278 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_278 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2368i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_278 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2368i)];
}
      condval_277 = exp((condval_278 - temp_max_shared[0i]));
} else {
      var condval_279 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_279 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2368i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_279 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2368i)];
}
      condval_277 = f32((condval_279 == temp_max_shared[0i]));
}
    condval_276 = condval_277;
} else {
    condval_276 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_276);
  var condval_280 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2432i) < podArgs.vocab_size)) {
    var condval_281 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_282 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_282 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2432i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_282 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2432i)];
}
      condval_281 = exp((condval_282 - temp_max_shared[0i]));
} else {
      var condval_283 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_283 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2432i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_283 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2432i)];
}
      condval_281 = f32((condval_283 == temp_max_shared[0i]));
}
    condval_280 = condval_281;
} else {
    condval_280 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_280);
  var condval_284 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2496i) < podArgs.vocab_size)) {
    var condval_285 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_286 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_286 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2496i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_286 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2496i)];
}
      condval_285 = exp((condval_286 - temp_max_shared[0i]));
} else {
      var condval_287 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_287 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2496i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_287 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2496i)];
}
      condval_285 = f32((condval_287 == temp_max_shared[0i]));
}
    condval_284 = condval_285;
} else {
    condval_284 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_284);
  var condval_288 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2560i) < podArgs.vocab_size)) {
    var condval_289 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_290 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_290 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2560i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_290 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2560i)];
}
      condval_289 = exp((condval_290 - temp_max_shared[0i]));
} else {
      var condval_291 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_291 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2560i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_291 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2560i)];
}
      condval_289 = f32((condval_291 == temp_max_shared[0i]));
}
    condval_288 = condval_289;
} else {
    condval_288 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_288);
  var condval_292 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2624i) < podArgs.vocab_size)) {
    var condval_293 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_294 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_294 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2624i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_294 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2624i)];
}
      condval_293 = exp((condval_294 - temp_max_shared[0i]));
} else {
      var condval_295 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_295 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2624i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_295 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2624i)];
}
      condval_293 = f32((condval_295 == temp_max_shared[0i]));
}
    condval_292 = condval_293;
} else {
    condval_292 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_292);
  var condval_296 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2688i) < podArgs.vocab_size)) {
    var condval_297 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_298 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_298 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2688i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_298 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2688i)];
}
      condval_297 = exp((condval_298 - temp_max_shared[0i]));
} else {
      var condval_299 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_299 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2688i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_299 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2688i)];
}
      condval_297 = f32((condval_299 == temp_max_shared[0i]));
}
    condval_296 = condval_297;
} else {
    condval_296 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_296);
  var condval_300 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2752i) < podArgs.vocab_size)) {
    var condval_301 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_302 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_302 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2752i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_302 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2752i)];
}
      condval_301 = exp((condval_302 - temp_max_shared[0i]));
} else {
      var condval_303 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_303 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2752i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_303 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2752i)];
}
      condval_301 = f32((condval_303 == temp_max_shared[0i]));
}
    condval_300 = condval_301;
} else {
    condval_300 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_300);
  var condval_304 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2816i) < podArgs.vocab_size)) {
    var condval_305 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_306 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_306 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2816i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_306 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2816i)];
}
      condval_305 = exp((condval_306 - temp_max_shared[0i]));
} else {
      var condval_307 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_307 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2816i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_307 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2816i)];
}
      condval_305 = f32((condval_307 == temp_max_shared[0i]));
}
    condval_304 = condval_305;
} else {
    condval_304 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_304);
  var condval_308 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2880i) < podArgs.vocab_size)) {
    var condval_309 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_310 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_310 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2880i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_310 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2880i)];
}
      condval_309 = exp((condval_310 - temp_max_shared[0i]));
} else {
      var condval_311 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_311 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2880i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_311 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2880i)];
}
      condval_309 = f32((condval_311 == temp_max_shared[0i]));
}
    condval_308 = condval_309;
} else {
    condval_308 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_308);
  var condval_312 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 2944i) < podArgs.vocab_size)) {
    var condval_313 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_314 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_314 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2944i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_314 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2944i)];
}
      condval_313 = exp((condval_314 - temp_max_shared[0i]));
} else {
      var condval_315 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_315 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2944i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_315 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 2944i)];
}
      condval_313 = f32((condval_315 == temp_max_shared[0i]));
}
    condval_312 = condval_313;
} else {
    condval_312 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_312);
  var condval_316 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3008i) < podArgs.vocab_size)) {
    var condval_317 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_318 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_318 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3008i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_318 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3008i)];
}
      condval_317 = exp((condval_318 - temp_max_shared[0i]));
} else {
      var condval_319 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_319 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3008i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_319 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3008i)];
}
      condval_317 = f32((condval_319 == temp_max_shared[0i]));
}
    condval_316 = condval_317;
} else {
    condval_316 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_316);
  var condval_320 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3072i) < podArgs.vocab_size)) {
    var condval_321 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_322 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_322 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3072i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_322 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3072i)];
}
      condval_321 = exp((condval_322 - temp_max_shared[0i]));
} else {
      var condval_323 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_323 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3072i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_323 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3072i)];
}
      condval_321 = f32((condval_323 == temp_max_shared[0i]));
}
    condval_320 = condval_321;
} else {
    condval_320 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_320);
  var condval_324 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3136i) < podArgs.vocab_size)) {
    var condval_325 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_326 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_326 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3136i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_326 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3136i)];
}
      condval_325 = exp((condval_326 - temp_max_shared[0i]));
} else {
      var condval_327 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_327 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3136i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_327 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3136i)];
}
      condval_325 = f32((condval_327 == temp_max_shared[0i]));
}
    condval_324 = condval_325;
} else {
    condval_324 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_324);
  var condval_328 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3200i) < podArgs.vocab_size)) {
    var condval_329 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_330 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_330 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3200i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_330 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3200i)];
}
      condval_329 = exp((condval_330 - temp_max_shared[0i]));
} else {
      var condval_331 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_331 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3200i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_331 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3200i)];
}
      condval_329 = f32((condval_331 == temp_max_shared[0i]));
}
    condval_328 = condval_329;
} else {
    condval_328 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_328);
  var condval_332 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3264i) < podArgs.vocab_size)) {
    var condval_333 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_334 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_334 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3264i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_334 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3264i)];
}
      condval_333 = exp((condval_334 - temp_max_shared[0i]));
} else {
      var condval_335 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_335 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3264i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_335 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3264i)];
}
      condval_333 = f32((condval_335 == temp_max_shared[0i]));
}
    condval_332 = condval_333;
} else {
    condval_332 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_332);
  var condval_336 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3328i) < podArgs.vocab_size)) {
    var condval_337 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_338 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_338 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3328i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_338 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3328i)];
}
      condval_337 = exp((condval_338 - temp_max_shared[0i]));
} else {
      var condval_339 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_339 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3328i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_339 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3328i)];
}
      condval_337 = f32((condval_339 == temp_max_shared[0i]));
}
    condval_336 = condval_337;
} else {
    condval_336 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_336);
  var condval_340 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3392i) < podArgs.vocab_size)) {
    var condval_341 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_342 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_342 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3392i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_342 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3392i)];
}
      condval_341 = exp((condval_342 - temp_max_shared[0i]));
} else {
      var condval_343 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_343 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3392i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_343 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3392i)];
}
      condval_341 = f32((condval_343 == temp_max_shared[0i]));
}
    condval_340 = condval_341;
} else {
    condval_340 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_340);
  var condval_344 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3456i) < podArgs.vocab_size)) {
    var condval_345 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_346 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_346 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3456i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_346 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3456i)];
}
      condval_345 = exp((condval_346 - temp_max_shared[0i]));
} else {
      var condval_347 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_347 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3456i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_347 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3456i)];
}
      condval_345 = f32((condval_347 == temp_max_shared[0i]));
}
    condval_344 = condval_345;
} else {
    condval_344 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_344);
  var condval_348 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3520i) < podArgs.vocab_size)) {
    var condval_349 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_350 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_350 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3520i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_350 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3520i)];
}
      condval_349 = exp((condval_350 - temp_max_shared[0i]));
} else {
      var condval_351 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_351 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3520i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_351 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3520i)];
}
      condval_349 = f32((condval_351 == temp_max_shared[0i]));
}
    condval_348 = condval_349;
} else {
    condval_348 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_348);
  var condval_352 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3584i) < podArgs.vocab_size)) {
    var condval_353 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_354 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_354 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3584i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_354 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3584i)];
}
      condval_353 = exp((condval_354 - temp_max_shared[0i]));
} else {
      var condval_355 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_355 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3584i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_355 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3584i)];
}
      condval_353 = f32((condval_355 == temp_max_shared[0i]));
}
    condval_352 = condval_353;
} else {
    condval_352 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_352);
  var condval_356 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3648i) < podArgs.vocab_size)) {
    var condval_357 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_358 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_358 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3648i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_358 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3648i)];
}
      condval_357 = exp((condval_358 - temp_max_shared[0i]));
} else {
      var condval_359 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_359 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3648i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_359 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3648i)];
}
      condval_357 = f32((condval_359 == temp_max_shared[0i]));
}
    condval_356 = condval_357;
} else {
    condval_356 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_356);
  var condval_360 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3712i) < podArgs.vocab_size)) {
    var condval_361 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_362 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_362 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3712i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_362 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3712i)];
}
      condval_361 = exp((condval_362 - temp_max_shared[0i]));
} else {
      var condval_363 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_363 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3712i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_363 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3712i)];
}
      condval_361 = f32((condval_363 == temp_max_shared[0i]));
}
    condval_360 = condval_361;
} else {
    condval_360 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_360);
  var condval_364 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3776i) < podArgs.vocab_size)) {
    var condval_365 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_366 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_366 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3776i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_366 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3776i)];
}
      condval_365 = exp((condval_366 - temp_max_shared[0i]));
} else {
      var condval_367 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_367 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3776i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_367 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3776i)];
}
      condval_365 = f32((condval_367 == temp_max_shared[0i]));
}
    condval_364 = condval_365;
} else {
    condval_364 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_364);
  var condval_368 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3840i) < podArgs.vocab_size)) {
    var condval_369 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_370 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_370 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3840i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_370 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3840i)];
}
      condval_369 = exp((condval_370 - temp_max_shared[0i]));
} else {
      var condval_371 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_371 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3840i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_371 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3840i)];
}
      condval_369 = f32((condval_371 == temp_max_shared[0i]));
}
    condval_368 = condval_369;
} else {
    condval_368 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_368);
  var condval_372 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3904i) < podArgs.vocab_size)) {
    var condval_373 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_374 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_374 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3904i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_374 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3904i)];
}
      condval_373 = exp((condval_374 - temp_max_shared[0i]));
} else {
      var condval_375 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_375 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3904i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_375 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3904i)];
}
      condval_373 = f32((condval_375 == temp_max_shared[0i]));
}
    condval_372 = condval_373;
} else {
    condval_372 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_372);
  var condval_376 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 3968i) < podArgs.vocab_size)) {
    var condval_377 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_378 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_378 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3968i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_378 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3968i)];
}
      condval_377 = exp((condval_378 - temp_max_shared[0i]));
} else {
      var condval_379 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_379 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3968i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_379 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 3968i)];
}
      condval_377 = f32((condval_379 == temp_max_shared[0i]));
}
    condval_376 = condval_377;
} else {
    condval_376 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_376);
  var condval_380 : f32;
  if ((((((v__1 % podArgs.num_chunks) * 4096i) + i32(threadIdx.x)) + 4032i) < podArgs.vocab_size)) {
    var condval_381 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      var condval_382 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_382 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 4032i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_382 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 4032i)];
}
      condval_381 = exp((condval_382 - temp_max_shared[0i]));
} else {
      var condval_383 : f32;
      if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
        condval_383 = (A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 4032i)] / temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)]);
} else {
        condval_383 = A[(((((v__1 % podArgs.num_chunks) * 4096i) + (((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks) * podArgs.vocab_size)) + i32(threadIdx.x)) + 4032i)];
}
      condval_381 = f32((condval_383 == temp_max_shared[0i]));
}
    condval_380 = condval_381;
} else {
    condval_380 = 0.000000e+00f;
}
  in_thread_temp_sum_shared[0i] = (in_thread_temp_sum_shared[0i] + condval_380);
  workgroupBarrier();
  red_buf0_1[i32(threadIdx.x)] = in_thread_temp_sum_shared[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 32i) {
    red_buf0_1[i32(threadIdx.x)] = (red_buf0_1[i32(threadIdx.x)] + red_buf0_1[(i32(threadIdx.x) + 32i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i) {
    red_buf0_1[i32(threadIdx.x)] = (red_buf0_1[i32(threadIdx.x)] + red_buf0_1[(i32(threadIdx.x) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i) {
    red_buf0_1[i32(threadIdx.x)] = (red_buf0_1[i32(threadIdx.x)] + red_buf0_1[(i32(threadIdx.x) + 8i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 4i) {
    red_buf0_1[i32(threadIdx.x)] = (red_buf0_1[i32(threadIdx.x)] + red_buf0_1[(i32(threadIdx.x) + 4i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0_1[i32(threadIdx.x)] = (red_buf0_1[i32(threadIdx.x)] + red_buf0_1[(i32(threadIdx.x) + 2i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i) {
    red_buf0_1[i32(threadIdx.x)] = (red_buf0_1[i32(threadIdx.x)] + red_buf0_1[(i32(threadIdx.x) + 1i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    temp_sum_shared[0i] = red_buf0_1[0i];
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i) {
    var condval_384 : f32;
    if ((1.000000e-05f < temperature[((v__1 % (podArgs.num_chunks * podArgs.batch_size)) / podArgs.num_chunks)])) {
      condval_384 = log(temp_sum_shared[0i]);
} else {
      condval_384 = temp_sum_shared[0i];
}
    chunked_sum[v__1] = condval_384;
    chunked_max[v__1] = temp_max_shared[0i];
  }
}
