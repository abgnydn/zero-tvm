// [69] gpu_2d_continuous_cumsum1_kernel (shader #69, 117 lines)
//----------------------------------------
// Function: gpu_2d_continuous_cumsum1_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read_write> Out : array<f32>;
@group(0) @binding(2) var<storage, read_write> Tmp : array<f32>;

struct PODArgs {
  m: i32,
  n: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

var<workgroup> shared_buf : array<f32, 512>;
@compute @workgroup_size(32, 4, 1)
fn gpu_2d_continuous_cumsum1_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  var local_buf : array<f32, 4>;
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var i_s : i32 = 0; i_s < 4i; i_s++) {
    var condval : f32;
    if ((((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + i_s) < podArgs.n)) {
      condval = A[(((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + (i32(blockIdx.y) * podArgs.n)) + i_s)];
} else {
      condval = 0.000000e+00f;
}
    local_buf[i_s] = condval;
  }
  local_buf[1i] = (local_buf[1i] + local_buf[0i]);
  local_buf[2i] = (local_buf[2i] + local_buf[1i]);
  local_buf[3i] = (local_buf[3i] + local_buf[2i]);
  let v__2 : i32 = ((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i));
  shared_buf[v__2 + 0] = vec4<f32>(local_buf[0i + 0], local_buf[0i + 1], local_buf[0i + 2], local_buf[0i + 3])[0];
  shared_buf[v__2 + 1] = vec4<f32>(local_buf[0i + 0], local_buf[0i + 1], local_buf[0i + 2], local_buf[0i + 3])[1];
  shared_buf[v__2 + 2] = vec4<f32>(local_buf[0i + 0], local_buf[0i + 1], local_buf[0i + 2], local_buf[0i + 3])[2];
  shared_buf[v__2 + 3] = vec4<f32>(local_buf[0i + 0], local_buf[0i + 1], local_buf[0i + 2], local_buf[0i + 3])[3];
  workgroupBarrier();
  if (1i <= i32(threadIdx.x)) {
    let v__3 : i32 = ((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i));
    shared_buf[v__3 + 0] = (vec4<f32>(shared_buf[v__3 + 0], shared_buf[v__3 + 1], shared_buf[v__3 + 2], shared_buf[v__3 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)]))[0];
    shared_buf[v__3 + 1] = (vec4<f32>(shared_buf[v__3 + 0], shared_buf[v__3 + 1], shared_buf[v__3 + 2], shared_buf[v__3 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)]))[1];
    shared_buf[v__3 + 2] = (vec4<f32>(shared_buf[v__3 + 0], shared_buf[v__3 + 1], shared_buf[v__3 + 2], shared_buf[v__3 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)]))[2];
    shared_buf[v__3 + 3] = (vec4<f32>(shared_buf[v__3 + 0], shared_buf[v__3 + 1], shared_buf[v__3 + 2], shared_buf[v__3 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 1i)]))[3];
  }
  workgroupBarrier();
  if (2i <= i32(threadIdx.x)) {
    let v__4 : i32 = ((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i));
    shared_buf[v__4 + 0] = (vec4<f32>(shared_buf[v__4 + 0], shared_buf[v__4 + 1], shared_buf[v__4 + 2], shared_buf[v__4 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)]))[0];
    shared_buf[v__4 + 1] = (vec4<f32>(shared_buf[v__4 + 0], shared_buf[v__4 + 1], shared_buf[v__4 + 2], shared_buf[v__4 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)]))[1];
    shared_buf[v__4 + 2] = (vec4<f32>(shared_buf[v__4 + 0], shared_buf[v__4 + 1], shared_buf[v__4 + 2], shared_buf[v__4 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)]))[2];
    shared_buf[v__4 + 3] = (vec4<f32>(shared_buf[v__4 + 0], shared_buf[v__4 + 1], shared_buf[v__4 + 2], shared_buf[v__4 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 5i)]))[3];
  }
  workgroupBarrier();
  if (4i <= i32(threadIdx.x)) {
    let v__5 : i32 = ((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i));
    shared_buf[v__5 + 0] = (vec4<f32>(shared_buf[v__5 + 0], shared_buf[v__5 + 1], shared_buf[v__5 + 2], shared_buf[v__5 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)]))[0];
    shared_buf[v__5 + 1] = (vec4<f32>(shared_buf[v__5 + 0], shared_buf[v__5 + 1], shared_buf[v__5 + 2], shared_buf[v__5 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)]))[1];
    shared_buf[v__5 + 2] = (vec4<f32>(shared_buf[v__5 + 0], shared_buf[v__5 + 1], shared_buf[v__5 + 2], shared_buf[v__5 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)]))[2];
    shared_buf[v__5 + 3] = (vec4<f32>(shared_buf[v__5 + 0], shared_buf[v__5 + 1], shared_buf[v__5 + 2], shared_buf[v__5 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 13i)]))[3];
  }
  workgroupBarrier();
  if (8i <= i32(threadIdx.x)) {
    let v__6 : i32 = ((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i));
    shared_buf[v__6 + 0] = (vec4<f32>(shared_buf[v__6 + 0], shared_buf[v__6 + 1], shared_buf[v__6 + 2], shared_buf[v__6 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)]))[0];
    shared_buf[v__6 + 1] = (vec4<f32>(shared_buf[v__6 + 0], shared_buf[v__6 + 1], shared_buf[v__6 + 2], shared_buf[v__6 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)]))[1];
    shared_buf[v__6 + 2] = (vec4<f32>(shared_buf[v__6 + 0], shared_buf[v__6 + 1], shared_buf[v__6 + 2], shared_buf[v__6 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)]))[2];
    shared_buf[v__6 + 3] = (vec4<f32>(shared_buf[v__6 + 0], shared_buf[v__6 + 1], shared_buf[v__6 + 2], shared_buf[v__6 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 29i)]))[3];
  }
  workgroupBarrier();
  if (16i <= i32(threadIdx.x)) {
    let v__7 : i32 = ((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i));
    shared_buf[v__7 + 0] = (vec4<f32>(shared_buf[v__7 + 0], shared_buf[v__7 + 1], shared_buf[v__7 + 2], shared_buf[v__7 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)]))[0];
    shared_buf[v__7 + 1] = (vec4<f32>(shared_buf[v__7 + 0], shared_buf[v__7 + 1], shared_buf[v__7 + 2], shared_buf[v__7 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)]))[1];
    shared_buf[v__7 + 2] = (vec4<f32>(shared_buf[v__7 + 0], shared_buf[v__7 + 1], shared_buf[v__7 + 2], shared_buf[v__7 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)]))[2];
    shared_buf[v__7 + 3] = (vec4<f32>(shared_buf[v__7 + 0], shared_buf[v__7 + 1], shared_buf[v__7 + 2], shared_buf[v__7 + 3]) + vec4<f32>(shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)], shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) - 61i)]))[3];
  }
  workgroupBarrier();
  if (i32(threadIdx.y) == 0i) {
    let v__8 : i32 = ((i32(threadIdx.x) * 4i) + 128i);
    shared_buf[v__8 + 0] = (vec4<f32>(shared_buf[v__8 + 0], shared_buf[v__8 + 1], shared_buf[v__8 + 2], shared_buf[v__8 + 3]) + vec4<f32>(shared_buf[127i], shared_buf[127i], shared_buf[127i], shared_buf[127i]))[0];
    shared_buf[v__8 + 1] = (vec4<f32>(shared_buf[v__8 + 0], shared_buf[v__8 + 1], shared_buf[v__8 + 2], shared_buf[v__8 + 3]) + vec4<f32>(shared_buf[127i], shared_buf[127i], shared_buf[127i], shared_buf[127i]))[1];
    shared_buf[v__8 + 2] = (vec4<f32>(shared_buf[v__8 + 0], shared_buf[v__8 + 1], shared_buf[v__8 + 2], shared_buf[v__8 + 3]) + vec4<f32>(shared_buf[127i], shared_buf[127i], shared_buf[127i], shared_buf[127i]))[2];
    shared_buf[v__8 + 3] = (vec4<f32>(shared_buf[v__8 + 0], shared_buf[v__8 + 1], shared_buf[v__8 + 2], shared_buf[v__8 + 3]) + vec4<f32>(shared_buf[127i], shared_buf[127i], shared_buf[127i], shared_buf[127i]))[3];
  }
  workgroupBarrier();
  if (i32(threadIdx.y) == 0i) {
    let v__9 : i32 = ((i32(threadIdx.x) * 4i) + 256i);
    shared_buf[v__9 + 0] = (vec4<f32>(shared_buf[v__9 + 0], shared_buf[v__9 + 1], shared_buf[v__9 + 2], shared_buf[v__9 + 3]) + vec4<f32>(shared_buf[255i], shared_buf[255i], shared_buf[255i], shared_buf[255i]))[0];
    shared_buf[v__9 + 1] = (vec4<f32>(shared_buf[v__9 + 0], shared_buf[v__9 + 1], shared_buf[v__9 + 2], shared_buf[v__9 + 3]) + vec4<f32>(shared_buf[255i], shared_buf[255i], shared_buf[255i], shared_buf[255i]))[1];
    shared_buf[v__9 + 2] = (vec4<f32>(shared_buf[v__9 + 0], shared_buf[v__9 + 1], shared_buf[v__9 + 2], shared_buf[v__9 + 3]) + vec4<f32>(shared_buf[255i], shared_buf[255i], shared_buf[255i], shared_buf[255i]))[2];
    shared_buf[v__9 + 3] = (vec4<f32>(shared_buf[v__9 + 0], shared_buf[v__9 + 1], shared_buf[v__9 + 2], shared_buf[v__9 + 3]) + vec4<f32>(shared_buf[255i], shared_buf[255i], shared_buf[255i], shared_buf[255i]))[3];
  }
  workgroupBarrier();
  if (i32(threadIdx.y) == 0i) {
    let v__10 : i32 = ((i32(threadIdx.x) * 4i) + 384i);
    shared_buf[v__10 + 0] = (vec4<f32>(shared_buf[v__10 + 0], shared_buf[v__10 + 1], shared_buf[v__10 + 2], shared_buf[v__10 + 3]) + vec4<f32>(shared_buf[383i], shared_buf[383i], shared_buf[383i], shared_buf[383i]))[0];
    shared_buf[v__10 + 1] = (vec4<f32>(shared_buf[v__10 + 0], shared_buf[v__10 + 1], shared_buf[v__10 + 2], shared_buf[v__10 + 3]) + vec4<f32>(shared_buf[383i], shared_buf[383i], shared_buf[383i], shared_buf[383i]))[1];
    shared_buf[v__10 + 2] = (vec4<f32>(shared_buf[v__10 + 0], shared_buf[v__10 + 1], shared_buf[v__10 + 2], shared_buf[v__10 + 3]) + vec4<f32>(shared_buf[383i], shared_buf[383i], shared_buf[383i], shared_buf[383i]))[2];
    shared_buf[v__10 + 3] = (vec4<f32>(shared_buf[v__10 + 0], shared_buf[v__10 + 1], shared_buf[v__10 + 2], shared_buf[v__10 + 3]) + vec4<f32>(shared_buf[383i], shared_buf[383i], shared_buf[383i], shared_buf[383i]))[3];
  }
  workgroupBarrier();
  for (var i_s_1 : i32 = 0; i_s_1 < 4i; i_s_1++) {
    if (((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + i_s_1) < podArgs.n) {
      Out[(((((v__1 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + (i32(blockIdx.y) * podArgs.n)) + i_s_1)] = shared_buf[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 4i)) + i_s_1)];
    }
  }
  if ((i32(threadIdx.x) == 0i) && (i32(threadIdx.y) == 0i)) {
    Tmp[((i32(blockIdx.y) * podArgs.n) + v__1)] = shared_buf[511i];
  }
}
