// [38] tir_attention_merge_state_kernel (shader #38, 60 lines)
//----------------------------------------
// Function: tir_attention_merge_state_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> S : array<f32>;
@group(0) @binding(1) var<storage, read> S_other : array<f32>;
@group(0) @binding(2) var<storage, read_write> V : array<f16>;
@group(0) @binding(3) var<storage, read> V_other : array<f16>;

struct PODArgs {
  D: i32,
  H: i32,
  N: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(24, 8, 1)
fn tir_attention_merge_state_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var s_val : array<f32, 1>;
  var s_other_val : array<f32, 1>;
  var s_max : array<f32, 1>;
  var scale : array<f32, 1>;
  var other_scale : array<f32, 1>;
  var v_vec : array<f16, 4>;
  var v_other_vec : array<f16, 4>;
  s_val[0i] = S[(((i32(blockIdx.y) * 8i) + (v__1 * podArgs.H)) + i32(threadIdx.y))];
  s_other_val[0i] = S_other[(((i32(blockIdx.y) * 8i) + (v__1 * podArgs.H)) + i32(threadIdx.y))];
  s_max[0i] = max(s_val[0i], s_other_val[0i]);
  s_val[0i] = exp2((s_val[0i] - s_max[0i]));
  s_other_val[0i] = exp2((s_other_val[0i] - s_max[0i]));
  scale[0i] = (s_val[0i] / (s_val[0i] + s_other_val[0i]));
  other_scale[0i] = (s_other_val[0i] / (s_val[0i] + s_other_val[0i]));
  let v__2 : i32 = ((i32(threadIdx.x) * 4i) + ((((i32(blockIdx.y) * 8i) + (v__1 * podArgs.H)) + i32(threadIdx.y)) * podArgs.D));
  v_vec[0i + 0] = vec4<f16>(V[v__2 + 0], V[v__2 + 1], V[v__2 + 2], V[v__2 + 3])[0];
  v_vec[0i + 1] = vec4<f16>(V[v__2 + 0], V[v__2 + 1], V[v__2 + 2], V[v__2 + 3])[1];
  v_vec[0i + 2] = vec4<f16>(V[v__2 + 0], V[v__2 + 1], V[v__2 + 2], V[v__2 + 3])[2];
  v_vec[0i + 3] = vec4<f16>(V[v__2 + 0], V[v__2 + 1], V[v__2 + 2], V[v__2 + 3])[3];
  v_other_vec[0i + 0] = vec4<f16>(V_other[v__2 + 0], V_other[v__2 + 1], V_other[v__2 + 2], V_other[v__2 + 3])[0];
  v_other_vec[0i + 1] = vec4<f16>(V_other[v__2 + 0], V_other[v__2 + 1], V_other[v__2 + 2], V_other[v__2 + 3])[1];
  v_other_vec[0i + 2] = vec4<f16>(V_other[v__2 + 0], V_other[v__2 + 1], V_other[v__2 + 2], V_other[v__2 + 3])[2];
  v_other_vec[0i + 3] = vec4<f16>(V_other[v__2 + 0], V_other[v__2 + 1], V_other[v__2 + 2], V_other[v__2 + 3])[3];
  for (var vec : i32 = 0; vec < 4i; vec++) {
    v_vec[vec] = f16(fma(f32(v_other_vec[vec]), other_scale[0i], (f32(v_vec[vec]) * scale[0i])));
  }
  V[v__2 + 0] = vec4<f16>(v_vec[0i + 0], v_vec[0i + 1], v_vec[0i + 2], v_vec[0i + 3])[0];
  V[v__2 + 1] = vec4<f16>(v_vec[0i + 0], v_vec[0i + 1], v_vec[0i + 2], v_vec[0i + 3])[1];
  V[v__2 + 2] = vec4<f16>(v_vec[0i + 0], v_vec[0i + 1], v_vec[0i + 2], v_vec[0i + 3])[2];
  V[v__2 + 3] = vec4<f16>(v_vec[0i + 0], v_vec[0i + 1], v_vec[0i + 2], v_vec[0i + 3])[3];
  S[(((i32(blockIdx.y) * 8i) + (v__1 * podArgs.H)) + i32(threadIdx.y))] = (log2((s_val[0i] + s_other_val[0i])) + s_max[0i]);
}
