// [40] fused_rope_kernel (shader #40, 65 lines)
//----------------------------------------
// Function: fused_rope_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> k : array<f16>;
@group(0) @binding(1) var<storage, read> position_map : array<i32>;
@group(0) @binding(2) var<storage, read_write> q : array<f16>;
@group(0) @binding(3) var<storage, read> qkv : array<f16>;
@group(0) @binding(4) var<storage, read_write> v : array<f16>;

struct PODArgs {
  apply_rope: i32,
  position_map_elem_offset: i32,
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn fused_rope_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if ((v__1 % 36i) < 12i) {
    var condval : f16;
    if ((0i < podArgs.apply_rope)) {
      let freq : f32 = (f32(position_map[((v__1 / 36i) + podArgs.position_map_elem_offset)]) / pow(1.000000e+04f, (f32(((((v__1 * 16i) + i32(threadIdx.x)) % 48i) * 2i)) / 9.600000e+01f)));
      var condval_1 : f16;
      if (((((v__1 * 4i) + (i32(threadIdx.x)>>4u)) % 6i) < 3i)) {
        condval_1 = (qkv[(((((v__1 / 36i) * 9216i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i)) + 48i)] * -1.000000e+00h);
} else {
        condval_1 = qkv[(((((v__1 / 36i) * 9216i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i)) - 48i)];
}
      condval = f16(fma(sin(freq), f32(condval_1), (cos(freq) * f32(qkv[((((v__1 / 36i) * 9216i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))]))));
} else {
      condval = qkv[((((v__1 / 36i) * 9216i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))];
}
    q[((((v__1 / 36i) * 3072i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))] = condval;
  } else {
    if ((v__1 % 36i) < 24i) {
      var condval_2 : f16;
      if ((0i < podArgs.apply_rope)) {
        let freq_1 : f32 = (f32(position_map[((v__1 / 36i) + podArgs.position_map_elem_offset)]) / pow(1.000000e+04f, (f32(((((v__1 * 16i) + i32(threadIdx.x)) % 48i) * 2i)) / 9.600000e+01f)));
        var condval_3 : f16;
        if (((((v__1 * 4i) + (i32(threadIdx.x)>>4u)) % 6i) < 3i)) {
          condval_3 = (qkv[(((((v__1 / 36i) * 9216i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i)) + 48i)] * -1.000000e+00h);
} else {
          condval_3 = qkv[(((((v__1 / 36i) * 9216i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i)) - 48i)];
}
        condval_2 = f16(fma(sin(freq_1), f32(condval_3), (cos(freq_1) * f32(qkv[((((v__1 / 36i) * 9216i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))]))));
} else {
        condval_2 = qkv[((((v__1 / 36i) * 9216i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))];
}
      k[(((((v__1 / 36i) * 3072i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i)) - 3072i)] = condval_2;
    } else {
      v[(((((v__1 / 36i) * 3072i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i)) - 6144i)] = qkv[((((v__1 / 36i) * 9216i) + (((((v__1 % 36i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))];
    }
  }
}
