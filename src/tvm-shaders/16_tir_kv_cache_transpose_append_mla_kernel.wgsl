// [16] tir_kv_cache_transpose_append_mla_kernel (shader #16, 35 lines)
//----------------------------------------
// Function: tir_kv_cache_transpose_append_mla_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> kv_data : array<f16>;
@group(0) @binding(1) var<storage, read_write> pages : array<f16>;
@group(0) @binding(2) var<storage, read> position_map : array<i32>;

struct PODArgs {
  ntoken: i32,
  num_pages: i32,
  pages_elem_offset: i32,
  position_map_elem_offset: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn tir_kv_cache_transpose_append_mla_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (position_map[((((v__1 * 8i) + (i32(threadIdx.x)>>5u)) / 3i) + podArgs.position_map_elem_offset)] != -1i) {
    if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.ntoken * 96i)) {
      let position : i32 = position_map[((((v__1 * 8i) + (i32(threadIdx.x)>>5u)) / 3i) + podArgs.position_map_elem_offset)];
      pages[(((position * 96i) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))] = kv_data[((v__1 * 256i) + i32(threadIdx.x))];
    }
  }
}
