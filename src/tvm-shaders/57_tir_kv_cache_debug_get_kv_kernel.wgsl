// [57] tir_kv_cache_debug_get_kv_kernel (shader #57, 35 lines)
//----------------------------------------
// Function: tir_kv_cache_debug_get_kv_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> k_data : array<f16>;
@group(0) @binding(1) var<storage, read> pages : array<f16>;
@group(0) @binding(2) var<storage, read> position_map : array<i32>;
@group(0) @binding(3) var<storage, read_write> v_data : array<f16>;

struct PODArgs {
  layer_id: i32,
  num_pages: i32,
  page_size: i32,
  pages_elem_offset: i32,
  position_map_elem_offset: i32,
  seqlen: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn tir_kv_cache_debug_get_kv_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  let position : i32 = position_map[((v__1 / 12i) + podArgs.position_map_elem_offset)];
  k_data[(((((v__1 / 12i) * 3072i) + ((podArgs.layer_id * podArgs.seqlen) * 3072i)) + (((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))] = pages[(((((((((position / podArgs.page_size) + ((position % podArgs.page_size)>>31u)) * 64i) + ((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i)) * podArgs.page_size) * 96i) + (((position % podArgs.page_size) + (podArgs.page_size & ((position % podArgs.page_size)>>31u))) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))];
  v_data[(((((v__1 / 12i) * 3072i) + ((podArgs.layer_id * podArgs.seqlen) * 3072i)) + (((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 96i)) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))] = pages[((((((((((position / podArgs.page_size) + ((position % podArgs.page_size)>>31u)) * 64i) + ((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i)) + 32i) * podArgs.page_size) * 96i) + (((position % podArgs.page_size) + (podArgs.page_size & ((position % podArgs.page_size)>>31u))) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))];
}
