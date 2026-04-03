// KV CACHE APPEND — write K,V vectors into paged cache.
//
// K and V are written to page[position] in the KV cache.
// Layout: pages[page_no * 98304 + head * 1536 + slot * 96 + dim]
//   where: 98304 = 32 heads * 16 slots * 96 dims * 2 (K+V)
//          1536 = 16 slots * 96 dims
//          K at offset 0, V at offset 49152 (= 32 * 1536)
//
// Matches TVM's tir_kv_cache_transpose_append_kernel.

enable f16;

@group(0) @binding(0) var<storage, read> k_data : array<f16>;
@group(0) @binding(1) var<storage, read> v_data : array<f16>;
@group(0) @binding(2) var<storage, read_write> pages : array<f16>;
@group(0) @binding(3) var<storage, read> position_map : array<i32>;

struct PODArgs {
  ntoken: i32,
  num_pages: i32,
  pages_elem_offset: i32,
  position_map_elem_offset: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn kv_append(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let global_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(global_id) >= podArgs.packGridDimX) { return; }

  let flat : i32 = global_id * 256 + i32(threadIdx.x);
  // flat covers ntoken * 32 heads * 96 dims = ntoken * 3072
  let token_idx : i32 = flat / 3072;
  let within : i32 = flat % 3072;
  let head : i32 = within / 96;
  let dim : i32 = within % 96;

  if (token_idx >= podArgs.ntoken) { return; }

  let position : i32 = position_map[token_idx + podArgs.position_map_elem_offset];
  if (position == -1) { return; }

  let page_no : i32 = position / 16;
  let slot : i32 = position % 16;

  // Write K
  let k_offset : i32 = page_no * 98304 + head * 1536 + slot * 96 + dim + podArgs.pages_elem_offset;
  pages[k_offset] = k_data[token_idx * 3072 + within];

  // Write V (offset by 49152 = 32 * 1536)
  let v_offset : i32 = k_offset + 49152;
  pages[v_offset] = v_data[token_idx * 3072 + within];
}
