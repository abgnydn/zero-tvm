// KV CACHE APPEND — write K,V vectors into paged cache.
//
// PREFILL-ONLY: the decode path fuses this into qkv_fused.wgsl (QKV matmul +
// RoPE + KV append in one dispatch); this kernel is kept for the sequential
// prefill path and for parity testing.
//
// K and V are written to page[position] in the KV cache.
// Layout: pages[page_no * KV_PAGE_STRIDE + head * HEAD_PAGE_STRIDE + slot * HEAD_DIM + dim]
//   where: KV_PAGE_STRIDE   = HEADS * PAGE_SIZE * HEAD_DIM * 2 (K+V)
//          HEAD_PAGE_STRIDE = PAGE_SIZE * HEAD_DIM
//          K at offset 0, V at offset V_PAGE_OFFSET (= HEADS * HEAD_PAGE_STRIDE)
//
// Matches TVM's tir_kv_cache_transpose_append_kernel.
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

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
  // flat covers ntoken * HEADS * HEAD_DIM elements (one K/V row per token)
  let token_idx : i32 = flat / (HEADS * HEAD_DIM);
  let within : i32 = flat % (HEADS * HEAD_DIM);
  let head : i32 = within / HEAD_DIM;
  let dim : i32 = within % HEAD_DIM;

  if (token_idx >= podArgs.ntoken) { return; }

  let position : i32 = position_map[token_idx + podArgs.position_map_elem_offset];
  if (position == -1) { return; }

  let page_no : i32 = position / PAGE_SIZE;
  let slot : i32 = position % PAGE_SIZE;

  // Write K
  let k_offset : i32 = page_no * KV_PAGE_STRIDE + head * HEAD_PAGE_STRIDE + slot * HEAD_DIM + dim
                       + podArgs.pages_elem_offset;
  pages[k_offset] = k_data[token_idx * (HEADS * HEAD_DIM) + within];

  // Write V (offset by V_PAGE_OFFSET = HEADS * HEAD_PAGE_STRIDE)
  let v_offset : i32 = k_offset + V_PAGE_OFFSET;
  pages[v_offset] = v_data[token_idx * (HEADS * HEAD_DIM) + within];
}
