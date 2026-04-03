// [7] compact_kv_copy_kernel (shader #7, 39 lines)
//----------------------------------------
// Function: compact_kv_copy_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> copy_length_indptr : array<i32>;
@group(0) @binding(1) var<storage, read> copy_src_dst_pos : array<i32>;
@group(0) @binding(2) var<storage, read_write> pages : array<f16>;

struct PODArgs {
  batch_size: i32,
  copy_length_indptr_elem_offset: i32,
  copy_src_dst_pos_elem_offset: i32,
  num_pages: i32,
  pages_elem_offset: i32,
  total_copy_length: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn compact_kv_copy_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if ((v__1 - (podArgs.batch_size * 12i)) < 0i) {
    for (var i : i32 = 0; i < (copy_length_indptr[(((v__1 / 12i) + podArgs.copy_length_indptr_elem_offset) + 1i)] - copy_length_indptr[((v__1 / 12i) + podArgs.copy_length_indptr_elem_offset)]); i++) {
      let src_pos : i32 = copy_src_dst_pos[((copy_length_indptr[((v__1 / 12i) + podArgs.copy_length_indptr_elem_offset)] + i) + podArgs.copy_src_dst_pos_elem_offset)];
      let dst_pos : i32 = copy_src_dst_pos[(((podArgs.total_copy_length + copy_length_indptr[((v__1 / 12i) + podArgs.copy_length_indptr_elem_offset)]) + i) + podArgs.copy_src_dst_pos_elem_offset)];
      pages[((((((dst_pos>>4u) * 98304i) + (((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 1536i)) + ((dst_pos & 15i) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))] = pages[((((((src_pos>>4u) * 98304i) + (((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 1536i)) + ((src_pos & 15i) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))];
      pages[(((((((dst_pos>>4u) * 98304i) + (((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 1536i)) + ((dst_pos & 15i) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i)) + 49152i)] = pages[(((((((src_pos>>4u) * 98304i) + (((((v__1 % 12i) * 8i) + (i32(threadIdx.x)>>5u)) / 3i) * 1536i)) + ((src_pos & 15i) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i)) + 49152i)];
    }
  }
}
