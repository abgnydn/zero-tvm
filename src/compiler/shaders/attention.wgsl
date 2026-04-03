// PAGED KV ATTENTION (DECODE) — FlashDecoding style.
//
// For single-token decode: Q has 1 token, K/V are in paged cache.
// One workgroup per (batch, head) pair.
// 32 threads, each owns 3 elements of head_dim=96.
//
// Algorithm: online softmax (FlashAttention)
//   for each page of KV cache:
//     for each slot in page:
//       score = dot(Q, K[slot]) * sm_scale
//       update running (max, sum, output) with online softmax
//   normalize output
//
// Pages layout: pages[page * 98304 + head * 1536 + slot * 96 + dim]
//   K at offset 0, V at offset 49152

enable f16;

@group(0) @binding(0) var<storage, read> Q : array<f16>;
@group(0) @binding(1) var<storage, read> page_table_indptr : array<i32>;
@group(0) @binding(2) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(3) var<storage, read> pages : array<f16>;
@group(0) @binding(4) var<storage, read> length_info : array<i32>;
@group(0) @binding(5) var<storage, read_write> output_buf : array<f16>;

struct PODArgs {
  B: i32,
  max_num_pages: i32,
  nnz_pages: i32,
  pages_elem_offset: i32,
  page_indptr_elem_offset: i32,
  page_values_elem_offset: i32,
  length_info_elem_offset: i32,
  sm_scale: f32,
  packGridDimX: u32
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

var<workgroup> score_reduce : array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn attention(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.x);
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  if (batch >= podArgs.B) { return; }

  // Each thread owns 3 elements of Q (32 threads × 3 = 96 = head_dim)
  var q0 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3]);
  var q1 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 1]);
  var q2 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 2]);

  // Page range for this batch
  let indptr_begin : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset];
  let indptr_end : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  // Online softmax state
  var m : f32 = -50000.0;
  var d : f32 = 0.0;
  var o0 : f32 = 0.0;
  var o1 : f32 = 0.0;
  var o2 : f32 = 0.0;

  // Iterate over all KV positions
  for (var page_idx : i32 = indptr_begin; page_idx < indptr_end; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
    let page_start : i32 = (page_idx - indptr_begin) * 16;
    let slots_in_page : i32 = min(16, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      let k_base : i32 = page_no * 98304 + head * 1536 + slot * 96 + podArgs.pages_elem_offset;

      // Partial dot product: each thread computes 3 multiplies
      let partial : f32 = q0 * f32(pages[k_base + tid * 3])
                        + q1 * f32(pages[k_base + tid * 3 + 1])
                        + q2 * f32(pages[k_base + tid * 3 + 2]);

      // Tree reduction across 32 threads to get full dot product
      score_reduce[tid] = partial;
      workgroupBarrier();
      if (tid < 16) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 16]; }
      workgroupBarrier();
      if (tid < 8) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 8]; }
      workgroupBarrier();
      if (tid < 4) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 4]; }
      workgroupBarrier();
      if (tid < 2) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 2]; }
      workgroupBarrier();
      if (tid < 1) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 1]; }
      workgroupBarrier();

      let s : f32 = score_reduce[0] * podArgs.sm_scale;

      // Online softmax update
      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      // Load V and accumulate
      let v_base : i32 = k_base + 49152;
      o0 = o0 * scale_prev + scale_new * f32(pages[v_base + tid * 3]);
      o1 = o1 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 1]);
      o2 = o2 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 2]);
    }
  }

  // Normalize and write output
  if (d > 0.0) {
    let inv_d : f32 = 1.0 / d;
    output_buf[batch * 3072 + head * 96 + tid * 3] = f16(o0 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 1] = f16(o1 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 2] = f16(o2 * inv_d);
  }
}
