// PAGED KV ATTENTION (subgroup variant) — same math as attention.wgsl,
// but replaces the 6-barrier tree reduction with a single subgroupAdd.
//
// Assumes subgroup size >= 32 (true on every shipping WebGPU backend that
// exposes the `subgroups` feature). The full 32-lane workgroup sits inside
// one subgroup, so subgroupAdd(partial) returns the full dot product with
// zero workgroup barriers per slot.
//
// Expected win: 6 workgroupBarrier() per KV slot → 0. Attention dominates
// tail latency on short-context decode; this is pure barrier elimination.

enable f16;
enable subgroups;

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

@compute @workgroup_size(32, 1, 1)
fn attention_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let batch : i32 = i32(blockIdx.x);
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  if (batch >= podArgs.B) { return; }

  // Each thread owns 3 elements of head_dim=96.
  var q0 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3]);
  var q1 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 1]);
  var q2 : f32 = f32(Q[batch * 3072 + head * 96 + tid * 3 + 2]);

  let indptr_begin : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset];
  let indptr_end : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  var m  : f32 = -50000.0;
  var d  : f32 = 0.0;
  var o0 : f32 = 0.0;
  var o1 : f32 = 0.0;
  var o2 : f32 = 0.0;

  for (var page_idx : i32 = indptr_begin; page_idx < indptr_end; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
    let page_start : i32 = (page_idx - indptr_begin) * 16;
    let slots_in_page : i32 = min(16, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      let k_base : i32 = page_no * 98304 + head * 1536 + slot * 96 + podArgs.pages_elem_offset;

      let partial : f32 = q0 * f32(pages[k_base + tid * 3])
                        + q1 * f32(pages[k_base + tid * 3 + 1])
                        + q2 * f32(pages[k_base + tid * 3 + 2]);

      // Single subgroupAdd replaces 6 tree-reduction barriers.
      let s : f32 = subgroupAdd(partial) * podArgs.sm_scale;

      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new  : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      let v_base : i32 = k_base + 49152;
      o0 = o0 * scale_prev + scale_new * f32(pages[v_base + tid * 3]);
      o1 = o1 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 1]);
      o2 = o2 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 2]);
    }
  }

  if (d > 0.0) {
    let inv_d : f32 = 1.0 / d;
    output_buf[batch * 3072 + head * 96 + tid * 3]     = f16(o0 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 1] = f16(o1 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 2] = f16(o2 * inv_d);
  }
}
