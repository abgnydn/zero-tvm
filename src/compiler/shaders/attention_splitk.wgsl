// PAGED KV ATTENTION, SPLIT-K PARTIAL PASS (?splitk=N experiment).
//
// attention.wgsl runs ONE workgroup per (batch, head) — 32 WGs × 32 threads
// = 1,024 threads for the whole GPU, serial over every KV slot. This kernel
// partitions the KV page range across `num_splits` workgroups per head:
// each partition runs the same online-softmax loop over its page slice and
// writes a merge-ready partial state (m, d, o[HEAD_DIM]) — the same (m, d, o)
// triple attention.wgsl carries in registers — to a scratch buffer.
// attention_combine.wgsl then merges the partials per head.
//
// Dispatch: (num_splits, HEADS, 1). Decode is always B=1 (the engine's
// writeStepState hard-codes batch 0), so the partials layout has no batch
// axis; podArgs.B is kept for uniform-layout parity with attention.wgsl.
//
// Partials layout (f32): [head][partition][m, d, o[0..HEAD_DIM-1]]
//   stride PARTIAL_STRIDE = HEAD_DIM + 2 floats per (head, partition).
//
// Empty partitions (kv_len covers fewer pages than num_splits) write
// m = -50000, d = 0 — the combine pass skips any partial with d == 0, so
// they cannot poison the merge.
//
// Built and correctness-tested; awaiting measurement (see BENCH.md
// "Pending experiments").
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read> Q : array<f16>;
@group(0) @binding(1) var<storage, read> page_table_indptr : array<i32>;
@group(0) @binding(2) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(3) var<storage, read> pages : array<f16>;
@group(0) @binding(4) var<storage, read> length_info : array<i32>;
@group(0) @binding(5) var<storage, read_write> partials : array<f32>;

struct PODArgs {
  B: i32,
  max_num_pages: i32,
  nnz_pages: i32,
  pages_elem_offset: i32,
  page_indptr_elem_offset: i32,
  page_values_elem_offset: i32,
  length_info_elem_offset: i32,
  sm_scale: f32,
  num_splits: u32
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const PARTIAL_STRIDE = HEAD_DIM + 2;   // per-(head, partition) f32 stride: m, d, o[HEAD_DIM]

var<workgroup> score_reduce : array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn attention_splitk(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let part : i32 = i32(blockIdx.x);
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);
  let num_splits : i32 = i32(podArgs.num_splits);
  let batch : i32 = 0;   // decode is always B=1

  if (part >= num_splits) { return; }

  // Each thread owns 3 elements of Q (32 threads × 3 = HEAD_DIM).
  var q0 : f32 = f32(Q[batch * (HEADS * HEAD_DIM) + head * HEAD_DIM + tid * 3]);
  var q1 : f32 = f32(Q[batch * (HEADS * HEAD_DIM) + head * HEAD_DIM + tid * 3 + 1]);
  var q2 : f32 = f32(Q[batch * (HEADS * HEAD_DIM) + head * HEAD_DIM + tid * 3 + 2]);

  let indptr_begin : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset];
  let indptr_end : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  // Partition the page range: contiguous ceil(total/num_splits)-page chunks.
  // Trailing partitions can be empty (p_begin >= p_end) — they still write
  // their (m=-50000, d=0) partial so the combine pass sees a defined state.
  let total_pages : i32 = indptr_end - indptr_begin;
  let pages_per_part : i32 = (total_pages + num_splits - 1) / num_splits;
  let p_begin : i32 = indptr_begin + part * pages_per_part;
  let p_end : i32 = min(indptr_end, p_begin + pages_per_part);

  // Online softmax state (identical to attention.wgsl lines 66-70).
  var m : f32 = -50000.0;
  var d : f32 = 0.0;
  var o0 : f32 = 0.0;
  var o1 : f32 = 0.0;
  var o2 : f32 = 0.0;

  for (var page_idx : i32 = p_begin; page_idx < p_end; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
    // page_start stays GLOBAL (relative to indptr_begin) so the last page's
    // partial-slot count is right regardless of which partition owns it.
    let page_start : i32 = (page_idx - indptr_begin) * PAGE_SIZE;
    let slots_in_page : i32 = min(PAGE_SIZE, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      let k_base : i32 = page_no * KV_PAGE_STRIDE + head * HEAD_PAGE_STRIDE + slot * HEAD_DIM
                         + podArgs.pages_elem_offset;

      let partial : f32 = q0 * f32(pages[k_base + tid * 3])
                        + q1 * f32(pages[k_base + tid * 3 + 1])
                        + q2 * f32(pages[k_base + tid * 3 + 2]);

      // Tree reduction across 32 threads to get the full dot product.
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
      // All threads must finish reading score_reduce[0] before the next slot
      // iteration overwrites score_reduce[tid] (see attention.wgsl).
      workgroupBarrier();

      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      let v_base : i32 = k_base + V_PAGE_OFFSET;
      o0 = o0 * scale_prev + scale_new * f32(pages[v_base + tid * 3]);
      o1 = o1 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 1]);
      o2 = o2 * scale_prev + scale_new * f32(pages[v_base + tid * 3 + 2]);
    }
  }

  // Write the unnormalized partial state; the combine pass rescales by
  // exp(m_p - M_global) and divides by the merged denominator.
  let pbase : i32 = (head * num_splits + part) * PARTIAL_STRIDE;
  partials[pbase + 2 + tid * 3] = o0;
  partials[pbase + 2 + tid * 3 + 1] = o1;
  partials[pbase + 2 + tid * 3 + 2] = o2;
  if (tid == 0) {
    partials[pbase] = m;
    partials[pbase + 1] = d;
  }
}
