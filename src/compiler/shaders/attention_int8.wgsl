// PAGED KV ATTENTION — int8 KV cache variant of attention.wgsl.
//
// Identical online-softmax math; the only change is reading K,V as int8 + f16
// scale instead of f16. Per-row scale (one scale per head-slot-side) keeps
// accuracy close to f16.
//
// Layout of pages_i8 (u32 words; 4 int8 per u32):
//   word_idx = page_no * (32*16*2*24) + head * (16*2*24)
//            + slot * (2*24) + side * 24 + (dim/4)
//
// Layout of scales (f16):
//   scale_idx = page_no * (32*16*2) + head * (16*2) + slot * 2 + side

enable f16;

@group(0) @binding(0) var<storage, read> Q                 : array<f16>;
@group(0) @binding(1) var<storage, read> page_table_indptr : array<i32>;
@group(0) @binding(2) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(3) var<storage, read> pages_i8          : array<u32>;
@group(0) @binding(4) var<storage, read> scales            : array<f16>;
@group(0) @binding(5) var<storage, read> length_info       : array<i32>;
@group(0) @binding(6) var<storage, read_write> output_buf  : array<f16>;

struct PODArgs {
  B: i32,
  max_num_pages: i32,
  nnz_pages: i32,
  pages_elem_offset: i32,
  page_indptr_elem_offset: i32,
  page_values_elem_offset: i32,
  length_info_elem_offset: i32,
  scales_elem_offset: i32,
  sm_scale: f32,
  packGridDimX: u32,
}
@group(0) @binding(7) var<uniform> podArgs : PODArgs;

var<workgroup> score_reduce : array<f32, 32>;

// Unpack signed int8 from a packed u32 word at byte index (0..3). Sign-extends.
fn unpack_i8(word : u32, byte_idx : u32) -> i32 {
  let raw : u32 = (word >> (byte_idx * 8u)) & 0xffu;
  // Sign-extend via arithmetic shift.
  return i32(raw << 24u) >> 24u;
}

@compute @workgroup_size(32, 1, 1)
fn attention_int8(
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
  let indptr_end   : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  var m  : f32 = -50000.0;
  var d  : f32 = 0.0;
  var o0 : f32 = 0.0;
  var o1 : f32 = 0.0;
  var o2 : f32 = 0.0;

  // tid * 3 in [0, 96). The 3 dims span either 1 u32 word (aligned) or
  // straddle 2 words. Precompute byte layout:
  let byte_0 : i32 = tid * 3;                 // 0..95
  let word_0 : i32 = byte_0 / 4;              // 0..23
  let lane_0 : u32 = u32(byte_0 - word_0 * 4);
  let word_1 : i32 = (byte_0 + 1) / 4;
  let lane_1 : u32 = u32((byte_0 + 1) - word_1 * 4);
  let word_2 : i32 = (byte_0 + 2) / 4;
  let lane_2 : u32 = u32((byte_0 + 2) - word_2 * 4);

  for (var page_idx : i32 = indptr_begin; page_idx < indptr_end; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
    let page_start : i32 = (page_idx - indptr_begin) * 16;
    let slots_in_page : i32 = min(16, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      // Base word indices for K and V within this (page, head, slot).
      let kv_word_base : i32 = page_no * (32 * 16 * 2 * 24)
                             + head * (16 * 2 * 24)
                             + slot * (2 * 24)
                             + podArgs.pages_elem_offset;
      let k_word_base : i32 = kv_word_base;          // side=0
      let v_word_base : i32 = kv_word_base + 24;     // side=1

      // Per-(head, slot, side) scales.
      let scale_base : i32 = page_no * (32 * 16 * 2)
                           + head * (16 * 2)
                           + slot * 2
                           + podArgs.scales_elem_offset;
      let k_scale : f32 = f32(scales[scale_base]);
      let v_scale : f32 = f32(scales[scale_base + 1]);

      // K dequant: read 3 int8 values, multiply by k_scale.
      let k0 : f32 = f32(unpack_i8(pages_i8[k_word_base + word_0], lane_0)) * k_scale;
      let k1 : f32 = f32(unpack_i8(pages_i8[k_word_base + word_1], lane_1)) * k_scale;
      let k2 : f32 = f32(unpack_i8(pages_i8[k_word_base + word_2], lane_2)) * k_scale;

      let partial : f32 = q0 * k0 + q1 * k1 + q2 * k2;

      score_reduce[tid] = partial;
      workgroupBarrier();
      if (tid < 16) { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 16]; }
      workgroupBarrier();
      if (tid < 8)  { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 8]; }
      workgroupBarrier();
      if (tid < 4)  { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 4]; }
      workgroupBarrier();
      if (tid < 2)  { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 2]; }
      workgroupBarrier();
      if (tid < 1)  { score_reduce[tid] = score_reduce[tid] + score_reduce[tid + 1]; }
      workgroupBarrier();

      let s : f32 = score_reduce[0] * podArgs.sm_scale;

      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new  : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      // V dequant + online softmax accumulate.
      let vv0 : f32 = f32(unpack_i8(pages_i8[v_word_base + word_0], lane_0)) * v_scale;
      let vv1 : f32 = f32(unpack_i8(pages_i8[v_word_base + word_1], lane_1)) * v_scale;
      let vv2 : f32 = f32(unpack_i8(pages_i8[v_word_base + word_2], lane_2)) * v_scale;

      o0 = o0 * scale_prev + scale_new * vv0;
      o1 = o1 * scale_prev + scale_new * vv1;
      o2 = o2 * scale_prev + scale_new * vv2;
    }
  }

  if (d > 0.0) {
    let inv_d : f32 = 1.0 / d;
    output_buf[batch * 3072 + head * 96 + tid * 3]     = f16(o0 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 1] = f16(o1 * inv_d);
    output_buf[batch * 3072 + head * 96 + tid * 3 + 2] = f16(o2 * inv_d);
  }
}
