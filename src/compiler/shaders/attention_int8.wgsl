// PAGED KV ATTENTION — int8 KV cache variant of attention.wgsl.
//
// Identical online-softmax math; the only change is reading K,V as int8 + f16
// scale instead of f16. Per-row scale (one scale per head-slot-side) keeps
// accuracy close to f16.
//
// GQA-aware: query head h reads the KV pages/scales of kv_head = h / GQA_GROUP.
//
// Layout of pages_i8 (u32 words; 4 int8 per u32):
//   word_idx = page_no * KV_I8_PAGE_WORDS + kv_head * KV_I8_HEAD_WORDS
//            + slot * KV_I8_SLOT_WORDS + side * KV_I8_ROW_WORDS + (dim/4)
//
// Layout of scales (f16):
//   scale_idx = page_no * KV_SCALES_PER_PAGE + kv_head * KV_SCALES_PER_HEAD
//             + slot * KV_SCALES_PER_SLOT + side
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

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

const EPT = HEAD_DIM / 32;   // elements per thread (Phi-3: 3, Qwen3: 4)

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

  let kv_head : i32 = head / GQA_GROUP;

  // Each thread owns EPT elements of HEAD_DIM (32 threads × EPT = HEAD_DIM).
  var q : array<f32, EPT>;
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    q[e] = f32(Q[batch * Q_DIM + head * HEAD_DIM + tid * EPT + e]);
  }

  let indptr_begin : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset];
  let indptr_end   : i32 = page_table_indptr[batch + podArgs.page_indptr_elem_offset + 1];
  let kv_len : i32 = length_info[batch + podArgs.length_info_elem_offset];

  var m  : f32 = -50000.0;
  var d  : f32 = 0.0;
  var o : array<f32, EPT>;
  for (var e : i32 = 0; e < EPT; e = e + 1) { o[e] = 0.0; }

  // tid * EPT in [0, HEAD_DIM). Each dim maps to a (word, byte-lane) inside
  // the packed int8 row; EPT=3 straddles words, EPT=4 is word-aligned.
  // Precompute byte layout:
  var words : array<i32, EPT>;
  var lanes : array<u32, EPT>;
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    let byte_e : i32 = tid * EPT + e;         // 0..HEAD_DIM-1
    words[e] = byte_e / 4;                    // 0..KV_I8_ROW_WORDS-1
    lanes[e] = u32(byte_e - words[e] * 4);
  }

  for (var page_idx : i32 = indptr_begin; page_idx < indptr_end; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx + podArgs.page_values_elem_offset];
    let page_start : i32 = (page_idx - indptr_begin) * PAGE_SIZE;
    let slots_in_page : i32 = min(PAGE_SIZE, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      // Base word indices for K and V within this (page, kv_head, slot).
      let kv_word_base : i32 = page_no * KV_I8_PAGE_WORDS
                             + kv_head * KV_I8_HEAD_WORDS
                             + slot * KV_I8_SLOT_WORDS
                             + podArgs.pages_elem_offset;
      let k_word_base : i32 = kv_word_base;                    // side=0
      let v_word_base : i32 = kv_word_base + KV_I8_ROW_WORDS;  // side=1

      // Per-(kv_head, slot, side) scales.
      let scale_base : i32 = page_no * KV_SCALES_PER_PAGE
                           + kv_head * KV_SCALES_PER_HEAD
                           + slot * KV_SCALES_PER_SLOT
                           + podArgs.scales_elem_offset;
      let k_scale : f32 = f32(scales[scale_base]);
      let v_scale : f32 = f32(scales[scale_base + 1]);

      // K dequant: read EPT int8 values; k_scale factored out of the dot.
      var qk : f32 = 0.0;
      for (var e : i32 = 0; e < EPT; e = e + 1) {
        qk = qk + q[e] * f32(unpack_i8(pages_i8[k_word_base + words[e]], lanes[e]));
      }
      let partial : f32 = qk * k_scale;

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
      // All threads must finish reading score_reduce[0] before the next slot
      // iteration overwrites score_reduce[tid] (see attention.wgsl).
      workgroupBarrier();

      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new  : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      // V dequant + online softmax accumulate; hoist scale_new * v_scale.
      let sv : f32 = scale_new * v_scale;
      for (var e : i32 = 0; e < EPT; e = e + 1) {
        let vv : f32 = f32(unpack_i8(pages_i8[v_word_base + words[e]], lanes[e]));
        o[e] = o[e] * scale_prev + sv * vv;
      }
    }
  }

  if (d > 0.0) {
    let inv_d : f32 = 1.0 / d;
    for (var e : i32 = 0; e < EPT; e = e + 1) {
      output_buf[batch * Q_DIM + head * HEAD_DIM + tid * EPT + e] = f16(o[e] * inv_d);
    }
  }
}
