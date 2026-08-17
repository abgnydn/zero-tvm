// PAGED int8-KV ATTENTION (CHUNKED PREFILL) — attention_prefill.wgsl's causal
// chunk loop reading the packed int8 cache instead of f16.
//
// This is a MERGE of two kernels that are already verified, and deliberately
// nothing more:
//   - the chunk/causal structure, page walk and online softmax come from
//     attention_prefill.wgsl verbatim (identity page mapping, kv_len =
//     len_base + t, so chunk token t sees exactly slots [0, chunkStart + t]);
//   - the unpacking, scale handling and dequant order come from
//     attention_int8.wgsl verbatim (one f16 scale per (page, kv_head, slot,
//     side); K's scale factored out of the dot, V's folded into scale_new).
// Keeping both halves byte-identical to their sources is what makes a
// divergence here mean "the wiring is wrong", never "the maths drifted".
//
// Why it exists: int8 KV forced per-token prefill, because the chunk path
// bound f16 append/attention kernels. That made the one quantized-KV feature
// unusable on exactly the long prompts it saves memory for.
//
// Grid: (seq_len, HEADS). Model-shape constants come from shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read> Q : array<f16>;                 // seq * Q_DIM
@group(0) @binding(1) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(2) var<storage, read> pages_i8 : array<u32>;          // packed int8
@group(0) @binding(3) var<storage, read> scales : array<f16>;
@group(0) @binding(4) var<storage, read_write> output_buf : array<f16>;  // seq * Q_DIM

struct PODArgs {
  seq_len: i32,     // chunk tokens (grid x)
  len_base: i32,    // kv_len of chunk token 0 (= chunkStart + 1)
  sm_scale: f32,
  pages_elem_offset: i32,   // u32-word offset
  scales_elem_offset: i32,
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

const EPT = HEAD_DIM / 32;   // elements per thread

var<workgroup> score_reduce : array<f32, 32>;

fn unpack_i8(word : u32, byte_idx : u32) -> i32 {
  let raw : u32 = (word >> (byte_idx * 8u)) & 0xffu;
  // Sign-extend via arithmetic shift.
  return i32(raw << 24u) >> 24u;
}

@compute @workgroup_size(32, 1, 1)
fn attention_prefill_int8(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.x);   // chunk token index
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  if (batch >= podArgs.seq_len) { return; }

  // GQA: this query head's KV pages live under its KV head.
  let kv_head : i32 = head / GQA_GROUP;

  var q : array<f32, EPT>;
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    q[e] = f32(Q[batch * Q_DIM + head * HEAD_DIM + tid * EPT + e]);
  }

  // Causal window for this token: positions [0, kv_len).
  let kv_len : i32 = podArgs.len_base + batch;
  let n_pages : i32 = (kv_len + PAGE_SIZE - 1) / PAGE_SIZE;

  // Online softmax state
  var m : f32 = -50000.0;
  var d : f32 = 0.0;
  var o : array<f32, EPT>;
  for (var e : i32 = 0; e < EPT; e = e + 1) { o[e] = 0.0; }

  // Byte layout of this thread's dims inside the packed row. EPT=3 straddles
  // words, EPT=4 is word-aligned; both are handled by indexing per dim.
  var words : array<i32, EPT>;
  var lanes : array<u32, EPT>;
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    let byte_e : i32 = tid * EPT + e;
    words[e] = byte_e / 4;
    lanes[e] = u32(byte_e - words[e] * 4);
  }

  for (var page_idx : i32 = 0; page_idx < n_pages; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx];
    let page_start : i32 = page_idx * PAGE_SIZE;
    let slots_in_page : i32 = min(PAGE_SIZE, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      let kv_word_base : i32 = page_no * KV_I8_PAGE_WORDS
                             + kv_head * KV_I8_HEAD_WORDS
                             + slot * KV_I8_SLOT_WORDS
                             + podArgs.pages_elem_offset;
      let k_word_base : i32 = kv_word_base;                    // side=0
      let v_word_base : i32 = kv_word_base + KV_I8_ROW_WORDS;  // side=1

      let scale_base : i32 = page_no * KV_SCALES_PER_PAGE
                           + kv_head * KV_SCALES_PER_HEAD
                           + slot * KV_SCALES_PER_SLOT
                           + podArgs.scales_elem_offset;
      let k_scale : f32 = f32(scales[scale_base]);
      let v_scale : f32 = f32(scales[scale_base + 1]);

      // K dequant: k_scale factored out of the dot, as the decode kernel does.
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
      // iteration overwrites score_reduce[tid] (see attention.wgsl — the
      // missing barrier here was a real race lavapipe caught once).
      workgroupBarrier();

      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      // V dequant + accumulate; hoist scale_new * v_scale.
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
