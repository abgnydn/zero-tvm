// PAGED KV ATTENTION (CHUNKED PREFILL) — attention.wgsl's decode loop run for
// a CHUNK of query tokens in one dispatch, causality enforced per token.
//
// The decode kernel already indexes Q/output by `batch`; this sibling reuses
// that axis as the chunk-token axis: workgroup (t, head) computes attention
// for chunk token t with kv_len = len_base + t (len_base = chunkStart + 1),
// so token t sees exactly the KV slots [0, chunkStart + t] — all of which the
// preceding kv_append dispatch (this chunk) or earlier chunks have written.
// Per (token, head) the slot loop, online-softmax order and f32 math are
// IDENTICAL to attention.wgsl with that kv_len, so outputs are bit-exact vs
// the per-token decode dispatch chain.
//
// Page table: identity mapping via page_table_values (same buffer the decode
// path binds); pages 0 .. ceil(kv_len/PAGE_SIZE)-1 are traversed directly —
// no per-batch indptr (a shared indptr cannot express per-token page counts).
//
// Grid: (seq_len, HEADS). Model-shape constants are injected by
// src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read> Q : array<f16>;                    // seq * Q_DIM
@group(0) @binding(1) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(2) var<storage, read> pages : array<f16>;
@group(0) @binding(3) var<storage, read_write> output_buf : array<f16>;     // seq * Q_DIM

struct PODArgs {
  seq_len: i32,     // chunk tokens (grid x)
  len_base: i32,    // kv_len of chunk token 0 (= chunkStart + 1)
  sm_scale: f32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const EPT = HEAD_DIM / 32;   // elements per thread

var<workgroup> score_reduce : array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn attention_prefill(
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

  for (var page_idx : i32 = 0; page_idx < n_pages; page_idx = page_idx + 1) {
    let page_no : i32 = page_table_values[page_idx];
    let page_start : i32 = page_idx * PAGE_SIZE;
    let slots_in_page : i32 = min(PAGE_SIZE, kv_len - page_start);

    for (var slot : i32 = 0; slot < slots_in_page; slot = slot + 1) {
      let k_base : i32 = page_no * KV_PAGE_STRIDE + kv_head * HEAD_PAGE_STRIDE + slot * HEAD_DIM;

      var partial : f32 = 0.0;
      for (var e : i32 = 0; e < EPT; e = e + 1) {
        partial = partial + q[e] * f32(pages[k_base + tid * EPT + e]);
      }

      // Tree reduction across 32 threads to get the full dot product
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

      // Online softmax update
      let m_prev : f32 = m;
      m = max(m, s);
      let scale_prev : f32 = exp(m_prev - m);
      let scale_new : f32 = exp(s - m);

      d = d * scale_prev + scale_new;

      let v_base : i32 = k_base + V_PAGE_OFFSET;
      for (var e : i32 = 0; e < EPT; e = e + 1) {
        o[e] = o[e] * scale_prev + scale_new * f32(pages[v_base + tid * EPT + e]);
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
