// PACKED-BRANCH PREFILL ATTENTION — attention_prefill.wgsl for a chunk that
// holds SEVERAL independent continuations of one cached prefix.
//
// A decision model asks N questions about one state. Each question is a
// branch that must see the state and itself only (kev/model.py's block-causal
// mask). Run branch by branch, every branch pays a full pass of ~10 dispatches
// per layer — measured ~31 ms per question on Kev-0.6B, flat in the branch
// length, because the pass is dispatch-bound at a dozen tokens. This kernel
// lets ALL branches ride one chunk: one GEMM per projection reads the weights
// once for every question.
//
// What differs from attention_prefill.wgsl:
//   - the prefix (the state) is read from the PAGED cache, slots [0, state_len)
//   - the chunk's own K/V are read from the chunk buffers rope.wgsl wrote —
//     NOT from the pages. Branch tokens share positions (every branch starts
//     at state_len), so the position-indexed page slots cannot hold more than
//     one of them; the chunk never runs kv_append.
//   - a query at chunk token t attends to a chunk token u only when
//     seg[u] == seg[t] and u <= t (causal within its own branch)
//
// Per (token, head) the slot order — prefix slots in order, then own-branch
// tokens in order — and the online-softmax arithmetic are IDENTICAL to
// attention_prefill.wgsl over `state + branch` as one causal row, and the K/V
// values are the same f16 that kv_append would have copied, so the output is
// bit-exact against the branch-by-branch path. tests/kernels pins that.
//
// Grid: (seq_len, HEADS). Model-shape constants are injected by
// src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read> Q : array<f16>;                    // seq * Q_DIM
@group(0) @binding(1) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(2) var<storage, read> pages : array<f16>;                // the cached prefix
@group(0) @binding(3) var<storage, read> k_chunk : array<f16>;              // seq * KV_DIM (RoPE'd)
@group(0) @binding(4) var<storage, read> v_chunk : array<f16>;              // seq * KV_DIM
@group(0) @binding(5) var<storage, read> seg : array<i32>;                  // seq: branch id per token
@group(0) @binding(6) var<storage, read_write> output_buf : array<f16>;     // seq * Q_DIM

struct PODArgs {
  seq_len: i32,     // chunk tokens (grid x)
  state_len: i32,   // prefix slots every token attends to: [0, state_len)
  sm_scale: f32
}
@group(0) @binding(7) var<uniform> podArgs : PODArgs;

const EPT = HEAD_DIM / 32;   // elements per thread

var<workgroup> score_reduce : array<f32, 32>;

// Full 32-thread dot product of this thread's q slice against a K slice, then
// the same online-softmax step attention_prefill.wgsl performs per slot.
fn reduce_score(partial : f32, tid : i32) -> f32 {
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
  // Everyone has read score_reduce[0] before the next call overwrites it.
  workgroupBarrier();
  return s;
}

@compute @workgroup_size(32, 1, 1)
fn attention_prefill_seg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let batch : i32 = i32(blockIdx.x);   // chunk token index
  let head : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  if (batch >= podArgs.seq_len) { return; }

  let kv_head : i32 = head / GQA_GROUP;

  var q : array<f32, EPT>;
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    q[e] = f32(Q[batch * Q_DIM + head * HEAD_DIM + tid * EPT + e]);
  }

  var m : f32 = -50000.0;
  var d : f32 = 0.0;
  var o : array<f32, EPT>;
  for (var e : i32 = 0; e < EPT; e = e + 1) { o[e] = 0.0; }

  // 1. The cached prefix: slots [0, state_len) through the page table.
  let kv_len : i32 = podArgs.state_len;
  let n_pages : i32 = (kv_len + PAGE_SIZE - 1) / PAGE_SIZE;
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
      let s : f32 = reduce_score(partial, tid);
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

  // 2. This token's own branch: chunk tokens u <= batch with the same segment.
  let my_seg : i32 = seg[batch];
  for (var u : i32 = 0; u <= batch; u = u + 1) {
    if (seg[u] != my_seg) { continue; }
    let kv_base : i32 = u * KV_DIM + kv_head * HEAD_DIM;
    var partial : f32 = 0.0;
    for (var e : i32 = 0; e < EPT; e = e + 1) {
      partial = partial + q[e] * f32(k_chunk[kv_base + tid * EPT + e]);
    }
    let s : f32 = reduce_score(partial, tid);
    let m_prev : f32 = m;
    m = max(m, s);
    let scale_prev : f32 = exp(m_prev - m);
    let scale_new : f32 = exp(s - m);
    d = d * scale_prev + scale_new;
    for (var e : i32 = 0; e < EPT; e = e + 1) {
      o[e] = o[e] * scale_prev + scale_new * f32(v_chunk[kv_base + tid * EPT + e]);
    }
  }

  if (d > 0.0) {
    let inv_d : f32 = 1.0 / d;
    for (var e : i32 = 0; e < EPT; e = e + 1) {
      output_buf[batch * Q_DIM + head * HEAD_DIM + tid * EPT + e] = f16(o[e] * inv_d);
    }
  }
}
