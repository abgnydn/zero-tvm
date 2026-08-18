// KV_QUANTIZE_INT8 — int8 quantize of one (K,V) slot per layer, per decode step.
//
// Input:  k_slot [KV_DIM f16], v_slot [KV_DIM f16]  (from qkv_fused_scratch;
//         KV_DIM = KV_HEADS * HEAD_DIM — == HEADS * HEAD_DIM for MHA)
// Output: kv_pages_i8 (packed int8 in u32), kv_scales (per-(head, side) f16)
//
// Granularity: one scale per (page, slot, head, side) — HEAD_DIM f16 values
// share one scale. Per-row (head-slot-side) scale keeps accuracy close to f16.
//
// Layout of kv_pages_i8 (u32 words; 4 int8 per u32):
//   page[page_no] · head[h] · slot[s] · side(K|V) · dim_quad[q]
//     word_idx = page_no * KV_I8_PAGE_WORDS
//              + h * KV_I8_HEAD_WORDS
//              + s * KV_I8_SLOT_WORDS
//              + side * KV_I8_ROW_WORDS
//              + q
//     (side = 0 for K, 1 for V; q = dim/4)
//
// Layout of kv_scales (f16):
//   scale_idx = page_no * KV_SCALES_PER_PAGE + h * KV_SCALES_PER_HEAD
//             + s * KV_SCALES_PER_SLOT + side
//
// One workgroup = one (kv head, side) — KV_HEADS × 2 workgroups. 32 threads
// cooperate to compute max over HEAD_DIM dims (EPT = HEAD_DIM/32 dims per
// thread), then threads 0..KV_I8_ROW_WORDS-1 each pack one u32 word (4 dims).
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read>       k_slot    : array<f16>;  // HEADS * HEAD_DIM
@group(0) @binding(1) var<storage, read>       v_slot    : array<f16>;  // HEADS * HEAD_DIM
@group(0) @binding(2) var<storage, read_write> pages_i8  : array<u32>;  // packed int8
@group(0) @binding(3) var<storage, read_write> scales    : array<f16>;
@group(0) @binding(4) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,  // u32-word offset
  scales_elem_offset       : i32,
  packGridDimX             : u32,  // KV_HEADS × 2 sides
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

const EPT = HEAD_DIM / 32;   // dims per thread in the max phase (Phi-3: 3, Qwen3: 4)

var<workgroup> max_reduce : array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn kv_quantize_int8(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let wg_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg_id) >= podArgs.packGridDimX) { return; }

  // TOKEN axis (grid y). Decode dispatches y=1 and lands on tok=0, which is
  // exactly the single-slot behaviour this kernel had before; chunked prefill
  // dispatches y=n and quantizes a whole chunk in one go. One kernel serves
  // both, so there is no second copy of the packing to keep in step.
  let tok : i32 = i32(blockIdx.y);
  let tid : i32 = i32(threadIdx.x);

  // Decompose wg_id → (kv head, side). side=0 for K, 1 for V.
  let head : i32 = wg_id / 2;
  let side : i32 = wg_id - head * 2;

  // Each thread reads EPT of the HEAD_DIM dims for this (head, side).
  // k_slot/v_slot are [token, KV_DIM]; tok is 0 on the decode path.
  let tok_base : i32 = tok * KV_DIM;
  let slot_base : i32 = tok_base + head * HEAD_DIM + tid * EPT;
  var m : f32 = 0.0;
  for (var e : i32 = 0; e < EPT; e = e + 1) {
    var v : f32;
    if (side == 0) {
      v = f32(k_slot[slot_base + e]);
    } else {
      v = f32(v_slot[slot_base + e]);
    }
    m = max(m, abs(v));
  }

  // Tree-reduce max(|x|) over 32 threads.
  max_reduce[tid] = m;
  workgroupBarrier();
  if (tid < 16) { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 16]); }
  workgroupBarrier();
  if (tid < 8)  { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 8]); }
  workgroupBarrier();
  if (tid < 4)  { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 4]); }
  workgroupBarrier();
  if (tid < 2)  { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 2]); }
  workgroupBarrier();
  if (tid < 1)  { max_reduce[tid] = max(max_reduce[tid], max_reduce[tid + 1]); }
  workgroupBarrier();

  // scale = max / 127; guard against zero (empty row).
  var scale : f32 = max_reduce[0] / 127.0;
  if (scale < 1e-8) { scale = 1e-8; }
  let inv_scale : f32 = 1.0 / scale;

  // Compute destination page/slot.
  let position : i32 = position_map[podArgs.position_map_elem_offset + tok];
  let page_no : i32 = position / PAGE_SIZE;
  let slot : i32 = position - page_no * PAGE_SIZE;

  // Thread 0 writes the scale.
  if (tid == 0) {
    let scale_idx : i32 = page_no * KV_SCALES_PER_PAGE + head * KV_SCALES_PER_HEAD
                          + slot * KV_SCALES_PER_SLOT + side
                          + podArgs.scales_elem_offset;
    scales[scale_idx] = f16(scale);
  }

  // Pack phase — thread `t` writes words t, t+32, t+64 …, so a row WIDER than
  // 32 words is still covered. It used to write exactly one word each and
  // return, which is correct only while KV_I8_ROW_WORDS (HEAD_DIM/4) <= 32.
  // That held for every spec int8 could reach while it was fused-only —
  // Phi-3 24, Qwen3 32 — and is false at HEAD_DIM 256, where 64 words meant
  // dims 128..255 were NEVER written and each row's top half kept whatever
  // the buffer already held. Coherent, wrong output; found the moment int8
  // reached a head-dim-256 hybrid.
  // The max-phase EPT split does not align to 4-dim groups, so re-read dims.
  for (var w : i32 = tid; w < KV_I8_ROW_WORDS; w = w + 32) {
    let d0 : i32 = w * 4;
    let pack_base : i32 = tok_base + head * HEAD_DIM + d0;
    var b0 : f32;
    var b1 : f32;
    var b2 : f32;
    var b3 : f32;
    if (side == 0) {
      b0 = f32(k_slot[pack_base]);
      b1 = f32(k_slot[pack_base + 1]);
      b2 = f32(k_slot[pack_base + 2]);
      b3 = f32(k_slot[pack_base + 3]);
    } else {
      b0 = f32(v_slot[pack_base]);
      b1 = f32(v_slot[pack_base + 1]);
      b2 = f32(v_slot[pack_base + 2]);
      b3 = f32(v_slot[pack_base + 3]);
    }

    let q0 : i32 = clamp(i32(round(b0 * inv_scale)), -127, 127);
    let q1 : i32 = clamp(i32(round(b1 * inv_scale)), -127, 127);
    let q2 : i32 = clamp(i32(round(b2 * inv_scale)), -127, 127);
    let q3 : i32 = clamp(i32(round(b3 * inv_scale)), -127, 127);

    let packed : u32 =
        (u32(q0) & 0xffu)
      | ((u32(q1) & 0xffu) << 8u)
      | ((u32(q2) & 0xffu) << 16u)
      | ((u32(q3) & 0xffu) << 24u);

    // word_idx: page[page_no] · head[h] · slot[s] · side · dim_quad[w]
    let word_idx : i32 = page_no * KV_I8_PAGE_WORDS
                       + head * KV_I8_HEAD_WORDS
                       + slot * KV_I8_SLOT_WORDS
                       + side * KV_I8_ROW_WORDS
                       + w
                       + podArgs.pages_elem_offset;
    pages_i8[word_idx] = packed;
  }
}
