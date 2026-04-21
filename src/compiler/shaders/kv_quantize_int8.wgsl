// KV_QUANTIZE_INT8 — int8 quantize of one (K,V) slot per layer, per decode step.
//
// Input:  k_slot [3072 f16], v_slot [3072 f16]  (from qkv_fused_scratch)
// Output: kv_pages_i8 (packed int8 in u32), kv_scales (per-(head, side) f16)
//
// Granularity: one scale per (page, slot, head, side) — 96 f16 values share
// one scale. Per-row (head-slot-side) scale keeps accuracy close to f16.
//
// Layout of kv_pages_i8 (u32 words; 4 int8 per u32):
//   page[page_no] · head[h] · slot[s] · side(K|V) · dim_quad[q]
//     word_idx = page_no * (32*16*2*24)
//              + h * (16*2*24)
//              + s * (2*24)
//              + side * 24
//              + q
//     (side = 0 for K, 1 for V; q = dim/4)
//
// Layout of kv_scales (f16):
//   scale_idx = page_no * (32*16*2)  +  h * (16*2)  +  s * 2  +  side
//
// One workgroup = one (head, side). 32 threads cooperate to compute max over
// 96 dims, then each thread quantizes 3 consecutive dims (96 = 32 × 3).

enable f16;

@group(0) @binding(0) var<storage, read>       k_slot    : array<f16>;  // 3072
@group(0) @binding(1) var<storage, read>       v_slot    : array<f16>;  // 3072
@group(0) @binding(2) var<storage, read_write> pages_i8  : array<u32>;  // packed int8
@group(0) @binding(3) var<storage, read_write> scales    : array<f16>;
@group(0) @binding(4) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,  // u32-word offset
  scales_elem_offset       : i32,
  packGridDimX             : u32,  // 64 = 32 heads × 2 sides
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> max_reduce : array<f32, 32>;

@compute @workgroup_size(32, 1, 1)
fn kv_quantize_int8(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let wg_id : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg_id) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);

  // Decompose wg_id → (head, side). side=0 for K, 1 for V.
  let head : i32 = wg_id / 2;
  let side : i32 = wg_id - head * 2;

  // Each thread reads 3 of the 96 dims for this (head, side).
  let slot_base : i32 = head * 96 + tid * 3;
  var v0 : f32;
  var v1 : f32;
  var v2 : f32;
  if (side == 0) {
    v0 = f32(k_slot[slot_base]);
    v1 = f32(k_slot[slot_base + 1]);
    v2 = f32(k_slot[slot_base + 2]);
  } else {
    v0 = f32(v_slot[slot_base]);
    v1 = f32(v_slot[slot_base + 1]);
    v2 = f32(v_slot[slot_base + 2]);
  }

  // Tree-reduce max(|x|) over 32 threads.
  var m : f32 = max(abs(v0), max(abs(v1), abs(v2)));
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
  let position : i32 = position_map[podArgs.position_map_elem_offset];
  let page_no : i32 = position / 16;
  let slot : i32 = position - page_no * 16;

  // Thread 0 writes the scale.
  if (tid == 0) {
    let scale_idx : i32 = page_no * (32 * 16 * 2) + head * (16 * 2) + slot * 2 + side
                          + podArgs.scales_elem_offset;
    scales[scale_idx] = f16(scale);
  }

  // Quantize 3 dims and pack into int8. Because 3 is not a multiple of 4,
  // we use atomic-free write-via-packing: each thread writes its own
  // contribution to one u32 word using read-modify-write (safe because
  // different threads touch non-overlapping 8-bit lanes within each word).
  //
  // Simpler: redistribute so each thread owns ONE u32 word = 4 dims.
  // 24 words per row, 32 threads → threads 0..23 each own one word.
  // We already have v0..v2 (dims tid*3, tid*3+1, tid*3+2). That does NOT
  // align to 4-dim groups. So we need to re-read.
  //
  // Switch to per-word indexing: thread `t` in [0, 24) writes one word
  // containing dims [t*4, t*4+4). Re-read those dims.
  if (tid >= 24) { return; }

  let d0 : i32 = tid * 4;
  var b0 : f32;
  var b1 : f32;
  var b2 : f32;
  var b3 : f32;
  if (side == 0) {
    b0 = f32(k_slot[head * 96 + d0]);
    b1 = f32(k_slot[head * 96 + d0 + 1]);
    b2 = f32(k_slot[head * 96 + d0 + 2]);
    b3 = f32(k_slot[head * 96 + d0 + 3]);
  } else {
    b0 = f32(v_slot[head * 96 + d0]);
    b1 = f32(v_slot[head * 96 + d0 + 1]);
    b2 = f32(v_slot[head * 96 + d0 + 2]);
    b3 = f32(v_slot[head * 96 + d0 + 3]);
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

  // word_idx: page[page_no] · head[h] · slot[s] · side · dim_quad[tid]
  let word_idx : i32 = page_no * (32 * 16 * 2 * 24)
                     + head * (16 * 2 * 24)
                     + slot * (2 * 24)
                     + side * 24
                     + tid
                     + podArgs.pages_elem_offset;
  pages_i8[word_idx] = packed;
}
