// QKV_FUSED — decode-path fusion of QKV matmul + RoPE + KV append.
//
// Replaces 3 dispatches per layer with 1:
//   int4_matmul (QKV_DIM WG) + rope + kv_append  →  qkv_fused (QKV_DIM/2 WG)
//
// Each workgroup computes TWO output rows of the QKV projection that form a
// RoPE pair (dim and dim+HALF_HEAD_DIM within the same head). The pair is
// rotated in registers, and K/V are written straight into the paged KV cache
// — the intermediate qkv/kOut/vOut buffers are skipped entirely.
//
// Workgroup index layout (pair_idx) — GQA-aware unequal [Q | K | V] groups:
//     [0, QKV_GROUP_PAIRS)             — Q: HEADS × HALF_HEAD_DIM dim-pairs, write to q_out
//     [.., + KV_GROUP_PAIRS)           — K: KV_HEADS heads, write K into kv_pages
//     [.., + KV_GROUP_PAIRS)           — V: KV_HEADS heads, write V into kv_pages
// (heads == kvHeads for Phi-3, so the historical equal-thirds layout is a
// special case of this decomposition.)
//
// Decode-only (ntoken=1). Prefill still uses the 3-dispatch path.
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> q_out        : array<f16>;  // D-wide Q
@group(0) @binding(1) var<storage, read_write> kv_pages     : array<f16>;  // paged KV cache
@group(0) @binding(2) var<storage, read>       hidden       : array<f16>;  // D-wide hidden state
@group(0) @binding(3) var<storage, read>       scales       : array<f16>;  // QKV_DIM × D_SCALES f16
@group(0) @binding(4) var<storage, read>       weights      : array<u32>;  // QKV_DIM × D_PACKED u32
@group(0) @binding(5) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,
  packGridDimX             : u32,
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const KV_GROUP_PAIRS = KV_DIM / 2;   // KV_HEADS * HALF_HEAD_DIM

var<workgroup> red0 : array<f32, 64>;
var<workgroup> red1 : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn qkv_fused(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let pair_idx : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(pair_idx) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);

  // Decompose pair_idx → (group, head, dim_lo) where group ∈ {0:Q, 1:K, 2:V}.
  // Groups are unequal under GQA: Q holds QKV_GROUP_PAIRS pairs, K and V hold
  // KV_GROUP_PAIRS each. `head` is a KV-head index inside the K/V groups.
  var group : i32 = 0;
  var pair_in_group : i32 = pair_idx;
  var row_base : i32 = 0;
  if (pair_in_group >= QKV_GROUP_PAIRS) {
    group = 1;
    pair_in_group = pair_in_group - QKV_GROUP_PAIRS;
    row_base = Q_DIM;
    if (pair_in_group >= KV_GROUP_PAIRS) {
      group = 2;
      pair_in_group = pair_in_group - KV_GROUP_PAIRS;
      row_base = Q_DIM + KV_DIM;
    }
  }
  let head   : i32 = pair_in_group / HALF_HEAD_DIM;
  let dim_lo : i32 = pair_in_group - head * HALF_HEAD_DIM;

  // Absolute rows in the QKV_DIM-wide QKV projection.
  let row_lo : i32 = row_base + head * HEAD_DIM + dim_lo;
  let row_hi : i32 = row_lo + HALF_HEAD_DIM;

  // Two dot products in parallel: acc0 = row_lo · hidden, acc1 = row_hi · hidden.
  var acc0 : f32 = 0.0;
  var acc1 : f32 = 0.0;

  // Per-group scale factored out of the 8 unpacked terms.
  for (var chunk : i32 = 0; chunk < D_PACKED / 64; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 64;
    let packed_lo : u32 = weights[row_lo * D_PACKED + w_offset];
    let packed_hi : u32 = weights[row_hi * D_PACKED + w_offset];
    let scale_lo : f32 = f32(scales[row_lo * D_SCALES + (w_offset >> 2)]);
    let scale_hi : f32 = f32(scales[row_hi * D_SCALES + (w_offset >> 2)]);
    let base : i32 = w_offset * 8;

    let x0 = f32(hidden[base    ]);
    let x1 = f32(hidden[base + 1]);
    let x2 = f32(hidden[base + 2]);
    let x3 = f32(hidden[base + 3]);
    let x4 = f32(hidden[base + 4]);
    let x5 = f32(hidden[base + 5]);
    let x6 = f32(hidden[base + 6]);
    let x7 = f32(hidden[base + 7]);

    acc0 = acc0 + scale_lo * (
        x0 * (f32((packed_lo >>  0u) & 15u) - 7.0)
      + x1 * (f32((packed_lo >>  4u) & 15u) - 7.0)
      + x2 * (f32((packed_lo >>  8u) & 15u) - 7.0)
      + x3 * (f32((packed_lo >> 12u) & 15u) - 7.0)
      + x4 * (f32((packed_lo >> 16u) & 15u) - 7.0)
      + x5 * (f32((packed_lo >> 20u) & 15u) - 7.0)
      + x6 * (f32((packed_lo >> 24u) & 15u) - 7.0)
      + x7 * (f32((packed_lo >> 28u) & 15u) - 7.0));

    acc1 = acc1 + scale_hi * (
        x0 * (f32((packed_hi >>  0u) & 15u) - 7.0)
      + x1 * (f32((packed_hi >>  4u) & 15u) - 7.0)
      + x2 * (f32((packed_hi >>  8u) & 15u) - 7.0)
      + x3 * (f32((packed_hi >> 12u) & 15u) - 7.0)
      + x4 * (f32((packed_hi >> 16u) & 15u) - 7.0)
      + x5 * (f32((packed_hi >> 20u) & 15u) - 7.0)
      + x6 * (f32((packed_hi >> 24u) & 15u) - 7.0)
      + x7 * (f32((packed_hi >> 28u) & 15u) - 7.0));
  }

  red0[tid] = acc0;
  red1[tid] = acc1;
  workgroupBarrier();

  if (tid < 32) {
    red0[tid] = red0[tid] + red0[tid + 32];
    red1[tid] = red1[tid] + red1[tid + 32];
  }
  workgroupBarrier();
  if (tid < 16) {
    red0[tid] = red0[tid] + red0[tid + 16];
    red1[tid] = red1[tid] + red1[tid + 16];
  }
  workgroupBarrier();
  if (tid < 8) {
    red0[tid] = red0[tid] + red0[tid + 8];
    red1[tid] = red1[tid] + red1[tid + 8];
  }
  workgroupBarrier();
  if (tid < 4) {
    red0[tid] = red0[tid] + red0[tid + 4];
    red1[tid] = red1[tid] + red1[tid + 4];
  }
  workgroupBarrier();
  if (tid < 2) {
    red0[tid] = red0[tid] + red0[tid + 2];
    red1[tid] = red1[tid] + red1[tid + 2];
  }
  workgroupBarrier();
  if (tid != 0) { return; }

  let v_lo : f32 = red0[0] + red0[1];
  let v_hi : f32 = red1[0] + red1[1];

  // V: no RoPE, copy into paged KV cache.
  if (group == 2) {
    let position : i32 = position_map[podArgs.position_map_elem_offset];
    let page_no : i32 = position / PAGE_SIZE;
    let slot : i32 = position - page_no * PAGE_SIZE;
    // V region starts at V_PAGE_OFFSET within each page (see kv_append.wgsl).
    let v_base : i32 = page_no * KV_PAGE_STRIDE + head * HEAD_PAGE_STRIDE + slot * HEAD_DIM
                       + V_PAGE_OFFSET + podArgs.pages_elem_offset;
    kv_pages[v_base + dim_lo]                 = f16(v_lo);
    kv_pages[v_base + dim_lo + HALF_HEAD_DIM] = f16(v_hi);
    return;
  }

  // Q or K: apply RoPE. dim_lo ∈ [0, HALF_HEAD_DIM) is the "low" index;
  // row_hi is "high". Matches rope.wgsl:
  //   dim < HALF_HEAD_DIM  : out = cos*x + sin*(-x_pair_hi)
  //   dim >= HALF_HEAD_DIM : out = cos*x + sin*( x_pair_lo)
  let pos  : f32 = f32(position_map[podArgs.position_map_elem_offset]);
  let freq : f32 = pos / pow(ROPE_THETA, f32(dim_lo * 2) / f32(HEAD_DIM));
  let c    : f32 = cos(freq);
  let s    : f32 = sin(freq);

  let rot_lo : f32 = c * v_lo + s * (-v_hi);
  let rot_hi : f32 = c * v_hi + s * ( v_lo);

  if (group == 0) {
    let base = head * HEAD_DIM + dim_lo;
    q_out[base                 ] = f16(rot_lo);
    q_out[base + HALF_HEAD_DIM ] = f16(rot_hi);
  } else {
    // K → paged KV cache.
    let position : i32 = position_map[podArgs.position_map_elem_offset];
    let page_no : i32 = position / PAGE_SIZE;
    let slot : i32 = position - page_no * PAGE_SIZE;
    let k_base : i32 = page_no * KV_PAGE_STRIDE + head * HEAD_PAGE_STRIDE + slot * HEAD_DIM
                       + podArgs.pages_elem_offset;
    kv_pages[k_base + dim_lo]                 = f16(rot_lo);
    kv_pages[k_base + dim_lo + HALF_HEAD_DIM] = f16(rot_hi);
  }
}
