// QKV_FUSED — decode-path fusion of QKV matmul + RoPE + KV append.
//
// Replaces 3 dispatches per layer with 1:
//   int4_matmul (9216 WG) + rope (36 WG) + kv_append (12 WG)  →  qkv_fused (4608 WG)
//
// Each workgroup computes TWO output rows of the QKV projection that form a
// RoPE pair (dim and dim+48 within the same head). The pair is rotated in
// registers, and K/V are written straight into the paged KV cache — the
// intermediate qkv/kOut/vOut buffers are skipped entirely.
//
// Workgroup index layout (pair_idx):
//     [0, 1536)        — Q heads: 32 heads × 48 dim-pairs, write to q_out
//     [1536, 3072)     — K heads: 32 heads × 48 dim-pairs, write K into kv_pages
//     [3072, 4608)     — V heads: 32 heads × 48 dim-pairs, write V into kv_pages
//
// Decode-only (ntoken=1). Prefill still uses the 3-dispatch path.

enable f16;

@group(0) @binding(0) var<storage, read_write> q_out        : array<f16>;  // 3072 Q
@group(0) @binding(1) var<storage, read_write> kv_pages     : array<f16>;  // paged KV cache
@group(0) @binding(2) var<storage, read>       hidden       : array<f16>;  // 3072 hidden state
@group(0) @binding(3) var<storage, read>       scales       : array<f16>;  // 9216 × 96 f16
@group(0) @binding(4) var<storage, read>       weights      : array<u32>;  // 9216 × 384 u32
@group(0) @binding(5) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,
  packGridDimX             : u32,
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const K_PACKED       : i32 = 384;  // K=3072 → K/8
const SCALES_PER_ROW : i32 = 96;   // K/32

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
  let group         : i32 = pair_idx / 1536;
  let pair_in_group : i32 = pair_idx - group * 1536;
  let head          : i32 = pair_in_group / 48;
  let dim_lo        : i32 = pair_in_group - head * 48;

  // Absolute rows in the 9216-wide QKV projection.
  let row_lo : i32 = group * 3072 + head * 96 + dim_lo;
  let row_hi : i32 = row_lo + 48;

  // Two dot products in parallel: acc0 = row_lo · hidden, acc1 = row_hi · hidden.
  var acc0 : f32 = 0.0;
  var acc1 : f32 = 0.0;

  for (var chunk : i32 = 0; chunk < K_PACKED / 64; chunk = chunk + 1) {
    let w_offset : i32 = tid + chunk * 64;
    let packed_lo : u32 = weights[row_lo * K_PACKED + w_offset];
    let packed_hi : u32 = weights[row_hi * K_PACKED + w_offset];
    let scale_lo : f32 = f32(scales[row_lo * SCALES_PER_ROW + (w_offset >> 2)]);
    let scale_hi : f32 = f32(scales[row_hi * SCALES_PER_ROW + (w_offset >> 2)]);
    let base : i32 = w_offset * 8;

    let x0 = f32(hidden[base    ]);
    let x1 = f32(hidden[base + 1]);
    let x2 = f32(hidden[base + 2]);
    let x3 = f32(hidden[base + 3]);
    let x4 = f32(hidden[base + 4]);
    let x5 = f32(hidden[base + 5]);
    let x6 = f32(hidden[base + 6]);
    let x7 = f32(hidden[base + 7]);

    acc0 = acc0 + x0 * (f32((packed_lo >>  0u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x1 * (f32((packed_lo >>  4u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x2 * (f32((packed_lo >>  8u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x3 * (f32((packed_lo >> 12u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x4 * (f32((packed_lo >> 16u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x5 * (f32((packed_lo >> 20u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x6 * (f32((packed_lo >> 24u) & 15u) - 7.0) * scale_lo;
    acc0 = acc0 + x7 * (f32((packed_lo >> 28u) & 15u) - 7.0) * scale_lo;

    acc1 = acc1 + x0 * (f32((packed_hi >>  0u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x1 * (f32((packed_hi >>  4u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x2 * (f32((packed_hi >>  8u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x3 * (f32((packed_hi >> 12u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x4 * (f32((packed_hi >> 16u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x5 * (f32((packed_hi >> 20u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x6 * (f32((packed_hi >> 24u) & 15u) - 7.0) * scale_hi;
    acc1 = acc1 + x7 * (f32((packed_hi >> 28u) & 15u) - 7.0) * scale_hi;
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
    let page_no : i32 = position / 16;
    let slot : i32 = position - page_no * 16;
    // V region starts at 49152 within each page (see kv_append.wgsl).
    let v_base : i32 = page_no * 98304 + head * 1536 + slot * 96 + 49152
                       + podArgs.pages_elem_offset;
    kv_pages[v_base + dim_lo]      = f16(v_lo);
    kv_pages[v_base + dim_lo + 48] = f16(v_hi);
    return;
  }

  // Q or K: apply RoPE. dim_lo ∈ [0, 48) is the "low" index; row_hi is "high".
  // Matches rope.wgsl:
  //   dim < 48  : out = cos*x + sin*(-x_pair_hi)
  //   dim >= 48 : out = cos*x + sin*( x_pair_lo)
  let pos  : f32 = f32(position_map[podArgs.position_map_elem_offset]);
  let freq : f32 = pos / pow(10000.0, f32(dim_lo * 2) / 96.0);
  let c    : f32 = cos(freq);
  let s    : f32 = sin(freq);

  let rot_lo : f32 = c * v_lo + s * (-v_hi);
  let rot_hi : f32 = c * v_hi + s * ( v_lo);

  if (group == 0) {
    let base = head * 96 + dim_lo;
    q_out[base     ] = f16(rot_lo);
    q_out[base + 48] = f16(rot_hi);
  } else {
    // K → paged KV cache.
    let position : i32 = position_map[podArgs.position_map_elem_offset];
    let page_no : i32 = position / 16;
    let slot : i32 = position - page_no * 16;
    let k_base : i32 = page_no * 98304 + head * 1536 + slot * 96
                       + podArgs.pages_elem_offset;
    kv_pages[k_base + dim_lo]      = f16(rot_lo);
    kv_pages[k_base + dim_lo + 48] = f16(rot_hi);
  }
}
