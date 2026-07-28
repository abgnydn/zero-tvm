// QKV_FUSED_TILED_SG — tiled + subgroup variant of qkv_fused_sg.wgsl.
//
// NEGATIVE RESULT on Apple M-series: both 4-pair (57% slower) and 2-pair
// (78% slower on the qkv kernel) versions regressed vs qkv_fused_sg. See
// BENCH.md "Negative result: tiled qkv_fused" for numbers. Kept compiled
// behind `?qkvtile=1` for re-testing on other GPUs; default stays `_sg`.
//
// The shader below is the 2-pair variant, kept as the cleanest reference
// implementation of the tiled pattern (4 output rows, 4 accumulators/thread,
// QKV_DIM/4 WGs, 1 subgroup of 32 threads, shared-mem input cache for the
// D-wide f16 hidden vector). Bind-group layout matches qkv_fused_sg.wgsl so
// chat.ts can swap the pipeline reference.
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> q_out        : array<f16>;
@group(0) @binding(1) var<storage, read_write> kv_pages     : array<f16>;
@group(0) @binding(2) var<storage, read>       hidden       : array<f16>;
@group(0) @binding(3) var<storage, read>       scales       : array<f16>;
@group(0) @binding(4) var<storage, read>       weights      : array<u32>;
@group(0) @binding(5) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,
  packGridDimX             : u32,
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const PAIRS_PER_WG : u32 = 2u;
const KV_GROUP_PAIRS = KV_DIM / 2;   // KV_HEADS * HALF_HEAD_DIM

var<workgroup> shared_input : array<f16, D>;

@compute @workgroup_size(32, 1, 1)
fn qkv_fused_tiled_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let wg_idx : u32 = blockIdx.z * gridDim.x + blockIdx.x;
  let pair_base : u32 = wg_idx * PAIRS_PER_WG;
  if (pair_base >= podArgs.packGridDimX) { return; }

  let tid : u32 = threadIdx.x;

  // Snapshot-load input → shared mem. 32 threads × D/32 elements = D.
  for (var i : u32 = 0u; i < D / 32u; i = i + 1u) {
    let idx : u32 = tid + i * 32u;
    shared_input[idx] = hidden[idx];
  }
  workgroupBarrier();

  // Both pairs in this WG share group and head (PAIRS_PER_WG=2 divides
  // HALF_HEAD_DIM = pairs per head, so consecutive pair_idx stay in one head).
  // Unequal [Q | K | V] groups under GQA (see qkv_fused.wgsl).
  var group_id : u32 = 0u;
  var pair_in_group : u32 = pair_base;
  var grp_base : u32 = 0u;
  if (pair_in_group >= u32(QKV_GROUP_PAIRS)) {
    group_id = 1u;
    pair_in_group = pair_in_group - u32(QKV_GROUP_PAIRS);
    grp_base = u32(Q_DIM);
    if (pair_in_group >= u32(KV_GROUP_PAIRS)) {
      group_id = 2u;
      pair_in_group = pair_in_group - u32(KV_GROUP_PAIRS);
      grp_base = u32(Q_DIM + KV_DIM);
    }
  }
  let head : u32 = pair_in_group / HALF_HEAD_DIM;
  let dim_lo_0 : u32 = pair_in_group - head * HALF_HEAD_DIM;

  let row_base : u32 = grp_base + head * HEAD_DIM + dim_lo_0;
  // 4 rows: 2 "lo" + 2 "hi" (hi = lo + HALF_HEAD_DIM).
  let rl0 = row_base;            let rl1 = row_base + 1u;
  let rh0 = rl0 + HALF_HEAD_DIM; let rh1 = rl1 + HALF_HEAD_DIM;

  var acc_l0 : f32 = 0.0; var acc_l1 : f32 = 0.0;
  var acc_h0 : f32 = 0.0; var acc_h1 : f32 = 0.0;

  // D_PACKED / 32 chunks. Per-group scale factored out of the 8 terms.
  for (var chunk : u32 = 0u; chunk < D_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * 32u;
    let base     : u32 = w_offset * 8u;
    let sc_idx   : u32 = w_offset >> 2u;

    let i0 = f32(shared_input[base     ]);
    let i1 = f32(shared_input[base + 1u]);
    let i2 = f32(shared_input[base + 2u]);
    let i3 = f32(shared_input[base + 3u]);
    let i4 = f32(shared_input[base + 4u]);
    let i5 = f32(shared_input[base + 5u]);
    let i6 = f32(shared_input[base + 6u]);
    let i7 = f32(shared_input[base + 7u]);

    { let p = weights[rl0 * D_PACKED + w_offset]; let s = f32(scales[rl0 * D_SCALES + sc_idx]);
      acc_l0 = acc_l0 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0) + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0) + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0) + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0) + i7 * (f32((p >> 28u) & 15u) - 7.0)); }
    { let p = weights[rl1 * D_PACKED + w_offset]; let s = f32(scales[rl1 * D_SCALES + sc_idx]);
      acc_l1 = acc_l1 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0) + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0) + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0) + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0) + i7 * (f32((p >> 28u) & 15u) - 7.0)); }
    { let p = weights[rh0 * D_PACKED + w_offset]; let s = f32(scales[rh0 * D_SCALES + sc_idx]);
      acc_h0 = acc_h0 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0) + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0) + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0) + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0) + i7 * (f32((p >> 28u) & 15u) - 7.0)); }
    { let p = weights[rh1 * D_PACKED + w_offset]; let s = f32(scales[rh1 * D_SCALES + sc_idx]);
      acc_h1 = acc_h1 + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0) + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0) + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0) + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0) + i7 * (f32((p >> 28u) & 15u) - 7.0)); }
  }

  let vl0 = subgroupAdd(acc_l0); let vl1 = subgroupAdd(acc_l1);
  let vh0 = subgroupAdd(acc_h0); let vh1 = subgroupAdd(acc_h1);

  if (tid != 0u) { return; }

  // Per-pair write-back (2 pairs, each with its own RoPE / KV-append path).
  // Both pairs share group_id / head; only dim_lo differs.
  let vl = array<f32, 2>(vl0, vl1);
  let vh = array<f32, 2>(vh0, vh1);

  let position : i32 = position_map[podArgs.position_map_elem_offset];
  let pos_f : f32 = f32(position);
  let page_no : i32 = position / PAGE_SIZE;
  let slot    : i32 = position - page_no * PAGE_SIZE;

  for (var k : u32 = 0u; k < PAIRS_PER_WG; k = k + 1u) {
    let dim_lo : u32 = dim_lo_0 + k;
    let v_lo : f32 = vl[k];
    let v_hi : f32 = vh[k];

    if (group_id == 2u) {
      // V: no RoPE; store raw.
      let v_base : i32 = page_no * KV_PAGE_STRIDE + i32(head) * HEAD_PAGE_STRIDE
                         + slot * HEAD_DIM + V_PAGE_OFFSET + podArgs.pages_elem_offset;
      kv_pages[v_base + i32(dim_lo)                ] = f16(v_lo);
      kv_pages[v_base + i32(dim_lo) + HALF_HEAD_DIM] = f16(v_hi);
      continue;
    }

    let freq : f32 = pos_f / pow(ROPE_THETA, f32(dim_lo * 2u) / f32(HEAD_DIM));
    let c    : f32 = cos(freq);
    let s    : f32 = sin(freq);
    let rot_lo : f32 = c * v_lo + s * (-v_hi);
    let rot_hi : f32 = c * v_hi + s * ( v_lo);

    if (group_id == 0u) {
      let b = i32(head) * HEAD_DIM + i32(dim_lo);
      q_out[b                ] = f16(rot_lo);
      q_out[b + HALF_HEAD_DIM] = f16(rot_hi);
    } else {
      // K
      let k_base : i32 = page_no * KV_PAGE_STRIDE + i32(head) * HEAD_PAGE_STRIDE
                         + slot * HEAD_DIM + podArgs.pages_elem_offset;
      kv_pages[k_base + i32(dim_lo)                ] = f16(rot_lo);
      kv_pages[k_base + i32(dim_lo) + HALF_HEAD_DIM] = f16(rot_hi);
    }
  }
}
