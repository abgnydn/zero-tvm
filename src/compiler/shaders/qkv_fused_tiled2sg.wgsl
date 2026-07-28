// QKV_FUSED_TILED_2SG — 2-subgroup, 2-pair tile variant.
//
// Motivation: previous qkv_fused_tiled_sg.wgsl tried 2 pairs / 32 threads /
// QKV_DIM/4 WGs; threads-in-flight dropped 4× vs the `_sg` baseline (9216 →
// 2304 active warps) and we regressed 20%+. This variant keeps 64 threads per
// WG (2 subgroups) so threads-in-flight only halve (9216 → 4608).
//
// Layout:
//   - 64 threads per WG, 2 subgroups of 32
//   - 2 pairs per WG = 4 output rows
//   - Subgroup 0 computes pair 0 (rows rl0, rh0)
//   - Subgroup 1 computes pair 1 (rows rl1, rh1)
//   - Input cached into workgroup shared memory (D f16 = 6 KB, fits)
//
// Result: it regressed too (BENCH.md) — hand-tuned QKV tiling is
// definitively not a lever on Apple. (Written in the M2 Pro era, when
// Zero-TVM still trailed WebLLM by 22%; the lever that did work on this
// kernel was load width — qkv_fused_sg_vec4, now the default.)
//
// Bind-group layout matches qkv_fused_sg.wgsl so chat.ts can swap pipelines.
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

@compute @workgroup_size(64, 1, 1)
fn qkv_fused_tiled2sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lane : u32,
  @builtin(subgroup_size) sg_size : u32,
) {
  let wg_idx : u32 = blockIdx.z * gridDim.x + blockIdx.x;
  let pair_base : u32 = wg_idx * PAIRS_PER_WG;
  if (pair_base >= podArgs.packGridDimX) { return; }

  let tid : u32 = threadIdx.x;
  let sg_id : u32 = tid / sg_size;  // 0 or 1 (assumes sg_size == 32)

  // Cooperative load of hidden[0..D] into shared mem.
  // 64 threads × D/64 elements = D.
  for (var i : u32 = 0u; i < D / 64u; i = i + 1u) {
    let idx : u32 = tid + i * 64u;
    shared_input[idx] = hidden[idx];
  }
  workgroupBarrier();

  // Both pairs share group and head (PAIRS_PER_WG=2 divides HALF_HEAD_DIM
  // pairs/head). Unequal [Q | K | V] groups under GQA (see qkv_fused.wgsl).
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
  // Subgroup 0 owns pair 0 (rows row_base+0, row_base+HALF_HEAD_DIM)
  // Subgroup 1 owns pair 1 (rows row_base+1, row_base+HALF_HEAD_DIM+1)
  let row_offset : u32 = sg_id;  // 0 or 1
  let my_rl : u32 = row_base + row_offset;
  let my_rh : u32 = my_rl + HALF_HEAD_DIM;

  var acc_lo : f32 = 0.0;
  var acc_hi : f32 = 0.0;

  // Each subgroup covers all D_PACKED / 32 chunks with its 32 lanes.
  // Per-group scale factored out of the 8 unpacked terms.
  for (var chunk : u32 = 0u; chunk < D_PACKED / 32u; chunk = chunk + 1u) {
    let w_offset : u32 = sg_lane + chunk * 32u;
    let base : u32 = w_offset * 8u;
    let sc_idx : u32 = w_offset >> 2u;

    let i0 = f32(shared_input[base     ]);
    let i1 = f32(shared_input[base + 1u]);
    let i2 = f32(shared_input[base + 2u]);
    let i3 = f32(shared_input[base + 3u]);
    let i4 = f32(shared_input[base + 4u]);
    let i5 = f32(shared_input[base + 5u]);
    let i6 = f32(shared_input[base + 6u]);
    let i7 = f32(shared_input[base + 7u]);

    {
      let p = weights[my_rl * D_PACKED + w_offset];
      let s = f32(scales[my_rl * D_SCALES + sc_idx]);
      acc_lo = acc_lo + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
    {
      let p = weights[my_rh * D_PACKED + w_offset];
      let s = f32(scales[my_rh * D_SCALES + sc_idx]);
      acc_hi = acc_hi + s * (
          i0 * (f32((p >>  0u) & 15u) - 7.0)
        + i1 * (f32((p >>  4u) & 15u) - 7.0)
        + i2 * (f32((p >>  8u) & 15u) - 7.0)
        + i3 * (f32((p >> 12u) & 15u) - 7.0)
        + i4 * (f32((p >> 16u) & 15u) - 7.0)
        + i5 * (f32((p >> 20u) & 15u) - 7.0)
        + i6 * (f32((p >> 24u) & 15u) - 7.0)
        + i7 * (f32((p >> 28u) & 15u) - 7.0));
    }
  }

  let v_lo : f32 = subgroupAdd(acc_lo);
  let v_hi : f32 = subgroupAdd(acc_hi);

  // Lane 0 of each subgroup writes its pair's output.
  if (sg_lane != 0u) { return; }

  let dim_lo : u32 = dim_lo_0 + row_offset;

  let position : i32 = position_map[podArgs.position_map_elem_offset];
  let page_no : i32 = position / PAGE_SIZE;
  let slot    : i32 = position - page_no * PAGE_SIZE;

  if (group_id == 2u) {
    let v_base : i32 = page_no * KV_PAGE_STRIDE + i32(head) * HEAD_PAGE_STRIDE
                       + slot * HEAD_DIM + V_PAGE_OFFSET + podArgs.pages_elem_offset;
    kv_pages[v_base + i32(dim_lo)                ] = f16(v_lo);
    kv_pages[v_base + i32(dim_lo) + HALF_HEAD_DIM] = f16(v_hi);
    return;
  }

  let pos_f : f32 = f32(position);
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
    let k_base : i32 = page_no * KV_PAGE_STRIDE + i32(head) * HEAD_PAGE_STRIDE
                       + slot * HEAD_DIM + podArgs.pages_elem_offset;
    kv_pages[k_base + i32(dim_lo)                ] = f16(rot_lo);
    kv_pages[k_base + i32(dim_lo) + HALF_HEAD_DIM] = f16(rot_hi);
  }
}
