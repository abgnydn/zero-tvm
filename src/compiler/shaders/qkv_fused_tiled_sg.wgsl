// QKV_FUSED_TILED_SG — tiled + subgroup variant of qkv_fused_sg.wgsl.
//
// NEGATIVE RESULT on Apple M-series: both 4-pair (57% slower) and 2-pair
// (78% slower on the qkv kernel) versions regressed vs qkv_fused_sg. See
// BENCH.md "Negative result: tiled qkv_fused" for numbers. Kept compiled
// behind `?qkvtile=1` for re-testing on other GPUs; default stays `_sg`.
//
// The shader below is the 2-pair variant, kept as the cleanest reference
// implementation of the tiled pattern (4 output rows, 4 accumulators/thread,
// 2304 WGs, 1 subgroup of 32 threads, shared-mem input cache for the 3072
// f16 hidden vector). Bind-group layout matches qkv_fused_sg.wgsl so chat.ts
// can swap the pipeline reference.

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

const K_PACKED       : u32 = 384u;
const SCALES_PER_ROW : u32 = 96u;
const PAIRS_PER_WG   : u32 = 2u;

var<workgroup> shared_input : array<f16, 3072>;

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

  // Snapshot-load input → shared mem. 32 threads × 96 elements = 3072.
  for (var i : u32 = 0u; i < 96u; i = i + 1u) {
    let idx : u32 = tid + i * 32u;
    shared_input[idx] = hidden[idx];
  }
  workgroupBarrier();

  // All 4 pairs in this WG share group and head (PAIRS_PER_WG=4 divides
  // 48 = pairs per head, so consecutive pair_idx stay in one head).
  let group_id : u32 = pair_base / 1536u;
  let pair_in_group : u32 = pair_base - group_id * 1536u;
  let head : u32 = pair_in_group / 48u;
  let dim_lo_0 : u32 = pair_in_group - head * 48u;

  let row_base : u32 = group_id * 3072u + head * 96u + dim_lo_0;
  // 4 rows: 2 "lo" + 2 "hi" (hi = lo + 48).
  let rl0 = row_base;  let rl1 = row_base + 1u;
  let rh0 = rl0 + 48u; let rh1 = rl1 + 48u;

  var acc_l0 : f32 = 0.0; var acc_l1 : f32 = 0.0;
  var acc_h0 : f32 = 0.0; var acc_h1 : f32 = 0.0;

  // K_PACKED / 32 = 12 chunks.
  for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
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

    { let p = weights[rl0 * K_PACKED + w_offset]; let s = f32(scales[rl0 * SCALES_PER_ROW + sc_idx]);
      acc_l0 = acc_l0
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[rl1 * K_PACKED + w_offset]; let s = f32(scales[rl1 * SCALES_PER_ROW + sc_idx]);
      acc_l1 = acc_l1
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[rh0 * K_PACKED + w_offset]; let s = f32(scales[rh0 * SCALES_PER_ROW + sc_idx]);
      acc_h0 = acc_h0
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
    { let p = weights[rh1 * K_PACKED + w_offset]; let s = f32(scales[rh1 * SCALES_PER_ROW + sc_idx]);
      acc_h1 = acc_h1
        + i0 * (f32((p >>  0u) & 15u) - 7.0) * s + i1 * (f32((p >>  4u) & 15u) - 7.0) * s
        + i2 * (f32((p >>  8u) & 15u) - 7.0) * s + i3 * (f32((p >> 12u) & 15u) - 7.0) * s
        + i4 * (f32((p >> 16u) & 15u) - 7.0) * s + i5 * (f32((p >> 20u) & 15u) - 7.0) * s
        + i6 * (f32((p >> 24u) & 15u) - 7.0) * s + i7 * (f32((p >> 28u) & 15u) - 7.0) * s; }
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
  let page_no : i32 = position / 16;
  let slot    : i32 = position - page_no * 16;

  for (var k : u32 = 0u; k < PAIRS_PER_WG; k = k + 1u) {
    let dim_lo : u32 = dim_lo_0 + k;
    let v_lo : f32 = vl[k];
    let v_hi : f32 = vh[k];

    if (group_id == 2u) {
      // V: no RoPE; store raw.
      let v_base : i32 = page_no * 98304 + i32(head) * 1536 + slot * 96 + 49152
                         + podArgs.pages_elem_offset;
      kv_pages[v_base + i32(dim_lo)     ] = f16(v_lo);
      kv_pages[v_base + i32(dim_lo) + 48] = f16(v_hi);
      continue;
    }

    let freq : f32 = pos_f / pow(10000.0, f32(dim_lo * 2u) / 96.0);
    let c    : f32 = cos(freq);
    let s    : f32 = sin(freq);
    let rot_lo : f32 = c * v_lo + s * (-v_hi);
    let rot_hi : f32 = c * v_hi + s * ( v_lo);

    if (group_id == 0u) {
      let b = i32(head) * 96 + i32(dim_lo);
      q_out[b     ] = f16(rot_lo);
      q_out[b + 48] = f16(rot_hi);
    } else {
      // K
      let k_base : i32 = page_no * 98304 + i32(head) * 1536 + slot * 96
                         + podArgs.pages_elem_offset;
      kv_pages[k_base + i32(dim_lo)     ] = f16(rot_lo);
      kv_pages[k_base + i32(dim_lo) + 48] = f16(rot_hi);
    }
  }
}
