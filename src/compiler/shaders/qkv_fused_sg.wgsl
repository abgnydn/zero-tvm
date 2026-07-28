// QKV_FUSED_SG — subgroup-reduction variant of qkv_fused.wgsl.
//
// Same fusion (QKV matmul + RoPE + KV append in one dispatch) and same bind
// group layout. The only difference is the tail reduction: the scalar variant
// does a 64→1 tree reduction with 5 workgroupBarriers; this variant does one
// subgroupAdd per subgroup (two subgroups of 32 lanes on Apple Metal-3), then a
// single workgroupBarrier + lane-0 finalisation.
//
// Requires sg_size = 32 (gated in chat.ts).
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

// Two subgroups per workgroup (64 threads / 32 lanes). Store per-subgroup
// partial sums so lane 0 can finalise.
var<workgroup> sg_sum0 : array<f32, 2>;
var<workgroup> sg_sum1 : array<f32, 2>;

@compute @workgroup_size(64, 1, 1)
fn qkv_fused_sg(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lane : u32,
  @builtin(subgroup_size) sg_size : u32,
) {
  let pair_idx : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(pair_idx) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);

  let group         : i32 = pair_idx / QKV_GROUP_PAIRS;
  let pair_in_group : i32 = pair_idx - group * QKV_GROUP_PAIRS;
  let head          : i32 = pair_in_group / HALF_HEAD_DIM;
  let dim_lo        : i32 = pair_in_group - head * HALF_HEAD_DIM;

  let row_lo : i32 = group * D + head * HEAD_DIM + dim_lo;
  let row_hi : i32 = row_lo + HALF_HEAD_DIM;

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

  // One subgroup-wide add per accumulator (all 32 lanes hold the same sum).
  let partial0 : f32 = subgroupAdd(acc0);
  let partial1 : f32 = subgroupAdd(acc1);

  let sg_id : u32 = u32(tid) / sg_size;  // 0 or 1 with sg_size=32, 64 threads
  if (sg_lane == 0u) {
    sg_sum0[sg_id] = partial0;
    sg_sum1[sg_id] = partial1;
  }
  workgroupBarrier();

  if (tid != 0) { return; }

  let v_lo : f32 = sg_sum0[0] + sg_sum0[1];
  let v_hi : f32 = sg_sum1[0] + sg_sum1[1];

  if (group == 2) {
    let position : i32 = position_map[podArgs.position_map_elem_offset];
    let page_no : i32 = position / PAGE_SIZE;
    let slot : i32 = position - page_no * PAGE_SIZE;
    let v_base : i32 = page_no * KV_PAGE_STRIDE + head * HEAD_PAGE_STRIDE + slot * HEAD_DIM
                       + V_PAGE_OFFSET + podArgs.pages_elem_offset;
    kv_pages[v_base + dim_lo]                 = f16(v_lo);
    kv_pages[v_base + dim_lo + HALF_HEAD_DIM] = f16(v_hi);
    return;
  }

  let pos  : f32 = f32(position_map[podArgs.position_map_elem_offset]);
  let freq : f32 = pos / pow(10000.0, f32(dim_lo * 2) / f32(HEAD_DIM));
  let c    : f32 = cos(freq);
  let s    : f32 = sin(freq);

  let rot_lo : f32 = c * v_lo + s * (-v_hi);
  let rot_hi : f32 = c * v_hi + s * ( v_lo);

  if (group == 0) {
    let base = head * HEAD_DIM + dim_lo;
    q_out[base                ] = f16(rot_lo);
    q_out[base + HALF_HEAD_DIM] = f16(rot_hi);
  } else {
    let position : i32 = position_map[podArgs.position_map_elem_offset];
    let page_no : i32 = position / PAGE_SIZE;
    let slot : i32 = position - page_no * PAGE_SIZE;
    let k_base : i32 = page_no * KV_PAGE_STRIDE + head * HEAD_PAGE_STRIDE + slot * HEAD_DIM
                       + podArgs.pages_elem_offset;
    kv_pages[k_base + dim_lo]                 = f16(rot_lo);
    kv_pages[k_base + dim_lo + HALF_HEAD_DIM] = f16(rot_hi);
  }
}
