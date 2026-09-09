// QKV_FUSED_SG_VEC4 — vec4-load variant of qkv_fused_sg.wgsl (?vec4qkv=1).
//
// Same fusion (QKV int4 matmul + RoPE + KV append) and same bind-group
// layout; two differences:
//
//   • Loads: weights and hidden are re-declared as array<vec4<u32>> (the
//     bound GPUBuffers are unchanged). Each thread moves 16 bytes of weights
//     (32 nibbles — exactly one int4 scale group, so the scale index IS the
//     vec4 index) and 4×16 bytes of activations per iteration, unpacked via
//     unpack2x16float.
//   • 32 threads = ONE subgroup (vs _sg's 64/2): D_PACKED/4 = 96 vec4 words
//     per row do not divide across 64 threads (96/64), so the vec4 layout
//     forces the narrower workgroup. The two-subgroup finalize is replaced
//     by a plain subgroupAdd + lane-0 tail. This entangles the load-width
//     experiment with a thread-count change — bisect against ?sgqkv=0 and
//     the qkv-tile negative results when measuring.
//
// Requires sg_size = 32 (gated in chat.ts).
//
// Measured 2026-07-25 (M2 Max): +4.2% alone, +7.1% combined with ?vec4=1
// — promoted to default (opt out with ?vec4qkv=0; see BENCH.md
// "Measured 2026-07-25").
//
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> q_out        : array<f16>;
@group(0) @binding(1) var<storage, read_write> kv_pages     : array<f16>;
@group(0) @binding(2) var<storage, read>       hidden       : array<vec4<u32>>;  // f16 data, 8 per element
@group(0) @binding(3) var<storage, read>       scales       : array<f16>;
@group(0) @binding(4) var<storage, read>       weights      : array<vec4<u32>>;  // 4 packed words per element
@group(0) @binding(5) var<storage, read>       position_map : array<i32>;

struct PODArgs {
  position_map_elem_offset : i32,
  pages_elem_offset        : i32,
  packGridDimX             : u32,
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

const D_PACKED_V4 = D_PACKED / 4;   // vec4<u32> words per K=D weight row
const KV_GROUP_PAIRS = KV_DIM / 2;  // KV_HEADS * HALF_HEAD_DIM

// Σ over one packed word: x0..x7 are the 8 activations, p the packed u32.
fn dq8(p : u32, x01 : vec2<f32>, x23 : vec2<f32>, x45 : vec2<f32>, x67 : vec2<f32>) -> f32 {
  return x01.x * (f32((p >>  0u) & 15u) - 7.0)
       + x01.y * (f32((p >>  4u) & 15u) - 7.0)
       + x23.x * (f32((p >>  8u) & 15u) - 7.0)
       + x23.y * (f32((p >> 12u) & 15u) - 7.0)
       + x45.x * (f32((p >> 16u) & 15u) - 7.0)
       + x45.y * (f32((p >> 20u) & 15u) - 7.0)
       + x67.x * (f32((p >> 24u) & 15u) - 7.0)
       + x67.y * (f32((p >> 28u) & 15u) - 7.0);
}

@compute @workgroup_size(32, 1, 1)
fn qkv_fused_sg_vec4(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
) {
  let pair_idx : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(pair_idx) >= podArgs.packGridDimX) { return; }

  let tid : i32 = i32(threadIdx.x);

  // Unequal [Q | K | V] groups under GQA (see qkv_fused.wgsl).
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

  let row_lo : i32 = row_base + head * HEAD_DIM + dim_lo;
  let row_hi : i32 = row_lo + HALF_HEAD_DIM;

  var acc0 : f32 = 0.0;
  var acc1 : f32 = 0.0;

  // D_PACKED_V4 / 32 iterations; the vec4 index doubles as the scale index
  // (4 packed words = 32 nibbles = one int4 scale group).
  for (var chunk : i32 = 0; chunk < D_PACKED_V4 / 32; chunk = chunk + 1) {
    let v_off : i32 = tid + chunk * 32;

    let w_lo : vec4<u32> = weights[row_lo * D_PACKED_V4 + v_off];
    let w_hi : vec4<u32> = weights[row_hi * D_PACKED_V4 + v_off];
    let scale_lo : f32 = f32(scales[row_lo * D_SCALES + v_off]);
    let scale_hi : f32 = f32(scales[row_hi * D_SCALES + v_off]);

    // 32 activations as 4 × vec4<u32>, unpacked to 16 vec2<f32>.
    let xa = hidden[v_off * 4];
    let xb = hidden[v_off * 4 + 1];
    let xc = hidden[v_off * 4 + 2];
    let xd = hidden[v_off * 4 + 3];
    let a0 = unpack2x16float(xa.x); let a1 = unpack2x16float(xa.y);
    let a2 = unpack2x16float(xa.z); let a3 = unpack2x16float(xa.w);
    let b0 = unpack2x16float(xb.x); let b1 = unpack2x16float(xb.y);
    let b2 = unpack2x16float(xb.z); let b3 = unpack2x16float(xb.w);
    let c0 = unpack2x16float(xc.x); let c1 = unpack2x16float(xc.y);
    let c2 = unpack2x16float(xc.z); let c3 = unpack2x16float(xc.w);
    let d0 = unpack2x16float(xd.x); let d1 = unpack2x16float(xd.y);
    let d2 = unpack2x16float(xd.z); let d3 = unpack2x16float(xd.w);

    acc0 = acc0 + scale_lo * (
        dq8(w_lo.x, a0, a1, a2, a3)
      + dq8(w_lo.y, b0, b1, b2, b3)
      + dq8(w_lo.z, c0, c1, c2, c3)
      + dq8(w_lo.w, d0, d1, d2, d3));

    acc1 = acc1 + scale_hi * (
        dq8(w_hi.x, a0, a1, a2, a3)
      + dq8(w_hi.y, b0, b1, b2, b3)
      + dq8(w_hi.z, c0, c1, c2, c3)
      + dq8(w_hi.w, d0, d1, d2, d3));
  }

  // One subgroup — a single subgroupAdd per accumulator, no barrier needed.
  let v_lo : f32 = subgroupAdd(acc0);
  let v_hi : f32 = subgroupAdd(acc1);

  if (tid != 0) { return; }

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
  let freq : f32 = pos / pow(ROPE_THETA, f32(dim_lo * 2) / f32(HEAD_DIM));
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
