// GATED_QKV_SPLIT — Qwen3.5 gated attention: unpack the fused c_attn output
// into the engine's [Q | K | V] qkv layout plus a per-head gate buffer, so the
// existing qk_norm → rope → kv_append → attention pipeline runs unchanged.
//
// c_attn row layout (mlc-llm qwen35_loader.py: c_attn = concat(q_proj, k_proj,
// v_proj); HF Qwen3_5Attention: q_proj outputs heads·headDim·2, viewed as
// (heads, 2·headDim) and chunked → per head [Q_dims | gate_dims] INTERLEAVED):
//   head h of the Q group: rows [h·2·HEAD_DIM, h·2·HEAD_DIM + HEAD_DIM) = Q_h,
//                          next HEAD_DIM rows = gate_h
//   then K at [2·Q_DIM, 2·Q_DIM + KV_DIM), V at [2·Q_DIM + KV_DIM, C_ATTN_DIM).
// The gate is raw (no norm, no RoPE); sigmoid is applied later by
// attn_gate.wgsl (HF: attn_output * sigmoid(gate) before o_proj).
//
// ATTN_GATE == 0 specs: C_ATTN_DIM == QKV_DIM and this collapses to a copy
// (compiles under every spec; only gated specs dispatch it).
//
// Grid: seq_len * C_ATTN_DIM / 256 workgroups (podArgs.packGridDimX).
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> qkv : array<f16>;   // seq * QKV_DIM ([Q|K|V])
@group(0) @binding(1) var<storage, read_write> gate : array<f16>;  // seq * Q_DIM (raw, pre-sigmoid)
@group(0) @binding(2) var<storage, read> c_attn : array<f16>;      // seq * C_ATTN_DIM

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn gated_qkv_split(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg) >= podArgs.packGridDimX) { return; }

  let flat : i32 = wg * 256 + i32(threadIdx.x);
  let seq_idx : i32 = flat / C_ATTN_DIM;
  let within : i32 = flat % C_ATTN_DIM;
  if (seq_idx >= podArgs.seq_len) { return; }

  let val : f16 = c_attn[flat];

  if (ATTN_GATE == 0) {
    // No gate rows — c_attn is already the [Q|K|V] layout.
    qkv[seq_idx * QKV_DIM + within] = val;
    return;
  }

  if (within < 2 * Q_DIM) {
    // Q group: per-head [Q | gate] interleave.
    let h : i32 = within / (2 * HEAD_DIM);
    let r : i32 = within % (2 * HEAD_DIM);
    if (r < HEAD_DIM) {
      qkv[seq_idx * QKV_DIM + h * HEAD_DIM + r] = val;
    } else {
      gate[seq_idx * Q_DIM + h * HEAD_DIM + (r - HEAD_DIM)] = val;
    }
  } else {
    // K and V blocks shift down by the Q_DIM gate rows.
    qkv[seq_idx * QKV_DIM + (within - Q_DIM)] = val;
  }
}
