// ATTN_GATE — Qwen3.5 gated attention output: attn_out *= sigmoid(gate),
// IN PLACE, before o_proj.
//
// Ground truth: HF modeling_qwen3_5.Qwen3_5Attention.forward:
//   attn_output = attn_output * torch.sigmoid(gate)   (then o_proj)
// MLC qwen35_model.Qwen35Attention: output * op.sigmoid(gate) — identical.
// The gate is the raw per-head half of the c_attn Q group, extracted by
// gated_qkv_split.wgsl (no norm, no RoPE). Sigmoid in f32.
//
// Grid: seq_len * Q_DIM / 256 workgroups (podArgs.packGridDimX).
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> attn_out : array<f16>;  // seq * Q_DIM
@group(0) @binding(1) var<storage, read> gate : array<f16>;            // seq * Q_DIM

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn attn_gate(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg) >= podArgs.packGridDimX) { return; }

  let flat : i32 = wg * 256 + i32(threadIdx.x);
  let g : f32 = f32(gate[flat]);
  attn_out[flat] = f16(f32(attn_out[flat]) * (1.0 / (1.0 + exp(-g))));
}
