// SILU_MUL — chunked-prefill FFN epilogue: out = SiLU(gate) * up.
//
// The decode path fuses gate+up+SiLU into fused_ffn.wgsl (one dispatch per
// token). Chunked prefill instead runs gate_up as ONE batched int4 matmul
// (int4_matmul_batched_dyn over the same [gate rows | up rows] weight buffer)
// producing [seq, 2*FFN_DIM], then this elementwise kernel applies
//   out[t, i] = SiLU(gu[t, i]) * gu[t, FFN_DIM + i]
// SiLU in f32, matching fused_ffn's phase-4 formula exactly (the only
// difference vs the decode path is the matmul reduction order, same as every
// other batched-vs-GEMV pairing).
//
// Grid: seq_len * FFN_DIM / 256 workgroups (podArgs.packGridDimX).
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_buf : array<f16>;  // seq * FFN_DIM
@group(0) @binding(1) var<storage, read> gate_up : array<f16>;        // seq * 2*FFN_DIM

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn silu_mul(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg) >= podArgs.packGridDimX) { return; }

  let flat : i32 = wg * 256 + i32(threadIdx.x);
  let t : i32 = flat / FFN_DIM;
  let i : i32 = flat % FFN_DIM;
  if (t >= podArgs.seq_len) { return; }

  let gate_val : f32 = f32(gate_up[t * 2 * FFN_DIM + i]);
  let up_val : f32 = f32(gate_up[t * 2 * FFN_DIM + FFN_DIM + i]);
  let silu_gate : f32 = gate_val * (1.0 / (1.0 + exp(-gate_val)));
  out_buf[t * FFN_DIM + i] = f16(up_val * silu_gate);
}
