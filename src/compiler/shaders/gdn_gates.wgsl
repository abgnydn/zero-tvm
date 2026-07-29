// GDN_GATES — Qwen3.5 GatedDeltaNet per-v-head decay + update-rate gates.
//
// Ground truth:
//   HF modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward:
//     beta = b.sigmoid()
//     g    = -exp(A_log.float()) * softplus(a.float() + dt_bias)
//   and the recurrence consumes exp(g) (torch_recurrent_gated_delta_rule:
//   g_t = g[:, :, i].exp()). MLC qwen35_model._compute_gate_beta emits the
//   exponentiated form directly:
//     gate = exp(-exp(A_log) * softplus(alpha + dt_bias))
//   with softplus(x) = x if x > 20 else log(1+exp(x))  (MLC TIR _softplus).
//
// This kernel matches MLC: output[h]              = exp(g)   (decay factor)
//                          output[GDN_V_HEADS+h]  = sigmoid(b) (beta)
// ab_raw is the a-then-b tail of the FUSED GDN input projection (in_proj_qkv
// ‖ z ‖ a ‖ b as one int4 matmul): a at ab_raw[h], b at ab_raw[GDN_V_HEADS+h].
// The engine binds the packed projection buffer at the a-region offset
// ((GDN_QKV_DIM + GDN_V_DIM)·2 bytes — 256-aligned), so this kernel sees the
// [a | b] pair as one 2·GDN_V_HEADS array. A_log and dt_bias are the raw f32
// records (must stay f32 — HF comment: in fp16, -exp(A_log) can overflow to
// -inf). All math here is f32.
//
// Grid: 1 workgroup (podArgs.packGridDimX = 1), 32 threads looping v-heads.
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> gates : array<f32>;  // [exp(g) | beta], 2*GDN_V_HEADS
@group(0) @binding(1) var<storage, read> ab_raw : array<f16>;       // [a | b], 2*GDN_V_HEADS
@group(0) @binding(2) var<storage, read> a_log : array<f32>;        // GDN_V_HEADS
@group(0) @binding(3) var<storage, read> dt_bias : array<f32>;      // GDN_V_HEADS

struct PODArgs { packGridDimX: u32 }
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(32, 1, 1)
fn gdn_gates(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg) >= podArgs.packGridDimX) { return; }

  for (var h : i32 = i32(threadIdx.x); h < GDN_V_HEADS; h = h + 32) {
    let x : f32 = f32(ab_raw[h]) + dt_bias[h];
    let sp : f32 = select(log(1.0 + exp(x)), x, x > 20.0);  // softplus, MLC threshold
    gates[h] = exp(-exp(a_log[h]) * sp);
    gates[GDN_V_HEADS + h] = 1.0 / (1.0 + exp(-f32(ab_raw[GDN_V_HEADS + h])));
  }
}
