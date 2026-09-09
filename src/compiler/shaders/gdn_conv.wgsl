// GDN_CONV — Qwen3.5 GatedDeltaNet depthwise causal Conv1d (width GDN_CONV_K)
// + SiLU. DECODE MODE: one token per dispatch, ring-buffer state holding the
// last GDN_CONV_K-1 RAW (pre-activation) qkv projections. Prefill v1 loops
// this kernel over tokens sequentially (documented cost: one dispatch/token).
//
// Ground truth:
//   HF modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward → causal_conv1d_update:
//     out = F.conv1d(cat([conv_state, x], -1), weight, groups=conv_dim)[..-1:]
//     then ACT2FN["silu"]; conv_state stores raw in_proj_qkv outputs.
//   MLC qwen35_model.Qwen35GatedDeltaNet._causal_conv1d_with_state + op.silu:
//     identical — conv over raw qkv, SiLU strictly after the convolution.
//
//   conv_out[c] = silu( Σ_{j=0}^{K-1} w[c,j] · x[t-(K-1)+j][c] ),  x[<0] = 0
//
// Ring buffer (RING = GDN_CONV_K-1 slots of GDN_QKV_DIM channels): slot
// s holds x[t'] with t' ≡ s (mod RING). At token t, window element j < RING
// is x[t-RING+j] in slot (t+j) % RING; after accumulating, x[t] overwrites
// slot t % RING (the expired x[t-RING]). Each thread owns one channel —
// read-then-write within the thread, no hazard. State must be zeroed before
// token 0 (engine reset = clear buffer).
//
// conv1d_weight is the MLC record [GDN_QKV_DIM, 1, GDN_CONV_K] f16 flattened
// channel-major: w[c,j] = conv_w[c * GDN_CONV_K + j]. Accumulation in f32.
//
// Grid: ceil(GDN_QKV_DIM / 256) workgroups (podArgs.packGridDimX).
// Model-shape constants are injected by src/compiler/shader-prelude.ts.

enable f16;

@group(0) @binding(0) var<storage, read_write> conv_out : array<f16>;   // GDN_QKV_DIM
@group(0) @binding(1) var<storage, read> qkv_raw : array<f16>;          // GDN_QKV_DIM (this token)
@group(0) @binding(2) var<storage, read_write> conv_state : array<f16>; // RING * GDN_QKV_DIM
@group(0) @binding(3) var<storage, read> conv_w : array<f16>;           // GDN_QKV_DIM * GDN_CONV_K

struct PODArgs {
  pos: i32,           // absolute token index (selects the ring slots)
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const RING = GDN_CONV_K - 1;   // history slots (kernel width minus current token)

@compute @workgroup_size(256, 1, 1)
fn gdn_conv(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let wg : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (u32(wg) >= podArgs.packGridDimX) { return; }

  let c : i32 = wg * 256 + i32(threadIdx.x);   // channel
  if (c >= GDN_QKV_DIM) { return; }

  let pos : i32 = podArgs.pos;

  // History taps: window j < RING is x[pos-RING+j], in ring slot (pos+j) % RING.
  var acc : f32 = 0.0;
  for (var j : i32 = 0; j < RING; j = j + 1) {
    let slot : i32 = (pos + j) % RING;
    acc = acc + f32(conv_w[c * GDN_CONV_K + j]) * f32(conv_state[slot * GDN_QKV_DIM + c]);
  }
  // Current token is the last tap.
  let x : f16 = qkv_raw[c];
  acc = acc + f32(conv_w[c * GDN_CONV_K + RING]) * f32(x);

  // Rotate the ring: x[pos] replaces the expired x[pos-RING].
  conv_state[(pos % RING) * GDN_QKV_DIM + c] = x;

  // silu(x) = x * sigmoid(x), in f32.
  conv_out[c] = f16(acc / (1.0 + exp(-acc)));
}
