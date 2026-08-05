// MOE_COMBINE — weighted sum of the slot outputs of a sparse MoE block.
//
//   out[i] = sum over slots of score[slot] * y[slot][i]
//
// The shared expert is just slot K with its own score (moe_router_topk writes
// sigmoid of its gate there), so there is no special case here — which is the
// whole point of stacking it into the expert tensors.
//
// Accumulates in f32: K+1 f16 terms of similar magnitude summed in f16 loses
// bits the block cannot spare, and the caller wants f32 for the residual add.
//
// Grid: ceil(N / 256) workgroups.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_buf : array<f32>;   // [N]
@group(0) @binding(1) var<storage, read> slots : array<f16>;           // [SLOTS, N]
@group(0) @binding(2) var<storage, read> score : array<f32>;           // [SLOTS]

struct PODArgs {
  N: u32,      // elements per slot (hidden size)
  SLOTS: u32   // routed top-K plus the shared expert
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn moe_combine(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i : u32 = gid.x;
  if (i >= podArgs.N) { return; }

  var acc : f32 = 0.0;
  for (var s : u32 = 0u; s < podArgs.SLOTS; s = s + 1u) {
    acc = acc + score[s] * f32(slots[s * podArgs.N + i]);
  }
  out_buf[i] = acc;
}
