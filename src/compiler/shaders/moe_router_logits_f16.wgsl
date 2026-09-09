// MOE_ROUTER_LOGITS_F16 — the router when the checkpoint leaves it in full
// precision.
//
// Quantizers routinely skip this one tensor. It is tiny (E x D: 64 x 2048 here,
// 256 KB) and it decides which experts run, so an error in it is not a small
// numerical error — it routes the token to the wrong specialists. DeepSeek-V2
// ships `mlp.gate.weight` with no `.scales` or `.biases` beside it; Qwen3.6
// quantizes it to 8 bits, Qwen3-30B leaves it at the model's 4.
//
// Its own file rather than a third entry point in moe_router_logits.wgsl: the
// bindings there carry `scales` and `biases`, and an entry point that never
// reads them gets an auto bind-group layout WITHOUT them, so the caller's
// six-buffer bind group would be rejected. Same reason the engine builds a
// four-buffer group for this path.
//
// Grid: one workgroup per row, matching the quantized siblings.

enable f16;

@group(0) @binding(0) var<storage, read_write> logits : array<f32>;    // [rows]
@group(0) @binding(1) var<storage, read> x : array<f16>;               // [D]
@group(0) @binding(2) var<storage, read> w : array<f16>;               // [rows, D]

struct PODArgs {
  D: u32,
  rows: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

var<workgroup> red : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn moe_router_logits_f16(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let e : u32 = blockIdx.x;
  if (e >= podArgs.rows) { return; }
  let t : u32 = blockIdx.y;          // token within the chunk (0 when decoding)
  let xBase : u32 = t * podArgs.D;
  let tid : u32 = threadIdx.x;

  // f32 accumulation over D terms — the same reason the quantized entries do
  // it. The output is a logit going into a softmax over experts.
  var acc : f32 = 0.0;
  for (var i : u32 = tid; i < podArgs.D; i = i + 64u) {
    acc = acc + f32(x[xBase + i]) * f32(w[e * podArgs.D + i]);
  }

  red[tid] = acc;
  workgroupBarrier();
  for (var st : u32 = 32u; st > 0u; st = st >> 1u) {
    if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
    workgroupBarrier();
  }
  if (tid == 0u) { logits[t * podArgs.rows + e] = red[0]; }
}
