// MOE_ROUTER_LOGITS — one router logit per expert, one workgroup per expert.
//
// Stage 1 of 2. The whole router is a [E, D] 8-bit matvec — small in bytes
// (256 x 2048 = 524 KB here) but it must finish before any expert matmul can
// start, so it sits on the critical path alone. Doing it in a SINGLE workgroup
// (one thread per expert, each thread walking its own row) costs 100 us on an
// M2 Max: one workgroup is one GPU core, and adjacent threads land D bytes
// apart, so every load is its own cache line. Splitting it per expert makes the
// reads consecutive within a workgroup and hands the work to the whole GPU.
//
// The router ships at 8 BITS (group 64), unlike every other weight in the model
// which is 4-bit — so it cannot reuse int4_matmul_affine. Same affine scheme
// though: w = scale*q + bias, with q an unsigned byte, 4 per u32 word,
// 16 words per group of 64. Reduced as
//
//   Sum xi*wi = s*Sum xi*qi + b*Sum xi
//
// so no offset trick is needed here (int4_matmul_affine folds in a -7 only
// because it reuses an existing symmetric dot).
//
// Row E is the shared-expert GATE, stacked onto the router by the loader so
// that one dispatch produces every logit the block needs — see moe_router_topk.
//
// Grid: one workgroup per row.

enable f16;

@group(0) @binding(0) var<storage, read_write> logits : array<f32>;     // [rows]
@group(0) @binding(1) var<storage, read> x : array<f16>;                // [D]
@group(0) @binding(2) var<storage, read> w : array<u32>;                // [rows, D/4]
@group(0) @binding(3) var<storage, read> scales : array<f16>;           // [rows, D/64]
@group(0) @binding(4) var<storage, read> biases : array<f16>;           // [rows, D/64]

struct PODArgs {
  D: u32,     // hidden size
  rows: u32   // router rows: E experts, plus the shared-expert gate at row E
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> red : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn moe_router_logits(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let e : u32 = blockIdx.x;
  if (e >= podArgs.rows) { return; }

  let WPR : u32 = podArgs.D / 4u;    // u32 words per expert row
  let GPR : u32 = podArgs.D / 64u;   // affine groups per row
  let tid : u32 = threadIdx.x;

  // Thread tid takes words tid, tid+64, tid+128, ... so the 64 lanes of an
  // iteration read 64 consecutive words. The group index rides along with the
  // word index (16 words per group of 64 values), which costs one extra
  // scale/bias load per word — both are already in cache by then.
  var acc : f32 = 0.0;
  for (var wi : u32 = tid; wi < WPR; wi = wi + 64u) {
    let word : u32 = w[e * WPR + wi];
    let g : u32 = wi >> 4u;
    let s : f32 = f32(scales[e * GPR + g]);
    let b : f32 = f32(biases[e * GPR + g]);
    let base : u32 = wi * 4u;
    var dot : f32 = 0.0;
    var xs : f32 = 0.0;
    for (var n : u32 = 0u; n < 4u; n = n + 1u) {
      let v : f32 = f32(x[base + n]);
      dot = dot + v * f32((word >> (8u * n)) & 255u);
      xs = xs + v;
    }
    acc = acc + s * dot + b * xs;
  }

  // Tree reduction in f32 across the 64 threads.
  red[tid] = acc;
  workgroupBarrier();
  for (var st : u32 = 32u; st > 0u; st = st >> 1u) {
    if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
    workgroupBarrier();
  }
  if (tid == 0u) { logits[e] = red[0]; }
}
