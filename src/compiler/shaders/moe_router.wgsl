// MOE_ROUTER — expert routing for a Qwen3.6-style sparse MoE block, one dispatch.
//
// One workgroup, one thread per expert (E must be <= 256). Each thread computes
// its own expert's router logit, then the workgroup cooperatively does the
// softmax and K rounds of arg-max to pick the top-K.
//
// The router ships at 8 BITS (group 64), unlike every other weight in the model
// which is 4-bit — so it cannot reuse int4_matmul_affine. Same affine scheme
// though: w = scale*q + bias, with q an unsigned byte, 4 per u32 word,
// 16 words per group of 64. Reduced as
//
//   Σ xᵢ·wᵢ = s·Σ xᵢ·qᵢ + b·Σ xᵢ
//
// so no offset trick is needed here (int4_matmul_affine folds in a -7 only
// because it reuses an existing symmetric dot).
//
// Softmax runs over ALL experts BEFORE top-K, and the K survivors are then
// renormalised — that order is load-bearing: softmax-after-top-K produces
// plausible-looking but wrong weights.
//
// Outputs are written in DESCENDING score order. mlx's argpartition happens to
// return ascending; the block output is order-invariant, so callers that
// compare against a reference should match on (index, score) pairs, not position.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_idx : array<u32>;    // [K]
@group(0) @binding(1) var<storage, read_write> out_score : array<f32>;  // [K]
@group(0) @binding(2) var<storage, read> x : array<f16>;                // [D]
@group(0) @binding(3) var<storage, read> w : array<u32>;                // [E, D/4]
@group(0) @binding(4) var<storage, read> scales : array<f16>;           // [E, D/64]
@group(0) @binding(5) var<storage, read> biases : array<f16>;           // [E, D/64]

struct PODArgs {
  D: u32,         // hidden size
  E: u32,         // number of experts (<= 256)
  K: u32,         // experts per token
  normTopk: u32   // 1 = renormalise the K scores to sum to 1
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

var<workgroup> prob : array<f32, 256>;
var<workgroup> rval : array<f32, 256>;
var<workgroup> ridx : array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn moe_router(@builtin(local_invocation_id) threadIdx : vec3<u32>) {
  let e : u32 = threadIdx.x;
  let WPR : u32 = podArgs.D / 4u;    // u32 words per expert row
  let GPR : u32 = podArgs.D / 64u;   // affine groups per row

  var acc : f32 = 0.0;
  if (e < podArgs.E) {
    for (var g : u32 = 0u; g < GPR; g = g + 1u) {
      let s : f32 = f32(scales[e * GPR + g]);
      let b : f32 = f32(biases[e * GPR + g]);
      var dot : f32 = 0.0;
      var xs : f32 = 0.0;
      for (var wi : u32 = 0u; wi < 16u; wi = wi + 1u) {   // 16 words = 64 values
        let word : u32 = w[e * WPR + g * 16u + wi];
        let base : u32 = g * 64u + wi * 4u;
        for (var n : u32 = 0u; n < 4u; n = n + 1u) {
          let v : f32 = f32(x[base + n]);
          dot = dot + v * f32((word >> (8u * n)) & 255u);
          xs = xs + v;
        }
      }
      acc = acc + s * dot + b * xs;
    }
  }
  // Inactive lanes must lose every max: -inf, not 0 (logits can be negative).
  prob[e] = select(-3.0e38, acc, e < podArgs.E);
  workgroupBarrier();

  rval[e] = prob[e];
  workgroupBarrier();
  for (var st : u32 = 128u; st > 0u; st = st >> 1u) {
    if (e < st) { rval[e] = max(rval[e], rval[e + st]); }
    workgroupBarrier();
  }
  let peak : f32 = rval[0];
  workgroupBarrier();

  let ex : f32 = select(0.0, exp(prob[e] - peak), e < podArgs.E);
  rval[e] = ex;
  workgroupBarrier();
  for (var st : u32 = 128u; st > 0u; st = st >> 1u) {
    if (e < st) { rval[e] = rval[e] + rval[e + st]; }
    workgroupBarrier();
  }
  let denom : f32 = rval[0];
  workgroupBarrier();
  prob[e] = ex / denom;
  workgroupBarrier();

  for (var k : u32 = 0u; k < podArgs.K; k = k + 1u) {
    rval[e] = prob[e];
    ridx[e] = e;
    workgroupBarrier();
    for (var st : u32 = 128u; st > 0u; st = st >> 1u) {
      if (e < st) {
        if (rval[e + st] > rval[e]) { rval[e] = rval[e + st]; ridx[e] = ridx[e + st]; }
      }
      workgroupBarrier();
    }
    if (e == 0u) {
      out_idx[k] = ridx[0];
      out_score[k] = rval[0];
    }
    workgroupBarrier();
    if (e == ridx[0]) { prob[e] = -1.0; }   // mask the winner for the next round
    workgroupBarrier();
  }

  if (podArgs.normTopk == 1u && e == 0u) {
    var total : f32 = 0.0;
    for (var k : u32 = 0u; k < podArgs.K; k = k + 1u) { total = total + out_score[k]; }
    for (var k : u32 = 0u; k < podArgs.K; k = k + 1u) { out_score[k] = out_score[k] / total; }
  }
}
