// INT8_AFFINE_MATVEC — GEMV over MLX-style 8-bit affine weights.
//
// Same scheme as moe_router_logits' inner loop, but exposed as a plain matvec with no
// softmax or top-k: out[r] = Σ xᵢ·wᵢ with w = scale*q + bias over groups of 64,
// q an unsigned byte, 4 per u32 word.
//
//   Σ xᵢ·wᵢ = s·Σ xᵢ·qᵢ + b·Σ xᵢ
//
// Written for the pieces of a MoE block that ship at 8 bits while the bulk of
// the model is 4-bit — currently just shared_expert_gate (a single row), which
// is why this is a simple one-workgroup-per-row tree reduction rather than
// anything tuned.
//
// Grid: one workgroup per output row.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_buf : array<f32>;   // [N]
@group(0) @binding(1) var<storage, read> x : array<f16>;               // [K]
@group(0) @binding(2) var<storage, read> w : array<u32>;               // [N, K/4]
@group(0) @binding(3) var<storage, read> scales : array<f16>;          // [N, K/64]
@group(0) @binding(4) var<storage, read> biases : array<f16>;          // [N, K/64]

struct PODArgs {
  K: u32,   // input dim
  N: u32    // output rows
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> part : array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn int8_affine_matvec(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row : u32 = blockIdx.z * gridDim.x + blockIdx.x;
  if (row >= podArgs.N) { return; }
  let tid : u32 = threadIdx.x;
  let GPR : u32 = podArgs.K / 64u;    // groups per row
  let WPR : u32 = podArgs.K / 4u;     // u32 words per row

  var acc : f32 = 0.0;
  // One group per thread per pass; a group is 64 values = 16 u32 words, so the
  // whole group lies inside one (scale, bias) pair and needs no cross-lane work.
  for (var g : u32 = tid; g < GPR; g = g + 64u) {
    let s : f32 = f32(scales[row * GPR + g]);
    let b : f32 = f32(biases[row * GPR + g]);
    var dot : f32 = 0.0;
    var xs : f32 = 0.0;
    for (var wi : u32 = 0u; wi < 16u; wi = wi + 1u) {
      let word : u32 = w[row * WPR + g * 16u + wi];
      let base : u32 = g * 64u + wi * 4u;
      for (var n : u32 = 0u; n < 4u; n = n + 1u) {
        let v : f32 = f32(x[base + n]);
        dot = dot + v * f32((word >> (8u * n)) & 255u);
        xs = xs + v;
      }
    }
    acc = acc + s * dot + b * xs;
  }
  part[tid] = acc;
  workgroupBarrier();
  for (var st : u32 = 32u; st > 0u; st = st >> 1u) {
    if (tid < st) { part[tid] = part[tid] + part[tid + st]; }
    workgroupBarrier();
  }
  if (tid == 0u) { out_buf[row] = part[0]; }
}
