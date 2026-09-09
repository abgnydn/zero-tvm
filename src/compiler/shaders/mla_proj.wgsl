// MLA_PROJ — the two small per-head projections that bracket latent attention.
//
//   in  : q_nope [heads, 128]   →  out : q_lat  [heads, 512]   (through Wk^T)
//   in  : o_lat  [heads, 512]   →  out : o_head [heads, 128]   (through Wv)
//
// Both are kv_b_proj wearing different hats. The checkpoint stores it as one
// [heads*(nope+v), kv_lora] matrix; the K half turns a head's query into a
// latent query, the V half turns a latent context back into that head's output.
//
// One kernel serves both because the weights are prepared to make them the same
// shape of contraction — Wk arrives TRANSPOSED. Untransposed, q_lat = Wk^T q
// would contract over the matrix's stored ROW axis, which a quantized kernel
// cannot do at all (the affine scale groups run along the other axis) and which
// even in f16 reads a column per output. Flipping it once when the layer loads
// costs 4 MB a layer and makes both of these an ordinary row-major matvec.
//
// f16 weights, deliberately: this is the one place MLA gives memory back. The
// KV cache it replaces was 7x larger, so ~1% of the model in dequantized
// projections is a trade worth making rather than a corner cut.
//
// Grid: (output rows / 64, heads). One thread per output element; each walks
// its own row and no reduction is needed.

enable f16;

@group(0) @binding(0) var<storage, read_write> out_v : array<f16>;   // [heads, N]
@group(0) @binding(1) var<storage, read> in_v : array<f16>;          // [heads, K]
@group(0) @binding(2) var<storage, read> w : array<f16>;             // [heads, N, K]

struct PODArgs {
  N: u32,   // outputs per head
  K: u32    // inputs per head — the contracted axis, always the LAST of w
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(64, 1, 1)
fn mla_proj(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n : u32 = gid.x;
  let h : u32 = gid.y;
  if (n >= podArgs.N) { return; }

  let wBase : u32 = (h * podArgs.N + n) * podArgs.K;
  let xBase : u32 = h * podArgs.K;
  // f32 accumulation over up to 512 terms; f16 would lose the tail of the sum
  // and this feeds a softmax, where a small bias moves every probability.
  var acc : f32 = 0.0;
  for (var k : u32 = 0u; k < podArgs.K; k = k + 1u) {
    acc = acc + f32(in_v[xBase + k]) * f32(w[wBase + k]);
  }
  out_v[h * podArgs.N + n] = f16(acc);
}
