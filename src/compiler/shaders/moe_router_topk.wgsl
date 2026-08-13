// MOE_ROUTER_TOPK — softmax over every expert, then the top-K survivors.
//
// Stage 2 of 2, over the [E+1] logits moe_router_logits produced. Reading 257
// floats is nothing; this exists as its own dispatch only because a top-K needs
// every logit and WebGPU has no grid-wide barrier. Which makes it pure latency
// on the critical path, and worth shrinking.
//
// Softmax runs over ALL experts BEFORE top-K, and the K survivors are then
// renormalised — that order is load-bearing: softmax-after-top-K produces
// plausible-looking but wrong weights.
//
// ONE SUBGROUP, EIGHT EXPERTS PER LANE. Two shapes were measured first and both
// lost on an M2 Max:
//   - 256 threads, tree reductions, K masked arg-max rounds: ~108 workgroupBarrier
//     calls, 22.5 us.
//   - 256 threads, no barriers, every thread scanning all 256 (rank selection):
//     38.5 us — trading 88 barriers for 256x the arithmetic is a bad trade;
//     a barrier here costs ~0.2 us and a full 256-wide scan ~17 us.
// 32 lanes x 8 experts held in registers needs NEITHER: subgroupMax/subgroupAdd
// reduce with no barrier at all, and the per-lane loops are constant-trip so the
// eight probabilities stay in registers instead of scratch memory.
//
// Requires subgroup size 32 — as does the int4 matmul this block dispatches next,
// so the whole MoE path lives or dies on the same probe.
//
// Ties are broken on the lower expert index (subgroupMin over the winners), so
// two identical probabilities can never both claim the same output slot.
//
// When the checkpoint has a SHARED expert (Qwen3.5/3.6 style) the block has one
// more slot than it has routed experts. It is not routed, so it skips the
// softmax entirely and takes sigmoid of its own logit — row E of the router,
// which stage 1 already computed. Emitting it here as slot K, with expert index
// E (where the loader stacked its weights), is what lets the expert matmul cover
// all K+1 slots in one dispatch and `moe_combine` stay a plain weighted sum.
//
// `hasShared = 0` (Mixtral / Qwen3-MoE style) is the whole of the difference:
// the router has E rows instead of E+1, the stacks have no index E, and this
// kernel writes exactly K slots. Every other kernel in the block already reads
// its slot count from a uniform, so none of them change.
//
// Grid: one workgroup per TOKEN.
//
// Outputs are written in DESCENDING score order (slot 0 = largest). mlx's
// argpartition happens to return ascending; the block output is order-invariant,
// so callers that compare against a reference should match on (index, score)
// pairs, not position.

enable subgroups;

@group(0) @binding(0) var<storage, read_write> out_idx : array<u32>;    // [K+1]
@group(0) @binding(1) var<storage, read_write> out_score : array<f32>;  // [K+1]
@group(0) @binding(2) var<storage, read> logits : array<f32>;           // [E+1]

struct PODArgs {
  E: u32,         // number of routed experts (<= 256)
  K: u32,         // experts per token (<= 32)
  normTopk: u32,  // 1 = renormalise the K scores to sum to 1
  hasShared: u32  // 1 = emit the shared expert as slot K (see the note above)
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

const PER : u32 = 8u;   // 32 lanes x 8 = the 256-expert ceiling this is written for

var<workgroup> top : array<f32, 32>;   // the K survivors, by rank

@compute @workgroup_size(32, 1, 1)
fn moe_router_topk(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let t : u32 = threadIdx.x;
  let E : u32 = podArgs.E;
  let K : u32 = podArgs.K;
  // ONE WORKGROUP PER TOKEN. Decode dispatches one; chunked prefill dispatches
  // the chunk. Both strides follow from the uniforms already here — the router
  // writes E+hasShared logits per token and this emits K+hasShared slots — so a
  // token's slots can never land in its neighbour's, which is the whole reason
  // MoE could not chunk before.
  let tok : u32 = blockIdx.x;
  let lBase : u32 = tok * (E + podArgs.hasShared);
  let oBase : u32 = tok * (K + podArgs.hasShared);

  // Eight logits per lane, kept in registers: every loop below is constant-trip
  // and indexed by the loop counter, so nothing spills to scratch.
  var pv : array<f32, 8>;
  var m : f32 = -3.0e38;
  for (var i : u32 = 0u; i < PER; i = i + 1u) {
    let e : u32 = t * PER + i;
    // Lanes past E must lose every max: -inf, not 0 (logits can be negative).
    let v : f32 = select(-3.0e38, logits[lBase + e], e < E);
    pv[i] = v;
    m = max(m, v);
  }
  let peak : f32 = subgroupMax(m);

  var s : f32 = 0.0;
  for (var i : u32 = 0u; i < PER; i = i + 1u) {
    let e : u32 = t * PER + i;
    let v : f32 = select(0.0, exp(pv[i] - peak), e < E);
    pv[i] = v;
    s = s + v;
  }
  let denom : f32 = subgroupAdd(s);

  // K rounds of arg-max, the winner masked out of its own lane's registers.
  var total : f32 = 0.0;
  for (var k : u32 = 0u; k < K; k = k + 1u) {
    var bv : f32 = -1.0;
    var bi : u32 = 0u;
    for (var i : u32 = 0u; i < PER; i = i + 1u) {
      if (pv[i] > bv) { bv = pv[i]; bi = t * PER + i; }
    }
    let best : f32 = subgroupMax(bv);
    let widx : u32 = subgroupMin(select(0xffffffffu, bi, bv == best));

    if (t == 0u) { top[k] = best; }
    total = total + best;
    for (var i : u32 = 0u; i < PER; i = i + 1u) {
      if (t * PER + i == widx) { pv[i] = -1.0; }
    }
    if (t == 0u) { out_idx[oBase + k] = widx; }
  }

  if (t == 0u) {
    let scale : f32 = select(1.0 / denom, 1.0 / total, podArgs.normTopk == 1u);
    for (var k : u32 = 0u; k < K; k = k + 1u) { out_score[oBase + k] = top[k] * scale; }
    // The shared slot exists only when the checkpoint has a shared expert.
    // Without one there is no row E in the router and no index E in the expert
    // stacks, so the block is exactly its K routed slots and nothing is
    // written past them — moe_combine sums SLOTS, which is K in that case.
    if (podArgs.hasShared == 1u) {
      out_idx[oBase + K] = E;                             // shared expert's slot in the stack
      out_score[oBase + K] = 1.0 / (1.0 + exp(-logits[lBase + E]));
    }
  }
}
