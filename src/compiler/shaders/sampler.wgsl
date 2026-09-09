// SAMPLER — temperature, top-p (nucleus) and min-p over the LM-head logits.
//
// Greedy decoding is argmax.wgsl and stays exactly what it was. This is a
// second sampler the engine dispatches INSTEAD of it, in two entry points over
// one binding set:
//
//   sample_reduce  (one workgroup per partition) — online-softmax partials
//   sample_select  (one workgroup)               — merge, threshold, draw
//
// SHAPE. The vocabulary runs to 248320 f32 logits (Qwen3.5). No workgroup can
// hold that distribution and WebGPU has no grid-wide barrier, so the max and
// the softmax denominator are reduced per partition and merged in a second
// dispatch — the (m, d) partial state attention_splitk hands attention_combine,
// minus the o[] vector. Empty partitions (a vocabulary too small to reach every
// partition) write d = 0 and the merge skips them, same as there: their m
// sentinel must not enter the max.
//
// DETERMINISM. There is no GPU randomness in here. The draw is
// hash32(seed ^ hash32(counter)), and the engine writes the token's POSITION
// as the counter — so a sampled token is a pure function of the logits and
// where it was sampled. That is what makes a run replayable, and it is why
// cross-turn prefix reuse and the prefill replay can re-run a position without
// changing the answer.
//
// TEMPERATURE 0 IS ARGMAX.WGSL, NOT MERELY AN ARGMAX. sample_select opens by
// running argmax.wgsl's kernel body verbatim — 256 threads over contiguous
// ceil(vocab/256) chunks of the whole vocabulary, the same -1e30 sentinel, the
// same tree reduction — rather than reusing an index the partition pass could
// have found more cheaply. The reason is ties: that tree reduction does NOT
// return the first index of the maximum (after the first round, slot t may
// hold index t+128, and "keep the lower slot" then keeps the LATER index), so
// which maximiser wins is a property of the exact partitioning. A sampler that
// tied differently would diverge from every greedy reference in this repo on
// the one input where it matters and agree everywhere else — silent by
// construction. Copying the loop makes the two the same token, tie for tie.
//
// The engine goes further and keeps dispatching argmax.wgsl itself at
// temperature 0; this branch is what a caller that passes 0 straight to the
// sampler gets, and tests/kernels/run.mjs pins the two together.
//
// TOP-P WITHOUT A SORT. With w_i = exp(z_i - M) over the temperature-scaled
// logits z and D = Σ w_i, the nucleus is a THRESHOLD set: the smallest set of
// largest probabilities whose mass reaches top_p is exactly { i : w_i >= τ }
// for the largest τ with mass(τ) >= top_p·D. mass() is monotonic in τ, so
// sample_select bisects for τ on the IEEE bit pattern of a positive f32 — 31
// halvings land ON a representable float, which means τ comes out bit-equal
// to the cutoff w rather than near it, and the kept set is the sorted answer,
// ties included. The price is that each halving re-reads the vocabulary, which
// is why it is skipped whole at top_p >= 1; min-p needs no search at all
// (p_i >= min_p·p_max reduces to w_i >= min_p, because p_max is 1/D).
//
// Both filters are computed against the full temperature-scaled distribution
// and intersected (τ = max of the two), then the survivors are renormalised
// once. Feeding one the other's renormalised output would be a different
// sampler, and a silently different one.
//
// Every accumulator here is f32 — the denominator sums up to 248320 terms and
// the CDF walk sums the whole nucleus (see moe_router_logits.wgsl for the
// house rule).

@group(0) @binding(0) var<storage, read> logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> partials : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<i32>;

struct Params {
  vocab_size: u32,
  parts: u32,        // partitions == sample_reduce workgroups
  temperature: f32,  // <= 0 → greedy
  top_p: f32,        // 1 → off
  min_p: f32,        // 0 → off
  seed: u32,
  counter: u32,      // the engine rewrites this per token (the position)
}
@group(0) @binding(3) var<uniform> params : Params;

const THREADS : u32 = 256u;
const PARTIAL_STRIDE : u32 = 2u;   // m, d per partition
const NEG_MAX : f32 = -3.4e38;

var<workgroup> red_val : array<f32, 256>;
var<workgroup> red_idx : array<i32, 256>;

// lowbias32 (Wellons). Any hash with decent avalanche would do; what matters
// is that tests/kernels/run.mjs reproduces it exactly in u32 JS arithmetic.
fn hash32(x : u32) -> u32 {
  var h = x;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

// 24 bits, not 32: f32 cannot represent every u32, and the ones it rounds up
// land on exactly 1.0 — a draw of 1.0 asks for mass past the end of the
// distribution and falls out of the CDF walk with nothing picked.
fn draw_unit() -> f32 {
  let h = hash32(params.seed ^ hash32(params.counter));
  return f32(h >> 8u) * (1.0 / 16777216.0);
}

// The four tree reductions across the two entry points are written out at
// every site rather than shared through a helper. A helper would work, but a
// barrier inside a called function moves the uniformity requirement onto the
// CALL SITE, and every other kernel here inlines its reduction — one house
// pattern is worth more than the lines it would save.
@compute @workgroup_size(256, 1, 1)
fn sample_reduce(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let part = blockIdx.x;
  let tid = threadIdx.x;
  let vocab = params.vocab_size;

  let per_part = (vocab + params.parts - 1u) / params.parts;
  let p_begin = part * per_part;
  let p_end = min(vocab, p_begin + per_part);
  let chunk = (per_part + THREADS - 1u) / THREADS;
  let start = min(p_end, p_begin + tid * chunk);
  let end = min(p_end, start + chunk);

  // Temperature 0 would divide by zero if taken literally. Scale by 1 instead
  // and let sample_select's greedy branch ignore the denominator this produces.
  let t = select(params.temperature, 1.0, params.temperature <= 0.0);
  let invT = 1.0 / t;

  var m : f32 = NEG_MAX;
  for (var i : u32 = start; i < end; i = i + 1u) {
    m = max(m, logits[i] * invT);
  }
  red_val[tid] = m;
  workgroupBarrier();
  for (var s : u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) { red_val[tid] = max(red_val[tid], red_val[tid + s]); }
    workgroupBarrier();
  }
  let pm = red_val[0];
  workgroupBarrier();   // the denominator below reuses red_val

  var d : f32 = 0.0;
  for (var i : u32 = start; i < end; i = i + 1u) {
    d = d + exp(logits[i] * invT - pm);
  }
  red_val[tid] = d;
  workgroupBarrier();
  for (var s : u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) { red_val[tid] = red_val[tid] + red_val[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let base = part * PARTIAL_STRIDE;
    partials[base] = pm;
    partials[base + 1u] = red_val[0];   // 0 exactly when the partition is empty
  }
  // A caller that runs this pass without the select pass would otherwise leave
  // the previous token sitting in `result` and decode it again, forever.
  if (part == 0u && tid == 0u) { result[0] = -1; }
}

@compute @workgroup_size(256, 1, 1)
fn sample_select(@builtin(local_invocation_id) threadIdx : vec3<u32>) {
  let tid = threadIdx.x;
  let vocab = params.vocab_size;
  let parts = params.parts;

  // The same chunking the mass passes below use, and argmax.wgsl's: a thread
  // owns one contiguous range, so the CDF can be walked in index order.
  let chunk = (vocab + THREADS - 1u) / THREADS;
  let start = tid * chunk;
  let end = min(start + chunk, vocab);

  // argmax.wgsl's kernel body, verbatim — see the temperature-0 note above.
  // This is also the fallback the draw lands on if the nucleus rounds empty,
  // which is why it is computed even when sampling.
  var best_val : f32 = -1e30;
  var best_idx : i32 = 0;
  for (var i : u32 = start; i < end; i = i + 1u) {
    let v = logits[i];
    if (v > best_val) { best_val = v; best_idx = i32(i); }
  }
  red_val[tid] = best_val;
  red_idx[tid] = best_idx;
  workgroupBarrier();
  for (var s : u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      if (red_val[tid + s] > red_val[tid]) {
        red_val[tid] = red_val[tid + s];
        red_idx[tid] = red_idx[tid + s];
      }
    }
    workgroupBarrier();
  }
  let greedy = red_idx[0];
  workgroupBarrier();   // the mass passes below reuse red_val

  if (params.temperature <= 0.0) {
    if (tid == 0u) { result[0] = greedy; }
    return;
  }

  // Every thread redundantly walks the (tiny) partition list rather than
  // reducing it — the same trade attention_combine makes, and it keeps M and D
  // identical across threads without a barrier.
  var M : f32 = NEG_MAX;
  for (var p : u32 = 0u; p < parts; p = p + 1u) {
    let base = p * PARTIAL_STRIDE;
    if (partials[base + 1u] > 0.0) { M = max(M, partials[base]); }
  }
  var D : f32 = 0.0;
  for (var p : u32 = 0u; p < parts; p = p + 1u) {
    let base = p * PARTIAL_STRIDE;
    let d_p = partials[base + 1u];
    if (d_p > 0.0) { D = D + d_p * exp(partials[base] - M); }
  }

  let invT = 1.0 / params.temperature;

  var tau : f32 = params.min_p;
  if (params.top_p < 1.0) {
    let want = params.top_p * D;
    // lo is always a threshold that keeps at least top_p of the mass, hi one
    // that keeps less. 31 halvings shrink the initial width to 1, so lo ends
    // up bit-equal to the cutoff w instead of near it.
    var lo : u32 = 0u;
    var hi : u32 = 0x3f800001u;   // just past 1.0; no w exceeds exp(0)
    // A fixed trip count, not a converged-early break: the decision below
    // comes out of workgroup memory, and a loop that exited on it would put
    // these barriers in non-uniform control flow.
    for (var it : u32 = 0u; it < 31u; it = it + 1u) {
      let mid = lo + (hi - lo) / 2u;
      let cut = bitcast<f32>(mid);
      var acc : f32 = 0.0;
      for (var i : u32 = start; i < end; i = i + 1u) {
        let w = exp(logits[i] * invT - M);
        if (w >= cut) { acc = acc + w; }
      }
      red_val[tid] = acc;
      workgroupBarrier();
      for (var s : u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { red_val[tid] = red_val[tid] + red_val[tid + s]; }
        workgroupBarrier();
      }
      let mass = red_val[0];
      // Not decoration: without this a fast thread overwrites red_val[tid] for
      // the next step while a slow one is still reading red_val[0].
      workgroupBarrier();
      if (mass >= want) { lo = mid; } else { hi = mid; }
    }
    tau = max(tau, bitcast<f32>(lo));
  }

  // Surviving mass per chunk, then ONE thread walks the CDF. The walk is
  // serial over a single chunk (vocab/256 logits), not over the vocabulary.
  var acc : f32 = 0.0;
  for (var i : u32 = start; i < end; i = i + 1u) {
    let w = exp(logits[i] * invT - M);
    if (w >= tau) { acc = acc + w; }
  }
  red_val[tid] = acc;
  workgroupBarrier();

  if (tid == 0u) {
    var kept : f32 = 0.0;
    for (var c : u32 = 0u; c < THREADS; c = c + 1u) { kept = kept + red_val[c]; }
    let draw_at = draw_unit() * kept;

    // Skip whole chunks, then walk the last one token by token.
    var run : f32 = 0.0;
    var owner : u32 = 0u;
    for (var c : u32 = 0u; c + 1u < THREADS; c = c + 1u) {
      if (run + red_val[c] > draw_at) { break; }
      run = run + red_val[c];
      owner = c + 1u;
    }

    // Falls back to the greedy token, never to an out-of-range id: the chunk
    // sums and this walk add the same terms in the same order but not in the
    // same groupings, so f32 rounding can leave the owning chunk with no
    // survivor that carries `run` past the draw at all.
    var picked : i32 = greedy;
    let w_start = owner * chunk;
    let w_end = min(w_start + chunk, vocab);
    for (var i : u32 = w_start; i < w_end; i = i + 1u) {
      let w = exp(logits[i] * invT - M);
      if (w >= tau) {
        run = run + w;
        picked = i32(i);
        if (run > draw_at) { break; }
      }
    }
    result[0] = picked;
  }
}
