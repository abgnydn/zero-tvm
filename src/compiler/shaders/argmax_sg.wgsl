// ARGMAX (subgroup variant) — same result as argmax.wgsl.
// Two-level: (1) per-subgroup butterfly reduction using subgroupShuffleXor,
// (2) thread 0 scans the 8 subgroup winners serially. Replaces the 8 barriers
// of the scalar tree reduction with 1.
//
// Branchless compare-and-swap via select() — subgroup shuffles want uniform
// control flow for portability, so we avoid `if` inside the butterfly.
//
// 256 threads per workgroup. With subgroup size 32 that's 8 subgroups.

enable subgroups;

@group(0) @binding(0) var<storage, read> logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> result : array<i32>;

struct Params {
  vocab_size: u32,
}
@group(0) @binding(2) var<uniform> params : Params;

// One slot per subgroup. 256 / 32 = 8; overprovisioned to 16 for safety.
var<workgroup> sg_val : array<f32, 16>;
var<workgroup> sg_idx : array<i32, 16>;

@compute @workgroup_size(256, 1, 1)
fn argmax_sg(
  @builtin(local_invocation_id) tid : vec3<u32>,
  @builtin(subgroup_invocation_id) lane : u32,
  @builtin(subgroup_size) sg_size : u32,
) {
  let thread_id = tid.x;
  let vocab = params.vocab_size;
  let chunk = (vocab + 255u) / 256u;
  let start = thread_id * chunk;
  let end = min(start + chunk, vocab);

  var best_val : f32 = -1e30;
  var best_idx : i32 = 0;
  for (var i = start; i < end; i = i + 1u) {
    let v = logits[i];
    if (v > best_val) { best_val = v; best_idx = i32(i); }
  }

  // Butterfly reduction within the subgroup. select() keeps control flow
  // uniform across the shuffles.
  for (var stride : u32 = 1u; stride < sg_size; stride = stride << 1u) {
    let other_val = subgroupShuffleXor(best_val, stride);
    let other_idx = subgroupShuffleXor(best_idx, stride);
    let take = other_val > best_val;
    best_val = select(best_val, other_val, take);
    best_idx = select(best_idx, other_idx, take);
  }

  // Lane 0 of each subgroup writes its winner to shared memory.
  let sg_id : u32 = thread_id / sg_size;
  if (lane == 0u) {
    sg_val[sg_id] = best_val;
    sg_idx[sg_id] = best_idx;
  }
  workgroupBarrier();

  // Cross-subgroup: thread 0 scans the 8 subgroup winners serially. Simpler
  // than a second butterfly and avoids subgroup ops in divergent control.
  if (thread_id == 0u) {
    let n_subgroups : u32 = 256u / sg_size;
    var v : f32 = sg_val[0];
    var idx : i32 = sg_idx[0];
    for (var i : u32 = 1u; i < n_subgroups; i = i + 1u) {
      if (sg_val[i] > v) { v = sg_val[i]; idx = sg_idx[i]; }
    }
    result[0] = idx;
  }
}
