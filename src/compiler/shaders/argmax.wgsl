// ARGMAX — Find the index of the maximum value in a f32 array.
// Replaces TVM's 20-dispatch hierarchical argsort + sampling pipeline.
//
// Input:  logits array (f32, length = vocab_size)
// Output: single i32 token ID
//
// Strategy: parallel reduction in shared memory.
// 256 threads, each scans vocab_size/256 elements, then tree reduce.

@group(0) @binding(0) var<storage, read> logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> result : array<i32>;

struct Params {
  vocab_size: u32,
}
@group(0) @binding(2) var<uniform> params : Params;

var<workgroup> shared_val : array<f32, 256>;
var<workgroup> shared_idx : array<i32, 256>;

@compute @workgroup_size(256, 1, 1)
fn argmax_kernel(@builtin(local_invocation_id) tid : vec3<u32>) {
  let thread_id = tid.x;
  let vocab = params.vocab_size;
  let chunk = (vocab + 255u) / 256u;
  let start = thread_id * chunk;
  let end = min(start + chunk, vocab);

  // Phase 1: each thread finds max in its chunk
  var best_val : f32 = -1e30;
  var best_idx : i32 = 0;

  for (var i = start; i < end; i = i + 1u) {
    let v = logits[i];
    if (v > best_val) {
      best_val = v;
      best_idx = i32(i);
    }
  }

  shared_val[thread_id] = best_val;
  shared_idx[thread_id] = best_idx;
  workgroupBarrier();

  // Phase 2: tree reduction (256 → 128 → 64 → ... → 1)
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (thread_id < stride) {
      if (shared_val[thread_id + stride] > shared_val[thread_id]) {
        shared_val[thread_id] = shared_val[thread_id + stride];
        shared_idx[thread_id] = shared_idx[thread_id + stride];
      }
    }
    workgroupBarrier();
  }

  // Thread 0 writes result
  if (thread_id == 0u) {
    result[0] = shared_idx[0];
  }
}
