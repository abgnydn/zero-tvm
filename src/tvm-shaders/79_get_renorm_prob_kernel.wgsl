// [79] get_renorm_prob_kernel (shader #79, 41 lines)
//----------------------------------------
// Function: get_renorm_prob_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> cumsum_sorted : array<f32>;
@group(0) @binding(1) var<storage, read_write> renorm_prob : array<f32>;
@group(0) @binding(2) var<storage, read> top_k : array<i32>;
@group(0) @binding(3) var<storage, read> top_p : array<f32>;

struct PODArgs {
  batch: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn get_renorm_prob_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.batch * podArgs.vocab_size)) {
    if ((top_p[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size)] <= cumsum_sorted[(((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size) * podArgs.vocab_size)]) || (top_k[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size)] <= 1i)) {
      renorm_prob[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size)] = cumsum_sorted[(((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size) * podArgs.vocab_size)];
    } else {
      if ((cumsum_sorted[((v__1 * 256i) + i32(threadIdx.x))] < top_p[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size)]) && (((((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size) + 1i) < top_k[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size)])) {
        if (((((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size) + 1i) == podArgs.vocab_size) {
          renorm_prob[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size)] = cumsum_sorted[((((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size) * podArgs.vocab_size) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size))];
        } else {
          if ((top_p[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size)] <= cumsum_sorted[(((((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size) * podArgs.vocab_size) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size)) + 1i)]) || (top_k[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size)] <= ((((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size) + 2i))) {
            renorm_prob[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size)] = cumsum_sorted[(((((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.batch)) / podArgs.vocab_size) * podArgs.vocab_size) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size)) + 1i)];
          }
        }
      }
    }
  }
}
