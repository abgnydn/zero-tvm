// [58] get_index_from_sorted_kernel (shader #58, 40 lines)
//----------------------------------------
// Function: get_index_from_sorted_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> cumsum_sorted : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read_write> output_index : array<i32>;
@group(0) @binding(3) var<storage, read> renorm_prob : array<f32>;
@group(0) @binding(4) var<storage, read> sample_indices : array<i32>;
@group(0) @binding(5) var<storage, read> usample : array<f32>;

struct PODArgs {
  batch: i32,
  out_batch: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(6) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn get_index_from_sorted_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.out_batch * podArgs.vocab_size)) {
    if ((usample[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)] < (cumsum_sorted[((sample_indices[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)] * podArgs.vocab_size) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size))] / renorm_prob[sample_indices[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)]])) || (((((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size) + 1i) == podArgs.vocab_size)) {
      if ((((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size) == 0i) {
        output_index[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)] = indices[(sample_indices[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)] * podArgs.vocab_size)];
      } else {
        if ((cumsum_sorted[(((sample_indices[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)] * podArgs.vocab_size) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size)) - 1i)] / renorm_prob[sample_indices[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)]]) <= usample[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)]) {
          output_index[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)] = indices[((sample_indices[((((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.vocab_size * podArgs.out_batch)) / podArgs.vocab_size)] * podArgs.vocab_size) + (((v__1 * 256i) + i32(threadIdx.x)) % podArgs.vocab_size))];
        }
      }
    }
  }
}
