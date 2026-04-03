// [81] argsort1_kernel (shader #82, 39 lines)
//----------------------------------------
// Function: argsort1_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> out_buf : array<i32>;
@group(0) @binding(1) var<storage, read> probs : array<f32>;
@group(0) @binding(2) var<storage, read_write> value_buf : array<f32>;

struct PODArgs {
  batch_size: i32,
  elem_offset: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn argsort1_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < podArgs.vocab_size) {
    let rmod : i32 = (i32(blockIdx.y) % podArgs.batch_size);
    let rmod_1 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
    let rdiv : i32 = (i32(blockIdx.y) / podArgs.batch_size);
    let rmod_2 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
    let rmod_3 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
    let rdiv_1 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
    value_buf[((((v__1 * 256i) + (select((rmod + podArgs.batch_size), rmod, (((podArgs.batch_size >= 0i) && (rmod >= 0i)) || ((podArgs.batch_size < 0i) && (rmod <= 0i)))) * podArgs.vocab_size)) + i32(threadIdx.x)) + select((rdiv - 1i), rdiv, (((podArgs.batch_size >= 0i) && (rmod_1 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_1 <= 0i)))))] = probs[(((((v__1 * 256i) + (select((rmod_2 + podArgs.batch_size), rmod_2, (((podArgs.batch_size >= 0i) && (rmod_2 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_2 <= 0i)))) * podArgs.vocab_size)) + i32(threadIdx.x)) + select((rdiv_1 - 1i), rdiv_1, (((podArgs.batch_size >= 0i) && (rmod_3 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_3 <= 0i))))) + podArgs.elem_offset)];
    let rmod_4 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
    let rmod_5 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
    let rdiv_2 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
    out_buf[((((v__1 * 256i) + (select((rmod_4 + podArgs.batch_size), rmod_4, (((podArgs.batch_size >= 0i) && (rmod_4 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_4 <= 0i)))) * podArgs.vocab_size)) + i32(threadIdx.x)) + select((rdiv_2 - 1i), rdiv_2, (((podArgs.batch_size >= 0i) && (rmod_5 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_5 <= 0i)))))] = ((v__1 * 256i) + i32(threadIdx.x));
  }
}
