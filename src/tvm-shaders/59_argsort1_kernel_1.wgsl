// [59] argsort1_kernel_1 (shader #59, 80 lines)
//----------------------------------------
// Function: argsort1_kernel_1
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> out_buf : array<i32>;
@group(0) @binding(1) var<storage, read_write> out_swap_buf : array<i32>;
@group(0) @binding(2) var<storage, read_write> value_buf : array<f32>;
@group(0) @binding(3) var<storage, read_write> value_swap_buf : array<f32>;

struct PODArgs {
  batch_size: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> temp_keys_swap : array<f32, 128>;
var<workgroup> temp_values_swap : array<i32, 128>;
@compute @workgroup_size(64, 1, 1)
fn argsort1_kernel_1(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  var temp_cond1 : array<f32, 1>;
  var temp_cond2 : array<f32, 1>;
  var temp_keys : array<f32, 1>;
  var temp_values : array<i32, 1>;
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var i : i32 = 0; i < 2i; i++) {
    if ((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + i) < podArgs.vocab_size) {
      let rmod : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_1 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      temp_keys_swap[((i32(threadIdx.x) * 2i) + i)] = value_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod + podArgs.batch_size), rmod, (((podArgs.batch_size >= 0i) && (rmod >= 0i)) || ((podArgs.batch_size < 0i) && (rmod <= 0i)))) * podArgs.vocab_size)) + select((rdiv - 1i), rdiv, (((podArgs.batch_size >= 0i) && (rmod_1 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_1 <= 0i))))) + i)];
      let rmod_2 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_3 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv_1 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      temp_values_swap[((i32(threadIdx.x) * 2i) + i)] = out_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod_2 + podArgs.batch_size), rmod_2, (((podArgs.batch_size >= 0i) && (rmod_2 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_2 <= 0i)))) * podArgs.vocab_size)) + select((rdiv_1 - 1i), rdiv_1, (((podArgs.batch_size >= 0i) && (rmod_3 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_3 <= 0i))))) + i)];
    }
  }
  workgroupBarrier();
  for (var j : i32 = 0; j < min(128i, (podArgs.vocab_size - (v__1 * 128i))); j++) {
    if ((((i32(threadIdx.x) * 2i) + (j & 1i)) < 127i) && ((((i32(threadIdx.x) * 2i) + (j & 1i)) + 1i) < (podArgs.vocab_size - (v__1 * 128i)))) {
      temp_cond1[0i] = temp_keys_swap[((i32(threadIdx.x) * 2i) + (j & 1i))];
      temp_cond2[0i] = temp_keys_swap[(((i32(threadIdx.x) * 2i) + (j & 1i)) + 1i)];
      if (temp_cond1[0i] < temp_cond2[0i]) {
        temp_keys[0i] = temp_keys_swap[((i32(threadIdx.x) * 2i) + (j & 1i))];
        temp_keys_swap[((i32(threadIdx.x) * 2i) + (j & 1i))] = temp_keys_swap[(((i32(threadIdx.x) * 2i) + (j & 1i)) + 1i)];
        temp_keys_swap[(((i32(threadIdx.x) * 2i) + (j & 1i)) + 1i)] = temp_keys[0i];
        temp_values[0i] = temp_values_swap[((i32(threadIdx.x) * 2i) + (j & 1i))];
        temp_values_swap[((i32(threadIdx.x) * 2i) + (j & 1i))] = temp_values_swap[(((i32(threadIdx.x) * 2i) + (j & 1i)) + 1i)];
        temp_values_swap[(((i32(threadIdx.x) * 2i) + (j & 1i)) + 1i)] = temp_values[0i];
      }
    }
    workgroupBarrier();
  }
  for (var k : i32 = 0; k < 2i; k++) {
    if ((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + k) < podArgs.vocab_size) {
      let rmod_4 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_5 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv_2 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      value_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod_4 + podArgs.batch_size), rmod_4, (((podArgs.batch_size >= 0i) && (rmod_4 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_4 <= 0i)))) * podArgs.vocab_size)) + select((rdiv_2 - 1i), rdiv_2, (((podArgs.batch_size >= 0i) && (rmod_5 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_5 <= 0i))))) + k)] = temp_keys_swap[((i32(threadIdx.x) * 2i) + k)];
      let rmod_6 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_7 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv_3 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      value_swap_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod_6 + podArgs.batch_size), rmod_6, (((podArgs.batch_size >= 0i) && (rmod_6 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_6 <= 0i)))) * podArgs.vocab_size)) + select((rdiv_3 - 1i), rdiv_3, (((podArgs.batch_size >= 0i) && (rmod_7 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_7 <= 0i))))) + k)] = temp_keys_swap[((i32(threadIdx.x) * 2i) + k)];
      let rmod_8 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_9 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv_4 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      out_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod_8 + podArgs.batch_size), rmod_8, (((podArgs.batch_size >= 0i) && (rmod_8 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_8 <= 0i)))) * podArgs.vocab_size)) + select((rdiv_4 - 1i), rdiv_4, (((podArgs.batch_size >= 0i) && (rmod_9 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_9 <= 0i))))) + k)] = temp_values_swap[((i32(threadIdx.x) * 2i) + k)];
      let rmod_10 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_11 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv_5 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      out_swap_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod_10 + podArgs.batch_size), rmod_10, (((podArgs.batch_size >= 0i) && (rmod_10 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_10 <= 0i)))) * podArgs.vocab_size)) + select((rdiv_5 - 1i), rdiv_5, (((podArgs.batch_size >= 0i) && (rmod_11 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_11 <= 0i))))) + k)] = temp_values_swap[((i32(threadIdx.x) * 2i) + k)];
    }
  }
}
