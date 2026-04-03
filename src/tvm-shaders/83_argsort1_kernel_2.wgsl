// [83] argsort1_kernel_2 (shader #74, 653 lines)
//----------------------------------------
// Function: argsort1_kernel_2
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> out_buf : array<i32>;
@group(0) @binding(1) var<storage, read_write> out_swap_buf : array<i32>;
@group(0) @binding(2) var<storage, read_write> value_buf : array<f32>;
@group(0) @binding(3) var<storage, read_write> value_swap_buf : array<f32>;

struct PODArgs {
  batch_size: i32,
  cse_v1: i32,
  i_0: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn argsort1_kernel_2(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  var first : array<i32, 1>;
  var last : array<i32, 1>;
  var first_1 : array<i32, 1>;
  var last_1 : array<i32, 1>;
  var first_2 : array<i32, 1>;
  var last_2 : array<i32, 1>;
  var first_3 : array<i32, 1>;
  var last_3 : array<i32, 1>;
  var first_4 : array<i32, 1>;
  var last_4 : array<i32, 1>;
  var first_5 : array<i32, 1>;
  var last_5 : array<i32, 1>;
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  let rmod : i32 = (i32(blockIdx.y) % podArgs.batch_size);
  let rdiv : i32 = (i32(blockIdx.y) / podArgs.batch_size);
  if (((2i<<u32(podArgs.cse_v1)) * select((rdiv - 1i), rdiv, (((podArgs.batch_size >= 0i) && (rmod >= 0i)) || ((podArgs.batch_size < 0i) && (rmod <= 0i))))) < podArgs.vocab_size) {
    if ((((2i<<u32(podArgs.cse_v1)) + 1023i)>>10u) == 1i) {
      if ((podArgs.i_0 % 2i) == 0i) {
        let rmod_1 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_1 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_2 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_2 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first[0i] = max(0i, ((min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_1 - 1i), rdiv_1, (((podArgs.batch_size >= 0i) && (rmod_1 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_1 <= 0i)))))), podArgs.vocab_size) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) - min(((select((rdiv_2 - 1i), rdiv_2, (((podArgs.batch_size >= 0i) && (rmod_2 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_2 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size)));
        let rmod_3 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_3 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last[0i] = min((i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u)), min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_3 - 1i), rdiv_3, (((podArgs.batch_size >= 0i) && (rmod_3 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_3 <= 0i))))))));
        while (true) {
          if (!((first[0i] < last[0i]))) { break; }
          let rmod_4 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_4 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_5 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_6 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_7 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_5 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if (value_buf[((((min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_4 - 1i), rdiv_4, (((podArgs.batch_size >= 0i) && (rmod_4 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_4 <= 0i)))))), podArgs.vocab_size) + (select((rmod_5 + podArgs.batch_size), rmod_5, (((podArgs.batch_size >= 0i) && (rmod_5 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_5 <= 0i)))) * podArgs.vocab_size)) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) - ((first[0i] + last[0i])>>1u)) - 1i)] <= value_buf[(((select((rmod_6 + podArgs.batch_size), rmod_6, (((podArgs.batch_size >= 0i) && (rmod_6 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_6 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_5 - 1i), rdiv_5, (((podArgs.batch_size >= 0i) && (rmod_7 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_7 <= 0i)))))) + ((first[0i] + last[0i])>>1u))]) {
            first[0i] = (((first[0i] + last[0i])>>1u) + 1i);
          } else {
            last[0i] = ((first[0i] + last[0i])>>1u);
          }
        }
        let rmod_8 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_6 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first[0i] = (((2i<<u32(podArgs.cse_v1)) * select((rdiv_6 - 1i), rdiv_6, (((podArgs.batch_size >= 0i) && (rmod_8 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_8 <= 0i))))) + first[0i]);
        let rmod_9 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_7 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last[0i] = ((min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_7 - 1i), rdiv_7, (((podArgs.batch_size >= 0i) && (rmod_9 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_9 <= 0i)))))), podArgs.vocab_size) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) - last[0i]);
        let rmod_10 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_8 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_11 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_9 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        for (var i_1 : i32 = 0; i_1 < min(((min(((select((rdiv_8 - 1i), rdiv_8, (((podArgs.batch_size >= 0i) && (rmod_10 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_10 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size) - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_9 - 1i), rdiv_9, (((podArgs.batch_size >= 0i) && (rmod_11 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_11 <= 0i)))))) - (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))), (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u)); i_1++) {
          let rmod_12 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_10 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_13 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_11 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if ((((first[0i] < (((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_10 - 1i), rdiv_10, (((podArgs.batch_size >= 0i) && (rmod_12 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_12 <= 0i))))))) && (first[0i] < podArgs.vocab_size)) && (last[0i] < ((select((rdiv_11 - 1i), rdiv_11, (((podArgs.batch_size >= 0i) && (rmod_13 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_13 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))))) && (last[0i] < podArgs.vocab_size)) {
            let rmod_14 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rmod_15 : i32 = (last[0i] % podArgs.vocab_size);
            let rdiv_12 : i32 = (last[0i] / podArgs.vocab_size);
                        let rmod_16 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_17 : i32 = (((select((rmod_16 + podArgs.batch_size), rmod_16, (((podArgs.batch_size >= 0i) && (rmod_16 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_16 <= 0i)))) * podArgs.vocab_size) + last[0i]) % podArgs.vocab_size);
            let rmod_18 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rmod_19 : i32 = (first[0i] % podArgs.vocab_size);
            let rdiv_13 : i32 = (first[0i] / podArgs.vocab_size);
                        let rmod_20 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_21 : i32 = (((select((rmod_20 + podArgs.batch_size), rmod_20, (((podArgs.batch_size >= 0i) && (rmod_20 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_20 <= 0i)))) * podArgs.vocab_size) + first[0i]) % podArgs.vocab_size);
            if (value_buf[(((select((rmod_14 + podArgs.batch_size), rmod_14, (((podArgs.batch_size >= 0i) && (rmod_14 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_14 <= 0i)))) + select((rdiv_12 - 1i), rdiv_12, (((podArgs.vocab_size >= 0i) && (rmod_15 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_15 <= 0i))))) * podArgs.vocab_size) + select((rmod_17 + podArgs.vocab_size), rmod_17, (((podArgs.vocab_size >= 0i) && (rmod_17 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_17 <= 0i)))))] <= value_buf[(((select((rmod_18 + podArgs.batch_size), rmod_18, (((podArgs.batch_size >= 0i) && (rmod_18 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_18 <= 0i)))) + select((rdiv_13 - 1i), rdiv_13, (((podArgs.vocab_size >= 0i) && (rmod_19 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_19 <= 0i))))) * podArgs.vocab_size) + select((rmod_21 + podArgs.vocab_size), rmod_21, (((podArgs.vocab_size >= 0i) && (rmod_21 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_21 <= 0i)))))]) {
              let rmod_22 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_23 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_14 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_24 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_25 : i32 = (first[0i] % podArgs.vocab_size);
              let rdiv_15 : i32 = (first[0i] / podArgs.vocab_size);
                            let rmod_26 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_27 : i32 = (((select((rmod_26 + podArgs.batch_size), rmod_26, (((podArgs.batch_size >= 0i) && (rmod_26 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_26 <= 0i)))) * podArgs.vocab_size) + first[0i]) % podArgs.vocab_size);
              value_swap_buf[((((select((rmod_22 + podArgs.batch_size), rmod_22, (((podArgs.batch_size >= 0i) && (rmod_22 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_22 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_14 - 1i), rdiv_14, (((podArgs.batch_size >= 0i) && (rmod_23 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_23 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_1)] = value_buf[(((select((rmod_24 + podArgs.batch_size), rmod_24, (((podArgs.batch_size >= 0i) && (rmod_24 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_24 <= 0i)))) + select((rdiv_15 - 1i), rdiv_15, (((podArgs.vocab_size >= 0i) && (rmod_25 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_25 <= 0i))))) * podArgs.vocab_size) + select((rmod_27 + podArgs.vocab_size), rmod_27, (((podArgs.vocab_size >= 0i) && (rmod_27 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_27 <= 0i)))))];
              let rmod_28 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_29 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_16 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_30 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_31 : i32 = (first[0i] % podArgs.vocab_size);
              let rdiv_17 : i32 = (first[0i] / podArgs.vocab_size);
                            let rmod_32 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_33 : i32 = (((select((rmod_32 + podArgs.batch_size), rmod_32, (((podArgs.batch_size >= 0i) && (rmod_32 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_32 <= 0i)))) * podArgs.vocab_size) + first[0i]) % podArgs.vocab_size);
              out_swap_buf[((((select((rmod_28 + podArgs.batch_size), rmod_28, (((podArgs.batch_size >= 0i) && (rmod_28 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_28 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_16 - 1i), rdiv_16, (((podArgs.batch_size >= 0i) && (rmod_29 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_29 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_1)] = out_buf[(((select((rmod_30 + podArgs.batch_size), rmod_30, (((podArgs.batch_size >= 0i) && (rmod_30 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_30 <= 0i)))) + select((rdiv_17 - 1i), rdiv_17, (((podArgs.vocab_size >= 0i) && (rmod_31 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_31 <= 0i))))) * podArgs.vocab_size) + select((rmod_33 + podArgs.vocab_size), rmod_33, (((podArgs.vocab_size >= 0i) && (rmod_33 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_33 <= 0i)))))];
              first[0i] = (first[0i] + 1i);
            } else {
              let rmod_34 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_35 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_18 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_36 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_37 : i32 = (last[0i] % podArgs.vocab_size);
              let rdiv_19 : i32 = (last[0i] / podArgs.vocab_size);
                            let rmod_38 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_39 : i32 = (((select((rmod_38 + podArgs.batch_size), rmod_38, (((podArgs.batch_size >= 0i) && (rmod_38 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_38 <= 0i)))) * podArgs.vocab_size) + last[0i]) % podArgs.vocab_size);
              value_swap_buf[((((select((rmod_34 + podArgs.batch_size), rmod_34, (((podArgs.batch_size >= 0i) && (rmod_34 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_34 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_18 - 1i), rdiv_18, (((podArgs.batch_size >= 0i) && (rmod_35 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_35 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_1)] = value_buf[(((select((rmod_36 + podArgs.batch_size), rmod_36, (((podArgs.batch_size >= 0i) && (rmod_36 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_36 <= 0i)))) + select((rdiv_19 - 1i), rdiv_19, (((podArgs.vocab_size >= 0i) && (rmod_37 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_37 <= 0i))))) * podArgs.vocab_size) + select((rmod_39 + podArgs.vocab_size), rmod_39, (((podArgs.vocab_size >= 0i) && (rmod_39 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_39 <= 0i)))))];
              let rmod_40 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_41 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_20 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_42 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_43 : i32 = (last[0i] % podArgs.vocab_size);
              let rdiv_21 : i32 = (last[0i] / podArgs.vocab_size);
                            let rmod_44 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_45 : i32 = (((select((rmod_44 + podArgs.batch_size), rmod_44, (((podArgs.batch_size >= 0i) && (rmod_44 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_44 <= 0i)))) * podArgs.vocab_size) + last[0i]) % podArgs.vocab_size);
              out_swap_buf[((((select((rmod_40 + podArgs.batch_size), rmod_40, (((podArgs.batch_size >= 0i) && (rmod_40 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_40 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_20 - 1i), rdiv_20, (((podArgs.batch_size >= 0i) && (rmod_41 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_41 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_1)] = out_buf[(((select((rmod_42 + podArgs.batch_size), rmod_42, (((podArgs.batch_size >= 0i) && (rmod_42 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_42 <= 0i)))) + select((rdiv_21 - 1i), rdiv_21, (((podArgs.vocab_size >= 0i) && (rmod_43 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_43 <= 0i))))) * podArgs.vocab_size) + select((rmod_45 + podArgs.vocab_size), rmod_45, (((podArgs.vocab_size >= 0i) && (rmod_45 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_45 <= 0i)))))];
              last[0i] = (last[0i] + 1i);
            }
          } else {
            let rmod_46 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rdiv_22 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
            if ((first[0i] < (((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_22 - 1i), rdiv_22, (((podArgs.batch_size >= 0i) && (rmod_46 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_46 <= 0i))))))) && (first[0i] < podArgs.vocab_size)) {
              let rmod_47 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_48 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_23 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_49 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_50 : i32 = (first[0i] % podArgs.vocab_size);
              let rdiv_24 : i32 = (first[0i] / podArgs.vocab_size);
                            let rmod_51 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_52 : i32 = (((select((rmod_51 + podArgs.batch_size), rmod_51, (((podArgs.batch_size >= 0i) && (rmod_51 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_51 <= 0i)))) * podArgs.vocab_size) + first[0i]) % podArgs.vocab_size);
              value_swap_buf[((((select((rmod_47 + podArgs.batch_size), rmod_47, (((podArgs.batch_size >= 0i) && (rmod_47 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_47 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_23 - 1i), rdiv_23, (((podArgs.batch_size >= 0i) && (rmod_48 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_48 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_1)] = value_buf[(((select((rmod_49 + podArgs.batch_size), rmod_49, (((podArgs.batch_size >= 0i) && (rmod_49 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_49 <= 0i)))) + select((rdiv_24 - 1i), rdiv_24, (((podArgs.vocab_size >= 0i) && (rmod_50 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_50 <= 0i))))) * podArgs.vocab_size) + select((rmod_52 + podArgs.vocab_size), rmod_52, (((podArgs.vocab_size >= 0i) && (rmod_52 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_52 <= 0i)))))];
              let rmod_53 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_54 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_25 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_55 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_56 : i32 = (first[0i] % podArgs.vocab_size);
              let rdiv_26 : i32 = (first[0i] / podArgs.vocab_size);
                            let rmod_57 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_58 : i32 = (((select((rmod_57 + podArgs.batch_size), rmod_57, (((podArgs.batch_size >= 0i) && (rmod_57 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_57 <= 0i)))) * podArgs.vocab_size) + first[0i]) % podArgs.vocab_size);
              out_swap_buf[((((select((rmod_53 + podArgs.batch_size), rmod_53, (((podArgs.batch_size >= 0i) && (rmod_53 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_53 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_25 - 1i), rdiv_25, (((podArgs.batch_size >= 0i) && (rmod_54 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_54 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_1)] = out_buf[(((select((rmod_55 + podArgs.batch_size), rmod_55, (((podArgs.batch_size >= 0i) && (rmod_55 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_55 <= 0i)))) + select((rdiv_26 - 1i), rdiv_26, (((podArgs.vocab_size >= 0i) && (rmod_56 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_56 <= 0i))))) * podArgs.vocab_size) + select((rmod_58 + podArgs.vocab_size), rmod_58, (((podArgs.vocab_size >= 0i) && (rmod_58 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_58 <= 0i)))))];
              first[0i] = (first[0i] + 1i);
            } else {
              let rmod_59 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_60 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_27 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_61 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_62 : i32 = (last[0i] % podArgs.vocab_size);
              let rdiv_28 : i32 = (last[0i] / podArgs.vocab_size);
                            let rmod_63 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_64 : i32 = (((select((rmod_63 + podArgs.batch_size), rmod_63, (((podArgs.batch_size >= 0i) && (rmod_63 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_63 <= 0i)))) * podArgs.vocab_size) + last[0i]) % podArgs.vocab_size);
              value_swap_buf[((((select((rmod_59 + podArgs.batch_size), rmod_59, (((podArgs.batch_size >= 0i) && (rmod_59 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_59 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_27 - 1i), rdiv_27, (((podArgs.batch_size >= 0i) && (rmod_60 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_60 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_1)] = value_buf[(((select((rmod_61 + podArgs.batch_size), rmod_61, (((podArgs.batch_size >= 0i) && (rmod_61 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_61 <= 0i)))) + select((rdiv_28 - 1i), rdiv_28, (((podArgs.vocab_size >= 0i) && (rmod_62 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_62 <= 0i))))) * podArgs.vocab_size) + select((rmod_64 + podArgs.vocab_size), rmod_64, (((podArgs.vocab_size >= 0i) && (rmod_64 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_64 <= 0i)))))];
              let rmod_65 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_66 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_29 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_67 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_68 : i32 = (last[0i] % podArgs.vocab_size);
              let rdiv_30 : i32 = (last[0i] / podArgs.vocab_size);
                            let rmod_69 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_70 : i32 = (((select((rmod_69 + podArgs.batch_size), rmod_69, (((podArgs.batch_size >= 0i) && (rmod_69 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_69 <= 0i)))) * podArgs.vocab_size) + last[0i]) % podArgs.vocab_size);
              out_swap_buf[((((select((rmod_65 + podArgs.batch_size), rmod_65, (((podArgs.batch_size >= 0i) && (rmod_65 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_65 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_29 - 1i), rdiv_29, (((podArgs.batch_size >= 0i) && (rmod_66 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_66 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_1)] = out_buf[(((select((rmod_67 + podArgs.batch_size), rmod_67, (((podArgs.batch_size >= 0i) && (rmod_67 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_67 <= 0i)))) + select((rdiv_30 - 1i), rdiv_30, (((podArgs.vocab_size >= 0i) && (rmod_68 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_68 <= 0i))))) * podArgs.vocab_size) + select((rmod_70 + podArgs.vocab_size), rmod_70, (((podArgs.vocab_size >= 0i) && (rmod_70 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_70 <= 0i)))))];
              last[0i] = (last[0i] + 1i);
            }
          }
        }
      } else {
        let rmod_71 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_31 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_72 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_32 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first_1[0i] = max(0i, ((min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_31 - 1i), rdiv_31, (((podArgs.batch_size >= 0i) && (rmod_71 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_71 <= 0i)))))), podArgs.vocab_size) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) - min(((select((rdiv_32 - 1i), rdiv_32, (((podArgs.batch_size >= 0i) && (rmod_72 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_72 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size)));
        let rmod_73 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_33 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last_1[0i] = min((i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u)), min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_33 - 1i), rdiv_33, (((podArgs.batch_size >= 0i) && (rmod_73 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_73 <= 0i))))))));
        while (true) {
          if (!((first_1[0i] < last_1[0i]))) { break; }
          let rmod_74 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_34 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_75 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_76 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_77 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_35 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if (value_swap_buf[((((min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_34 - 1i), rdiv_34, (((podArgs.batch_size >= 0i) && (rmod_74 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_74 <= 0i)))))), podArgs.vocab_size) + (select((rmod_75 + podArgs.batch_size), rmod_75, (((podArgs.batch_size >= 0i) && (rmod_75 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_75 <= 0i)))) * podArgs.vocab_size)) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) - ((first_1[0i] + last_1[0i])>>1u)) - 1i)] <= value_swap_buf[(((select((rmod_76 + podArgs.batch_size), rmod_76, (((podArgs.batch_size >= 0i) && (rmod_76 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_76 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_35 - 1i), rdiv_35, (((podArgs.batch_size >= 0i) && (rmod_77 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_77 <= 0i)))))) + ((first_1[0i] + last_1[0i])>>1u))]) {
            first_1[0i] = (((first_1[0i] + last_1[0i])>>1u) + 1i);
          } else {
            last_1[0i] = ((first_1[0i] + last_1[0i])>>1u);
          }
        }
        let rmod_78 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_36 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first_1[0i] = (((2i<<u32(podArgs.cse_v1)) * select((rdiv_36 - 1i), rdiv_36, (((podArgs.batch_size >= 0i) && (rmod_78 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_78 <= 0i))))) + first_1[0i]);
        let rmod_79 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_37 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last_1[0i] = ((min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_37 - 1i), rdiv_37, (((podArgs.batch_size >= 0i) && (rmod_79 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_79 <= 0i)))))), podArgs.vocab_size) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) - last_1[0i]);
        let rmod_80 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_38 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_81 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_39 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        for (var i_2 : i32 = 0; i_2 < min(((min(((select((rdiv_38 - 1i), rdiv_38, (((podArgs.batch_size >= 0i) && (rmod_80 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_80 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size) - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_39 - 1i), rdiv_39, (((podArgs.batch_size >= 0i) && (rmod_81 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_81 <= 0i)))))) - (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))), (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u)); i_2++) {
          let rmod_82 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_40 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_83 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_41 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if ((((first_1[0i] < (((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_40 - 1i), rdiv_40, (((podArgs.batch_size >= 0i) && (rmod_82 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_82 <= 0i))))))) && (first_1[0i] < podArgs.vocab_size)) && (last_1[0i] < ((select((rdiv_41 - 1i), rdiv_41, (((podArgs.batch_size >= 0i) && (rmod_83 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_83 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))))) && (last_1[0i] < podArgs.vocab_size)) {
            let rmod_84 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rmod_85 : i32 = (last_1[0i] % podArgs.vocab_size);
            let rdiv_42 : i32 = (last_1[0i] / podArgs.vocab_size);
                        let rmod_86 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_87 : i32 = (((select((rmod_86 + podArgs.batch_size), rmod_86, (((podArgs.batch_size >= 0i) && (rmod_86 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_86 <= 0i)))) * podArgs.vocab_size) + last_1[0i]) % podArgs.vocab_size);
            let rmod_88 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rmod_89 : i32 = (first_1[0i] % podArgs.vocab_size);
            let rdiv_43 : i32 = (first_1[0i] / podArgs.vocab_size);
                        let rmod_90 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_91 : i32 = (((select((rmod_90 + podArgs.batch_size), rmod_90, (((podArgs.batch_size >= 0i) && (rmod_90 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_90 <= 0i)))) * podArgs.vocab_size) + first_1[0i]) % podArgs.vocab_size);
            if (value_swap_buf[(((select((rmod_84 + podArgs.batch_size), rmod_84, (((podArgs.batch_size >= 0i) && (rmod_84 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_84 <= 0i)))) + select((rdiv_42 - 1i), rdiv_42, (((podArgs.vocab_size >= 0i) && (rmod_85 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_85 <= 0i))))) * podArgs.vocab_size) + select((rmod_87 + podArgs.vocab_size), rmod_87, (((podArgs.vocab_size >= 0i) && (rmod_87 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_87 <= 0i)))))] <= value_swap_buf[(((select((rmod_88 + podArgs.batch_size), rmod_88, (((podArgs.batch_size >= 0i) && (rmod_88 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_88 <= 0i)))) + select((rdiv_43 - 1i), rdiv_43, (((podArgs.vocab_size >= 0i) && (rmod_89 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_89 <= 0i))))) * podArgs.vocab_size) + select((rmod_91 + podArgs.vocab_size), rmod_91, (((podArgs.vocab_size >= 0i) && (rmod_91 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_91 <= 0i)))))]) {
              let rmod_92 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_93 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_44 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_94 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_95 : i32 = (first_1[0i] % podArgs.vocab_size);
              let rdiv_45 : i32 = (first_1[0i] / podArgs.vocab_size);
                            let rmod_96 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_97 : i32 = (((select((rmod_96 + podArgs.batch_size), rmod_96, (((podArgs.batch_size >= 0i) && (rmod_96 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_96 <= 0i)))) * podArgs.vocab_size) + first_1[0i]) % podArgs.vocab_size);
              value_buf[((((select((rmod_92 + podArgs.batch_size), rmod_92, (((podArgs.batch_size >= 0i) && (rmod_92 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_92 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_44 - 1i), rdiv_44, (((podArgs.batch_size >= 0i) && (rmod_93 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_93 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_2)] = value_swap_buf[(((select((rmod_94 + podArgs.batch_size), rmod_94, (((podArgs.batch_size >= 0i) && (rmod_94 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_94 <= 0i)))) + select((rdiv_45 - 1i), rdiv_45, (((podArgs.vocab_size >= 0i) && (rmod_95 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_95 <= 0i))))) * podArgs.vocab_size) + select((rmod_97 + podArgs.vocab_size), rmod_97, (((podArgs.vocab_size >= 0i) && (rmod_97 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_97 <= 0i)))))];
              let rmod_98 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_99 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_46 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_100 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_101 : i32 = (first_1[0i] % podArgs.vocab_size);
              let rdiv_47 : i32 = (first_1[0i] / podArgs.vocab_size);
                            let rmod_102 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_103 : i32 = (((select((rmod_102 + podArgs.batch_size), rmod_102, (((podArgs.batch_size >= 0i) && (rmod_102 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_102 <= 0i)))) * podArgs.vocab_size) + first_1[0i]) % podArgs.vocab_size);
              out_buf[((((select((rmod_98 + podArgs.batch_size), rmod_98, (((podArgs.batch_size >= 0i) && (rmod_98 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_98 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_46 - 1i), rdiv_46, (((podArgs.batch_size >= 0i) && (rmod_99 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_99 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_2)] = out_swap_buf[(((select((rmod_100 + podArgs.batch_size), rmod_100, (((podArgs.batch_size >= 0i) && (rmod_100 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_100 <= 0i)))) + select((rdiv_47 - 1i), rdiv_47, (((podArgs.vocab_size >= 0i) && (rmod_101 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_101 <= 0i))))) * podArgs.vocab_size) + select((rmod_103 + podArgs.vocab_size), rmod_103, (((podArgs.vocab_size >= 0i) && (rmod_103 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_103 <= 0i)))))];
              first_1[0i] = (first_1[0i] + 1i);
            } else {
              let rmod_104 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_105 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_48 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_106 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_107 : i32 = (last_1[0i] % podArgs.vocab_size);
              let rdiv_49 : i32 = (last_1[0i] / podArgs.vocab_size);
                            let rmod_108 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_109 : i32 = (((select((rmod_108 + podArgs.batch_size), rmod_108, (((podArgs.batch_size >= 0i) && (rmod_108 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_108 <= 0i)))) * podArgs.vocab_size) + last_1[0i]) % podArgs.vocab_size);
              value_buf[((((select((rmod_104 + podArgs.batch_size), rmod_104, (((podArgs.batch_size >= 0i) && (rmod_104 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_104 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_48 - 1i), rdiv_48, (((podArgs.batch_size >= 0i) && (rmod_105 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_105 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_2)] = value_swap_buf[(((select((rmod_106 + podArgs.batch_size), rmod_106, (((podArgs.batch_size >= 0i) && (rmod_106 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_106 <= 0i)))) + select((rdiv_49 - 1i), rdiv_49, (((podArgs.vocab_size >= 0i) && (rmod_107 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_107 <= 0i))))) * podArgs.vocab_size) + select((rmod_109 + podArgs.vocab_size), rmod_109, (((podArgs.vocab_size >= 0i) && (rmod_109 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_109 <= 0i)))))];
              let rmod_110 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_111 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_50 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_112 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_113 : i32 = (last_1[0i] % podArgs.vocab_size);
              let rdiv_51 : i32 = (last_1[0i] / podArgs.vocab_size);
                            let rmod_114 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_115 : i32 = (((select((rmod_114 + podArgs.batch_size), rmod_114, (((podArgs.batch_size >= 0i) && (rmod_114 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_114 <= 0i)))) * podArgs.vocab_size) + last_1[0i]) % podArgs.vocab_size);
              out_buf[((((select((rmod_110 + podArgs.batch_size), rmod_110, (((podArgs.batch_size >= 0i) && (rmod_110 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_110 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_50 - 1i), rdiv_50, (((podArgs.batch_size >= 0i) && (rmod_111 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_111 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_2)] = out_swap_buf[(((select((rmod_112 + podArgs.batch_size), rmod_112, (((podArgs.batch_size >= 0i) && (rmod_112 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_112 <= 0i)))) + select((rdiv_51 - 1i), rdiv_51, (((podArgs.vocab_size >= 0i) && (rmod_113 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_113 <= 0i))))) * podArgs.vocab_size) + select((rmod_115 + podArgs.vocab_size), rmod_115, (((podArgs.vocab_size >= 0i) && (rmod_115 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_115 <= 0i)))))];
              last_1[0i] = (last_1[0i] + 1i);
            }
          } else {
            let rmod_116 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rdiv_52 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
            if ((first_1[0i] < (((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_52 - 1i), rdiv_52, (((podArgs.batch_size >= 0i) && (rmod_116 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_116 <= 0i))))))) && (first_1[0i] < podArgs.vocab_size)) {
              let rmod_117 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_118 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_53 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_119 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_120 : i32 = (first_1[0i] % podArgs.vocab_size);
              let rdiv_54 : i32 = (first_1[0i] / podArgs.vocab_size);
                            let rmod_121 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_122 : i32 = (((select((rmod_121 + podArgs.batch_size), rmod_121, (((podArgs.batch_size >= 0i) && (rmod_121 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_121 <= 0i)))) * podArgs.vocab_size) + first_1[0i]) % podArgs.vocab_size);
              value_buf[((((select((rmod_117 + podArgs.batch_size), rmod_117, (((podArgs.batch_size >= 0i) && (rmod_117 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_117 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_53 - 1i), rdiv_53, (((podArgs.batch_size >= 0i) && (rmod_118 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_118 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_2)] = value_swap_buf[(((select((rmod_119 + podArgs.batch_size), rmod_119, (((podArgs.batch_size >= 0i) && (rmod_119 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_119 <= 0i)))) + select((rdiv_54 - 1i), rdiv_54, (((podArgs.vocab_size >= 0i) && (rmod_120 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_120 <= 0i))))) * podArgs.vocab_size) + select((rmod_122 + podArgs.vocab_size), rmod_122, (((podArgs.vocab_size >= 0i) && (rmod_122 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_122 <= 0i)))))];
              let rmod_123 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_124 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_55 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_125 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_126 : i32 = (first_1[0i] % podArgs.vocab_size);
              let rdiv_56 : i32 = (first_1[0i] / podArgs.vocab_size);
                            let rmod_127 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_128 : i32 = (((select((rmod_127 + podArgs.batch_size), rmod_127, (((podArgs.batch_size >= 0i) && (rmod_127 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_127 <= 0i)))) * podArgs.vocab_size) + first_1[0i]) % podArgs.vocab_size);
              out_buf[((((select((rmod_123 + podArgs.batch_size), rmod_123, (((podArgs.batch_size >= 0i) && (rmod_123 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_123 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_55 - 1i), rdiv_55, (((podArgs.batch_size >= 0i) && (rmod_124 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_124 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_2)] = out_swap_buf[(((select((rmod_125 + podArgs.batch_size), rmod_125, (((podArgs.batch_size >= 0i) && (rmod_125 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_125 <= 0i)))) + select((rdiv_56 - 1i), rdiv_56, (((podArgs.vocab_size >= 0i) && (rmod_126 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_126 <= 0i))))) * podArgs.vocab_size) + select((rmod_128 + podArgs.vocab_size), rmod_128, (((podArgs.vocab_size >= 0i) && (rmod_128 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_128 <= 0i)))))];
              first_1[0i] = (first_1[0i] + 1i);
            } else {
              let rmod_129 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_130 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_57 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_131 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_132 : i32 = (last_1[0i] % podArgs.vocab_size);
              let rdiv_58 : i32 = (last_1[0i] / podArgs.vocab_size);
                            let rmod_133 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_134 : i32 = (((select((rmod_133 + podArgs.batch_size), rmod_133, (((podArgs.batch_size >= 0i) && (rmod_133 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_133 <= 0i)))) * podArgs.vocab_size) + last_1[0i]) % podArgs.vocab_size);
              value_buf[((((select((rmod_129 + podArgs.batch_size), rmod_129, (((podArgs.batch_size >= 0i) && (rmod_129 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_129 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_57 - 1i), rdiv_57, (((podArgs.batch_size >= 0i) && (rmod_130 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_130 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_2)] = value_swap_buf[(((select((rmod_131 + podArgs.batch_size), rmod_131, (((podArgs.batch_size >= 0i) && (rmod_131 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_131 <= 0i)))) + select((rdiv_58 - 1i), rdiv_58, (((podArgs.vocab_size >= 0i) && (rmod_132 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_132 <= 0i))))) * podArgs.vocab_size) + select((rmod_134 + podArgs.vocab_size), rmod_134, (((podArgs.vocab_size >= 0i) && (rmod_134 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_134 <= 0i)))))];
              let rmod_135 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_136 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_59 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_137 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_138 : i32 = (last_1[0i] % podArgs.vocab_size);
              let rdiv_60 : i32 = (last_1[0i] / podArgs.vocab_size);
                            let rmod_139 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_140 : i32 = (((select((rmod_139 + podArgs.batch_size), rmod_139, (((podArgs.batch_size >= 0i) && (rmod_139 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_139 <= 0i)))) * podArgs.vocab_size) + last_1[0i]) % podArgs.vocab_size);
              out_buf[((((select((rmod_135 + podArgs.batch_size), rmod_135, (((podArgs.batch_size >= 0i) && (rmod_135 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_135 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_59 - 1i), rdiv_59, (((podArgs.batch_size >= 0i) && (rmod_136 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_136 <= 0i)))))) + (i32(threadIdx.x) * (((2i<<u32(podArgs.cse_v1)) + 255i)>>8u))) + i_2)] = out_swap_buf[(((select((rmod_137 + podArgs.batch_size), rmod_137, (((podArgs.batch_size >= 0i) && (rmod_137 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_137 <= 0i)))) + select((rdiv_60 - 1i), rdiv_60, (((podArgs.vocab_size >= 0i) && (rmod_138 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_138 <= 0i))))) * podArgs.vocab_size) + select((rmod_140 + podArgs.vocab_size), rmod_140, (((podArgs.vocab_size >= 0i) && (rmod_140 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_140 <= 0i)))))];
              last_1[0i] = (last_1[0i] + 1i);
            }
          }
        }
      }
    } else {
      if ((podArgs.i_0 % 2i) == 0i) {
        let rmod_141 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_61 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_142 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_62 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first_2[0i] = max(0i, (((v__1 * 1024i) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_61 - 1i), rdiv_61, (((podArgs.batch_size >= 0i) && (rmod_141 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_141 <= 0i)))))), podArgs.vocab_size)) - min(((select((rdiv_62 - 1i), rdiv_62, (((podArgs.batch_size >= 0i) && (rmod_142 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_142 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size)));
        let rmod_143 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_63 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last_2[0i] = min((v__1 * 1024i), min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_63 - 1i), rdiv_63, (((podArgs.batch_size >= 0i) && (rmod_143 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_143 <= 0i))))))));
        while (true) {
          if (!((first_2[0i] < last_2[0i]))) { break; }
          let rmod_144 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_64 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_145 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_146 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_147 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_65 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if (value_buf[(((((v__1 * 1024i) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_64 - 1i), rdiv_64, (((podArgs.batch_size >= 0i) && (rmod_144 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_144 <= 0i)))))), podArgs.vocab_size)) + (select((rmod_145 + podArgs.batch_size), rmod_145, (((podArgs.batch_size >= 0i) && (rmod_145 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_145 <= 0i)))) * podArgs.vocab_size)) - ((first_2[0i] + last_2[0i])>>1u)) - 1i)] <= value_buf[(((select((rmod_146 + podArgs.batch_size), rmod_146, (((podArgs.batch_size >= 0i) && (rmod_146 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_146 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_65 - 1i), rdiv_65, (((podArgs.batch_size >= 0i) && (rmod_147 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_147 <= 0i)))))) + ((first_2[0i] + last_2[0i])>>1u))]) {
            first_2[0i] = (((first_2[0i] + last_2[0i])>>1u) + 1i);
          } else {
            last_2[0i] = ((first_2[0i] + last_2[0i])>>1u);
          }
        }
        let rmod_148 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_66 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_149 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_67 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first_3[0i] = max(0i, ((i32(threadIdx.x) * 4i) - min((((min(((select((rdiv_66 - 1i), rdiv_66, (((podArgs.batch_size >= 0i) && (rmod_148 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_148 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size) + last_2[0i]) - min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_67 - 1i), rdiv_67, (((podArgs.batch_size >= 0i) && (rmod_149 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_149 <= 0i)))))), podArgs.vocab_size)) - (v__1 * 1024i)), 1024i)));
        let rmod_150 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_68 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last_3[0i] = min((i32(threadIdx.x) * 4i), min((min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_68 - 1i), rdiv_68, (((podArgs.batch_size >= 0i) && (rmod_150 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_150 <= 0i))))))) - first_2[0i]), 1024i));
        while (true) {
          if (!((first_3[0i] < last_3[0i]))) { break; }
          let rmod_151 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_69 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_152 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_153 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_154 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_70 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if (value_buf[(((((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_69 - 1i), rdiv_69, (((podArgs.batch_size >= 0i) && (rmod_151 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_151 <= 0i)))))), podArgs.vocab_size)) + (select((rmod_152 + podArgs.batch_size), rmod_152, (((podArgs.batch_size >= 0i) && (rmod_152 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_152 <= 0i)))) * podArgs.vocab_size)) - last_2[0i]) - ((first_3[0i] + last_3[0i])>>1u)) - 1i)] <= value_buf[((((select((rmod_153 + podArgs.batch_size), rmod_153, (((podArgs.batch_size >= 0i) && (rmod_153 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_153 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_70 - 1i), rdiv_70, (((podArgs.batch_size >= 0i) && (rmod_154 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_154 <= 0i)))))) + first_2[0i]) + ((first_3[0i] + last_3[0i])>>1u))]) {
            first_3[0i] = (((first_3[0i] + last_3[0i])>>1u) + 1i);
          } else {
            last_3[0i] = ((first_3[0i] + last_3[0i])>>1u);
          }
        }
        let rmod_155 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_71 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first_3[0i] = ((((2i<<u32(podArgs.cse_v1)) * select((rdiv_71 - 1i), rdiv_71, (((podArgs.batch_size >= 0i) && (rmod_155 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_155 <= 0i))))) + first_2[0i]) + first_3[0i]);
        let rmod_156 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_72 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last_3[0i] = (((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_72 - 1i), rdiv_72, (((podArgs.batch_size >= 0i) && (rmod_156 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_156 <= 0i)))))), podArgs.vocab_size)) - last_2[0i]) - last_3[0i]);
        let rmod_157 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_73 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_158 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_74 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_159 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_75 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        for (var i_3 : i32 = 0; i_3 < min(((min((min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_73 - 1i), rdiv_73, (((podArgs.batch_size >= 0i) && (rmod_157 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_157 <= 0i))))))) - first_2[0i]), 1024i) + min((((min(((select((rdiv_74 - 1i), rdiv_74, (((podArgs.batch_size >= 0i) && (rmod_158 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_158 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size) + last_2[0i]) - min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_75 - 1i), rdiv_75, (((podArgs.batch_size >= 0i) && (rmod_159 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_159 <= 0i)))))), podArgs.vocab_size)) - (v__1 * 1024i)), 1024i)) - (i32(threadIdx.x) * 4i)), 4i); i_3++) {
          let rmod_160 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_76 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_161 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_77 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_162 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_78 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_163 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_79 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if ((((first_3[0i] < ((min((min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_76 - 1i), rdiv_76, (((podArgs.batch_size >= 0i) && (rmod_160 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_160 <= 0i))))))) - first_2[0i]), 1024i) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_77 - 1i), rdiv_77, (((podArgs.batch_size >= 0i) && (rmod_161 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_161 <= 0i)))))) + first_2[0i])) && (last_3[0i] < ((((v__1 * 1024i) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_78 - 1i), rdiv_78, (((podArgs.batch_size >= 0i) && (rmod_162 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_162 <= 0i)))))), podArgs.vocab_size)) + 1024i) - last_2[0i]))) && (last_3[0i] < ((select((rdiv_79 - 1i), rdiv_79, (((podArgs.batch_size >= 0i) && (rmod_163 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_163 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))))) && (last_3[0i] < podArgs.vocab_size)) {
            let rmod_164 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rmod_165 : i32 = (last_3[0i] % podArgs.vocab_size);
            let rdiv_80 : i32 = (last_3[0i] / podArgs.vocab_size);
                        let rmod_166 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_167 : i32 = (((select((rmod_166 + podArgs.batch_size), rmod_166, (((podArgs.batch_size >= 0i) && (rmod_166 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_166 <= 0i)))) * podArgs.vocab_size) + last_3[0i]) % podArgs.vocab_size);
            let rmod_168 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rmod_169 : i32 = (first_3[0i] % podArgs.vocab_size);
            let rdiv_81 : i32 = (first_3[0i] / podArgs.vocab_size);
                        let rmod_170 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_171 : i32 = (((select((rmod_170 + podArgs.batch_size), rmod_170, (((podArgs.batch_size >= 0i) && (rmod_170 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_170 <= 0i)))) * podArgs.vocab_size) + first_3[0i]) % podArgs.vocab_size);
            if (value_buf[(((select((rmod_164 + podArgs.batch_size), rmod_164, (((podArgs.batch_size >= 0i) && (rmod_164 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_164 <= 0i)))) + select((rdiv_80 - 1i), rdiv_80, (((podArgs.vocab_size >= 0i) && (rmod_165 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_165 <= 0i))))) * podArgs.vocab_size) + select((rmod_167 + podArgs.vocab_size), rmod_167, (((podArgs.vocab_size >= 0i) && (rmod_167 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_167 <= 0i)))))] <= value_buf[(((select((rmod_168 + podArgs.batch_size), rmod_168, (((podArgs.batch_size >= 0i) && (rmod_168 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_168 <= 0i)))) + select((rdiv_81 - 1i), rdiv_81, (((podArgs.vocab_size >= 0i) && (rmod_169 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_169 <= 0i))))) * podArgs.vocab_size) + select((rmod_171 + podArgs.vocab_size), rmod_171, (((podArgs.vocab_size >= 0i) && (rmod_171 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_171 <= 0i)))))]) {
              let rmod_172 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_173 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_82 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_174 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_175 : i32 = (first_3[0i] % podArgs.vocab_size);
              let rdiv_83 : i32 = (first_3[0i] / podArgs.vocab_size);
                            let rmod_176 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_177 : i32 = (((select((rmod_176 + podArgs.batch_size), rmod_176, (((podArgs.batch_size >= 0i) && (rmod_176 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_176 <= 0i)))) * podArgs.vocab_size) + first_3[0i]) % podArgs.vocab_size);
              value_swap_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_172 + podArgs.batch_size), rmod_172, (((podArgs.batch_size >= 0i) && (rmod_172 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_172 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_82 - 1i), rdiv_82, (((podArgs.batch_size >= 0i) && (rmod_173 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_173 <= 0i)))))) + i_3)] = value_buf[(((select((rmod_174 + podArgs.batch_size), rmod_174, (((podArgs.batch_size >= 0i) && (rmod_174 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_174 <= 0i)))) + select((rdiv_83 - 1i), rdiv_83, (((podArgs.vocab_size >= 0i) && (rmod_175 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_175 <= 0i))))) * podArgs.vocab_size) + select((rmod_177 + podArgs.vocab_size), rmod_177, (((podArgs.vocab_size >= 0i) && (rmod_177 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_177 <= 0i)))))];
              let rmod_178 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_179 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_84 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_180 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_181 : i32 = (first_3[0i] % podArgs.vocab_size);
              let rdiv_85 : i32 = (first_3[0i] / podArgs.vocab_size);
                            let rmod_182 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_183 : i32 = (((select((rmod_182 + podArgs.batch_size), rmod_182, (((podArgs.batch_size >= 0i) && (rmod_182 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_182 <= 0i)))) * podArgs.vocab_size) + first_3[0i]) % podArgs.vocab_size);
              out_swap_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_178 + podArgs.batch_size), rmod_178, (((podArgs.batch_size >= 0i) && (rmod_178 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_178 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_84 - 1i), rdiv_84, (((podArgs.batch_size >= 0i) && (rmod_179 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_179 <= 0i)))))) + i_3)] = out_buf[(((select((rmod_180 + podArgs.batch_size), rmod_180, (((podArgs.batch_size >= 0i) && (rmod_180 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_180 <= 0i)))) + select((rdiv_85 - 1i), rdiv_85, (((podArgs.vocab_size >= 0i) && (rmod_181 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_181 <= 0i))))) * podArgs.vocab_size) + select((rmod_183 + podArgs.vocab_size), rmod_183, (((podArgs.vocab_size >= 0i) && (rmod_183 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_183 <= 0i)))))];
              first_3[0i] = (first_3[0i] + 1i);
            } else {
              let rmod_184 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_185 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_86 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_186 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_187 : i32 = (last_3[0i] % podArgs.vocab_size);
              let rdiv_87 : i32 = (last_3[0i] / podArgs.vocab_size);
                            let rmod_188 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_189 : i32 = (((select((rmod_188 + podArgs.batch_size), rmod_188, (((podArgs.batch_size >= 0i) && (rmod_188 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_188 <= 0i)))) * podArgs.vocab_size) + last_3[0i]) % podArgs.vocab_size);
              value_swap_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_184 + podArgs.batch_size), rmod_184, (((podArgs.batch_size >= 0i) && (rmod_184 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_184 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_86 - 1i), rdiv_86, (((podArgs.batch_size >= 0i) && (rmod_185 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_185 <= 0i)))))) + i_3)] = value_buf[(((select((rmod_186 + podArgs.batch_size), rmod_186, (((podArgs.batch_size >= 0i) && (rmod_186 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_186 <= 0i)))) + select((rdiv_87 - 1i), rdiv_87, (((podArgs.vocab_size >= 0i) && (rmod_187 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_187 <= 0i))))) * podArgs.vocab_size) + select((rmod_189 + podArgs.vocab_size), rmod_189, (((podArgs.vocab_size >= 0i) && (rmod_189 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_189 <= 0i)))))];
              let rmod_190 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_191 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_88 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_192 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_193 : i32 = (last_3[0i] % podArgs.vocab_size);
              let rdiv_89 : i32 = (last_3[0i] / podArgs.vocab_size);
                            let rmod_194 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_195 : i32 = (((select((rmod_194 + podArgs.batch_size), rmod_194, (((podArgs.batch_size >= 0i) && (rmod_194 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_194 <= 0i)))) * podArgs.vocab_size) + last_3[0i]) % podArgs.vocab_size);
              out_swap_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_190 + podArgs.batch_size), rmod_190, (((podArgs.batch_size >= 0i) && (rmod_190 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_190 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_88 - 1i), rdiv_88, (((podArgs.batch_size >= 0i) && (rmod_191 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_191 <= 0i)))))) + i_3)] = out_buf[(((select((rmod_192 + podArgs.batch_size), rmod_192, (((podArgs.batch_size >= 0i) && (rmod_192 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_192 <= 0i)))) + select((rdiv_89 - 1i), rdiv_89, (((podArgs.vocab_size >= 0i) && (rmod_193 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_193 <= 0i))))) * podArgs.vocab_size) + select((rmod_195 + podArgs.vocab_size), rmod_195, (((podArgs.vocab_size >= 0i) && (rmod_195 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_195 <= 0i)))))];
              last_3[0i] = (last_3[0i] + 1i);
            }
          } else {
            let rmod_196 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rdiv_90 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
            let rmod_197 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rdiv_91 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
            if (first_3[0i] < ((min((min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_90 - 1i), rdiv_90, (((podArgs.batch_size >= 0i) && (rmod_196 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_196 <= 0i))))))) - first_2[0i]), 1024i) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_91 - 1i), rdiv_91, (((podArgs.batch_size >= 0i) && (rmod_197 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_197 <= 0i)))))) + first_2[0i])) {
              let rmod_198 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_199 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_92 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_200 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_201 : i32 = (first_3[0i] % podArgs.vocab_size);
              let rdiv_93 : i32 = (first_3[0i] / podArgs.vocab_size);
                            let rmod_202 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_203 : i32 = (((select((rmod_202 + podArgs.batch_size), rmod_202, (((podArgs.batch_size >= 0i) && (rmod_202 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_202 <= 0i)))) * podArgs.vocab_size) + first_3[0i]) % podArgs.vocab_size);
              value_swap_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_198 + podArgs.batch_size), rmod_198, (((podArgs.batch_size >= 0i) && (rmod_198 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_198 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_92 - 1i), rdiv_92, (((podArgs.batch_size >= 0i) && (rmod_199 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_199 <= 0i)))))) + i_3)] = value_buf[(((select((rmod_200 + podArgs.batch_size), rmod_200, (((podArgs.batch_size >= 0i) && (rmod_200 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_200 <= 0i)))) + select((rdiv_93 - 1i), rdiv_93, (((podArgs.vocab_size >= 0i) && (rmod_201 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_201 <= 0i))))) * podArgs.vocab_size) + select((rmod_203 + podArgs.vocab_size), rmod_203, (((podArgs.vocab_size >= 0i) && (rmod_203 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_203 <= 0i)))))];
              let rmod_204 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_205 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_94 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_206 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_207 : i32 = (first_3[0i] % podArgs.vocab_size);
              let rdiv_95 : i32 = (first_3[0i] / podArgs.vocab_size);
                            let rmod_208 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_209 : i32 = (((select((rmod_208 + podArgs.batch_size), rmod_208, (((podArgs.batch_size >= 0i) && (rmod_208 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_208 <= 0i)))) * podArgs.vocab_size) + first_3[0i]) % podArgs.vocab_size);
              out_swap_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_204 + podArgs.batch_size), rmod_204, (((podArgs.batch_size >= 0i) && (rmod_204 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_204 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_94 - 1i), rdiv_94, (((podArgs.batch_size >= 0i) && (rmod_205 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_205 <= 0i)))))) + i_3)] = out_buf[(((select((rmod_206 + podArgs.batch_size), rmod_206, (((podArgs.batch_size >= 0i) && (rmod_206 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_206 <= 0i)))) + select((rdiv_95 - 1i), rdiv_95, (((podArgs.vocab_size >= 0i) && (rmod_207 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_207 <= 0i))))) * podArgs.vocab_size) + select((rmod_209 + podArgs.vocab_size), rmod_209, (((podArgs.vocab_size >= 0i) && (rmod_209 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_209 <= 0i)))))];
              first_3[0i] = (first_3[0i] + 1i);
            } else {
              let rmod_210 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_211 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_96 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_212 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_213 : i32 = (last_3[0i] % podArgs.vocab_size);
              let rdiv_97 : i32 = (last_3[0i] / podArgs.vocab_size);
                            let rmod_214 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_215 : i32 = (((select((rmod_214 + podArgs.batch_size), rmod_214, (((podArgs.batch_size >= 0i) && (rmod_214 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_214 <= 0i)))) * podArgs.vocab_size) + last_3[0i]) % podArgs.vocab_size);
              value_swap_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_210 + podArgs.batch_size), rmod_210, (((podArgs.batch_size >= 0i) && (rmod_210 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_210 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_96 - 1i), rdiv_96, (((podArgs.batch_size >= 0i) && (rmod_211 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_211 <= 0i)))))) + i_3)] = value_buf[(((select((rmod_212 + podArgs.batch_size), rmod_212, (((podArgs.batch_size >= 0i) && (rmod_212 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_212 <= 0i)))) + select((rdiv_97 - 1i), rdiv_97, (((podArgs.vocab_size >= 0i) && (rmod_213 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_213 <= 0i))))) * podArgs.vocab_size) + select((rmod_215 + podArgs.vocab_size), rmod_215, (((podArgs.vocab_size >= 0i) && (rmod_215 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_215 <= 0i)))))];
              let rmod_216 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_217 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_98 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_218 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_219 : i32 = (last_3[0i] % podArgs.vocab_size);
              let rdiv_99 : i32 = (last_3[0i] / podArgs.vocab_size);
                            let rmod_220 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_221 : i32 = (((select((rmod_220 + podArgs.batch_size), rmod_220, (((podArgs.batch_size >= 0i) && (rmod_220 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_220 <= 0i)))) * podArgs.vocab_size) + last_3[0i]) % podArgs.vocab_size);
              out_swap_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_216 + podArgs.batch_size), rmod_216, (((podArgs.batch_size >= 0i) && (rmod_216 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_216 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_98 - 1i), rdiv_98, (((podArgs.batch_size >= 0i) && (rmod_217 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_217 <= 0i)))))) + i_3)] = out_buf[(((select((rmod_218 + podArgs.batch_size), rmod_218, (((podArgs.batch_size >= 0i) && (rmod_218 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_218 <= 0i)))) + select((rdiv_99 - 1i), rdiv_99, (((podArgs.vocab_size >= 0i) && (rmod_219 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_219 <= 0i))))) * podArgs.vocab_size) + select((rmod_221 + podArgs.vocab_size), rmod_221, (((podArgs.vocab_size >= 0i) && (rmod_221 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_221 <= 0i)))))];
              last_3[0i] = (last_3[0i] + 1i);
            }
          }
        }
      } else {
        let rmod_222 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_100 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_223 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_101 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first_4[0i] = max(0i, (((v__1 * 1024i) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_100 - 1i), rdiv_100, (((podArgs.batch_size >= 0i) && (rmod_222 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_222 <= 0i)))))), podArgs.vocab_size)) - min(((select((rdiv_101 - 1i), rdiv_101, (((podArgs.batch_size >= 0i) && (rmod_223 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_223 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size)));
        let rmod_224 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_102 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last_4[0i] = min((v__1 * 1024i), min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_102 - 1i), rdiv_102, (((podArgs.batch_size >= 0i) && (rmod_224 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_224 <= 0i))))))));
        while (true) {
          if (!((first_4[0i] < last_4[0i]))) { break; }
          let rmod_225 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_103 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_226 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_227 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_228 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_104 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if (value_swap_buf[(((((v__1 * 1024i) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_103 - 1i), rdiv_103, (((podArgs.batch_size >= 0i) && (rmod_225 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_225 <= 0i)))))), podArgs.vocab_size)) + (select((rmod_226 + podArgs.batch_size), rmod_226, (((podArgs.batch_size >= 0i) && (rmod_226 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_226 <= 0i)))) * podArgs.vocab_size)) - ((first_4[0i] + last_4[0i])>>1u)) - 1i)] <= value_swap_buf[(((select((rmod_227 + podArgs.batch_size), rmod_227, (((podArgs.batch_size >= 0i) && (rmod_227 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_227 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_104 - 1i), rdiv_104, (((podArgs.batch_size >= 0i) && (rmod_228 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_228 <= 0i)))))) + ((first_4[0i] + last_4[0i])>>1u))]) {
            first_4[0i] = (((first_4[0i] + last_4[0i])>>1u) + 1i);
          } else {
            last_4[0i] = ((first_4[0i] + last_4[0i])>>1u);
          }
        }
        let rmod_229 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_105 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_230 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_106 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first_5[0i] = max(0i, ((i32(threadIdx.x) * 4i) - min((((min(((select((rdiv_105 - 1i), rdiv_105, (((podArgs.batch_size >= 0i) && (rmod_229 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_229 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size) + last_4[0i]) - min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_106 - 1i), rdiv_106, (((podArgs.batch_size >= 0i) && (rmod_230 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_230 <= 0i)))))), podArgs.vocab_size)) - (v__1 * 1024i)), 1024i)));
        let rmod_231 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_107 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last_5[0i] = min((i32(threadIdx.x) * 4i), min((min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_107 - 1i), rdiv_107, (((podArgs.batch_size >= 0i) && (rmod_231 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_231 <= 0i))))))) - first_4[0i]), 1024i));
        while (true) {
          if (!((first_5[0i] < last_5[0i]))) { break; }
          let rmod_232 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_108 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_233 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_234 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rmod_235 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_109 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if (value_swap_buf[(((((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_108 - 1i), rdiv_108, (((podArgs.batch_size >= 0i) && (rmod_232 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_232 <= 0i)))))), podArgs.vocab_size)) + (select((rmod_233 + podArgs.batch_size), rmod_233, (((podArgs.batch_size >= 0i) && (rmod_233 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_233 <= 0i)))) * podArgs.vocab_size)) - last_4[0i]) - ((first_5[0i] + last_5[0i])>>1u)) - 1i)] <= value_swap_buf[((((select((rmod_234 + podArgs.batch_size), rmod_234, (((podArgs.batch_size >= 0i) && (rmod_234 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_234 <= 0i)))) * podArgs.vocab_size) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_109 - 1i), rdiv_109, (((podArgs.batch_size >= 0i) && (rmod_235 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_235 <= 0i)))))) + first_4[0i]) + ((first_5[0i] + last_5[0i])>>1u))]) {
            first_5[0i] = (((first_5[0i] + last_5[0i])>>1u) + 1i);
          } else {
            last_5[0i] = ((first_5[0i] + last_5[0i])>>1u);
          }
        }
        let rmod_236 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_110 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        first_5[0i] = ((((2i<<u32(podArgs.cse_v1)) * select((rdiv_110 - 1i), rdiv_110, (((podArgs.batch_size >= 0i) && (rmod_236 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_236 <= 0i))))) + first_4[0i]) + first_5[0i]);
        let rmod_237 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_111 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        last_5[0i] = (((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_111 - 1i), rdiv_111, (((podArgs.batch_size >= 0i) && (rmod_237 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_237 <= 0i)))))), podArgs.vocab_size)) - last_4[0i]) - last_5[0i]);
        let rmod_238 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_112 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_239 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_113 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        let rmod_240 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
        let rdiv_114 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
        for (var i_6 : i32 = 0; i_6 < min(((min((min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_112 - 1i), rdiv_112, (((podArgs.batch_size >= 0i) && (rmod_238 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_238 <= 0i))))))) - first_4[0i]), 1024i) + min((((min(((select((rdiv_113 - 1i), rdiv_113, (((podArgs.batch_size >= 0i) && (rmod_239 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_239 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))), podArgs.vocab_size) + last_4[0i]) - min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_114 - 1i), rdiv_114, (((podArgs.batch_size >= 0i) && (rmod_240 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_240 <= 0i)))))), podArgs.vocab_size)) - (v__1 * 1024i)), 1024i)) - (i32(threadIdx.x) * 4i)), 4i); i_6++) {
          let rmod_241 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_115 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_242 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_116 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_243 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_117 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          let rmod_244 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
          let rdiv_118 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
          if ((((first_5[0i] < ((min((min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_115 - 1i), rdiv_115, (((podArgs.batch_size >= 0i) && (rmod_241 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_241 <= 0i))))))) - first_4[0i]), 1024i) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_116 - 1i), rdiv_116, (((podArgs.batch_size >= 0i) && (rmod_242 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_242 <= 0i)))))) + first_4[0i])) && (last_5[0i] < ((((v__1 * 1024i) + min((((2i<<u32(podArgs.cse_v1))>>1u) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_117 - 1i), rdiv_117, (((podArgs.batch_size >= 0i) && (rmod_243 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_243 <= 0i)))))), podArgs.vocab_size)) + 1024i) - last_4[0i]))) && (last_5[0i] < ((select((rdiv_118 - 1i), rdiv_118, (((podArgs.batch_size >= 0i) && (rmod_244 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_244 <= 0i)))) + 1i) * (2i<<u32(podArgs.cse_v1))))) && (last_5[0i] < podArgs.vocab_size)) {
            let rmod_245 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rmod_246 : i32 = (last_5[0i] % podArgs.vocab_size);
            let rdiv_119 : i32 = (last_5[0i] / podArgs.vocab_size);
                        let rmod_247 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_248 : i32 = (((select((rmod_247 + podArgs.batch_size), rmod_247, (((podArgs.batch_size >= 0i) && (rmod_247 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_247 <= 0i)))) * podArgs.vocab_size) + last_5[0i]) % podArgs.vocab_size);
            let rmod_249 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rmod_250 : i32 = (first_5[0i] % podArgs.vocab_size);
            let rdiv_120 : i32 = (first_5[0i] / podArgs.vocab_size);
                        let rmod_251 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_252 : i32 = (((select((rmod_251 + podArgs.batch_size), rmod_251, (((podArgs.batch_size >= 0i) && (rmod_251 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_251 <= 0i)))) * podArgs.vocab_size) + first_5[0i]) % podArgs.vocab_size);
            if (value_swap_buf[(((select((rmod_245 + podArgs.batch_size), rmod_245, (((podArgs.batch_size >= 0i) && (rmod_245 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_245 <= 0i)))) + select((rdiv_119 - 1i), rdiv_119, (((podArgs.vocab_size >= 0i) && (rmod_246 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_246 <= 0i))))) * podArgs.vocab_size) + select((rmod_248 + podArgs.vocab_size), rmod_248, (((podArgs.vocab_size >= 0i) && (rmod_248 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_248 <= 0i)))))] <= value_swap_buf[(((select((rmod_249 + podArgs.batch_size), rmod_249, (((podArgs.batch_size >= 0i) && (rmod_249 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_249 <= 0i)))) + select((rdiv_120 - 1i), rdiv_120, (((podArgs.vocab_size >= 0i) && (rmod_250 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_250 <= 0i))))) * podArgs.vocab_size) + select((rmod_252 + podArgs.vocab_size), rmod_252, (((podArgs.vocab_size >= 0i) && (rmod_252 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_252 <= 0i)))))]) {
              let rmod_253 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_254 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_121 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_255 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_256 : i32 = (first_5[0i] % podArgs.vocab_size);
              let rdiv_122 : i32 = (first_5[0i] / podArgs.vocab_size);
                            let rmod_257 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_258 : i32 = (((select((rmod_257 + podArgs.batch_size), rmod_257, (((podArgs.batch_size >= 0i) && (rmod_257 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_257 <= 0i)))) * podArgs.vocab_size) + first_5[0i]) % podArgs.vocab_size);
              value_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_253 + podArgs.batch_size), rmod_253, (((podArgs.batch_size >= 0i) && (rmod_253 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_253 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_121 - 1i), rdiv_121, (((podArgs.batch_size >= 0i) && (rmod_254 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_254 <= 0i)))))) + i_6)] = value_swap_buf[(((select((rmod_255 + podArgs.batch_size), rmod_255, (((podArgs.batch_size >= 0i) && (rmod_255 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_255 <= 0i)))) + select((rdiv_122 - 1i), rdiv_122, (((podArgs.vocab_size >= 0i) && (rmod_256 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_256 <= 0i))))) * podArgs.vocab_size) + select((rmod_258 + podArgs.vocab_size), rmod_258, (((podArgs.vocab_size >= 0i) && (rmod_258 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_258 <= 0i)))))];
              let rmod_259 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_260 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_123 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_261 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_262 : i32 = (first_5[0i] % podArgs.vocab_size);
              let rdiv_124 : i32 = (first_5[0i] / podArgs.vocab_size);
                            let rmod_263 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_264 : i32 = (((select((rmod_263 + podArgs.batch_size), rmod_263, (((podArgs.batch_size >= 0i) && (rmod_263 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_263 <= 0i)))) * podArgs.vocab_size) + first_5[0i]) % podArgs.vocab_size);
              out_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_259 + podArgs.batch_size), rmod_259, (((podArgs.batch_size >= 0i) && (rmod_259 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_259 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_123 - 1i), rdiv_123, (((podArgs.batch_size >= 0i) && (rmod_260 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_260 <= 0i)))))) + i_6)] = out_swap_buf[(((select((rmod_261 + podArgs.batch_size), rmod_261, (((podArgs.batch_size >= 0i) && (rmod_261 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_261 <= 0i)))) + select((rdiv_124 - 1i), rdiv_124, (((podArgs.vocab_size >= 0i) && (rmod_262 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_262 <= 0i))))) * podArgs.vocab_size) + select((rmod_264 + podArgs.vocab_size), rmod_264, (((podArgs.vocab_size >= 0i) && (rmod_264 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_264 <= 0i)))))];
              first_5[0i] = (first_5[0i] + 1i);
            } else {
              let rmod_265 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_266 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_125 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_267 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_268 : i32 = (last_5[0i] % podArgs.vocab_size);
              let rdiv_126 : i32 = (last_5[0i] / podArgs.vocab_size);
                            let rmod_269 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_270 : i32 = (((select((rmod_269 + podArgs.batch_size), rmod_269, (((podArgs.batch_size >= 0i) && (rmod_269 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_269 <= 0i)))) * podArgs.vocab_size) + last_5[0i]) % podArgs.vocab_size);
              value_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_265 + podArgs.batch_size), rmod_265, (((podArgs.batch_size >= 0i) && (rmod_265 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_265 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_125 - 1i), rdiv_125, (((podArgs.batch_size >= 0i) && (rmod_266 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_266 <= 0i)))))) + i_6)] = value_swap_buf[(((select((rmod_267 + podArgs.batch_size), rmod_267, (((podArgs.batch_size >= 0i) && (rmod_267 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_267 <= 0i)))) + select((rdiv_126 - 1i), rdiv_126, (((podArgs.vocab_size >= 0i) && (rmod_268 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_268 <= 0i))))) * podArgs.vocab_size) + select((rmod_270 + podArgs.vocab_size), rmod_270, (((podArgs.vocab_size >= 0i) && (rmod_270 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_270 <= 0i)))))];
              let rmod_271 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_272 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_127 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_273 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_274 : i32 = (last_5[0i] % podArgs.vocab_size);
              let rdiv_128 : i32 = (last_5[0i] / podArgs.vocab_size);
                            let rmod_275 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_276 : i32 = (((select((rmod_275 + podArgs.batch_size), rmod_275, (((podArgs.batch_size >= 0i) && (rmod_275 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_275 <= 0i)))) * podArgs.vocab_size) + last_5[0i]) % podArgs.vocab_size);
              out_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_271 + podArgs.batch_size), rmod_271, (((podArgs.batch_size >= 0i) && (rmod_271 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_271 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_127 - 1i), rdiv_127, (((podArgs.batch_size >= 0i) && (rmod_272 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_272 <= 0i)))))) + i_6)] = out_swap_buf[(((select((rmod_273 + podArgs.batch_size), rmod_273, (((podArgs.batch_size >= 0i) && (rmod_273 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_273 <= 0i)))) + select((rdiv_128 - 1i), rdiv_128, (((podArgs.vocab_size >= 0i) && (rmod_274 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_274 <= 0i))))) * podArgs.vocab_size) + select((rmod_276 + podArgs.vocab_size), rmod_276, (((podArgs.vocab_size >= 0i) && (rmod_276 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_276 <= 0i)))))];
              last_5[0i] = (last_5[0i] + 1i);
            }
          } else {
            let rmod_277 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rdiv_129 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
            let rmod_278 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
            let rdiv_130 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
            if (first_5[0i] < ((min((min(((2i<<u32(podArgs.cse_v1))>>1u), (podArgs.vocab_size - ((2i<<u32(podArgs.cse_v1)) * select((rdiv_129 - 1i), rdiv_129, (((podArgs.batch_size >= 0i) && (rmod_277 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_277 <= 0i))))))) - first_4[0i]), 1024i) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_130 - 1i), rdiv_130, (((podArgs.batch_size >= 0i) && (rmod_278 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_278 <= 0i)))))) + first_4[0i])) {
              let rmod_279 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_280 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_131 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_281 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_282 : i32 = (first_5[0i] % podArgs.vocab_size);
              let rdiv_132 : i32 = (first_5[0i] / podArgs.vocab_size);
                            let rmod_283 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_284 : i32 = (((select((rmod_283 + podArgs.batch_size), rmod_283, (((podArgs.batch_size >= 0i) && (rmod_283 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_283 <= 0i)))) * podArgs.vocab_size) + first_5[0i]) % podArgs.vocab_size);
              value_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_279 + podArgs.batch_size), rmod_279, (((podArgs.batch_size >= 0i) && (rmod_279 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_279 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_131 - 1i), rdiv_131, (((podArgs.batch_size >= 0i) && (rmod_280 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_280 <= 0i)))))) + i_6)] = value_swap_buf[(((select((rmod_281 + podArgs.batch_size), rmod_281, (((podArgs.batch_size >= 0i) && (rmod_281 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_281 <= 0i)))) + select((rdiv_132 - 1i), rdiv_132, (((podArgs.vocab_size >= 0i) && (rmod_282 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_282 <= 0i))))) * podArgs.vocab_size) + select((rmod_284 + podArgs.vocab_size), rmod_284, (((podArgs.vocab_size >= 0i) && (rmod_284 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_284 <= 0i)))))];
              let rmod_285 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_286 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_133 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_287 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_288 : i32 = (first_5[0i] % podArgs.vocab_size);
              let rdiv_134 : i32 = (first_5[0i] / podArgs.vocab_size);
                            let rmod_289 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_290 : i32 = (((select((rmod_289 + podArgs.batch_size), rmod_289, (((podArgs.batch_size >= 0i) && (rmod_289 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_289 <= 0i)))) * podArgs.vocab_size) + first_5[0i]) % podArgs.vocab_size);
              out_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_285 + podArgs.batch_size), rmod_285, (((podArgs.batch_size >= 0i) && (rmod_285 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_285 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_133 - 1i), rdiv_133, (((podArgs.batch_size >= 0i) && (rmod_286 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_286 <= 0i)))))) + i_6)] = out_swap_buf[(((select((rmod_287 + podArgs.batch_size), rmod_287, (((podArgs.batch_size >= 0i) && (rmod_287 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_287 <= 0i)))) + select((rdiv_134 - 1i), rdiv_134, (((podArgs.vocab_size >= 0i) && (rmod_288 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_288 <= 0i))))) * podArgs.vocab_size) + select((rmod_290 + podArgs.vocab_size), rmod_290, (((podArgs.vocab_size >= 0i) && (rmod_290 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_290 <= 0i)))))];
              first_5[0i] = (first_5[0i] + 1i);
            } else {
              let rmod_291 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_292 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_135 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_293 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_294 : i32 = (last_5[0i] % podArgs.vocab_size);
              let rdiv_136 : i32 = (last_5[0i] / podArgs.vocab_size);
                            let rmod_295 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_296 : i32 = (((select((rmod_295 + podArgs.batch_size), rmod_295, (((podArgs.batch_size >= 0i) && (rmod_295 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_295 <= 0i)))) * podArgs.vocab_size) + last_5[0i]) % podArgs.vocab_size);
              value_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_291 + podArgs.batch_size), rmod_291, (((podArgs.batch_size >= 0i) && (rmod_291 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_291 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_135 - 1i), rdiv_135, (((podArgs.batch_size >= 0i) && (rmod_292 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_292 <= 0i)))))) + i_6)] = value_swap_buf[(((select((rmod_293 + podArgs.batch_size), rmod_293, (((podArgs.batch_size >= 0i) && (rmod_293 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_293 <= 0i)))) + select((rdiv_136 - 1i), rdiv_136, (((podArgs.vocab_size >= 0i) && (rmod_294 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_294 <= 0i))))) * podArgs.vocab_size) + select((rmod_296 + podArgs.vocab_size), rmod_296, (((podArgs.vocab_size >= 0i) && (rmod_296 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_296 <= 0i)))))];
              let rmod_297 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_298 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rdiv_137 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
              let rmod_299 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
              let rmod_300 : i32 = (last_5[0i] % podArgs.vocab_size);
              let rdiv_138 : i32 = (last_5[0i] / podArgs.vocab_size);
                            let rmod_301 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
let rmod_302 : i32 = (((select((rmod_301 + podArgs.batch_size), rmod_301, (((podArgs.batch_size >= 0i) && (rmod_301 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_301 <= 0i)))) * podArgs.vocab_size) + last_5[0i]) % podArgs.vocab_size);
              out_buf[(((((v__1 * 1024i) + (i32(threadIdx.x) * 4i)) + (select((rmod_297 + podArgs.batch_size), rmod_297, (((podArgs.batch_size >= 0i) && (rmod_297 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_297 <= 0i)))) * podArgs.vocab_size)) + ((2i<<u32(podArgs.cse_v1)) * select((rdiv_137 - 1i), rdiv_137, (((podArgs.batch_size >= 0i) && (rmod_298 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_298 <= 0i)))))) + i_6)] = out_swap_buf[(((select((rmod_299 + podArgs.batch_size), rmod_299, (((podArgs.batch_size >= 0i) && (rmod_299 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_299 <= 0i)))) + select((rdiv_138 - 1i), rdiv_138, (((podArgs.vocab_size >= 0i) && (rmod_300 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_300 <= 0i))))) * podArgs.vocab_size) + select((rmod_302 + podArgs.vocab_size), rmod_302, (((podArgs.vocab_size >= 0i) && (rmod_302 >= 0i)) || ((podArgs.vocab_size < 0i) && (rmod_302 <= 0i)))))];
              last_5[0i] = (last_5[0i] + 1i);
            }
          }
        }
      }
    }
  }
}
