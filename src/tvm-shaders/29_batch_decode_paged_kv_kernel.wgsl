// [29] batch_decode_paged_kv_kernel (shader #29, 192 lines)
//----------------------------------------
// Function: batch_decode_paged_kv_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> Q : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> k_rope_pos_offset : array<i32>;
@group(0) @binding(2) var<storage, read> length_info : array<i32>;
@group(0) @binding(3) var<storage, read_write> lse : array<f32>;
@group(0) @binding(4) var<storage, read_write> output : array<vec4<f16>>;
@group(0) @binding(5) var<storage, read> page_table_indptr : array<i32>;
@group(0) @binding(6) var<storage, read> page_table_values : array<i32>;
@group(0) @binding(7) var<storage, read> pages : array<f16>;
@group(0) @binding(8) var<storage, read> q_rope_position : array<i32>;

struct PODArgs {
  B: i32,
  k_rope_pos_offset_elem_offset: i32,
  length_info_elem_offset: i32,
  max_num_pages: i32,
  nnz_pages: i32,
  page_indptr_elem_offset: i32,
  page_values_elem_offset: i32,
  pages_elem_offset: i32,
  q_rope_position_elem_offset: i32,
  rope_scale: f32,
  rope_theta: f32,
  rotary_mode: i32,
  sm_scale: f32,
  packGridDimX: u32
}
@group(0) @binding(9) var<uniform> podArgs : PODArgs;

var<workgroup> O_allreduce : array<vec4<f32>, 240>;
var<workgroup> md_allreduce : array<f32, 20>;
var<workgroup> K_smem : array<vec4<f16>, 480>;
var<workgroup> V_smem : array<vec4<f16>, 480>;
var<workgroup> red_buf0 : array<f32, 240>;
@compute @workgroup_size(24, 1, 10)
fn batch_decode_paged_kv_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var kv_chunk_len : array<i32, 1>;
  var st_m : array<f32, 1>;
  var st_d : array<f32, 1>;
  var O_local : array<vec4<f32>, 1>;
  var Q_local : array<vec4<f16>, 1>;
  var m_prev : array<f32, 1>;
  var S_local : array<f32, 2>;
  var QK_local : array<vec4<f32>, 1>;
  var S_reduce_local : array<f32, 1>;
  var V_local : array<vec4<f16>, 1>;
  var d_prev : array<f32, 1>;
  var other_m : array<f32, 1>;
  var other_d : array<f32, 1>;
  var other_o : array<vec4<f32>, 1>;
  var exp_mprev : array<f32, 1>;
  let cur_page_indptr_begin : i32 = page_table_indptr[(v__1 + podArgs.page_indptr_elem_offset)];
  let cur_page_indptr_end : i32 = page_table_indptr[((v__1 + podArgs.page_indptr_elem_offset) + 1i)];
  var condval : i32;
  if ((cur_page_indptr_begin != cur_page_indptr_end)) {
    condval = ((((cur_page_indptr_end * 16i) + length_info[(v__1 + podArgs.length_info_elem_offset)]) - (cur_page_indptr_begin * 16i)) - 16i);
} else {
    condval = 0i;
}
  kv_chunk_len[0i] = condval;
  st_m[0i] = -5.000000e+04f;
  st_d[0i] = 1.000000e+00f;
  O_local[0i] = vec4<f32>(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  var condval_1 : vec4<f16>;
  if ((podArgs.rotary_mode == 1i)) {
    let freq : vec4<f32> = (vec4<f32>((f32(q_rope_position[(v__1 + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale), (f32(q_rope_position[(v__1 + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale), (f32(q_rope_position[(v__1 + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale), (f32(q_rope_position[(v__1 + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale)) / pow(vec4<f32>(podArgs.rope_theta, podArgs.rope_theta, podArgs.rope_theta, podArgs.rope_theta), (vec4<f32>(vec4<i32>((((i32(threadIdx.x) % 12i) * 8i))+(2i*0), (((i32(threadIdx.x) % 12i) * 8i))+(2i*1), (((i32(threadIdx.x) % 12i) * 8i))+(2i*2), (((i32(threadIdx.x) % 12i) * 8i))+(2i*3))) / vec4<f32>(9.600000e+01f, 9.600000e+01f, 9.600000e+01f, 9.600000e+01f))));
    var condval_2 : vec4<f16>;
    if ((i32(threadIdx.x) < 12i)) {
      condval_2 = (Q[(((((v__1 * 3072i) + (i32(blockIdx.y) * 96i)) + (i32(threadIdx.x) * 4i)) + 48i) / 4i)] * vec4<f16>(-1.000000e+00h, -1.000000e+00h, -1.000000e+00h, -1.000000e+00h));
} else {
      condval_2 = Q[(((((v__1 * 3072i) + (i32(blockIdx.y) * 96i)) + (i32(threadIdx.x) * 4i)) - 48i) / 4i)];
}
    condval_1 = vec4<f16>(fma(sin(freq), vec4<f32>(condval_2), (cos(freq) * vec4<f32>(Q[((((v__1 * 3072i) + (i32(blockIdx.y) * 96i)) + (i32(threadIdx.x) * 4i)) / 4i)]))));
} else {
    condval_1 = Q[((((v__1 * 3072i) + (i32(blockIdx.y) * 96i)) + (i32(threadIdx.x) * 4i)) / 4i)];
}
  Q_local[0i] = condval_1;
  for (var iterator : i32 = 0; iterator < (((kv_chunk_len[0i] - 2147483621i) / 20i) - -107374182i); iterator++) {
    workgroupBarrier();
    for (var j : i32 = 0; j < 2i; j++) {
      if ((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + j) < kv_chunk_len[0i]) {
        let page_no : i32 = page_table_values[(((((iterator * 5i) + (i32(threadIdx.z)>>1u))>>2u) + cur_page_indptr_begin) + podArgs.page_values_elem_offset)];
        var condval_3 : vec4<f16>;
        if ((podArgs.rotary_mode == 1i)) {
          let freq_1 : vec4<f32> = (vec4<f32>((f32(((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + k_rope_pos_offset[(v__1 + podArgs.k_rope_pos_offset_elem_offset)]) + j)) * podArgs.rope_scale), (f32(((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + k_rope_pos_offset[(v__1 + podArgs.k_rope_pos_offset_elem_offset)]) + j)) * podArgs.rope_scale), (f32(((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + k_rope_pos_offset[(v__1 + podArgs.k_rope_pos_offset_elem_offset)]) + j)) * podArgs.rope_scale), (f32(((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + k_rope_pos_offset[(v__1 + podArgs.k_rope_pos_offset_elem_offset)]) + j)) * podArgs.rope_scale)) / pow(vec4<f32>(podArgs.rope_theta, podArgs.rope_theta, podArgs.rope_theta, podArgs.rope_theta), (vec4<f32>(vec4<i32>((((i32(threadIdx.x) % 12i) * 8i))+(2i*0), (((i32(threadIdx.x) % 12i) * 8i))+(2i*1), (((i32(threadIdx.x) % 12i) * 8i))+(2i*2), (((i32(threadIdx.x) % 12i) * 8i))+(2i*3))) / vec4<f32>(9.600000e+01f, 9.600000e+01f, 9.600000e+01f, 9.600000e+01f))));
          var condval_4 : vec4<f16>;
          if ((i32(threadIdx.x) < 12i)) {
            let v__2 : i32 = ((((((page_no * 98304i) + (i32(blockIdx.y) * 1536i)) + (((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + j) & 15i) * 96i)) + (i32(threadIdx.x) * 4i)) + podArgs.pages_elem_offset) + 48i);
            condval_4 = (vec4<f16>(pages[v__2 + 0], pages[v__2 + 1], pages[v__2 + 2], pages[v__2 + 3]) * vec4<f16>(-1.000000e+00h, -1.000000e+00h, -1.000000e+00h, -1.000000e+00h));
} else {
            let v__3 : i32 = ((((((page_no * 98304i) + (i32(blockIdx.y) * 1536i)) + (((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + j) & 15i) * 96i)) + (i32(threadIdx.x) * 4i)) + podArgs.pages_elem_offset) - 48i);
            condval_4 = vec4<f16>(pages[v__3 + 0], pages[v__3 + 1], pages[v__3 + 2], pages[v__3 + 3]);
}
          let v__4 : i32 = (((((page_no * 98304i) + (i32(blockIdx.y) * 1536i)) + (((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + j) & 15i) * 96i)) + (i32(threadIdx.x) * 4i)) + podArgs.pages_elem_offset);
          condval_3 = vec4<f16>(fma(sin(freq_1), vec4<f32>(condval_4), (cos(freq_1) * vec4<f32>(vec4<f16>(pages[v__4 + 0], pages[v__4 + 1], pages[v__4 + 2], pages[v__4 + 3])))));
} else {
          let v__5 : i32 = (((((page_no * 98304i) + (i32(blockIdx.y) * 1536i)) + (((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + j) & 15i) * 96i)) + (i32(threadIdx.x) * 4i)) + podArgs.pages_elem_offset);
          condval_3 = vec4<f16>(pages[v__5 + 0], pages[v__5 + 1], pages[v__5 + 2], pages[v__5 + 3]);
}
        K_smem[(((i32(threadIdx.z) * 48i) + (j * 24i)) + i32(threadIdx.x))] = condval_3;
        let v__6 : i32 = ((((((page_no * 98304i) + (i32(blockIdx.y) * 1536i)) + (((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + j) & 15i) * 96i)) + (i32(threadIdx.x) * 4i)) + podArgs.pages_elem_offset) + 49152i);
        V_smem[(((i32(threadIdx.z) * 48i) + (j * 24i)) + i32(threadIdx.x))] = vec4<f16>(pages[v__6 + 0], pages[v__6 + 1], pages[v__6 + 2], pages[v__6 + 3]);
      } else {
        K_smem[(((i32(threadIdx.z) * 48i) + (j * 24i)) + i32(threadIdx.x))] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h);
        V_smem[(((i32(threadIdx.z) * 48i) + (j * 24i)) + i32(threadIdx.x))] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h);
      }
    }
    workgroupBarrier();
    m_prev[0i] = st_m[0i];
    for (var j_1 : i32 = 0; j_1 < 2i; j_1++) {
      QK_local[0i] = (((vec4<f32>(Q_local[0i]) * vec4<f32>(K_smem[(((i32(threadIdx.z) * 48i) + (j_1 * 24i)) + i32(threadIdx.x))])) * vec4<f32>(podArgs.sm_scale, podArgs.sm_scale, podArgs.sm_scale, podArgs.sm_scale)) * vec4<f32>(1.442695e+00f, 1.442695e+00f, 1.442695e+00f, 1.442695e+00f));
      S_reduce_local[0i] = 0.000000e+00f;
      S_reduce_local[0i] = (S_reduce_local[0i] + (QK_local[0i])[0]);
      S_reduce_local[0i] = (S_reduce_local[0i] + (QK_local[0i])[1]);
      S_reduce_local[0i] = (S_reduce_local[0i] + (QK_local[0i])[2]);
      S_reduce_local[0i] = (S_reduce_local[0i] + (QK_local[0i])[3]);
      workgroupBarrier();
      red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] = S_reduce_local[0i];
      workgroupBarrier();
      if (i32(threadIdx.x) < 8i) {
        red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] = (red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] + red_buf0[(((i32(threadIdx.z) * 24i) + i32(threadIdx.x)) + 16i)]);
      }
      workgroupBarrier();
      if (i32(threadIdx.x) < 8i) {
        red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] = (red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] + red_buf0[(((i32(threadIdx.z) * 24i) + i32(threadIdx.x)) + 8i)]);
      }
      workgroupBarrier();
      if (i32(threadIdx.x) < 4i) {
        red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] = (red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] + red_buf0[(((i32(threadIdx.z) * 24i) + i32(threadIdx.x)) + 4i)]);
      }
      workgroupBarrier();
      if (i32(threadIdx.x) < 2i) {
        red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] = (red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] + red_buf0[(((i32(threadIdx.z) * 24i) + i32(threadIdx.x)) + 2i)]);
      }
      workgroupBarrier();
      if (i32(threadIdx.x) < 1i) {
        red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] = (red_buf0[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] + red_buf0[(((i32(threadIdx.z) * 24i) + i32(threadIdx.x)) + 1i)]);
      }
      workgroupBarrier();
      S_local[j_1] = -5.000000e+04f;
      if ((((iterator * 20i) + (i32(threadIdx.z) * 2i)) + j_1) < kv_chunk_len[0i]) {
        S_local[j_1] = red_buf0[(i32(threadIdx.z) * 24i)];
      }
      st_m[0i] = max(st_m[0i], S_local[j_1]);
    }
    let o_scale : f32 = exp2((m_prev[0i] - st_m[0i]));
    st_d[0i] = (st_d[0i] * o_scale);
    for (var j_2 : i32 = 0; j_2 < 2i; j_2++) {
      S_local[j_2] = exp2((S_local[j_2] - st_m[0i]));
      st_d[0i] = (st_d[0i] + S_local[j_2]);
    }
    O_local[0i] = (O_local[0i] * vec4<f32>(o_scale, o_scale, o_scale, o_scale));
    for (var j_3 : i32 = 0; j_3 < 2i; j_3++) {
      V_local[0i] = V_smem[(((i32(threadIdx.z) * 48i) + (j_3 * 24i)) + i32(threadIdx.x))];
      O_local[0i] = fma(vec4<f32>(V_local[0i]), vec4<f32>(S_local[j_3], S_local[j_3], S_local[j_3], S_local[j_3]), O_local[0i]);
    }
  }
  O_allreduce[((i32(threadIdx.z) * 24i) + i32(threadIdx.x))] = O_local[0i];
  md_allreduce[(i32(threadIdx.z) * 2i)] = st_m[0i];
  md_allreduce[((i32(threadIdx.z) * 2i) + 1i)] = st_d[0i];
  workgroupBarrier();
  st_m[0i] = -5.000000e+04f;
  st_d[0i] = 1.000000e+00f;
  O_local[0i] = vec4<f32>(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  for (var j_4 : i32 = 0; j_4 < 10i; j_4++) {
    m_prev[0i] = st_m[0i];
    d_prev[0i] = st_d[0i];
    other_m[0i] = md_allreduce[(j_4 * 2i)];
    other_d[0i] = md_allreduce[((j_4 * 2i) + 1i)];
    other_o[0i] = O_allreduce[((j_4 * 24i) + i32(threadIdx.x))];
    st_m[0i] = max(st_m[0i], other_m[0i]);
    st_d[0i] = fma(other_d[0i], exp2((other_m[0i] - st_m[0i])), (d_prev[0i] * exp2((m_prev[0i] - st_m[0i]))));
    exp_mprev[0i] = exp2((m_prev[0i] - st_m[0i]));
    other_m[0i] = exp2((other_m[0i] - st_m[0i]));
    O_local[0i] = fma(other_o[0i], vec4<f32>(other_m[0i], other_m[0i], other_m[0i], other_m[0i]), (O_local[0i] * vec4<f32>(exp_mprev[0i], exp_mprev[0i], exp_mprev[0i], exp_mprev[0i])));
  }
  O_local[0i] = (O_local[0i] / vec4<f32>(st_d[0i], st_d[0i], st_d[0i], st_d[0i]));
  output[((((v__1 * 3072i) + (i32(blockIdx.y) * 96i)) + (i32(threadIdx.x) * 4i)) / 4i)] = vec4<f16>(O_local[0i]);
  lse[((v__1 * 32i) + i32(blockIdx.y))] = (st_m[0i] + log2(st_d[0i]));
}
