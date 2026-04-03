// [43] batch_prefill_ragged_kv_kernel (shader #43, 294 lines)
//----------------------------------------
// Function: batch_prefill_ragged_kv_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> k : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> k_rope_pos_offset : array<i32>;
@group(0) @binding(2) var<storage, read> kv_indptr : array<i32>;
@group(0) @binding(3) var<storage, read_write> lse : array<f32>;
@group(0) @binding(4) var<storage, read_write> output : array<vec2<f16>>;
@group(0) @binding(5) var<storage, read> q : array<vec4<f16>>;
@group(0) @binding(6) var<storage, read> q_indptr : array<i32>;
@group(0) @binding(7) var<storage, read> q_rope_position : array<i32>;
@group(0) @binding(8) var<storage, read> v : array<vec4<f16>>;

struct PODArgs {
  batch_size: i32,
  causal: i32,
  k_rope_pos_offset_elem_offset: i32,
  kv_indptr_elem_offset: i32,
  kv_len: i32,
  q_indptr_elem_offset: i32,
  q_rope_position_elem_offset: i32,
  qo_len: i32,
  rope_scale: f32,
  rope_theta: f32,
  rotary_mode: i32,
  sm_scale: f32,
  packGridDimX: u32
}
@group(0) @binding(9) var<uniform> podArgs : PODArgs;

var<workgroup> m_smem : array<f32, 32>;
var<workgroup> d_smem : array<f32, 32>;
var<workgroup> Q_smem : array<f16, 3072>;
var<workgroup> K_smem : array<f16, 3072>;
var<workgroup> V_smem : array<f16, 3072>;
var<workgroup> S_smem : array<f32, 1024>;
var<workgroup> m_prev_smem : array<f32, 32>;
@compute @workgroup_size(32, 4, 1)
fn batch_prefill_ragged_kv_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var tile_id : array<i32, 1>;
  var batch_idx : array<i32, 1>;
  var batch_rows : array<i32, 1>;
  var batch_tiles : array<i32, 1>;
  var kv_chunk_len : array<i32, 1>;
  var O_local : array<vec2<f32>, 12>;
  var S_local : array<vec4<f32>, 2>;
  var m_prev : array<f32, 1>;
  var m_new : array<f32, 1>;
  var d_new : array<f32, 1>;
  tile_id[0i] = v__1;
  batch_idx[0i] = 0i;
  batch_rows[0i] = (q_indptr[(podArgs.q_indptr_elem_offset + 1i)] - q_indptr[podArgs.q_indptr_elem_offset]);
  batch_tiles[0i] = ((batch_rows[0i] + 31i)>>5u);
  while (true) {
    if (!(((batch_idx[0i] < podArgs.batch_size)))) { break; }
    while (true) {
      if (!(((batch_tiles[0i] <= tile_id[0i]) && (batch_idx[0i] < podArgs.batch_size)))) { break; }
      tile_id[0i] = (tile_id[0i] - batch_tiles[0i]);
      batch_idx[0i] = (batch_idx[0i] + 1i);
      if (batch_idx[0i] < podArgs.batch_size) {
        let b_idx : i32 = batch_idx[0i];
        batch_rows[0i] = (q_indptr[((b_idx + podArgs.q_indptr_elem_offset) + 1i)] - q_indptr[(b_idx + podArgs.q_indptr_elem_offset)]);
        batch_tiles[0i] = ((batch_rows[0i] + 31i)>>5u);
      }
    }
    if ((batch_idx[0i] < podArgs.batch_size)) {
      let b_idx_1 : i32 = batch_idx[0i];
      let q_indptr_val : i32 = q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)];
      let LH_start : i32 = (tile_id[0i] * 32i);
      kv_chunk_len[0i] = (kv_indptr[((b_idx_1 + podArgs.kv_indptr_elem_offset) + 1i)] - kv_indptr[(b_idx_1 + podArgs.kv_indptr_elem_offset)]);
      workgroupBarrier();
      if (i32(threadIdx.y) < 1i) {
        m_smem[i32(threadIdx.x)] = -5.000000e+04f;
        d_smem[i32(threadIdx.x)] = 1.000000e+00f;
      }
      for (var li_1 : i32 = 0; li_1 < 4i; li_1++) {
        O_local[(li_1 * 3i)] = vec2<f32>(0.000000e+00f, 0.000000e+00f);
        O_local[((li_1 * 3i) + 1i)] = vec2<f32>(0.000000e+00f, 0.000000e+00f);
        O_local[((li_1 * 3i) + 2i)] = vec2<f32>(0.000000e+00f, 0.000000e+00f);
      }
      workgroupBarrier();
      for (var li_0 : i32 = 0; li_0 < 2i; li_0++) {
        for (var lj_0_0 : i32 = 0; lj_0_0 < 3i; lj_0_0++) {
          if ((((((li_0 * 16i) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) + q_indptr_val) + LH_start) < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
            var condval : vec4<f16>;
            if ((podArgs.rotary_mode == 1i)) {
              let freq : vec4<f32> = (vec4<f32>((f32(q_rope_position[((((((li_0 * 16i) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) + q_indptr_val) + LH_start) + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale), (f32(q_rope_position[((((((li_0 * 16i) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) + q_indptr_val) + LH_start) + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale), (f32(q_rope_position[((((((li_0 * 16i) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) + q_indptr_val) + LH_start) + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale), (f32(q_rope_position[((((((li_0 * 16i) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) + q_indptr_val) + LH_start) + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale)) / pow(vec4<f32>(podArgs.rope_theta, podArgs.rope_theta, podArgs.rope_theta, podArgs.rope_theta), (vec4<f32>(vec4<i32>(((((lj_0_0 * 64i) + ((i32(threadIdx.x) & 7i) * 8i)) % 96i))+(2i*0), ((((lj_0_0 * 64i) + ((i32(threadIdx.x) & 7i) * 8i)) % 96i))+(2i*1), ((((lj_0_0 * 64i) + ((i32(threadIdx.x) & 7i) * 8i)) % 96i))+(2i*2), ((((lj_0_0 * 64i) + ((i32(threadIdx.x) & 7i) * 8i)) % 96i))+(2i*3))) / vec4<f32>(9.600000e+01f, 9.600000e+01f, 9.600000e+01f, 9.600000e+01f))));
              var condval_1 : vec4<f16>;
              if ((((lj_0_0 * 2i) + ((i32(threadIdx.x) & 7i)>>2u)) < 3i)) {
                condval_1 = (q[((((((((((li_0 * 49152i) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (q_indptr_val * 3072i)) + (LH_start * 3072i)) + (i32(blockIdx.y) * 96i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) + 48i) / 4i)] * vec4<f16>(-1.000000e+00h, -1.000000e+00h, -1.000000e+00h, -1.000000e+00h));
} else {
                condval_1 = q[((((((((((li_0 * 49152i) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (q_indptr_val * 3072i)) + (LH_start * 3072i)) + (i32(blockIdx.y) * 96i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) - 48i) / 4i)];
}
              condval = vec4<f16>(fma(sin(freq), vec4<f32>(condval_1), (cos(freq) * vec4<f32>(q[(((((((((li_0 * 49152i) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (q_indptr_val * 3072i)) + (LH_start * 3072i)) + (i32(blockIdx.y) * 96i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) / 4i)]))));
} else {
              condval = q[(((((((((li_0 * 49152i) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (q_indptr_val * 3072i)) + (LH_start * 3072i)) + (i32(blockIdx.y) * 96i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) / 4i)];
}
            let v__2 : i32 = (((((li_0 * 1536i) + (i32(threadIdx.y) * 384i)) + ((i32(threadIdx.x)>>3u) * 96i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i));
            Q_smem[v__2 + 0] = condval[0];
            Q_smem[v__2 + 1] = condval[1];
            Q_smem[v__2 + 2] = condval[2];
            Q_smem[v__2 + 3] = condval[3];
          } else {
            let v__3 : i32 = (((((li_0 * 1536i) + (i32(threadIdx.y) * 384i)) + ((i32(threadIdx.x)>>3u) * 96i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i));
            Q_smem[v__3 + 0] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[0];
            Q_smem[v__3 + 1] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[1];
            Q_smem[v__3 + 2] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[2];
            Q_smem[v__3 + 3] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[3];
          }
        }
      }
      workgroupBarrier();
      for (var iterator : i32 = 0; iterator < ((kv_chunk_len[0i] + 31i)>>5u); iterator++) {
        let L_kv_base : i32 = kv_indptr[(b_idx_1 + podArgs.kv_indptr_elem_offset)];
        for (var lz_0 : i32 = 0; lz_0 < 2i; lz_0++) {
          for (var ly_0_0 : i32 = 0; ly_0_0 < 3i; ly_0_0++) {
            if (((((iterator * 32i) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) < kv_chunk_len[0i]) {
              var condval_2 : vec4<f16>;
              if ((podArgs.rotary_mode == 1i)) {
                let freq_1 : vec4<f32> = (vec4<f32>((f32((((((iterator * 32i) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) + k_rope_pos_offset[(b_idx_1 + podArgs.k_rope_pos_offset_elem_offset)])) * podArgs.rope_scale), (f32((((((iterator * 32i) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) + k_rope_pos_offset[(b_idx_1 + podArgs.k_rope_pos_offset_elem_offset)])) * podArgs.rope_scale), (f32((((((iterator * 32i) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) + k_rope_pos_offset[(b_idx_1 + podArgs.k_rope_pos_offset_elem_offset)])) * podArgs.rope_scale), (f32((((((iterator * 32i) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) + k_rope_pos_offset[(b_idx_1 + podArgs.k_rope_pos_offset_elem_offset)])) * podArgs.rope_scale)) / pow(vec4<f32>(podArgs.rope_theta, podArgs.rope_theta, podArgs.rope_theta, podArgs.rope_theta), (vec4<f32>(vec4<i32>(((((ly_0_0 * 64i) + ((i32(threadIdx.x) & 7i) * 8i)) % 96i))+(2i*0), ((((ly_0_0 * 64i) + ((i32(threadIdx.x) & 7i) * 8i)) % 96i))+(2i*1), ((((ly_0_0 * 64i) + ((i32(threadIdx.x) & 7i) * 8i)) % 96i))+(2i*2), ((((ly_0_0 * 64i) + ((i32(threadIdx.x) & 7i) * 8i)) % 96i))+(2i*3))) / vec4<f32>(9.600000e+01f, 9.600000e+01f, 9.600000e+01f, 9.600000e+01f))));
                var condval_3 : vec4<f16>;
                if ((((ly_0_0 * 2i) + ((i32(threadIdx.x) & 7i)>>2u)) < 3i)) {
                  condval_3 = (k[((((((((((iterator * 98304i) + (lz_0 * 49152i)) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (ly_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) + 48i) / 4i)] * vec4<f16>(-1.000000e+00h, -1.000000e+00h, -1.000000e+00h, -1.000000e+00h));
} else {
                  condval_3 = k[((((((((((iterator * 98304i) + (lz_0 * 49152i)) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (ly_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) - 48i) / 4i)];
}
                condval_2 = vec4<f16>(fma(sin(freq_1), vec4<f32>(condval_3), (cos(freq_1) * vec4<f32>(k[(((((((((iterator * 98304i) + (lz_0 * 49152i)) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (ly_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) / 4i)]))));
} else {
                condval_2 = k[(((((((((iterator * 98304i) + (lz_0 * 49152i)) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (ly_0_0 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) / 4i)];
}
              let v__4 : vec4<i32> = vec4<i32>(((((((ly_0_0 * 1024i) + ((i32(threadIdx.x) & 7i) * 128i)) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)))+(32i*0), ((((((ly_0_0 * 1024i) + ((i32(threadIdx.x) & 7i) * 128i)) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)))+(32i*1), ((((((ly_0_0 * 1024i) + ((i32(threadIdx.x) & 7i) * 128i)) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)))+(32i*2), ((((((ly_0_0 * 1024i) + ((i32(threadIdx.x) & 7i) * 128i)) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)))+(32i*3));
              K_smem[v__4[0]] = condval_2[0];
              K_smem[v__4[1]] = condval_2[1];
              K_smem[v__4[2]] = condval_2[2];
              K_smem[v__4[3]] = condval_2[3];
            } else {
              let v__5 : vec4<i32> = vec4<i32>(((((((ly_0_0 * 1024i) + ((i32(threadIdx.x) & 7i) * 128i)) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)))+(32i*0), ((((((ly_0_0 * 1024i) + ((i32(threadIdx.x) & 7i) * 128i)) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)))+(32i*1), ((((((ly_0_0 * 1024i) + ((i32(threadIdx.x) & 7i) * 128i)) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)))+(32i*2), ((((((ly_0_0 * 1024i) + ((i32(threadIdx.x) & 7i) * 128i)) + (lz_0 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)))+(32i*3));
              K_smem[v__5[0]] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[0];
              K_smem[v__5[1]] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[1];
              K_smem[v__5[2]] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[2];
              K_smem[v__5[3]] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[3];
            }
          }
        }
        workgroupBarrier();
        for (var lz_0_1 : i32 = 0; lz_0_1 < 2i; lz_0_1++) {
          for (var ly_0_0_1 : i32 = 0; ly_0_0_1 < 3i; ly_0_0_1++) {
            if (((((iterator * 32i) + (lz_0_1 * 16i)) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>3u)) < kv_chunk_len[0i]) {
              let v__6 : i32 = (((((lz_0_1 * 1536i) + (i32(threadIdx.y) * 384i)) + ((i32(threadIdx.x)>>3u) * 96i)) + (ly_0_0_1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i));
              V_smem[v__6 + 0] = v[(((((((((iterator * 98304i) + (lz_0_1 * 49152i)) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (ly_0_0_1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) / 4i)][0];
              V_smem[v__6 + 1] = v[(((((((((iterator * 98304i) + (lz_0_1 * 49152i)) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (ly_0_0_1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) / 4i)][1];
              V_smem[v__6 + 2] = v[(((((((((iterator * 98304i) + (lz_0_1 * 49152i)) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (ly_0_0_1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) / 4i)][2];
              V_smem[v__6 + 3] = v[(((((((((iterator * 98304i) + (lz_0_1 * 49152i)) + (i32(threadIdx.y) * 12288i)) + ((i32(threadIdx.x)>>3u) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (ly_0_0_1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) / 4i)][3];
            } else {
              let v__7 : i32 = (((((lz_0_1 * 1536i) + (i32(threadIdx.y) * 384i)) + ((i32(threadIdx.x)>>3u) * 96i)) + (ly_0_0_1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i));
              V_smem[v__7 + 0] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[0];
              V_smem[v__7 + 1] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[1];
              V_smem[v__7 + 2] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[2];
              V_smem[v__7 + 3] = vec4<f16>(0.000000e+00h, 0.000000e+00h, 0.000000e+00h, 0.000000e+00h)[3];
            }
          }
        }
        workgroupBarrier();
        S_local[0i] = vec4<f32>(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
        S_local[1i] = vec4<f32>(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
        for (var lk_0 : i32 = 0; lk_0 < 6i; lk_0++) {
          for (var lk_1 : i32 = 0; lk_1 < 16i; lk_1++) {
            let v__8 : i32 = (((lk_0 * 512i) + (lk_1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i));
            S_local[0i] = fma(((vec4<f32>(f32(Q_smem[((((i32(threadIdx.y) * 768i) + ((i32(threadIdx.x)>>3u) * 192i)) + (lk_0 * 16i)) + lk_1)]), f32(Q_smem[((((i32(threadIdx.y) * 768i) + ((i32(threadIdx.x)>>3u) * 192i)) + (lk_0 * 16i)) + lk_1)]), f32(Q_smem[((((i32(threadIdx.y) * 768i) + ((i32(threadIdx.x)>>3u) * 192i)) + (lk_0 * 16i)) + lk_1)]), f32(Q_smem[((((i32(threadIdx.y) * 768i) + ((i32(threadIdx.x)>>3u) * 192i)) + (lk_0 * 16i)) + lk_1)])) * vec4<f32>(vec4<f16>(K_smem[v__8 + 0], K_smem[v__8 + 1], K_smem[v__8 + 2], K_smem[v__8 + 3]))) * vec4<f32>(podArgs.sm_scale, podArgs.sm_scale, podArgs.sm_scale, podArgs.sm_scale)), vec4<f32>(1.442695e+00f, 1.442695e+00f, 1.442695e+00f, 1.442695e+00f), S_local[0i]);
          }
          for (var lk_1_1 : i32 = 0; lk_1_1 < 16i; lk_1_1++) {
            let v__9 : i32 = (((lk_0 * 512i) + (lk_1_1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i));
            S_local[1i] = fma(((vec4<f32>(f32(Q_smem[(((((i32(threadIdx.y) * 768i) + ((i32(threadIdx.x)>>3u) * 192i)) + (lk_0 * 16i)) + lk_1_1) + 96i)]), f32(Q_smem[(((((i32(threadIdx.y) * 768i) + ((i32(threadIdx.x)>>3u) * 192i)) + (lk_0 * 16i)) + lk_1_1) + 96i)]), f32(Q_smem[(((((i32(threadIdx.y) * 768i) + ((i32(threadIdx.x)>>3u) * 192i)) + (lk_0 * 16i)) + lk_1_1) + 96i)]), f32(Q_smem[(((((i32(threadIdx.y) * 768i) + ((i32(threadIdx.x)>>3u) * 192i)) + (lk_0 * 16i)) + lk_1_1) + 96i)])) * vec4<f32>(vec4<f16>(K_smem[v__9 + 0], K_smem[v__9 + 1], K_smem[v__9 + 2], K_smem[v__9 + 3]))) * vec4<f32>(podArgs.sm_scale, podArgs.sm_scale, podArgs.sm_scale, podArgs.sm_scale)), vec4<f32>(1.442695e+00f, 1.442695e+00f, 1.442695e+00f, 1.442695e+00f), S_local[1i]);
          }
        }
        workgroupBarrier();
        for (var li_1_1 : i32 = 0; li_1_1 < 2i; li_1_1++) {
          let v__10 : i32 = ((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>3u) * 64i)) + (li_1_1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i));
          S_smem[v__10 + 0] = S_local[li_1_1][0];
          S_smem[v__10 + 1] = S_local[li_1_1][1];
          S_smem[v__10 + 2] = S_local[li_1_1][2];
          S_smem[v__10 + 3] = S_local[li_1_1][3];
        }
        workgroupBarrier();
        if (i32(threadIdx.y) < 1i) {
          m_prev[0i] = m_smem[i32(threadIdx.x)];
          m_new[0i] = m_smem[i32(threadIdx.x)];
          for (var j : i32 = 0; j < 32i; j++) {
            let cse_v5 : i32 = ((iterator * 32i) + j);
            var condval_4 : bool;
            if ((0i < podArgs.causal)) {
              condval_4 = (cse_v5 <= ((((((i32(threadIdx.y) * 32i) + kv_chunk_len[0i]) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) + LH_start) + i32(threadIdx.x)) - q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]));
} else {
              condval_4 = (cse_v5 < kv_chunk_len[0i]);
}
            if (condval_4) {
              m_new[0i] = max(m_new[0i], S_smem[(((i32(threadIdx.y) * 1024i) + (i32(threadIdx.x) * 32i)) + j)]);
            }
          }
          d_new[0i] = (d_smem[i32(threadIdx.x)] * exp2((m_prev[0i] - m_new[0i])));
        }
        for (var j_1 : i32 = 0; j_1 < 32i; j_1++) {
          workgroupBarrier();
          if (i32(threadIdx.y) < 1i) {
            let cse_v6 : i32 = ((iterator * 32i) + j_1);
            var condval_5 : bool;
            if ((0i < podArgs.causal)) {
              condval_5 = (cse_v6 <= ((((((i32(threadIdx.y) * 32i) + kv_chunk_len[0i]) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) + LH_start) + i32(threadIdx.x)) - q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]));
} else {
              condval_5 = (cse_v6 < kv_chunk_len[0i]);
}
            if (condval_5) {
              S_smem[(((i32(threadIdx.y) * 1024i) + (i32(threadIdx.x) * 32i)) + j_1)] = exp2((S_smem[(((i32(threadIdx.y) * 1024i) + (i32(threadIdx.x) * 32i)) + j_1)] - m_new[0i]));
            } else {
              S_smem[((i32(threadIdx.x) * 32i) + j_1)] = exp2((-5.000000e+04f - m_new[0i]));
            }
          }
        }
        workgroupBarrier();
        if (i32(threadIdx.y) < 1i) {
          for (var j_2 : i32 = 0; j_2 < 32i; j_2++) {
            d_new[0i] = (d_new[0i] + S_smem[((i32(threadIdx.x) * 32i) + j_2)]);
          }
          m_smem[i32(threadIdx.x)] = m_new[0i];
          d_smem[i32(threadIdx.x)] = d_new[0i];
          m_prev_smem[i32(threadIdx.x)] = m_prev[0i];
        }
        workgroupBarrier();
        O_local[0i] = (O_local[0i] * vec2<f32>(exp2((m_prev_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))] - m_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))])), exp2((m_prev_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))] - m_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))]))));
        O_local[1i] = (O_local[1i] * vec2<f32>(exp2((m_prev_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))] - m_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))])), exp2((m_prev_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))] - m_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))]))));
        O_local[2i] = (O_local[2i] * vec2<f32>(exp2((m_prev_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))] - m_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))])), exp2((m_prev_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))] - m_smem[((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i))]))));
        O_local[3i] = (O_local[3i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)]))));
        O_local[4i] = (O_local[4i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)]))));
        O_local[5i] = (O_local[5i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 1i)]))));
        O_local[6i] = (O_local[6i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)]))));
        O_local[7i] = (O_local[7i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)]))));
        O_local[8i] = (O_local[8i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 2i)]))));
        O_local[9i] = (O_local[9i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)]))));
        O_local[10i] = (O_local[10i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)]))));
        O_local[11i] = (O_local[11i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + 3i)]))));
        for (var lk_0_1 : i32 = 0; lk_0_1 < 2i; lk_0_1++) {
          for (var lk_1_2 : i32 = 0; lk_1_2 < 16i; lk_1_2++) {
            let v__11 : i32 = (((lk_0_1 * 1536i) + (lk_1_2 * 96i)) + ((i32(threadIdx.x) & 15i) * 6i));
            O_local[0i] = fma(vec2<f32>(S_smem[((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2)], S_smem[((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2)]), vec2<f32>(vec2<f16>(V_smem[v__11 + 0], V_smem[v__11 + 1])), O_local[0i]);
            let v__12 : i32 = ((((lk_0_1 * 1536i) + (lk_1_2 * 96i)) + ((i32(threadIdx.x) & 15i) * 6i)) + 2i);
            O_local[1i] = fma(vec2<f32>(S_smem[((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2)], S_smem[((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2)]), vec2<f32>(vec2<f16>(V_smem[v__12 + 0], V_smem[v__12 + 1])), O_local[1i]);
            let v__13 : i32 = ((((lk_0_1 * 1536i) + (lk_1_2 * 96i)) + ((i32(threadIdx.x) & 15i) * 6i)) + 4i);
            O_local[2i] = fma(vec2<f32>(S_smem[((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2)], S_smem[((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2)]), vec2<f32>(vec2<f16>(V_smem[v__13 + 0], V_smem[v__13 + 1])), O_local[2i]);
            O_local[3i] = fma(vec2<f32>(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 32i)], S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 32i)]), vec2<f32>(vec2<f16>(V_smem[v__11 + 0], V_smem[v__11 + 1])), O_local[3i]);
            O_local[4i] = fma(vec2<f32>(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 32i)], S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 32i)]), vec2<f32>(vec2<f16>(V_smem[v__12 + 0], V_smem[v__12 + 1])), O_local[4i]);
            O_local[5i] = fma(vec2<f32>(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 32i)], S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 32i)]), vec2<f32>(vec2<f16>(V_smem[v__13 + 0], V_smem[v__13 + 1])), O_local[5i]);
            O_local[6i] = fma(vec2<f32>(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 64i)], S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 64i)]), vec2<f32>(vec2<f16>(V_smem[v__11 + 0], V_smem[v__11 + 1])), O_local[6i]);
            O_local[7i] = fma(vec2<f32>(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 64i)], S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 64i)]), vec2<f32>(vec2<f16>(V_smem[v__12 + 0], V_smem[v__12 + 1])), O_local[7i]);
            O_local[8i] = fma(vec2<f32>(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 64i)], S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 64i)]), vec2<f32>(vec2<f16>(V_smem[v__13 + 0], V_smem[v__13 + 1])), O_local[8i]);
            O_local[9i] = fma(vec2<f32>(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 96i)], S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 96i)]), vec2<f32>(vec2<f16>(V_smem[v__11 + 0], V_smem[v__11 + 1])), O_local[9i]);
            O_local[10i] = fma(vec2<f32>(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 96i)], S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 96i)]), vec2<f32>(vec2<f16>(V_smem[v__12 + 0], V_smem[v__12 + 1])), O_local[10i]);
            O_local[11i] = fma(vec2<f32>(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 96i)], S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (lk_0_1 * 16i)) + lk_1_2) + 96i)]), vec2<f32>(vec2<f16>(V_smem[v__13 + 0], V_smem[v__13 + 1])), O_local[11i]);
          }
        }
      }
      for (var li_1_2 : i32 = 0; li_1_2 < 4i; li_1_2++) {
        let cur_L : i32 = (((((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) + LH_start) + li_1_2);
        if (cur_L < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
          output[((((cur_L * 3072i) + (i32(blockIdx.y) * 96i)) + ((i32(threadIdx.x) & 15i) * 6i)) / 2i)] = vec2<f16>((O_local[(li_1_2 * 3i)] / vec2<f32>(d_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + li_1_2)], d_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + li_1_2)])));
        }
        let cur_L_1 : i32 = (((((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) + LH_start) + li_1_2);
        if (cur_L_1 < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
          output[(((((cur_L_1 * 3072i) + (i32(blockIdx.y) * 96i)) + ((i32(threadIdx.x) & 15i) * 6i)) + 2i) / 2i)] = vec2<f16>((O_local[((li_1_2 * 3i) + 1i)] / vec2<f32>(d_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + li_1_2)], d_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + li_1_2)])));
        }
        let cur_L_2 : i32 = (((((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) + LH_start) + li_1_2);
        if (cur_L_2 < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
          output[(((((cur_L_2 * 3072i) + (i32(blockIdx.y) * 96i)) + ((i32(threadIdx.x) & 15i) * 6i)) + 4i) / 2i)] = vec2<f16>((O_local[((li_1_2 * 3i) + 2i)] / vec2<f32>(d_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + li_1_2)], d_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + li_1_2)])));
        }
      }
      if (i32(threadIdx.y) < 1i) {
        let cur_L_3 : i32 = ((((i32(threadIdx.y) * 32i) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) + LH_start) + i32(threadIdx.x));
        if (cur_L_3 < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
          lse[((cur_L_3 * 32i) + i32(blockIdx.y))] = (m_smem[i32(threadIdx.x)] + log2(d_smem[i32(threadIdx.x)]));
        }
      }
      tile_id[0i] = (tile_id[0i] + 16i);
    }
  }
}
