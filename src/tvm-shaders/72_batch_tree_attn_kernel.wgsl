// [72] batch_tree_attn_kernel (shader #72, 232 lines)
//----------------------------------------
// Function: batch_tree_attn_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> k : array<f16>;
@group(0) @binding(1) var<storage, read> kv_indptr : array<i32>;
@group(0) @binding(2) var<storage, read_write> lse : array<f32>;
@group(0) @binding(3) var<storage, read> mask : array<i32>;
@group(0) @binding(4) var<storage, read> mn_indptr : array<i32>;
@group(0) @binding(5) var<storage, read_write> output : array<f16>;
@group(0) @binding(6) var<storage, read> q : array<f16>;
@group(0) @binding(7) var<storage, read> q_indptr : array<i32>;
@group(0) @binding(8) var<storage, read> q_rope_position : array<i32>;
@group(0) @binding(9) var<storage, read> v : array<f16>;

struct PODArgs {
  batch_size_plus_1: i32,
  kv_indptr_elem_offset: i32,
  kv_len: i32,
  mask_elem_offset: i32,
  mn_indptr_elem_offset: i32,
  q_indptr_elem_offset: i32,
  q_rope_position_elem_offset: i32,
  qo_len: i32,
  rope_scale: f32,
  rope_theta: f32,
  rotary_mode: i32,
  sm_scale: f32,
  tree_size: i32,
  packGridDimX: u32
}
@group(0) @binding(10) var<uniform> podArgs : PODArgs;

var<workgroup> m_smem : array<f32, 32>;
var<workgroup> d_smem : array<f32, 32>;
var<workgroup> Q_smem : array<f16, 3072>;
var<workgroup> K_smem : array<f16, 3072>;
var<workgroup> V_smem : array<f16, 3072>;
var<workgroup> S_smem : array<f32, 1024>;
var<workgroup> m_prev_smem : array<f32, 32>;
@compute @workgroup_size(32, 4, 1)
fn batch_tree_attn_kernel(
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
  var O_local : array<f32, 24>;
  var S_local : array<f32, 8>;
  var m_prev : array<f32, 1>;
  var m_new : array<f32, 1>;
  var d_new : array<f32, 1>;
  tile_id[0i] = v__1;
  batch_idx[0i] = 0i;
  batch_rows[0i] = (q_indptr[(podArgs.q_indptr_elem_offset + 1i)] - q_indptr[podArgs.q_indptr_elem_offset]);
  batch_tiles[0i] = ((batch_rows[0i] + 31i)>>5u);
  while (true) {
    if (!((((batch_idx[0i] + 1i) < podArgs.batch_size_plus_1)))) { break; }
    while (true) {
      if (!(((batch_tiles[0i] <= tile_id[0i]) && ((batch_idx[0i] + 1i) < podArgs.batch_size_plus_1)))) { break; }
      tile_id[0i] = (tile_id[0i] - batch_tiles[0i]);
      batch_idx[0i] = (batch_idx[0i] + 1i);
      if ((batch_idx[0i] + 1i) < podArgs.batch_size_plus_1) {
        let b_idx : i32 = batch_idx[0i];
        batch_rows[0i] = (q_indptr[((b_idx + podArgs.q_indptr_elem_offset) + 1i)] - q_indptr[(b_idx + podArgs.q_indptr_elem_offset)]);
        batch_tiles[0i] = ((batch_rows[0i] + 31i)>>5u);
      }
    }
    if (((batch_idx[0i] + 1i) < podArgs.batch_size_plus_1)) {
      let b_idx_1 : i32 = batch_idx[0i];
      let LH_start : i32 = (tile_id[0i] * 32i);
      let q_indptr_val : i32 = q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)];
      kv_chunk_len[0i] = (kv_indptr[((b_idx_1 + podArgs.kv_indptr_elem_offset) + 1i)] - kv_indptr[(b_idx_1 + podArgs.kv_indptr_elem_offset)]);
      workgroupBarrier();
      if (i32(threadIdx.y) < 1i) {
        m_smem[i32(threadIdx.x)] = -5.000000e+04f;
        d_smem[i32(threadIdx.x)] = 1.000000e+00f;
      }
      for (var li_1 : i32 = 0; li_1 < 4i; li_1++) {
        for (var lj_1 : i32 = 0; lj_1 < 6i; lj_1++) {
          O_local[((li_1 * 6i) + lj_1)] = 0.000000e+00f;
        }
      }
      workgroupBarrier();
      for (var li_lj_fused_0 : i32 = 0; li_lj_fused_0 < 6i; li_lj_fused_0++) {
        for (var li_lj_fused_3_s : i32 = 0; li_lj_fused_3_s < 4i; li_lj_fused_3_s++) {
          if ((((((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) / 96i) + q_indptr_val) + LH_start) < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
            var condval : f16;
            if ((podArgs.rotary_mode == 1i)) {
              let freq : f32 = ((f32(q_rope_position[((((((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) / 96i) + q_indptr_val) + LH_start) + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale) / pow(podArgs.rope_theta, (f32(((((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) % 48i) * 2i)) / 9.600000e+01f)));
              var condval_1 : f16;
              if (((((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) % 96i) < 48i)) {
                condval_1 = (q[(((((((((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) / 96i) * 3072i) + (q_indptr_val * 3072i)) + (LH_start * 3072i)) + (i32(blockIdx.y) * 96i)) + (((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) % 96i)) + 48i)] * -1.000000e+00h);
} else {
                condval_1 = q[(((((((((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) / 96i) * 3072i) + (q_indptr_val * 3072i)) + (LH_start * 3072i)) + (i32(blockIdx.y) * 96i)) + (((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) % 96i)) - 48i)];
}
              condval = f16(fma(sin(freq), f32(condval_1), (cos(freq) * f32(q[((((((((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) / 96i) * 3072i) + (q_indptr_val * 3072i)) + (LH_start * 3072i)) + (i32(blockIdx.y) * 96i)) + (((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) % 96i))]))));
} else {
              condval = q[((((((((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) / 96i) * 3072i) + (q_indptr_val * 3072i)) + (LH_start * 3072i)) + (i32(blockIdx.y) * 96i)) + (((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s) % 96i))];
}
            Q_smem[((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s)] = condval;
          } else {
            Q_smem[((((li_lj_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + li_lj_fused_3_s)] = 0.000000e+00h;
          }
        }
      }
      workgroupBarrier();
      for (var iterator : i32 = 0; iterator < ((kv_chunk_len[0i] + 31i)>>5u); iterator++) {
        let L_kv_base : i32 = kv_indptr[(b_idx_1 + podArgs.kv_indptr_elem_offset)];
        workgroupBarrier();
        for (var lz_ly_fused_0 : i32 = 0; lz_ly_fused_0 < 6i; lz_ly_fused_0++) {
          for (var lz_ly_fused_3_s : i32 = 0; lz_ly_fused_3_s < 4i; lz_ly_fused_3_s++) {
            if (((iterator * 32i) + (((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) / 96i)) < kv_chunk_len[0i]) {
              var condval_2 : f16;
              if ((podArgs.rotary_mode == 1i)) {
                let freq_1 : f32 = ((f32(q_rope_position[((((iterator * 32i) + (((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) / 96i)) + L_kv_base) + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale) / pow(podArgs.rope_theta, (f32(((((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) % 48i) * 2i)) / 9.600000e+01f)));
                var condval_3 : f16;
                if (((((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) % 96i) < 48i)) {
                  condval_3 = (k[((((((iterator * 98304i) + ((((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) / 96i) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) % 96i)) + 48i)] * -1.000000e+00h);
} else {
                  condval_3 = k[((((((iterator * 98304i) + ((((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) / 96i) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) % 96i)) - 48i)];
}
                condval_2 = f16(fma(sin(freq_1), f32(condval_3), (cos(freq_1) * f32(k[(((((iterator * 98304i) + ((((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) / 96i) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) % 96i))]))));
} else {
                condval_2 = k[(((((iterator * 98304i) + ((((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) / 96i) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) % 96i))];
}
              K_smem[((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s)] = condval_2;
              V_smem[((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s)] = v[(((((iterator * 98304i) + ((((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) / 96i) * 3072i)) + (L_kv_base * 3072i)) + (i32(blockIdx.y) * 96i)) + (((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s) % 96i))];
            } else {
              K_smem[((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s)] = 0.000000e+00h;
              V_smem[((((lz_ly_fused_0 * 512i) + (i32(threadIdx.y) * 128i)) + (i32(threadIdx.x) * 4i)) + lz_ly_fused_3_s)] = 0.000000e+00h;
            }
          }
        }
        workgroupBarrier();
        for (var li_1_init : i32 = 0; li_1_init < 2i; li_1_init++) {
          for (var lj_1_init : i32 = 0; lj_1_init < 4i; lj_1_init++) {
            S_local[((li_1_init * 4i) + lj_1_init)] = 0.000000e+00f;
          }
        }
        for (var lk_0 : i32 = 0; lk_0 < 12i; lk_0++) {
          for (var li_1_1 : i32 = 0; li_1_1 < 2i; li_1_1++) {
            for (var lj_1_1 : i32 = 0; lj_1_1 < 4i; lj_1_1++) {
              for (var lk_1 : i32 = 0; lk_1 < 8i; lk_1++) {
                S_local[((li_1_1 * 4i) + lj_1_1)] = fma(((f32(Q_smem[(((((i32(threadIdx.y) * 768i) + ((i32(threadIdx.x)>>3u) * 192i)) + (li_1_1 * 96i)) + (lk_0 * 8i)) + lk_1)]) * f32(K_smem[(((((i32(threadIdx.x) & 7i) * 384i) + (lj_1_1 * 96i)) + (lk_0 * 8i)) + lk_1)])) * podArgs.sm_scale), 1.442695e+00f, S_local[((li_1_1 * 4i) + lj_1_1)]);
              }
            }
          }
        }
        workgroupBarrier();
        for (var li_1_2 : i32 = 0; li_1_2 < 2i; li_1_2++) {
          for (var lj_1_2 : i32 = 0; lj_1_2 < 4i; lj_1_2++) {
            S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>3u) * 64i)) + (li_1_2 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) + lj_1_2)] = S_local[((li_1_2 * 4i) + lj_1_2)];
          }
        }
        workgroupBarrier();
        if (i32(threadIdx.y) < 1i) {
          m_prev[0i] = m_smem[i32(threadIdx.x)];
          m_new[0i] = m_smem[i32(threadIdx.x)];
          for (var j : i32 = 0; j < 32i; j++) {
            let cse_v4 : i32 = ((iterator * 32i) + j);
            let cse_v3 : i32 = ((iterator * 64i) + (j * 2i));
            if ((cse_v4 < kv_chunk_len[0i]) && ((cse_v4 < ((kv_chunk_len[0i] + mn_indptr[(b_idx_1 + podArgs.mn_indptr_elem_offset)]) - mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)])) || ((mask[((((mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)] * 2i) + cse_v3) + podArgs.mask_elem_offset) - (kv_chunk_len[0i] * 2i))] <= mask[((((((i32(threadIdx.x) * 2i) + (LH_start * 2i)) + (mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)] * 2i)) + (q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)] * 2i)) + podArgs.mask_elem_offset) - (q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)] * 2i))]) && (mask[((((((i32(threadIdx.x) * 2i) + (LH_start * 2i)) + (mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)] * 2i)) + (q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)] * 2i)) + podArgs.mask_elem_offset) - (q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)] * 2i))] < mask[(((((mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)] * 2i) + cse_v3) + podArgs.mask_elem_offset) + 1i) - (kv_chunk_len[0i] * 2i))])))) {
              m_new[0i] = max(m_new[0i], S_smem[(((i32(threadIdx.y) * 1024i) + (i32(threadIdx.x) * 32i)) + j)]);
            }
          }
          d_new[0i] = (d_smem[i32(threadIdx.x)] * exp2((m_prev[0i] - m_new[0i])));
        }
        for (var j_1 : i32 = 0; j_1 < 32i; j_1++) {
          workgroupBarrier();
          if (i32(threadIdx.y) < 1i) {
            let cse_v6 : i32 = ((iterator * 32i) + j_1);
            let cse_v5 : i32 = ((iterator * 64i) + (j_1 * 2i));
            if ((cse_v6 < kv_chunk_len[0i]) && ((cse_v6 < ((kv_chunk_len[0i] + mn_indptr[(b_idx_1 + podArgs.mn_indptr_elem_offset)]) - mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)])) || ((mask[((((mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)] * 2i) + cse_v5) + podArgs.mask_elem_offset) - (kv_chunk_len[0i] * 2i))] <= mask[((((((i32(threadIdx.x) * 2i) + (LH_start * 2i)) + (mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)] * 2i)) + (q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)] * 2i)) + podArgs.mask_elem_offset) - (q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)] * 2i))]) && (mask[((((((i32(threadIdx.x) * 2i) + (LH_start * 2i)) + (mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)] * 2i)) + (q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)] * 2i)) + podArgs.mask_elem_offset) - (q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)] * 2i))] < mask[(((((mn_indptr[((b_idx_1 + podArgs.mn_indptr_elem_offset) + 1i)] * 2i) + cse_v5) + podArgs.mask_elem_offset) + 1i) - (kv_chunk_len[0i] * 2i))])))) {
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
        for (var li_1_init_1 : i32 = 0; li_1_init_1 < 4i; li_1_init_1++) {
          for (var lj_1_init_1 : i32 = 0; lj_1_init_1 < 6i; lj_1_init_1++) {
            O_local[((li_1_init_1 * 6i) + lj_1_init_1)] = (O_local[((li_1_init_1 * 6i) + lj_1_init_1)] * exp2((m_prev_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + li_1_init_1)] - m_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + li_1_init_1)])));
          }
        }
        for (var lk_0_1 : i32 = 0; lk_0_1 < 4i; lk_0_1++) {
          for (var lk_1_1 : i32 = 0; lk_1_1 < 8i; lk_1_1++) {
            for (var li_1_3 : i32 = 0; li_1_3 < 4i; li_1_3++) {
              for (var lj_1_3 : i32 = 0; lj_1_3 < 6i; lj_1_3++) {
                O_local[((li_1_3 * 6i) + lj_1_3)] = fma(S_smem[(((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>4u) * 128i)) + (li_1_3 * 32i)) + (lk_0_1 * 8i)) + lk_1_1)], f32(V_smem[((((lk_0_1 * 768i) + (lk_1_1 * 96i)) + ((i32(threadIdx.x) & 15i) * 6i)) + lj_1_3)]), O_local[((li_1_3 * 6i) + lj_1_3)]);
              }
            }
          }
        }
      }
      for (var li_1_4 : i32 = 0; li_1_4 < 4i; li_1_4++) {
        for (var lj_1_4 : i32 = 0; lj_1_4 < 6i; lj_1_4++) {
          let cur_L : i32 = (((((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) + LH_start) + li_1_4);
          if (cur_L < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
            output[((((cur_L * 3072i) + (i32(blockIdx.y) * 96i)) + ((i32(threadIdx.x) & 15i) * 6i)) + lj_1_4)] = f16((O_local[((li_1_4 * 6i) + lj_1_4)] / d_smem[(((i32(threadIdx.y) * 8i) + ((i32(threadIdx.x)>>4u) * 4i)) + li_1_4)]));
          }
        }
      }
      if (i32(threadIdx.y) < 1i) {
        let cur_L_1 : i32 = ((((i32(threadIdx.y) * 32i) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) + LH_start) + i32(threadIdx.x));
        if (cur_L_1 < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
          lse[((cur_L_1 * 32i) + i32(blockIdx.y))] = (m_smem[i32(threadIdx.x)] + log2(d_smem[i32(threadIdx.x)]));
        }
      }
      tile_id[0i] = (tile_id[0i] + 16i);
    }
  }
}
