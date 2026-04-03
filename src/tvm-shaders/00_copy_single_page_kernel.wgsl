// [0] copy_single_page_kernel (shader #0, 49 lines)
//----------------------------------------
// Function: copy_single_page_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read_write> pages : array<f16>;

struct PODArgs {
  copy_length: i32,
  num_pages: i32,
  page_size: i32,
  pages_elem_offset: i32,
  src_page_id: i32,
  tgt_page_id: i32,
  packGridDimX: u32
}
@group(0) @binding(1) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn copy_single_page_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if ((v__1 - (podArgs.copy_length * 12i)) < 0i) {
    let rmod : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rmod_1 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rmod_2 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rdiv : i32 = (((v__1 * 256i) + i32(threadIdx.x)) / (podArgs.copy_length * 96i));
    let rmod_3 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rmod_4 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rmod_5 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rdiv_1 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) / (podArgs.copy_length * 96i));
    pages[((((((select((rmod + (podArgs.copy_length * 96i)), rmod, ((((podArgs.copy_length * 96i) >= 0i) && (rmod >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod <= 0i)))) / 96i) + ((select((rmod_1 + (podArgs.copy_length * 96i)), rmod_1, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_1 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_1 <= 0i)))) % 96i)>>31u)) * 96i) + ((((podArgs.tgt_page_id * 64i) + select((rdiv - 1i), rdiv, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_2 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_2 <= 0i))))) * podArgs.page_size) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))] = pages[((((((select((rmod_3 + (podArgs.copy_length * 96i)), rmod_3, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_3 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_3 <= 0i)))) / 96i) + ((select((rmod_4 + (podArgs.copy_length * 96i)), rmod_4, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_4 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_4 <= 0i)))) % 96i)>>31u)) * 96i) + ((((podArgs.src_page_id * 64i) + select((rdiv_1 - 1i), rdiv_1, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_5 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_5 <= 0i))))) * podArgs.page_size) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))];
    let rmod_6 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rmod_7 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rmod_8 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rdiv_2 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) / (podArgs.copy_length * 96i));
    let rmod_9 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rmod_10 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rmod_11 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) % (podArgs.copy_length * 96i));
    let rdiv_3 : i32 = (((v__1 * 256i) + i32(threadIdx.x)) / (podArgs.copy_length * 96i));
    pages[((((((select((rmod_6 + (podArgs.copy_length * 96i)), rmod_6, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_6 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_6 <= 0i)))) / 96i) + ((select((rmod_7 + (podArgs.copy_length * 96i)), rmod_7, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_7 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_7 <= 0i)))) % 96i)>>31u)) * 96i) + (((((podArgs.tgt_page_id * 64i) + select((rdiv_2 - 1i), rdiv_2, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_8 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_8 <= 0i))))) + 32i) * podArgs.page_size) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))] = pages[((((((select((rmod_9 + (podArgs.copy_length * 96i)), rmod_9, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_9 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_9 <= 0i)))) / 96i) + ((select((rmod_10 + (podArgs.copy_length * 96i)), rmod_10, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_10 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_10 <= 0i)))) % 96i)>>31u)) * 96i) + (((((podArgs.src_page_id * 64i) + select((rdiv_3 - 1i), rdiv_3, ((((podArgs.copy_length * 96i) >= 0i) && (rmod_11 >= 0i)) || (((podArgs.copy_length * 96i) < 0i) && (rmod_11 <= 0i))))) + 32i) * podArgs.page_size) * 96i)) + podArgs.pages_elem_offset) + (((v__1 * 64i) + i32(threadIdx.x)) % 96i))];
  }
}
