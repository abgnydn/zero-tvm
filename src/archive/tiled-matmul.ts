/**
 * TILED INT4 MATMUL — Drop-in replacement for TVM's matmul shaders.
 *
 * TVM's matmul: 64 threads, 1 output per workgroup, reloads input vector every time.
 * Ours: 256 threads, TILE_ROWS outputs per workgroup, input cached in shared memory.
 *
 * Same bindings, same data layout, same results — just faster.
 *
 * Binding layout (matches TVM exactly):
 *   @binding(0) output     : array<f16>         (read_write)
 *   @binding(1) input      : array<f16>         (read)
 *   @binding(2) scales     : array<f16>         (read)
 *   @binding(3) weights    : array<u32>         (read)  — int4 packed
 *   @binding(4) podArgs    : { packGridDimX }   (uniform)
 *
 * Weight format: each u32 holds 8 int4 values, dequant = (nibble - 7) * scale
 * Scale grouping: 1 scale per 32 input elements (4 packed u32s)
 */

export const TILE_ROWS = 4

/**
 * Generate a tiled int4 matmul WGSL shader.
 * @param dPacked - number of packed u32s per weight row (D/8). Extracted from TVM shader.
 * @param entryPoint - function name to match TVM's pipeline
 */
export function generateTiledMatmul(dPacked: number, entryPoint: string): string {
  const D = dPacked * 8
  const chunksPerThread = dPacked / 64  // TVM uses 64 threads for the dot product
  const scalesPerRow = D / 32

  return `
enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const TILE_ROWS : u32 = ${TILE_ROWS}u;
const THREADS_PER_ROW : u32 = 64u;
const D : u32 = ${D}u;
const D_PACKED : u32 = ${dPacked}u;
const CHUNKS : u32 = ${chunksPerThread}u;
const SCALES_PER_ROW : u32 = ${scalesPerRow}u;

var<workgroup> input_smem : array<f16, ${D}>;
var<workgroup> red_buf : array<f16, ${TILE_ROWS * 64}>;

@compute @workgroup_size(${TILE_ROWS * 64}, 1, 1)
fn ${entryPoint}(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  // Map workgroup to TILE_ROWS consecutive output rows
  let wg_id : u32 = blockIdx.z * gridDim.x + blockIdx.x;
  let base_row : i32 = i32(wg_id * TILE_ROWS);
  let row_offset : u32 = threadIdx.x / THREADS_PER_ROW;
  let lane : u32 = threadIdx.x % THREADS_PER_ROW;
  let row : i32 = base_row + i32(row_offset);

  // Bounds check: skip entire workgroup if past output range
  if (base_row > i32(podArgs.packGridDimX)) { return; }

  // === PHASE 1: Cooperatively load input vector into shared memory ===
  // ${TILE_ROWS * 64} threads load ${D} f16 values
  let elems_per_thread : u32 = (D + ${TILE_ROWS * 64}u - 1u) / ${TILE_ROWS * 64}u;
  for (var i : u32 = 0u; i < elems_per_thread; i = i + 1u) {
    let idx : u32 = threadIdx.x + i * ${TILE_ROWS * 64}u;
    if (idx < D) {
      input_smem[idx] = input_buf[idx];
    }
  }
  workgroupBarrier();

  // === PHASE 2: Compute partial dot product for this row ===
  var acc : f16 = 0.000000e+00h;

  if (row <= i32(podArgs.packGridDimX)) {
    for (var chunk : u32 = 0u; chunk < CHUNKS; chunk = chunk + 1u) {
      let w_idx : u32 = u32(row) * D_PACKED + lane + chunk * 64u;
      let packed : u32 = weights[w_idx];
      let s : f16 = scales[u32(row) * SCALES_PER_ROW + (lane + chunk * 64u) / 4u];
      let base : u32 = (lane + chunk * 64u) * 8u;

      acc = fma(input_smem[base],        (f16(((packed >>  0u) & 15u)) - 7.000000e+00h) * s, acc);
      acc = fma(input_smem[base + 1u],   (f16(((packed >>  4u) & 15u)) - 7.000000e+00h) * s, acc);
      acc = fma(input_smem[base + 2u],   (f16(((packed >>  8u) & 15u)) - 7.000000e+00h) * s, acc);
      acc = fma(input_smem[base + 3u],   (f16(((packed >> 12u) & 15u)) - 7.000000e+00h) * s, acc);
      acc = fma(input_smem[base + 4u],   (f16(((packed >> 16u) & 15u)) - 7.000000e+00h) * s, acc);
      acc = fma(input_smem[base + 5u],   (f16(((packed >> 20u) & 15u)) - 7.000000e+00h) * s, acc);
      acc = fma(input_smem[base + 6u],   (f16(((packed >> 24u) & 15u)) - 7.000000e+00h) * s, acc);
      acc = fma(input_smem[base + 7u],   (f16(((packed >> 28u) & 15u)) - 7.000000e+00h) * s, acc);
    }
  }

  // === PHASE 3: Workgroup reduction (64 threads → 1 per row) ===
  let red_idx : u32 = row_offset * THREADS_PER_ROW + lane;
  red_buf[red_idx] = acc;
  workgroupBarrier();
  if (lane < 32u) { red_buf[red_idx] = red_buf[red_idx] + red_buf[red_idx + 32u]; }
  workgroupBarrier();
  if (lane < 16u) { red_buf[red_idx] = red_buf[red_idx] + red_buf[red_idx + 16u]; }
  workgroupBarrier();
  if (lane < 8u) { red_buf[red_idx] = red_buf[red_idx] + red_buf[red_idx + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { red_buf[red_idx] = red_buf[red_idx] + red_buf[red_idx + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red_buf[red_idx] = red_buf[red_idx] + red_buf[red_idx + 2u]; }
  workgroupBarrier();
  if (lane < 1u) { red_buf[red_idx] = red_buf[red_idx] + red_buf[red_idx + 1u]; }
  workgroupBarrier();

  // === PHASE 4: Write output ===
  if (lane == 0u && row <= i32(podArgs.packGridDimX)) {
    output_buf[row] = red_buf[row_offset * THREADS_PER_ROW];
  }
}
`.trim()
}

/**
 * Detect if a WGSL shader is a TVM int4 matmul and extract its parameters.
 * Returns null if not a matmul, or { dPacked, entryPoint } if it is.
 */
export function detectMatmul(code: string, entryPoint: string): { dPacked: number } | null {
  // Skip shaders with type casts — they have different output semantics
  if (entryPoint.includes('cast')) return null
  // Must have the matmul reduction pattern with f16 workgroup buffer
  if (!code.includes('red_buf0 : array<f16, 64>')) return null
  // Must have int4 dequant pattern (4-bit with zero_point=7)
  if (!code.includes('& 15u)) - 7.000000e+00h)')) return null
  // Must use fma for accumulation
  if (!code.includes('fma(')) return null
  // Must be a simple matmul (5 bindings: out, in, scale, weight, uniform)
  const bindingCount = (code.match(/@binding\(/g) || []).length
  if (bindingCount !== 5) return null

  // Extract D_PACKED from weight access stride: `v__1 * XXXi`
  const strideMatch = code.match(/v__1 \* (\d+)i\) \+ i32\(threadIdx\.x\)\)/)
  if (!strideMatch) return null

  const dPacked = parseInt(strideMatch[1], 10)
  if (dPacked < 64 || dPacked > 4096) return null  // sanity check

  return { dPacked }
}
