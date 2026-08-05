/**
 * INT4 DEQUANT MATMUL — generator for the whole int4_matmul family.
 *
 * Replaces 9 hand-maintained .wgsl files that differed on exactly four knobs:
 *   outF32     f16 vs f32 output (one cast at the write — LM head wants f32)
 *   rowsPerWG  1 / 4 / 8 output rows per workgroup (input-bandwidth tiling)
 *   subgroups  subgroupAdd (32 threads) vs tree reduction (64 threads)
 *   m          1 (GEMV decode) or 4 (batched GEMM for spec-decode/prefill)
 *
 * Everything else — bindings, PODArgs, dequant math — is identical, so the
 * shaders stay auditable: the emitted WGSL is plain readable text, and
 * `debugDumpAll()` prints every shipped variant. To read one from a shell:
 *
 *   node -e "import('./src/compiler/shaders/int4_matmul.gen.ts')
 *              .then(m => console.log(m.debugDumpAll()))"
 *
 * Math: output[row] = dot(input, dequant(weights[row])) with
 * dequant(nibble) = (nibble - 7) * scale. The per-group scale is factored out
 * of the 8 unpacked terms — s * (Σ xᵢ·(nᵢ-7)) — one multiply per u32 word
 * instead of eight, and slightly more accurate. Accumulation is f32
 * (fixes TVM's f16 precision loss). Dimensions arrive via PODArgs uniforms,
 * so the same shader serves K=3072 and K=8192; no model-shape literals here.
 *
 * Bindings (match the old int4_matmul.wgsl 1:1 so bind groups are reused):
 *   @binding(0) output   array<f16|f32> read_write
 *   @binding(1) input    array<f16>     read
 *   @binding(2) scales   array<f16>     read
 *   @binding(3) weights  array<u32>     read
 *   @binding(4) podArgs  uniform {K_PACKED, SCALES_PER_ROW, packGridDimX}
 */

export interface Int4MatmulOpts {
  /** f32 output (LM head logits) instead of f16. */
  outF32?: boolean
  /** Output rows per workgroup: 1, 4 or 8. >1 requires subgroups. */
  rowsPerWG?: 1 | 4 | 8
  /** Use subgroupAdd (32-thread WG) instead of a 64-thread tree reduction. */
  subgroups?: boolean
  /** Batched input rows: 1 (GEMV) or 4 (M=4 GEMM). m=4 requires subgroups. */
  m?: 1 | 4
  /**
   * vec4<u32> loads (?vec4=1 experiment): weights read 16 bytes (32 nibbles)
   * per load, activations read as vec4<u32> reinterpreted via unpack2x16float
   * (8 f16 per load). The bound GPUBuffers are unchanged — only the WGSL
   * array element type differs. Requires subgroups (32-thread WG) and m=1;
   * K must be a multiple of 1024 (each thread consumes 32 K-elements per
   * loop iteration). Built and correctness-tested; not yet measured.
   */
  vec4?: boolean
  /**
   * Half-unroll sibling of `vec4` for K % 512 == 0 shapes (Qwen3's d=2560
   * and ffn=9728 fail the K % 1024 gate; both are multiples of 512). Each
   * thread consumes 16 K-elements per iteration instead of 32: weights read
   * as vec2<u32> (8 bytes = 16 nibbles = half an int4 scale group, so the
   * scale index is the vec2 index >> 1), activations still as vec4<u32>
   * (two loads). Same bindings/buffers as every other variant. Requires
   * subgroups + m=1; mutually exclusive with `vec4`.
   */
  vec4Half?: boolean
  /**
   * Runtime-M batched GEMM (chunked prefill): the m=4 register block loops
   * over ceil(M/4) row blocks, with M supplied at dispatch time via a 4th
   * PODArgs field (M_ROWS). Weight nibbles are unpacked once per weight word
   * per block and reused across the 4 batch rows (4× weight-traffic
   * amortization; the 4-row × K weight tile stays L2-resident across
   * blocks). Loads past M_ROWS read in-bounds garbage (the caller sizes the
   * input buffer to the chunk CAPACITY, a multiple of 4); writes are guarded.
   * Requires subgroups + rowsPerWG=4.
   */
  mDyn?: boolean
  /**
   * Affine (MLX-style) quantization instead of the symmetric MLC scheme:
   * `w = scale·q + bias` over groups of 64, rather than `w = (nibble−7)·scale`
   * over groups of 32. Adds a per-group bias tensor at `@binding(5)`.
   *
   * The nibbles are read unchanged — only the metadata differs — via
   *
   *   Σ xᵢwᵢ = s·Σ xᵢqᵢ + b·Σ xᵢ
   *
   * `Σ xᵢ` is accumulated per thread over exactly the 8 values that thread
   * already loaded, and multiplied by that thread's own group bias — summing
   * across threads then lands each group's bias term exactly once. (Adding a
   * whole-group Σx per thread would double-count it 8×, since 8 consecutive
   * threads share one group-64 scale.)
   *
   * The symmetric path's `−7` per nibble is NOT applied here. It exists only
   * because MLC stores `w = (nibble−7)·scale`; folding it in as
   * `s·Σx(q−7) + (7s+b)·Σx` is algebraically identical and was how this variant
   * first shipped. Dropping it is NOT a speed win — an A/B of the two shaders in
   * one session, 9 paired runs on the Qwen3.6 gate projection, measured 27.00 µs
   * both ways (0.0%); the subtract folds into the multiply-add. It is kept out
   * because the caller then hands MLX's bias straight through instead of
   * materialising a derived `7·scale + bias` tensor at load time.
   *
   * `SCALES_PER_ROW` means K/64 for this variant, not K/32.
   */
  affine?: boolean
  /**
   * MoE indirection: `workgroup_id.z` is a SLOT (0..top_k-1), and the expert row
   * for that slot is read from an `ids` buffer at `@binding(6)`, so one dispatch
   * covers every selected expert instead of one dispatch each. Measured: 960
   * per-expert dispatches cost 7.6 ms/token against 1.0 ms when the expert is a
   * grid dimension, at 7.9 µs of launch overhead apiece.
   *
   * Note this changes what `z` means. Every other variant flattens the grid as
   * `(z * gridDim.x + x)`; here `z` is the slot and the row comes from `x` alone.
   *
   * Slot striding for the activation and output buffers comes from PODArgs, so
   * one kernel serves gate/up (input shared across slots, output strided) and
   * down (both strided). Requires `affine`.
   */
  moe?: boolean
}

/** Entry-point name for a variant (same names the old .wgsl files used). */
export function int4MatmulEntry(opts: Int4MatmulOpts = {}): string {
  const { outF32 = false, rowsPerWG = 1, subgroups = false, m = 1, vec4 = false, vec4Half = false, mDyn = false, affine = false, moe = false } = opts
  if (mDyn) return 'int4_matmul_batched_dyn'
  if (m === 4) return 'int4_matmul_batched_m4'
  let name = 'int4_matmul'
  if (outF32) name += '_f32'
  if (rowsPerWG === 4) name += '_tiled'
  else if (rowsPerWG === 8) name += '_tiled8'
  else if (subgroups) name += '_sg'
  if (vec4) name += '_vec4'
  else if (vec4Half) name += '_vec4h'
  if (affine) name += '_affine'
  if (moe) name += '_moe'
  return name
}

// Σ xᵢ·(nibble(p, i) - 7), scale factored out by the caller. With
// `offset = false` the -7 is dropped: affine (MLX) weights are `s·q + b`, so
// the nibble is used raw and the bias rides along in its own term.
function dequantDot(p: string, x: (i: number) => string, indent: string, offset = true): string {
  const terms: string[] = []
  for (let i = 0; i < 8; i++) {
    const shift = String(i * 4).padStart(2)
    const q = `f32((${p} >> ${shift}u) & 15u)`
    terms.push(offset ? `${x(i)} * (${q} - 7.0)` : `${x(i)} * ${q}`)
  }
  const pairs: string[] = []
  for (let i = 0; i < 8; i += 2) pairs.push(`${terms[i]} + ${terms[i + 1]}`)
  return pairs.join(`\n${indent}+ `)
}

/** Render the WGSL for one variant. */
export function int4MatmulWGSL(opts: Int4MatmulOpts = {}): string {
  const { outF32 = false, rowsPerWG = 1, subgroups = false, m = 1, vec4 = false, vec4Half = false, mDyn = false, affine = false, moe = false } = opts
  if (moe && !affine) throw new Error('moe requires affine (the MoE weights are MLX affine-quantised)')
  if (affine && (m !== 1 || vec4 || vec4Half || mDyn))
    throw new Error('affine is implemented for the scalar path only (m=1, no vec4/vec4Half/mDyn)')
  if (rowsPerWG !== 1 && !subgroups) throw new Error('rowsPerWG > 1 requires subgroups')
  if (m === 4 && (!subgroups || rowsPerWG !== 4)) throw new Error('m=4 requires subgroups + rowsPerWG=4')
  if (vec4 && !subgroups) throw new Error('vec4 requires subgroups (32-thread WG keeps K=3072 divisible)')
  if (vec4 && m !== 1) throw new Error('vec4 not implemented for m=4')
  if (vec4Half && (vec4 || !subgroups || m !== 1))
    throw new Error('vec4Half requires subgroups + m=1 and is mutually exclusive with vec4')
  if (mDyn) {
    if (!subgroups || rowsPerWG !== 4 || m !== 1 || vec4 || vec4Half || outF32)
      throw new Error('mDyn requires subgroups + rowsPerWG=4 (no vec4/outF32/m=4)')
    return int4MatmulBatchedDynWGSL()
  }

  const entry = int4MatmulEntry(opts)
  const outType = outF32 ? 'f32' : 'f16'
  const wgSize = subgroups ? 32 : 64
  const cast = (v: string) => (outF32 ? v : `f16(${v})`)
  const rows = [...Array(rowsPerWG).keys()]
  const batches = [...Array(m).keys()]

  const header = `// GENERATED by src/compiler/shaders/int4_matmul.gen.ts — do not edit by hand.
// Variant: ${entry} (out=${outType}, rowsPerWG=${rowsPerWG}, ${subgroups ? 'subgroupAdd' : 'tree reduction'}${m > 1 ? `, M=${m}` : ''}${vec4 ? ', vec4 loads' : ''}${vec4Half ? ', vec2 weight loads (K%512)' : ''})
${subgroups ? '// Requires subgroup size 32 (gated in chat.ts via a shader probe).\n' : ''}${vec4 ? '// vec4 experiment: same GPUBuffers as the scalar variant — input (f16 data) and\n// weights (u32 data) are merely re-declared as array<vec4<u32>> so each load\n// moves 16 bytes; activations unpack via unpack2x16float.\n' : ''}${vec4Half ? '// vec4h sibling: same GPUBuffers as the scalar variant — input (f16 data) is\n// re-declared as array<vec4<u32>> (16-byte loads) and weights (u32 data) as\n// array<vec2<u32>> (8-byte loads). Half the per-thread unroll of _vec4, so the\n// K-divisibility constraint relaxes from 1024 to 512.\n' : ''}
enable f16;
${subgroups ? 'enable subgroups;\n' : ''}
@group(0) @binding(0) var<storage, read_write> output_buf : array<${outType}>;
@group(0) @binding(1) var<storage, read> input_buf : array<${vec4 || vec4Half ? 'vec4<u32>' : 'f16'}>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<${vec4 ? 'vec4<u32>' : vec4Half ? 'vec2<u32>' : 'u32'}>;
${affine ? '@group(0) @binding(5) var<storage, read> biases : array<f16>;   // MLX bias, verbatim\n' : ''}${moe ? '@group(0) @binding(6) var<storage, read> ids : array<u32>;       // expert row per slot\n' : ''}
struct PODArgs {
  K_PACKED: u32,        // K / 8  (u32 words per weight row)
  SCALES_PER_ROW: u32,  // K / ${affine ? '64' : '32'} (int4 group scales per weight row)
  packGridDimX: u32${moe ? ',' : ''}     // N (number of output ${m > 1 ? 'columns' : 'elements'})${moe ? `
  IN_SLOT_STRIDE: u32,  // activation elements per slot (0 = all slots share one input)
  OUT_SLOT_STRIDE: u32  // output elements per slot` : ''}
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;
`

  const prologue = `
@compute @workgroup_size(${wgSize}, 1, 1)
fn ${entry}(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
${moe ? `  // z is the SLOT here, not part of the row flattening the other variants use.
  let slot : u32 = blockIdx.z;
  let row_base : u32 = blockIdx.x * ${rowsPerWG}u;` : `  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * ${rowsPerWG}u;`}
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let tid : u32 = threadIdx.x;
${moe ? `  let expert : u32 = ids[slot];
  let wBase : u32 = expert * podArgs.packGridDimX * K_PACKED;
  let sBase : u32 = expert * podArgs.packGridDimX * SCALES_PER_ROW;
  let inBase : u32 = slot * podArgs.IN_SLOT_STRIDE;
  let outBase : u32 = slot * podArgs.OUT_SLOT_STRIDE;
` : ''}`

  // Per-thread K-chunk loop. Each thread strides by the workgroup width.
  const rowDecl = rows.map((r) => `  let r${r} = row_base${r ? ` + ${r}u` : ''};`).join('\n')

  let body: string
  if (vec4) {
    // vec4 path: each thread consumes one vec4<u32> of weights per row per
    // iteration (4 packed words = 32 nibbles = exactly one int4 scale group,
    // so the scale index IS the vec4 index) and 4 vec4<u32> of activations
    // (32 f16). Loads drop 4× (weights) / 8× (activations) vs the scalar path.
    const accDecl = rows.map((r) => `  var acc${r} : f32 = 0.0;`).join('\n')
    const vecs = ['xa', 'xb', 'xc', 'xd']
    const loads = vecs
      .map((v, j) => `    let ${v} = input_buf[v_off * 4u${j ? ` + ${j}u` : ''}];`)
      .join('\n')
    const unpacks = vecs
      .map(
        (v) =>
          '    ' +
          ['x', 'y', 'z', 'w']
            .map((c, k) => `let ${v}${k} = unpack2x16float(${v}.${c});`)
            .join(' '),
      )
      .join('\n')
    // Activation accessor for word j (0..3), element i (0..7).
    const x = (j: number) => (i: number) => `${vecs[j]}${i >> 1}.${i & 1 ? 'y' : 'x'}`
    const rowBlocks = rows
      .map(
        (r) => `    { // row r${r}: one vec4 weight load + one scale (4 words = 1 scale group)
      let w = weights[r${r} * KPV + v_off];
      let s = f32(scales[r${r} * SCALES_PER_ROW + v_off]);
      acc${r} = acc${r} + s * (
          ${dequantDot('w.x', x(0), '        ')}
        + ${dequantDot('w.y', x(1), '        ')}
        + ${dequantDot('w.z', x(2), '        ')}
        + ${dequantDot('w.w', x(3), '        ')});
    }`,
      )
      .join('\n')
    body = `
  // vec4<u32> words per weight row. Requires K % ${wgSize * 32} == 0
  // (each of the ${wgSize} threads consumes 32 K-elements per iteration).
  let KPV : u32 = K_PACKED / 4u;

${accDecl}

${rowDecl}

  for (var chunk : u32 = 0u; chunk < KPV / ${wgSize}u; chunk = chunk + 1u) {
    let v_off : u32 = tid + chunk * ${wgSize}u;   // vec4 index == int4 scale-group index

${loads}
${unpacks}

${rowBlocks}
  }
`
  } else if (vec4Half) {
    // vec4h path: each thread consumes one vec2<u32> of weights per row per
    // iteration (2 packed words = 16 nibbles = HALF an int4 scale group, so
    // the scale index is the vec2 index >> 1) and 2 vec4<u32> of activations
    // (16 f16). Weight loads drop 2×, activation loads 8× vs the scalar path.
    const accDecl = rows.map((r) => `  var acc${r} : f32 = 0.0;`).join('\n')
    const vecs = ['xa', 'xb']
    const loads = vecs
      .map((v, j) => `    let ${v} = input_buf[v_off * 2u${j ? ` + ${j}u` : ''}];`)
      .join('\n')
    const unpacks = vecs
      .map(
        (v) =>
          '    ' +
          ['x', 'y', 'z', 'w']
            .map((c, k) => `let ${v}${k} = unpack2x16float(${v}.${c});`)
            .join(' '),
      )
      .join('\n')
    // Activation accessor for word j (0..1), element i (0..7).
    const x = (j: number) => (i: number) => `${vecs[j]}${i >> 1}.${i & 1 ? 'y' : 'x'}`
    const rowBlocks = rows
      .map(
        (r) => `    { // row r${r}: one vec2 weight load + one scale (2 words = half a scale group)
      let w = weights[r${r} * KPV + v_off];
      let s = f32(scales[r${r} * SCALES_PER_ROW + (v_off >> 1u)]);
      acc${r} = acc${r} + s * (
          ${dequantDot('w.x', x(0), '        ')}
        + ${dequantDot('w.y', x(1), '        ')});
    }`,
      )
      .join('\n')
    body = `
  // vec2<u32> words per weight row. Requires K % ${wgSize * 16} == 0
  // (each of the ${wgSize} threads consumes 16 K-elements per iteration).
  let KPV : u32 = K_PACKED / 2u;

${accDecl}

${rowDecl}

  for (var chunk : u32 = 0u; chunk < KPV / ${wgSize}u; chunk = chunk + 1u) {
    let v_off : u32 = tid + chunk * ${wgSize}u;   // vec2 index; scale group = v_off >> 1

${loads}
${unpacks}

${rowBlocks}
  }
`
  } else if (m === 1) {
    const accDecl = rows.map((r) => `  var acc${r} : f32 = 0.0;`).join('\n')
    const loads =
      rowsPerWG === 1
        ? [...Array(8).keys()]
            .map((i) => `    let i${i} = f32(input_buf[base${i ? ` + ${i}u` : ''}]);`)
            .join('\n')
        : `    // Load 8 inputs once, reuse across all ${rowsPerWG} rows.\n` +
          [...Array(8).keys()]
            .map((i) => `    let i${i} = f32(input_buf[base${i ? ` + ${i}u` : ''}]);`)
            .join('\n')
    const rowBlocks = rows
      .map(
        (r) => `    { // row r${r}: one (packed, scale${affine ? ', bias' : ''}) load, scale factored out of the 8 terms
      let p = weights[${moe ? 'wBase + ' : ''}r${r} * K_PACKED + w_offset];
      let s = f32(scales[${moe ? 'sBase + ' : ''}r${r} * SCALES_PER_ROW + sc_idx]);
${affine ? `      let b = f32(biases[${moe ? 'sBase + ' : ''}r${r} * SCALES_PER_ROW + sc_idx]);\n` : ''}      acc${r} = acc${r} + s * (
          ${dequantDot('p', (i) => `i${i}`, '        ', !affine)})${affine ? ' + b * xs' : ''};
    }`,
      )
      .join('\n')
    body = `
${accDecl}

${rowDecl}

  for (var chunk : u32 = 0u; chunk < K_PACKED / ${wgSize}u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * ${wgSize}u;
    let base : u32 = ${moe ? 'inBase + ' : ''}w_offset * 8u;
    let sc_idx : u32 = w_offset >> ${affine ? '3' : '2'}u;

${loads}
${affine ? '\n    // Σx over exactly this thread\'s 8 values — paired with this thread\'s own\n    // group bias, so the group term is counted once across the 8 threads.\n    let xs : f32 = i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7;\n' : ''}
${rowBlocks}
  }
`
  } else {
    // M=4: load the 8 inputs of every batch row, dequantize each weight row
    // once (nibbles only — the scale multiplies the batch-row dot), and
    // accumulate m × rowsPerWG cells.
    const accDecl = batches
      .map((b) => '  ' + rows.map((r) => `var a${b}${r} : f32 = 0.0;`).join(' '))
      .join('\n')
    const loads = batches
      .map((b) =>
        [...Array(8).keys()]
          .map((i) => `    let b${b}_${i} = f32(input_buf[${b}u * K + base${i ? ` + ${i}u` : ''}]);`)
          .join('\n'),
      )
      .join('\n')
    const rowBlocks = rows
      .map((r) => {
        const nibbles = [...Array(8).keys()]
          .map((i) => `      let n${i} = f32((p >> ${String(i * 4).padStart(2)}u) & 15u) - 7.0;`)
          .join('\n')
        const accs = batches
          .map(
            (b) =>
              `      a${b}${r} = a${b}${r} + s * (b${b}_0*n0 + b${b}_1*n1 + b${b}_2*n2 + b${b}_3*n3 + b${b}_4*n4 + b${b}_5*n5 + b${b}_6*n6 + b${b}_7*n7);`,
          )
          .join('\n')
        return `    { // weight row r${r}: unpack nibbles once, reuse across all ${m} batch rows
      let p = weights[r${r} * K_PACKED + w_offset];
      let s = f32(scales[r${r} * SCALES_PER_ROW + sc_idx]);
${nibbles}
${accs}
    }`
      })
      .join('\n')
    body = `
  let K : u32 = K_PACKED * 8u;  // elements per input row

${accDecl}

${rowDecl}

  for (var chunk : u32 = 0u; chunk < K_PACKED / ${wgSize}u; chunk = chunk + 1u) {
    let w_offset : u32 = tid + chunk * ${wgSize}u;
    let base : u32 = w_offset * 8u;
    let sc_idx : u32 = w_offset >> 2u;

${loads}

${rowBlocks}
  }
`
  }

  let tail: string
  if (subgroups) {
    const sums =
      m === 1
        ? rows.map((r) => `  let sum${r} = subgroupAdd(acc${r});`).join('\n')
        : batches
            .map((b) => '  ' + rows.map((r) => `let s${b}${r} = subgroupAdd(a${b}${r});`).join(' '))
            .join('\n')
    const writes =
      m === 1
        ? rows.map((r) => `    output_buf[${moe ? 'outBase + ' : ''}r${r}] = ${cast(`sum${r}`)};`).join('\n')
        : `    let N = podArgs.packGridDimX;\n` +
          batches
            .map((b) =>
              rows.map((r) => `    output_buf[${b}u * N + r${r}] = ${cast(`s${b}${r}`)};`).join('\n'),
            )
            .join('\n')
    tail = `
  // Uniform subgroup sums (all 32 lanes always present and active).
${sums}

  if (tid == 0u) {
${writes}
  }
}
`
  } else {
    tail = `
  // Tree reduction in f32 across the 64 threads.
  red_buf[tid] = acc0;
  workgroupBarrier();
  if (tid < 32u) { red_buf[tid] = red_buf[tid] + red_buf[tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { red_buf[tid] = red_buf[tid] + red_buf[tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { red_buf[tid] = red_buf[tid] + red_buf[tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { red_buf[tid] = red_buf[tid] + red_buf[tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { red_buf[tid] = red_buf[tid] + red_buf[tid + 2u]; }
  workgroupBarrier();
  if (tid < 1u) { red_buf[tid] = red_buf[tid] + red_buf[tid + 1u]; }
  workgroupBarrier();

  if (tid == 0u) {
    output_buf[${moe ? 'outBase + ' : ''}r0] = ${cast('red_buf[0]')};
  }
}
`
  }

  const sharedMem = subgroups ? '' : '\nvar<workgroup> red_buf : array<f32, 64>;\n'
  return header + sharedMem + prologue + body + tail
}

/**
 * Runtime-M batched GEMM for chunked prefill: the m=4 register block of
 * int4_matmul_batched_m4 wrapped in a loop over ceil(M/4) row blocks, with M
 * arriving at dispatch time (PODArgs.M_ROWS). Same bindings as every other
 * variant; PODArgs gains the 4th field (it fits inside the existing 16-byte
 * uniform padding).
 */
function int4MatmulBatchedDynWGSL(): string {
  const rows = [0, 1, 2, 3]
  const batches = [0, 1, 2, 3]
  const loads = batches
    .map((b) =>
      [...Array(8).keys()]
        .map((i) => `      let b${b}_${i} = f32(input_buf[(mb * 4u + ${b}u) * K + base${i ? ` + ${i}u` : ''}]);`)
        .join('\n'),
    )
    .join('\n')
  const rowBlocks = rows
    .map((r) => {
      const nibbles = [...Array(8).keys()]
        .map((i) => `        let n${i} = f32((p >> ${String(i * 4).padStart(2)}u) & 15u) - 7.0;`)
        .join('\n')
      const accs = batches
        .map(
          (b) =>
            `        a${b}${r} = a${b}${r} + s * (b${b}_0*n0 + b${b}_1*n1 + b${b}_2*n2 + b${b}_3*n3 + b${b}_4*n4 + b${b}_5*n5 + b${b}_6*n6 + b${b}_7*n7);`,
        )
        .join('\n')
      return `      { // weight row r${r}: unpack nibbles once, reuse across the 4 batch rows
        let p = weights[r${r} * K_PACKED + w_offset];
        let s = f32(scales[r${r} * SCALES_PER_ROW + sc_idx]);
${nibbles}
${accs}
      }`
    })
    .join('\n')
  const sums = batches
    .map((b) => '    ' + rows.map((r) => `let s${b}${r} = subgroupAdd(a${b}${r});`).join(' '))
    .join('\n')
  const writes = batches
    .map(
      (b) => `      if (mb * 4u + ${b}u < M) {
${rows.map((r) => `        output_buf[(mb * 4u + ${b}u) * N + r${r}] = f16(s${b}${r});`).join('\n')}
      }`,
    )
    .join('\n')
  return `// GENERATED by src/compiler/shaders/int4_matmul.gen.ts — do not edit by hand.
// Variant: int4_matmul_batched_dyn (out=f16, rowsPerWG=4, subgroupAdd, runtime M)
// Requires subgroup size 32 (gated in chat.ts via a shader probe).
// Chunked-prefill GEMM: loops the m=4 register block over ceil(M/4) input-row
// blocks. Rows past M_ROWS are read (in-bounds — the caller sizes input to
// the chunk capacity, a multiple of 4) but never written.

enable f16;
enable subgroups;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;

struct PODArgs {
  K_PACKED: u32,        // K / 8  (u32 words per weight row)
  SCALES_PER_ROW: u32,  // K / 32 (int4 group scales per weight row)
  packGridDimX: u32,    // N (number of output columns)
  M_ROWS: u32           // valid input rows this dispatch (chunk seq_len)
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(32, 1, 1)
fn int4_matmul_batched_dyn(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  let row_base : u32 = (blockIdx.z * gridDim.x + blockIdx.x) * 4u;
  if (row_base >= podArgs.packGridDimX) { return; }

  let K_PACKED : u32 = podArgs.K_PACKED;
  let SCALES_PER_ROW : u32 = podArgs.SCALES_PER_ROW;
  let N : u32 = podArgs.packGridDimX;
  let M : u32 = podArgs.M_ROWS;
  let K : u32 = K_PACKED * 8u;  // elements per input row
  let tid : u32 = threadIdx.x;

${rows.map((r) => `  let r${r} = row_base${r ? ` + ${r}u` : ''};`).join('\n')}

  // Uniform loop bound (M is uniform) — subgroupAdd below stays uniform.
  for (var mb : u32 = 0u; mb * 4u < M; mb = mb + 1u) {
${batches.map((b) => '    ' + rows.map((r) => `var a${b}${r} : f32 = 0.0;`).join(' ')).join('\n')}

    for (var chunk : u32 = 0u; chunk < K_PACKED / 32u; chunk = chunk + 1u) {
      let w_offset : u32 = tid + chunk * 32u;
      let base : u32 = w_offset * 8u;
      let sc_idx : u32 = w_offset >> 2u;

${loads}

${rowBlocks}
    }

    // Uniform subgroup sums (all 32 lanes always present and active).
${sums}

    if (tid == 0u) {
${writes}
    }
  }
}
`
}

/** The shipped variants (the first 9 are 1:1 with the deleted .wgsl files;
 *  the 4 `_vec4` entries are the vec4-load variants — measured +4.5% on
 *  M2 Max, 2026-07-25, and now the default where sg32 holds; ?vec4=0 opts
 *  out — see BENCH.md). */
export const INT4_MATMUL_VARIANTS: ReadonlyArray<Int4MatmulOpts> = [
  {},                                            // int4_matmul
  { subgroups: true },                           // int4_matmul_sg
  { subgroups: true, rowsPerWG: 4 },             // int4_matmul_tiled
  { subgroups: true, rowsPerWG: 8 },             // int4_matmul_tiled8
  { outF32: true },                              // int4_matmul_f32
  { outF32: true, subgroups: true },             // int4_matmul_f32_sg
  { outF32: true, subgroups: true, rowsPerWG: 4 }, // int4_matmul_f32_tiled
  { outF32: true, subgroups: true, rowsPerWG: 8 }, // int4_matmul_f32_tiled8
  { subgroups: true, rowsPerWG: 4, m: 4 },       // int4_matmul_batched_m4
  { subgroups: true, rowsPerWG: 4, mDyn: true }, // int4_matmul_batched_dyn (chunked prefill, runtime M),
  { subgroups: true, vec4: true },                            // int4_matmul_sg_vec4
  { subgroups: true, rowsPerWG: 4, vec4: true },              // int4_matmul_tiled_vec4
  { outF32: true, subgroups: true, vec4: true },              // int4_matmul_f32_sg_vec4
  { outF32: true, subgroups: true, rowsPerWG: 4, vec4: true }, // int4_matmul_f32_tiled_vec4
  // K%512 half-unroll siblings (Qwen3's d=2560 / ffn=9728 miss the K%1024
  // gate; resolution falls back to these before giving up on wide loads):
  { subgroups: true, vec4Half: true },                            // int4_matmul_sg_vec4h
  { subgroups: true, rowsPerWG: 4, vec4Half: true },              // int4_matmul_tiled_vec4h
  { outF32: true, subgroups: true, vec4Half: true },              // int4_matmul_f32_sg_vec4h
  { outF32: true, subgroups: true, rowsPerWG: 4, vec4Half: true }, // int4_matmul_f32_tiled_vec4h
  // MLX affine (w = scale·q + bias, group 64) — second per-group tensor at @binding(5).
  { affine: true }, // int4_matmul_affine
  { outF32: true, affine: true }, // int4_matmul_f32_affine
  { subgroups: true, affine: true }, // int4_matmul_sg_affine
  { subgroups: true, rowsPerWG: 4, affine: true }, // int4_matmul_tiled_affine
  // The f32-output siblings are not optional extras: variants.ts's resolveMatmul
  // only returns a tiled pipeline when BOTH the f16 and the f32 sibling exist,
  // and falls through to the scalar path otherwise. Without _f32_tiled_affine an
  // untied lm_head lands on the scalar kernel, which is 248320 workgroups —
  // past the 65535 per-dimension limit, so it needs the z-fold; the tiled one is
  // 62080 and does not.
  { outF32: true, subgroups: true, affine: true }, // int4_matmul_f32_sg_affine
  { outF32: true, subgroups: true, rowsPerWG: 4, affine: true }, // int4_matmul_f32_tiled_affine
  // MoE: workgroup_id.z is the SLOT and the expert row comes from ids[] at
  // @binding(6), so one dispatch covers every selected expert.
  { affine: true, moe: true, subgroups: true, rowsPerWG: 4 } // int4_matmul_tiled_affine_moe
]

/** Human-auditable dump of every shipped variant. */
export function debugDumpAll(): string {
  return INT4_MATMUL_VARIANTS.map(
    (v) => `// ${'═'.repeat(70)}\n// ${int4MatmulEntry(v)}\n// ${'═'.repeat(70)}\n${int4MatmulWGSL(v)}`,
  ).join('\n')
}
