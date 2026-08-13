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
  /**
   * 3-bit weights (MLX bits=3) instead of 4-bit nibbles. The values are a
   * CONTINUOUS LSB-first bitstream per row — value i occupies bits
   * [3i, 3i+3), crossing u32 boundaries freely (discovered empirically:
   * 10 values in a word's low 30 bits, the 11th straddling into the next).
   *
   * The kernel walks 8-VALUE BLOCKS: one block is 24 bits starting at
   * bit w_offset*24, so the in-word shift cycles {0,8,16,24} and a block
   * spans at most two words — and never past the row's final word, because
   * rows are 3K/32 words and the sh=24 case cannot fall on the last block
   * (K % 64 == 0 makes the row's bit length a multiple of 192).
   *
   * Group-64 boundaries stay word-aligned (192 bits = 6 words), so
   * SCALES_PER_ROW keeps meaning K/64 and sc_idx keeps being
   * blockIndex >> 3. PODArgs.K_PACKED becomes 3K/32 (words per row) —
   * the LOOP bound is derived from SCALES_PER_ROW instead, since K_PACKED
   * no longer counts 8-value blocks. Requires `affine` (the q3 experts are
   * MLX-affine); K % 256 == 0 for the 32-thread variants.
   */
  q3?: boolean
}

/** Entry-point name for a variant (same names the old .wgsl files used). */
export function int4MatmulEntry(opts: Int4MatmulOpts = {}): string {
  const { outF32 = false, rowsPerWG = 1, subgroups = false, m = 1, vec4 = false, vec4Half = false, mDyn = false, affine = false, moe = false, q3 = false } = opts
  if (mDyn) return affine ? 'int4_matmul_batched_dyn_affine' : 'int4_matmul_batched_dyn'
  if (m === 4) return 'int4_matmul_batched_m4'
  let name = 'int4_matmul'
  if (outF32) name += '_f32'
  if (rowsPerWG === 4) name += '_tiled'
  else if (rowsPerWG === 8) name += '_tiled8'
  else if (subgroups) name += '_sg'
  if (vec4) name += '_vec4'
  else if (vec4Half) name += '_vec4h'
  if (affine) name += '_affine'
  if (q3) name += '_q3'
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
  const { outF32 = false, rowsPerWG = 1, subgroups = false, m = 1, vec4 = false, vec4Half = false, mDyn = false, affine = false, moe = false, q3 = false } = opts
  if (moe && !affine) throw new Error('moe requires affine (the MoE weights are MLX affine-quantised)')
  if (q3 && !affine) throw new Error('q3 requires affine (3-bit ships only in the MLX-affine layout)')
  // mDyn is the exception: chunked prefill needs a batched affine GEMM, and
  // without it every MLX checkpoint prefills one token at a time.
  if (affine && (m !== 1 || vec4 || vec4Half))
    throw new Error('affine is implemented for the scalar and mDyn paths only (m=1, no vec4/vec4Half)')
  if (rowsPerWG !== 1 && !subgroups) throw new Error('rowsPerWG > 1 requires subgroups')
  if (m === 4 && (!subgroups || rowsPerWG !== 4)) throw new Error('m=4 requires subgroups + rowsPerWG=4')
  if (vec4 && !subgroups) throw new Error('vec4 requires subgroups (32-thread WG keeps K=3072 divisible)')
  if (vec4 && m !== 1) throw new Error('vec4 not implemented for m=4')
  if (vec4Half && (vec4 || !subgroups || m !== 1))
    throw new Error('vec4Half requires subgroups + m=1 and is mutually exclusive with vec4')
  if (mDyn) {
    if (!subgroups || rowsPerWG !== 4 || m !== 1 || vec4 || vec4Half || outF32)
      throw new Error('mDyn requires subgroups + rowsPerWG=4 (no vec4/outF32/m=4)')
    return int4MatmulBatchedDynWGSL(affine)
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
    const rowBlocksQ3 = rows
      .map(
        (r) => `    { // row r${r}: 24-bit window from the row's bitstream, one (scale, bias) load
      let wi${r} : u32 = ${moe ? 'wBase + ' : ''}r${r} * K_PACKED + (bit >> 5u);
      var win${r} : u32 = weights[wi${r}] >> sh;
      if (sh > 8u) { win${r} = win${r} | (weights[wi${r} + 1u] << (32u - sh)); }
      let s${r} = f32(scales[${moe ? 'sBase + ' : ''}r${r} * SCALES_PER_ROW + sc_idx]);
      let b${r} = f32(biases[${moe ? 'sBase + ' : ''}r${r} * SCALES_PER_ROW + sc_idx]);
      acc${r} = acc${r} + s${r} * (
          i0 * f32(win${r} & 7u)          + i1 * f32((win${r} >>  3u) & 7u)
        + i2 * f32((win${r} >>  6u) & 7u) + i3 * f32((win${r} >>  9u) & 7u)
        + i4 * f32((win${r} >> 12u) & 7u) + i5 * f32((win${r} >> 15u) & 7u)
        + i6 * f32((win${r} >> 18u) & 7u) + i7 * f32((win${r} >> 21u) & 7u)) + b${r} * xs;
    }`,
      )
      .join('\n')
    body = q3
      ? `
${accDecl}

${rowDecl}

  // 8-value blocks; K_PACKED is 3K/32 here, so the block count comes from
  // SCALES_PER_ROW (= K/64): K/8 blocks in total. Strided-and-bounded rather
  // than a trip count, so a K that is not a multiple of ${wgSize * 8} keeps its
  // tail instead of having it silently truncated away by the division.
  for (var w_offset : u32 = tid; w_offset < SCALES_PER_ROW * 8u; w_offset = w_offset + ${wgSize}u) {
    let base : u32 = ${moe ? 'inBase + ' : ''}w_offset * 8u;
    let sc_idx : u32 = w_offset >> 3u;
    let bit : u32 = w_offset * 24u;
    let sh : u32 = bit & 31u;   // cycles 0,24,16,8

${loads}

    // Σx over exactly this thread's 8 values — see the affine note above.
    let xs : f32 = i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7;

${rowBlocksQ3}
  }
`
      : `
${accDecl}

${rowDecl}

  // Strided-and-bounded, not a trip count: K_PACKED / ${wgSize} would truncate
  // the tail of any K that is not a multiple of ${wgSize * 8}.
  for (var w_offset : u32 = tid; w_offset < K_PACKED; w_offset = w_offset + ${wgSize}u) {
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

  // Strided-and-bounded — see the M=1 path: a trip-count division drops the
  // tail of any K that is not a multiple of ${wgSize * 8}.
  for (var w_offset : u32 = tid; w_offset < K_PACKED; w_offset = w_offset + ${wgSize}u) {
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
function int4MatmulBatchedDynWGSL(affine = false): string {
  const rows = [0, 1, 2, 3]
  const batches = [0, 1, 2, 3]
  const loads = batches
    .map((b) =>
      [...Array(8).keys()]
        .map((i) => `      let b${b}_${i} = f32(input_buf[(mb * 4u + ${b}u) * K + base${i ? ` + ${i}u` : ''}]);`)
        .join('\n'),
    )
    .join('\n')
  // Σx over exactly this thread's 8 values, PER BATCH ROW. Paired with this
  // thread's own group bias below, so the group term is counted once across the
  // 8 threads that share a group-64 — the same argument the scalar affine path
  // makes, one dimension wider.
  const xsums = affine
    ? '\n' + batches
        .map((b) => `      let xs${b} : f32 = b${b}_0 + b${b}_1 + b${b}_2 + b${b}_3 + b${b}_4 + b${b}_5 + b${b}_6 + b${b}_7;`)
        .join('\n') + '\n'
    : ''
  const rowBlocks = rows
    .map((r) => {
      // MLC is symmetric (w = s·(q−7)); MLX affine is w = s·q + b. Dropping the
      // −7 and adding b·Σx is the whole difference.
      const nibbles = [...Array(8).keys()]
        .map((i) => `        let n${i} = f32((p >> ${String(i * 4).padStart(2)}u) & 15u)${affine ? '' : ' - 7.0'};`)
        .join('\n')
      const accs = batches
        .map(
          (b) =>
            `        a${b}${r} = a${b}${r} + s * (b${b}_0*n0 + b${b}_1*n1 + b${b}_2*n2 + b${b}_3*n3 + b${b}_4*n4 + b${b}_5*n5 + b${b}_6*n6 + b${b}_7*n7)${affine ? ` + bi * xs${b}` : ''};`,
        )
        .join('\n')
      return `      { // weight row r${r}: unpack nibbles once, reuse across the 4 batch rows
        let p = weights[r${r} * K_PACKED + w_offset];
        let s = f32(scales[r${r} * SCALES_PER_ROW + sc_idx]);
${affine ? `        let bi = f32(biases[r${r} * SCALES_PER_ROW + sc_idx]);\n` : ''}${nibbles}
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
// Variant: ${affine ? 'int4_matmul_batched_dyn_affine' : 'int4_matmul_batched_dyn'} (out=f16, rowsPerWG=4, subgroupAdd, runtime M${affine ? ', MLX affine w = s*q + b, group 64' : ''})
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
${affine ? '@group(0) @binding(5) var<storage, read> biases : array<f16>;   // MLX bias, verbatim\n' : ''}
struct PODArgs {
  K_PACKED: u32,        // K / 8  (u32 words per weight row)
  SCALES_PER_ROW: u32,  // K / ${affine ? '64' : '32'} (int4 group scales per weight row)
  packGridDimX: u32,    // N (number of output columns)
  M_ROWS: u32           // valid input rows this dispatch (chunk seq_len)
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(32, 1, 1)
fn ${affine ? 'int4_matmul_batched_dyn_affine' : 'int4_matmul_batched_dyn'}(
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

    // Strided-and-bounded. The subgroupAdds are OUTSIDE this loop, so lanes
    // taking one fewer iteration on a ragged K stay uniform where it matters.
    for (var w_offset : u32 = tid; w_offset < K_PACKED; w_offset = w_offset + 32u) {
      let base : u32 = w_offset * 8u;
      let sc_idx : u32 = w_offset >> ${affine ? '3' : '2'}u;

${loads}${xsums}

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
  // ...and its MLX-affine sibling. Without this every affine checkpoint
  // (every MLX model here) prefills one token at a time.
  { subgroups: true, rowsPerWG: 4, mDyn: true, affine: true },
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
  { affine: true, moe: true, subgroups: true, rowsPerWG: 4 }, // int4_matmul_tiled_affine_moe
  // 3-bit experts (MLX bits=3, continuous bitstream — see the q3 option note).
  { affine: true, q3: true }, // int4_matmul_affine_q3
  { affine: true, q3: true, subgroups: true, rowsPerWG: 4 }, // int4_matmul_tiled_affine_q3
  { affine: true, q3: true, moe: true, subgroups: true, rowsPerWG: 4 } // int4_matmul_tiled_affine_q3_moe
]

/** Human-auditable dump of every shipped variant. */
export function debugDumpAll(): string {
  return INT4_MATMUL_VARIANTS.map(
    (v) => `// ${'═'.repeat(70)}\n// ${int4MatmulEntry(v)}\n// ${'═'.repeat(70)}\n${int4MatmulWGSL(v)}`,
  ).join('\n')
}

/**
 * Workgroup-memory tiled batched GEMM — the prefill kernel.
 *
 * int4_matmul_batched_dyn is a 4-row × 4-batch SUBGROUP MATVEC: every
 * workgroup covering 4 output rows re-reads the full activation block from
 * global memory, nothing is staged in shared memory, and each output pays a
 * subgroupAdd. That shape is why prefill measured 231 tok/s against LM
 * Studio's 1,385 on the same machine (BENCH.md 2026-08-12) while the CHUNK_CAP
 * sweep moved nothing: the GEMM itself was the cost.
 *
 * This is the classic tiling instead. One workgroup owns a 32(M)×32(N) output
 * tile; the K loop stages a 32×64 activation tile in workgroup memory ONCE per
 * 64-element step and every thread reuses it for a 4×4 register block of
 * outputs. Activation traffic drops from N/4 global re-reads to N/32 staged
 * ones; weights are still read exactly once per output tile; no subgroups
 * feature is required at all.
 *
 * Constraints, checked by the engine before it selects this pipeline:
 *   - K % 64 == 0 (the staged tile step; every current chunked spec passes —
 *     d/qDim/ffn/gdnVDim are all %64 on qwen35, qwen3mlx and llama32)
 *   - chunk capacity % 32 == 0 (grid.y tiles of 32 batch rows; caller-sized
 *     activation buffers make out-of-M reads in-bounds, and writes are guarded
 *     by M_ROWS so rows past M stay untouched)
 * Ragged N is handled with a per-row guard: an edge tile's out-of-range rows
 * skip their loads and writes, costing divergence only on the last tile.
 *
 * Same PODArgs and binding order as batched_dyn, so the engine's uniforms and
 * dynBg() bind groups work unchanged — only the pipeline and the 2-D dispatch
 * differ.
 */
export function int4MatmulTiledMWGSL(affine = false): string {
  const entry = affine ? 'int4_matmul_tiled_m_affine' : 'int4_matmul_tiled_m'
  const nsub = [0, 1, 2, 3]
  const msub = [0, 1, 2, 3]

  // Per (n-row, word): load the packed u32 + its group scale (+bias), unpack 8
  // nibbles, then FMA against the 4 staged activation rows.
  const rowBlock = (n: number) => `      if (row${n} < N) {
        let p = weights[row${n} * K_PACKED + wordIdx];
        let s = f32(scales[row${n} * SCALES_PER_ROW + scIdx]);
${affine ? `        let bi = f32(biases[row${n} * SCALES_PER_ROW + scIdx]);\n` : ''}${[...Array(8).keys()]
    .map((i) => `        let q${i} = f32((p >> ${String(i * 4).padStart(2)}u) & 15u)${affine ? '' : ' - 7.0'};`)
    .join('\n')}
${msub
    .map(
      (m) => `        acc${m}_${n} = acc${m}_${n} + s * (a${m}_0*q0 + a${m}_1*q1 + a${m}_2*q2 + a${m}_3*q3 + a${m}_4*q4 + a${m}_5*q5 + a${m}_6*q6 + a${m}_7*q7)${affine ? ` + bi * asum${m}` : ''};`,
    )
    .join('\n')}
      }`

  return `// GENERATED by src/compiler/shaders/int4_matmul.gen.ts — do not edit by hand.
// Variant: ${entry} (tiled batched GEMM, 32x32 output tile, TK=64, no subgroups${affine ? ', MLX affine w = s*q + b, group 64' : ''})
// K must be a multiple of 64 and the caller's batch capacity a multiple of 32 —
// the engine checks both before selecting this pipeline.

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;
${affine ? '@group(0) @binding(5) var<storage, read> biases : array<f16>;   // MLX bias, verbatim\n' : ''}
struct PODArgs {
  K_PACKED: u32,        // K / 8  (u32 words per weight row)
  SCALES_PER_ROW: u32,  // K / ${affine ? '64' : '32'} (int4 group scales per weight row)
  packGridDimX: u32,    // N (number of output columns)
  M_ROWS: u32           // valid input rows this dispatch (chunk seq_len)
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

// 32 batch rows x 64 K-elements, staged once per K step and reused by every
// thread in the tile.
var<workgroup> Ash : array<f16, 2048>;

@compute @workgroup_size(64, 1, 1)
fn ${entry}(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
) {
  let K_PACKED = podArgs.K_PACKED;
  let SCALES_PER_ROW = podArgs.SCALES_PER_ROW;
  let N = podArgs.packGridDimX;
  let M = podArgs.M_ROWS;
  let K = K_PACKED * 8u;

  let nBase = wid.x * 32u;
  let mBase = wid.y * 32u;
  let tid = lid.x;
  // 8x8 thread grid over the 32x32 tile: each thread owns 4 consecutive N rows
  // and 4 consecutive M rows.
  let tn = (tid % 8u) * 4u;
  let tm = (tid / 8u) * 4u;
${nsub.map((n) => `  let row${n} = nBase + tn + ${n}u;`).join('\n')}

${msub.flatMap((m) => nsub.map((n) => `  var acc${m}_${n} : f32 = 0.0;`)).join('\n')}

  for (var k0 : u32 = 0u; k0 < K; k0 = k0 + 64u) {
    // Stage the activation tile: 2048 halves, 32 per thread. Rows past M hold
    // whatever the buffer holds — read (in-bounds by caller sizing), never
    // written back, and each output row's accumulation is independent of them.
    for (var i : u32 = 0u; i < 32u; i = i + 1u) {
      let idx = tid * 32u + i;
      let am = idx / 64u;
      let ak = idx % 64u;
      Ash[idx] = input_buf[(mBase + am) * K + k0 + ak];
    }
    workgroupBarrier();

    for (var w : u32 = 0u; w < 8u; w = w + 1u) {
      let wordIdx = (k0 >> 3u) + w;
      let scIdx = wordIdx >> ${affine ? '3u' : '2u'};
      // This thread's 4 activation rows for this word, hoisted out of the
      // N-row blocks so each staged half is read once per thread, not 4x.
${msub
    .map(
      (m) => `      let aBase${m} = (tm + ${m}u) * 64u + w * 8u;\n${[...Array(8).keys()]
        .map((i) => `      let a${m}_${i} = f32(Ash[aBase${m} + ${i}u]);`)
        .join('\n')}`,
    )
    .join('\n')}
${affine ? msub.map((m) => `      let asum${m} : f32 = a${m}_0 + a${m}_1 + a${m}_2 + a${m}_3 + a${m}_4 + a${m}_5 + a${m}_6 + a${m}_7;`).join('\n') + '\n' : ''}
${nsub.map(rowBlock).join('\n')}
    }
    workgroupBarrier();
  }

${msub
    .map(
      (m) => `  if (mBase + tm + ${m}u < M) {
${nsub.map((n) => `    if (row${n} < N) { output_buf[(mBase + tm + ${m}u) * N + row${n}] = f16(acc${m}_${n}); }`).join('\n')}
  }`,
    )
    .join('\n')}
}
`
}

/**
 * Tiled batched GEMM v2 — dequantized weights STAGED in workgroup memory.
 *
 * Why v1 only tied the matvec: at M<=64 this GEMM is COMPUTE bound
 * (arithmetic intensity ~230 FLOP/byte against a ~30 FLOP/byte roofline
 * knee), and the compute is dominated by DEQUANT — unpack nibbles, convert,
 * scale — which both the matvec and v1 amortize over exactly 4 batch rows.
 * Same amortization, same speed. Measured parity, roofline-explained.
 *
 * v2 moves dequant out of the inner loop entirely: each K-step, the workgroup
 * cooperatively dequantizes the 32x64 weight tile into shared memory (4 words
 * per thread), so every unpacked value is reused by all 32 batch rows, and
 * the AFFINE BIAS FOLDS INTO THE STAGED VALUE — Wsh holds s·q+b (or s·(q−7))
 * as a finished f16, making the inner loop identical for both quant flavors:
 * pure shared-memory FMA, zero dequant, zero bias bookkeeping, no asum.
 *
 * Same constraints, PODArgs and binding order as v1/batched_dyn.
 */
export function int4MatmulTiledStWGSL(affine = false): string {
  const entry = affine ? 'int4_matmul_tiled_st_affine' : 'int4_matmul_tiled_st'
  const msub = [0, 1, 2, 3]
  const nsub = [0, 1, 2, 3]
  return `// GENERATED by src/compiler/shaders/int4_matmul.gen.ts — do not edit by hand.
// Variant: ${entry} (tiled GEMM, staged dequantized weights, 32x32 tile, TK=64${affine ? ', MLX affine folded into the stage' : ''})
// K % 64 == 0 and batch capacity % 32 == 0 — engine-checked.

enable f16;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;
${affine ? '@group(0) @binding(5) var<storage, read> biases : array<f16>;   // MLX bias, verbatim\n' : ''}
struct PODArgs {
  K_PACKED: u32,        // K / 8
  SCALES_PER_ROW: u32,  // K / ${affine ? '64' : '32'}
  packGridDimX: u32,    // N
  M_ROWS: u32           // valid batch rows
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> Ash : array<f16, 2048>;   // 32 m-rows x 64 k
// f32, not f16, and the correctness gate is why: staging w as f16 added a
// rounding the per-token path never takes, and under cancellation it measured
// 3.4e-2 against the CPU reference — void. f32 staging costs 4 KB more shared
// memory (12 KB total, inside the 16 KB floor) and restores 4.7e-4.
var<workgroup> Wsh : array<f32, 2048>;   // 32 n-rows x 64 k, DEQUANTIZED

@compute @workgroup_size(64, 1, 1)
fn ${entry}(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
) {
  let K_PACKED = podArgs.K_PACKED;
  let SCALES_PER_ROW = podArgs.SCALES_PER_ROW;
  let N = podArgs.packGridDimX;
  let M = podArgs.M_ROWS;
  let K = K_PACKED * 8u;

  let nBase = wid.x * 32u;
  let mBase = wid.y * 32u;
  let tid = lid.x;
  let tn = (tid % 8u) * 4u;
  let tm = (tid / 8u) * 4u;

${msub.flatMap((m) => nsub.map((n) => `  var acc${m}_${n} : f32 = 0.0;`)).join('\n')}

  for (var k0 : u32 = 0u; k0 < K; k0 = k0 + 64u) {
    // Stage A: 2048 halves, 32 per thread (reads in-bounds by caller sizing).
    for (var i : u32 = 0u; i < 32u; i = i + 1u) {
      let idx = tid * 32u + i;
      Ash[idx] = input_buf[(mBase + idx / 64u) * K + k0 + (idx % 64u)];
    }
    // Stage W, dequantized: 32 rows x 8 words = 256 words, 4 per thread. Each
    // unpacked value is written ONCE and read by all 8 m-thread-rows — the
    // whole point of v2. Out-of-range N rows stage zeros so the inner loop
    // stays branch-free; their outputs are discarded by the write guard.
    for (var wq : u32 = 0u; wq < 4u; wq = wq + 1u) {
      let widx = tid * 4u + wq;          // 0..255
      let wrow = widx / 8u;              // n-row within the tile
      let wcol = widx % 8u;              // word within the row's K-step
      let row = nBase + wrow;
      let base = wrow * 64u + wcol * 8u;
      if (row < N) {
        let wordIdx = (k0 >> 3u) + wcol;
        let p = weights[row * K_PACKED + wordIdx];
        let s = f32(scales[row * SCALES_PER_ROW + (wordIdx >> ${affine ? '3u' : '2u'})]);
${affine
    ? `        let bi = f32(biases[row * SCALES_PER_ROW + (wordIdx >> 3u)]);
${[...Array(8).keys()].map((i) => `        Wsh[base + ${i}u] = s * f32((p >> ${String(i * 4).padStart(2)}u) & 15u) + bi;`).join('\n')}`
    : `${[...Array(8).keys()].map((i) => `        Wsh[base + ${i}u] = s * (f32((p >> ${String(i * 4).padStart(2)}u) & 15u) - 7.0);`).join('\n')}`}
      } else {
${[...Array(8).keys()].map((i) => `        Wsh[base + ${i}u] = 0.0;`).join('\n')}
      }
    }
    workgroupBarrier();

    // Pure FMA. 64 k-steps x 16 cells; a-values hoisted per k so each shared
    // half is read once per thread per k.
    for (var k : u32 = 0u; k < 64u; k = k + 1u) {
${msub.map((m) => `      let a${m} = f32(Ash[(tm + ${m}u) * 64u + k]);`).join('\n')}
${nsub.map((n) => `      let w${n} = Wsh[(tn + ${n}u) * 64u + k];`).join('\n')}
${msub.flatMap((m) => nsub.map((n) => `      acc${m}_${n} = fma(a${m}, w${n}, acc${m}_${n});`)).join('\n')}
    }
    workgroupBarrier();
  }

${msub
    .map(
      (m) => `  if (mBase + tm + ${m}u < M) {
${nsub.map((n) => `    if (nBase + tn + ${n}u < N) { output_buf[(mBase + tm + ${m}u) * N + nBase + tn + ${n}u] = f16(acc${m}_${n}); }`).join('\n')}
  }`,
    )
    .join('\n')}
}
`
}

/**
 * Subgroup-matrix GEMM — Metal's matrix unit (simdgroup mats) from WGSL, via
 * `chromium-experimental-subgroup-matrix`. THE path past the ~2.2-2.4 TF
 * hand-written-FMA ceiling: measured in scripts/gemm-bench.mjs it beats
 * tiled-v2 on every chunk shape (1.15-1.6x) and the shipped matvec by up to
 * 2.2x at M=64, same run, same thermal state.
 *
 * Three facts the bench iterations paid for, recorded so nobody re-learns
 * them:
 *   - Fragment-op offsets must be WORKGROUP-uniform: the validator rejects a
 *     subgroup-indexed offset outright, so subgroups cannot split one
 *     workgroup's tile and multi-subgroup workgroups buy nothing. One
 *     subgroup per workgroup, every offset from workgroup_id + loop counters.
 *   - f32 fragments are supported and CORRECT but run ~5x slower than f16
 *     (~270 GF) — the hardware's fast path is f16. There is no free
 *     precision: f16 fragments round the staged weights, which measures
 *     3.4e-2 against a from-the-formula reference under cancellation. That is
 *     MLX-class arithmetic (their GEMMs run the same unit the same way), but
 *     it is NOT bit-comparable to the per-token path, so this kernel cannot
 *     silently replace it — it ships opt-in until the equivalence policy is
 *     decided.
 *   - Loading A fragments STRAIGHT FROM STORAGE (stride K) beats staging A in
 *     shared memory — the matrix unit's loads cover the latency, and dropping
 *     the stage removes a barrier's worth of work per K-step.
 *
 * Weights still stage through shared memory dequantized (bias folded), same
 * as tiled-v2 — the unit consumes f16 tiles, not packed nibbles.
 */
export function int4MatmulSgMatWGSL(affine = false): string {
  const entry = affine ? 'int4_matmul_sgmat_affine' : 'int4_matmul_sgmat'
  return `// GENERATED by src/compiler/shaders/int4_matmul.gen.ts — do not edit by hand.
// Variant: ${entry} (subgroup-matrix GEMM, 32x32 tile, f16 fragments, f32 acc${affine ? ', MLX affine folded into the stage' : ''})
// Requires chromium-experimental-subgroup-matrix. K % 64 == 0, capacity % 32 == 0.

enable f16;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;
${affine ? '@group(0) @binding(5) var<storage, read> biases : array<f16>;   // MLX bias, verbatim\n' : ''}
struct PODArgs {
  K_PACKED: u32,        // K / 8
  SCALES_PER_ROW: u32,  // K / ${affine ? '64' : '32'}
  packGridDimX: u32,    // N
  M_ROWS: u32           // valid batch rows
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> Wsh : array<f16, 2048>;   // 32 n-rows x 64 k, dequantized
var<workgroup> Osh : array<f32, 1024>;   // 32 x 32 result staging (ragged writes)

@compute @workgroup_size(32, 1, 1)
fn ${entry}(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
) {
  let K_PACKED = podArgs.K_PACKED;
  let SPR = podArgs.SCALES_PER_ROW;
  let N = podArgs.packGridDimX;
  let M = podArgs.M_ROWS;
  let K = K_PACKED * 8u;
  let nBase = wid.x * 32u;
  let mBase = wid.y * 32u;
  let tid = lid.x;

  var acc : array<subgroup_matrix_result<f32, 8, 8>, 16>;

  for (var k0 : u32 = 0u; k0 < K; k0 = k0 + 64u) {
    // Dequantize the 32x64 weight tile into shared: 8 words per thread, bias
    // folded into the staged value so the MAC loop is quant-flavor-blind.
    for (var wq : u32 = 0u; wq < 8u; wq = wq + 1u) {
      let widx = tid * 8u + wq;
      let wrow = widx / 8u;
      let wcol = widx % 8u;
      let row = nBase + wrow;
      let base = wrow * 64u + wcol * 8u;
      if (row < N) {
        let wordIdx = (k0 >> 3u) + wcol;
        let p = weights[row * K_PACKED + wordIdx];
        let sv = f32(scales[row * SPR + (wordIdx >> ${affine ? '3u' : '2u'})]);
${affine ? '        let bv = f32(biases[row * SPR + (wordIdx >> 3u)]);\n' : ''}        for (var b2 : u32 = 0u; b2 < 8u; b2 = b2 + 1u) {
          Wsh[base + b2] = f16(sv * ${affine ? 'f32((p >> (b2 * 4u)) & 15u) + bv' : '(f32((p >> (b2 * 4u)) & 15u) - 7.0)'});
        }
      } else {
        for (var b2 : u32 = 0u; b2 < 8u; b2 = b2 + 1u) { Wsh[base + b2] = 0.0h; }
      }
    }
    workgroupBarrier();

    for (var k8 : u32 = 0u; k8 < 8u; k8 = k8 + 1u) {
      var L : array<subgroup_matrix_left<f16, 8, 8>, 4>;
      var R : array<subgroup_matrix_right<f16, 8, 8>, 4>;
      for (var i : u32 = 0u; i < 4u; i = i + 1u) {
        // A straight from storage — measured faster than staging it.
        L[i] = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&input_buf, (mBase + i * 8u) * K + k0 + k8 * 8u, false, K);
        R[i] = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Wsh, (i * 8u) * 64u + k8 * 8u, true, 64u);
      }
      for (var mi : u32 = 0u; mi < 4u; mi = mi + 1u) {
        for (var ni : u32 = 0u; ni < 4u; ni = ni + 1u) {
          acc[mi * 4u + ni] = subgroupMatrixMultiplyAccumulate(L[mi], R[ni], acc[mi * 4u + ni]);
        }
      }
    }
    workgroupBarrier();
  }

  for (var mi : u32 = 0u; mi < 4u; mi = mi + 1u) {
    for (var ni : u32 = 0u; ni < 4u; ni = ni + 1u) {
      subgroupMatrixStore(&Osh, (mi * 8u) * 32u + ni * 8u, acc[mi * 4u + ni], false, 32u);
    }
  }
  workgroupBarrier();
  for (var i : u32 = 0u; i < 32u; i = i + 1u) {
    let idx = tid * 32u + i;
    if (mBase + idx / 32u < M && nBase + (idx % 32u) < N) {
      output_buf[(mBase + idx / 32u) * N + nBase + (idx % 32u)] = f16(Osh[idx]);
    }
  }
}
`
}

/**
 * E1 — the convergent subgroup-matrix GEMM (docs/PREFILL_RESEARCH.md).
 *
 * 128 threads / 4 subgroups over a 32(M)x64(N) tile, TILE_K 32, both operands
 * staged at stride 40 (TK+8, the MLX/ORT anti-bank-conflict pad), 8 NAMED
 * accumulators per subgroup (a fragment ARRAY trips an MSL stack blowout —
 * crbug 443794633), vectorized nibble dequant, and the line that unlocked all
 * of it: `diagnostic(off, chromium.subgroup_matrix_uniformity)` — the
 * "offsets must be workgroup-uniform" wall was Tint being conservative, not
 * the platform; ORT and llama.cpp both disable it and index by subgroup.
 *
 * Measured (gemm-bench, correctness-gated, same run): 2.8-3.8 TF across the
 * chunk shapes — 1.5-1.6x the single-subgroup sgmat, past the 3.5 TF
 * plain-WGSL bar. llama.cpp's 256-thread/8-subgroup config LOST on this
 * hardware (2.1-3.0 TF). Remaining arms (direct store, TILE_K sweep, swizzle)
 * are docs/PREFILL_RESEARCH.md E3/E5.
 */
export function int4MatmulSgE1WGSL(affine = false): string {
  const entry = affine ? 'int4_matmul_sg_e1_affine' : 'int4_matmul_sg_e1'
  const deq = affine ? `        let sv = f16(scales[n2 * SPR + (wordIdx >> 3u)]);
        let bv = f16(biases[n2 * SPR + (wordIdx >> 3u)]);
        // value i sits in bits 4i: low-nibble bytes are k = 0,2,4,6; high are
        // 1,3,5,7 — hence the interleaved stores.
        let lo = vec4<f16>(unpack4xU8(p & 0x0F0F0F0Fu)) * sv + vec4<f16>(bv);
        let hi = vec4<f16>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) * sv + vec4<f16>(bv);` : `        let sv = f16(scales[n2 * SPR + (wordIdx >> 2u)]);
        let lo = (vec4<f16>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4<f16>(7.0h)) * sv;
        let hi = (vec4<f16>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) - vec4<f16>(7.0h)) * sv;`
  const biasBind = affine ? '@group(0) @binding(5) var<storage, read> biases : array<f16>;\n' : ''
  return `// GENERATED by src/compiler/shaders/int4_matmul.gen.ts — do not edit by hand.
// Variant: ${entry} (E1 subgroup-matrix GEMM, 32x64 tile, 4 subgroups${affine ? ', MLX affine' : ''})

diagnostic(off, chromium.subgroup_matrix_uniformity);
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;
__BIASBIND__struct PODArgs { K_PACKED: u32, SCALES_PER_ROW: u32, packGridDimX: u32, M_ROWS: u32 }
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const TK : u32 = 32u;      // K-tile
const STRIDE : u32 = 40u;  // TK + 8 pad

var<workgroup> Ash : array<f16, 1280>;   // 32 m-rows x STRIDE
var<workgroup> Bsh : array<f16, 2560>;   // 64 n-rows x STRIDE, dequantized
var<workgroup> Osh : array<f32, 2048>;   // 32 x 64 staging for ragged tiles

@compute @workgroup_size(128, 1, 1)
fn __ENTRY__(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
  @builtin(subgroup_size) sgSize : u32
) {
  let K_PACKED = podArgs.K_PACKED;
  let SPR = podArgs.SCALES_PER_ROW;
  let N = podArgs.packGridDimX;
  let M = podArgs.M_ROWS;
  let K = K_PACKED * 8u;
  let nBase = wid.x * 64u;
  let mBase = wid.y * 32u;
  let tid = lid.x;
  let sg = tid / sgSize;          // 0..3 at sgSize 32
  let sgRow = sg / 2u;            // 16-row M slice
  let sgCol = sg % 2u;            // 32-col N slice

  var c00 : subgroup_matrix_result<f32, 8, 8>; var c01 : subgroup_matrix_result<f32, 8, 8>;
  var c02 : subgroup_matrix_result<f32, 8, 8>; var c03 : subgroup_matrix_result<f32, 8, 8>;
  var c10 : subgroup_matrix_result<f32, 8, 8>; var c11 : subgroup_matrix_result<f32, 8, 8>;
  var c12 : subgroup_matrix_result<f32, 8, 8>; var c13 : subgroup_matrix_result<f32, 8, 8>;

  for (var k0 : u32 = 0u; k0 < K; k0 = k0 + TK) {
    // Stage A: 32 x 32 halves, 8 per thread.
    for (var i : u32 = 0u; i < 8u; i = i + 1u) {
      let idx = tid * 8u + i;
      let r = idx / TK;
      let c = idx % TK;
      Ash[r * STRIDE + c] = input_buf[(mBase + r) * K + k0 + c];
    }
    // Stage B dequantized: 64 rows x 4 words, 2 words per thread.
    for (var wq : u32 = 0u; wq < 2u; wq = wq + 1u) {
      let widx = tid * 2u + wq;   // 0..255
      let row = widx / 4u;
      let wcol = widx % 4u;
      let n2 = nBase + row;
      let base = row * STRIDE + wcol * 8u;
      if (n2 < N) {
        let wordIdx = (k0 >> 3u) + wcol;
        let p = weights[n2 * K_PACKED + wordIdx];
__DEQ__
        Bsh[base + 0u] = lo.x; Bsh[base + 2u] = lo.y; Bsh[base + 4u] = lo.z; Bsh[base + 6u] = lo.w;
        Bsh[base + 1u] = hi.x; Bsh[base + 3u] = hi.y; Bsh[base + 5u] = hi.z; Bsh[base + 7u] = hi.w;
      } else {
        for (var i : u32 = 0u; i < 8u; i = i + 1u) { Bsh[base + i] = 0.0h; }
      }
    }
    workgroupBarrier();

    for (var step : u32 = 0u; step < TK; step = step + 8u) {
      let aOff = sgRow * 16u * STRIDE + step;
      let a0 = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&Ash, aOff, false, STRIDE);
      let a1 = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&Ash, aOff + 8u * STRIDE, false, STRIDE);
      let bOff = sgCol * 32u * STRIDE + step;
      let b0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, bOff, true, STRIDE);
      let b1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, bOff + 8u * STRIDE, true, STRIDE);
      let b2 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, bOff + 16u * STRIDE, true, STRIDE);
      let b3 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, bOff + 24u * STRIDE, true, STRIDE);
      c00 = subgroupMatrixMultiplyAccumulate(a0, b0, c00);
      c01 = subgroupMatrixMultiplyAccumulate(a0, b1, c01);
      c02 = subgroupMatrixMultiplyAccumulate(a0, b2, c02);
      c03 = subgroupMatrixMultiplyAccumulate(a0, b3, c03);
      c10 = subgroupMatrixMultiplyAccumulate(a1, b0, c10);
      c11 = subgroupMatrixMultiplyAccumulate(a1, b1, c11);
      c12 = subgroupMatrixMultiplyAccumulate(a1, b2, c12);
      c13 = subgroupMatrixMultiplyAccumulate(a1, b3, c13);
    }
    workgroupBarrier();
  }

  // Store through Osh (f32), then guarded copy-out. The full-tile direct
  // store is a later arm of the sweep; correctness first.
  let oBase = sgRow * 16u * 64u + sgCol * 32u;
  subgroupMatrixStore(&Osh, oBase, c00, false, 64u);
  subgroupMatrixStore(&Osh, oBase + 8u, c01, false, 64u);
  subgroupMatrixStore(&Osh, oBase + 16u, c02, false, 64u);
  subgroupMatrixStore(&Osh, oBase + 24u, c03, false, 64u);
  subgroupMatrixStore(&Osh, oBase + 8u * 64u, c10, false, 64u);
  subgroupMatrixStore(&Osh, oBase + 8u * 64u + 8u, c11, false, 64u);
  subgroupMatrixStore(&Osh, oBase + 8u * 64u + 16u, c12, false, 64u);
  subgroupMatrixStore(&Osh, oBase + 8u * 64u + 24u, c13, false, 64u);
  workgroupBarrier();
  for (var i : u32 = 0u; i < 16u; i = i + 1u) {
    let idx = tid * 16u + i;
    let m = idx / 64u;
    let n2 = idx % 64u;
    if (mBase + m < M && nBase + n2 < N) {
      output_buf[(mBase + m) * N + nBase + n2] = f16(Osh[idx]);
    }
  }
}`
    .replace('__ENTRY__', entry)
    .replace('__BIASBIND__', biasBind)
    .replace('__DEQ__', deq)
}

/**
 * E5 — E1's tile transposed, with llama.cpp's swizzled B layout.
 *
 * 128 threads / 4 subgroups over a 64(M)x32(N) tile, TILE_K 32. Each subgroup
 * owns a 16-row M slice and they all read the SAME 32 N-columns, so B is
 * staged once and reused four times while A carries the tile's width — the
 * inverse of E1, which stages 64 N-rows of B for 32 M-rows of A.
 *
 * B is not stored row-major-with-pad. Each (n8, k8) 8x8 block sits contiguous
 * in 64 elements, k-major inside, so a fragment load is a dense stride-8 read
 * with no padding column and no transpose (E1 loads B col-major at stride 40).
 *
 * Measured against the shipped E1 in the same process, three runs, AC power
 * (scripts/gemm-sweep-native.mjs, BENCH.md "Sweep round 3"): +18-22% at
 * M=256 and +16-37% at M=64, winning every one of the three chunk shapes —
 * gate_up 5.78 vs 4.83, ffn_down 5.09 vs 4.26, o_proj 4.62 vs 4.00 TF.
 *
 * Read that history before trusting any sweep number: rounds 1 and 2 ranked
 * configs against the harness's own reconstruction of E1 rather than the
 * shipped kernel, and this config placed near LAST in round 2. The builder
 * emitted an always-true bounds guard per staged element, and the cost scaled
 * with elements staged per thread — which is exactly what a 64-row A tile has
 * most of.
 *
 * Same constraint as E1 plus one: the A stage reads a full 64 rows regardless
 * of M, so the activation buffer must hold a multiple of 64 rows (engine-core
 * checks CHUNK_CAP % 64).
 */
export function int4MatmulSgE5WGSL(affine = false): string {
  const entry = affine ? 'int4_matmul_sg_e5_affine' : 'int4_matmul_sg_e5'
  const deq = affine ? `        let sv = f16(scales[n2 * SPR + (wordIdx >> 3u)]);
        let bv = f16(biases[n2 * SPR + (wordIdx >> 3u)]);
        let lo = vec4<f16>(unpack4xU8(p & 0x0F0F0F0Fu)) * sv + vec4<f16>(bv);
        let hi = vec4<f16>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) * sv + vec4<f16>(bv);` : `        let sv = f16(scales[n2 * SPR + (wordIdx >> 2u)]);
        let lo = (vec4<f16>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4<f16>(7.0h)) * sv;
        let hi = (vec4<f16>(unpack4xU8((p >> 4u) & 0x0F0F0F0Fu)) - vec4<f16>(7.0h)) * sv;`
  const biasBind = affine ? '@group(0) @binding(5) var<storage, read> biases : array<f16>;\n' : ''
  return `// GENERATED by src/compiler/shaders/int4_matmul.gen.ts — do not edit by hand.
// Variant: ${entry} (E5 subgroup-matrix GEMM, 64x32 tile, swizzled B${affine ? ', MLX affine' : ''})

diagnostic(off, chromium.subgroup_matrix_uniformity);
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> output_buf : array<f16>;
@group(0) @binding(1) var<storage, read> input_buf : array<f16>;
@group(0) @binding(2) var<storage, read> scales : array<f16>;
@group(0) @binding(3) var<storage, read> weights : array<u32>;
__BIASBIND__struct PODArgs { K_PACKED: u32, SCALES_PER_ROW: u32, packGridDimX: u32, M_ROWS: u32 }
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const TK : u32 = 32u;      // K-tile
const ASTRIDE : u32 = 40u; // TK + 8 pad, A only — B needs no pad when swizzled

var<workgroup> Ash : array<f16, 2560>;   // 64 m-rows x ASTRIDE
var<workgroup> Bsh : array<f16, 1024>;   // 4x4 dense 8x8 blocks, dequantized
var<workgroup> Osh : array<f32, 2048>;   // 64 x 32 staging for ragged tiles

@compute @workgroup_size(128, 1, 1)
fn __ENTRY__(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>,
  @builtin(subgroup_size) sgSize : u32
) {
  let K_PACKED = podArgs.K_PACKED;
  let SPR = podArgs.SCALES_PER_ROW;
  let N = podArgs.packGridDimX;
  let M = podArgs.M_ROWS;
  let K = K_PACKED * 8u;
  let nBase = wid.x * 32u;
  let mBase = wid.y * 64u;
  let tid = lid.x;
  let sg = tid / sgSize;          // 0..3 at sgSize 32; each owns 16 M-rows

  var c00 : subgroup_matrix_result<f32, 8, 8>; var c01 : subgroup_matrix_result<f32, 8, 8>;
  var c02 : subgroup_matrix_result<f32, 8, 8>; var c03 : subgroup_matrix_result<f32, 8, 8>;
  var c10 : subgroup_matrix_result<f32, 8, 8>; var c11 : subgroup_matrix_result<f32, 8, 8>;
  var c12 : subgroup_matrix_result<f32, 8, 8>; var c13 : subgroup_matrix_result<f32, 8, 8>;

  for (var k0 : u32 = 0u; k0 < K; k0 = k0 + TK) {
    // Stage A: 64 rows x 32 k = 2048 elements, 16 per thread, no guard (the
    // division is exact). Rows past M are staged too — see the CHUNK_CAP % 64
    // requirement; their results are masked at store.
    for (var i : u32 = 0u; i < 16u; i = i + 1u) {
      let idx = tid * 16u + i;
      let r = idx / TK;
      let c = idx % TK;
      Ash[r * ASTRIDE + c] = input_buf[(mBase + r) * K + k0 + c];
    }
    // Stage B dequantized into 8x8 blocks: 32 rows x 4 words = 128 = one word
    // per thread. Element k of this word lands at blk*64 + k*8 + (row % 8).
    {
      let row = tid / 4u;         // 0..31, n within the tile
      let wcol = tid % 4u;        // 0..3, which 8-k group
      let n2 = nBase + row;
      let base = ((row / 8u) * 4u + wcol) * 64u + (row % 8u);
      if (n2 < N) {
        let wordIdx = (k0 >> 3u) + wcol;
        let p = weights[n2 * K_PACKED + wordIdx];
__DEQ__
        Bsh[base] = lo.x;        Bsh[base + 8u] = hi.x;
        Bsh[base + 16u] = lo.y;  Bsh[base + 24u] = hi.y;
        Bsh[base + 32u] = lo.z;  Bsh[base + 40u] = hi.z;
        Bsh[base + 48u] = lo.w;  Bsh[base + 56u] = hi.w;
      } else {
        for (var i : u32 = 0u; i < 8u; i = i + 1u) { Bsh[base + i * 8u] = 0.0h; }
      }
    }
    workgroupBarrier();

    for (var k8 : u32 = 0u; k8 < 4u; k8 = k8 + 1u) {
      let aOff = sg * 16u * ASTRIDE + k8 * 8u;
      let a0 = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&Ash, aOff, false, ASTRIDE);
      let a1 = subgroupMatrixLoad<subgroup_matrix_left<f16, 8, 8>>(&Ash, aOff + 8u * ASTRIDE, false, ASTRIDE);
      // Dense blocks: stride 8, no transpose. E1's B loads are col-major at
      // stride 40; that difference is the whole point of the swizzle.
      let b0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, k8 * 64u, false, 8u);
      let b1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, (4u + k8) * 64u, false, 8u);
      let b2 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, (8u + k8) * 64u, false, 8u);
      let b3 = subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&Bsh, (12u + k8) * 64u, false, 8u);
      c00 = subgroupMatrixMultiplyAccumulate(a0, b0, c00);
      c01 = subgroupMatrixMultiplyAccumulate(a0, b1, c01);
      c02 = subgroupMatrixMultiplyAccumulate(a0, b2, c02);
      c03 = subgroupMatrixMultiplyAccumulate(a0, b3, c03);
      c10 = subgroupMatrixMultiplyAccumulate(a1, b0, c10);
      c11 = subgroupMatrixMultiplyAccumulate(a1, b1, c11);
      c12 = subgroupMatrixMultiplyAccumulate(a1, b2, c12);
      c13 = subgroupMatrixMultiplyAccumulate(a1, b3, c13);
    }
    workgroupBarrier();
  }

  let oBase = sg * 16u * 32u;
  subgroupMatrixStore(&Osh, oBase, c00, false, 32u);
  subgroupMatrixStore(&Osh, oBase + 8u, c01, false, 32u);
  subgroupMatrixStore(&Osh, oBase + 16u, c02, false, 32u);
  subgroupMatrixStore(&Osh, oBase + 24u, c03, false, 32u);
  subgroupMatrixStore(&Osh, oBase + 8u * 32u, c10, false, 32u);
  subgroupMatrixStore(&Osh, oBase + 8u * 32u + 8u, c11, false, 32u);
  subgroupMatrixStore(&Osh, oBase + 8u * 32u + 16u, c12, false, 32u);
  subgroupMatrixStore(&Osh, oBase + 8u * 32u + 24u, c13, false, 32u);
  workgroupBarrier();
  for (var i : u32 = 0u; i < 16u; i = i + 1u) {
    let idx = tid * 16u + i;
    let m = idx / 32u;
    let n2 = idx % 32u;
    if (mBase + m < M && nBase + n2 < N) {
      output_buf[(mBase + m) * N + nBase + n2] = f16(Osh[idx]);
    }
  }
}`
    .replace('__ENTRY__', entry)
    .replace('__BIASBIND__', biasBind)
    .replace('__DEQ__', deq)
}

