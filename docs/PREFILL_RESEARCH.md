# zero-tvm prefill plan — ranked experiments to close the 2.3 → 7-9 TF gap

Verified against the live kernel before writing this plan: `/Users/ahmetbarisgunaydin/dev/zero-tvm/src/compiler/shaders/int4_matmul.gen.ts` lines 993-1087 (`int4MatmulSgMatWGSL`): `@workgroup_size(32)` = 1 subgroup; `acc : array<subgroup_matrix_result<f32,8,8>, 16>`; A fragment-loaded from `input_buf` (device storage, stride K) inside the k8 loop with the comment "A straight from storage — measured faster than staging it" (line 1060); Wsh 32x64 f16 (4 KB) + Osh 32x32 f32 (4 KB) = 8 KB shared per 32 threads; per-nibble shift+mask+f32-convert dequant done by 32 threads; every output staged through Osh. All reference sources cited below are cached locally in `/private/tmp/claude-501/-Users-ahmetbarisgunaydin/371bd978-f5ed-4e2f-8676-47deaccff18e/scratchpad/` (`sgmat/` for WGSL/Dawn files, `mlx/` @ 596dc79f, `llama.cpp/` @ 6e62ba53).

## 0. Headline verdict and ceiling

**MLX-level (7-9 TF) is NOT reachable from Chrome WebGPU today, but ~5-6.5 TF is.** Evidence:

- Best demonstrated WGSL-vs-native-Metal prefill ratio on Apple Silicon: **71-74%** — llama.cpp WebGPU backend, M3, pp512: F16 1014.2 t/s vs 1368.5 native (74%); Q4_0 960.5 vs 1346.7 (71%) (https://github.com/ggml-org/llama.cpp/pull/17031, tuned in PR #22241).
- LlamaWeb (arXiv 2605.20706): native Metal beats their WebGPU by >2x on M4 prefill; WebGPU robustness/bounds checks alone cost **14-23%** on prefill. This tax is structural in shipping Chrome.
- Native ceiling on this machine class: llama.cpp Metal M2 Max Q4_0 pp512 = 671.3 t/s (38-core) ≈ **9.0 effective TF** / 537.6 t/s (30-core) ≈ **7.2 TF** — ~66% of fp32 peak (https://github.com/ggml-org/llama.cpp/discussions/4167, build 8e672ef). MLX's 7-9 TF is the same class.

Realistic in-browser ceiling: **0.70-0.75 x native ≈ 5.0-6.5 TF ≈ 620-810 prefill tok/s** on Qwen3.5-4B — i.e. 2.2-2.8x above today's 2.3 TF / 287 tok/s, still ~1.7-2x behind MLX's 1385. That residual is Chrome tax (robustness clamps, no occupancy/scheduling control), not kernel skill. Reference milestones to publish against: gpu.cpp's ~3.5 TF plain-WGSL f16 (M1 Max) and ORT's 184.5 tok/s (Phi-3.5-mini) — beating both makes zero-tvm the citable fastest WebGPU int4 GEMM on Apple.

## 1. Corrections to our "learned facts" (contradictions, adjudicated)

1. **"Fragment op offsets must be workgroup-uniform" — WRONG as stated; this is the single biggest unlock.** The requirement is *subgroup*-uniform. Tint's conservative analysis emits a default-error diagnostic, `chromium.subgroup_matrix_uniformity` (Dawn CL https://dawn-review.googlesource.com/c/dawn/+/236054), which both production users disable: ORT `shader_helper.cc` ~407-414 emits `diagnostic(off, chromium.subgroup_matrix_uniformity);` explicitly "since we use subgroup_id"; llama.cpp `mul_mat_subgroup_matrix.wgsl` line 1 does the same. The reports are right; our constraint was the diagnostic, not the platform. This is why we're stuck at 1 subgroup/workgroup.
2. **"f32 fragments ~5x slower" — real measurement, but not hardware, and partially self-inflicted.** Apple silicon runs simdgroup matrix f32 at the same rate as f16 (~102 ops/cycle both, philipturner/metal-benchmarks), and MLX's 7-9 TF kernel uses `simdgroup_float8x8` for A, B and C (`mlx/backend/metal/steel/gemm/mma.h:439-483`). Note our kernel *already* uses f16 operands with **f32 result fragments** at 2.3 TF — so the 5x number presumably measured the all-f32 operand config. Two likely culprits: Dawn/Tint lowering of the f32 config (file a bug if it survives restructuring), and our 16-element *array* of result fragments — Dawn's own e2e test exists to cover an MSL compiler stack-blowout with fragment arrays (crbug.com/443794633, `SubgroupMatrixTests.cpp:1265,1417`). Practical call: **stay on f16 operands + f32 results** (we get f32 accumulation for free — no Qwen NaN hazard, unlike llama.cpp's f16-acc path, issue #21602), and re-measure f32 operands after Experiment 1.
3. **webgpu-engines' "we are already near the WebGPU frontier" is too pessimistic.** It missed llama.cpp's WebGPU backend (71-74% of native). The wgsl-gemm-sota report is right: ~2-2.8x headroom exists. Its "ORT's 184.5 tok/s trails our 287" comparison is real but crosses hardware/models — don't lean on it.
4. **Double buffering (webgpu-engines/CubeCL suggestion) is premature.** MLX (5 KB smem), llama.cpp Metal (6 KB), llama.cpp WGSL, and ORT all reach 66-75% of peak/native with a single stage and two barriers per K-tile. They hide latency with *occupancy* (40-48 B shared/thread vs our 256 B/thread). Defer to last.
5. **In-code comment "A straight from storage — measured faster than staging" (line 1060) — true then, expected to invert.** That was measured under a 32-thread workgroup where the staging copy itself was serial. Storage-vs-workgroup `subgroupMatrixLoad` codegen is literally identical (`tint/lang/msl/writer/raise/builtin_polyfill.cc:1108-1238`), so staging's win is 128-thread amortization + reuse across 4 subgroups + small-stride conflict-free loads — which is how both native references run at 7-9 TF. Re-test under the new structure.

## 2. Ranked experiments — (a) doable in current WGSL + chromium-experimental-subgroup-matrix

### E1. The convergent reshape: 4 subgroups / 128 threads, 32x64 tile, 8 named accumulators, both operands staged (DO FIRST)
**Expected: 1.8-2.5x (2.3 → ~4.5-6 TF). Confidence: high — ORT, llama.cpp-Metal, llama.cpp-WGSL and MLX all independently converged on this exact shape. Effort: 1-2 days (three reference kernels to copy from, cached in scratchpad/sgmat/).**

This is one coherent kernel, not incremental toggles — the pieces interlock (staging only pays at 128 threads; 8 accumulators only fit when each subgroup owns a subtile). Exact spec:

- `diagnostic(off, chromium.subgroup_matrix_uniformity);` at the top.
- `@workgroup_size(128)` = 4 subgroups. Tile = 32(M) x 64(N), TILE_K = 32. Grid: `ceil(N/64) x ceil(M/32)`.
- Subgroup grid 2x2: `sgRow = subgroup_id >> 1`, `sgCol = subgroup_id & 1`; each subgroup owns a 16(M) x 32(N) subtile = **8 named accumulators** `c00..c13` (2 A-rows x 4 B-cols), f32 result type as today. Named vars, NOT an array — avoids crbug 443794633 spill (our current 16-frag array is squarely in the danger zone).
- Inner loop per K-tile (4 k-steps of 8): load `a0,a1` from shared A, `b0..b3` from shared B with `col_major=true`, then 8 MMAs reusing each A fragment across all 4 B fragments (ORT `subgroup_matrix_matmul_nbits_8x8x8.wgsl.template:176-210` is line-for-line the template).
- **Stage A**: cooperative 128-thread copy of the 32x32 f16 A chunk into `Ash` once per K-tile (bind `input_buf` as `array<vec4<f16>>` for the copy — K%64==0 already guaranteed). Fragment-load A from shared with the small stride.
- **Stage B (dequant)**: 64 rows x 32 K = 256 u32 words per tile; 2 words/thread across 128 threads (vs 8 words over 32 threads today). Replace the shift-chain with the vectorized nibble trick (ORT): `(vec4<f16>(unpack4xU8(p & 0x0F0F0F0Fu)) - vec4(7.0h)) * s` for low nibbles, `(p >> 4u) & 0x0F0F0F0Fu` for high — or MLX's folded-scale form with `s` and `s/16` (`quantized.h:486-528`), masks + FMA only. For the affine variant fold the MLX bias exactly as now.
- **Shared layout**: pad both strides to BK+8: `Ash: array<f16, 32*40>`, `Bsh: array<f16, 64*40>` = 7.5 KB per 128 threads = **60 B/thread** (vs 256 today; MLX is 40, llama.cpp 48). MLX's `BK_padded = BK + 16/sizeof(T)` (`quantized.h:1219`); ORT's Intel kernel documents the same `SHMEM_STRIDE = 40`.
- **Kill Osh on the common path**: when the tile is fully in-bounds (`mBase+32 <= M && nBase+64 <= N`), `subgroupMatrixStore` straight to `output_buf`; keep the scratch path only for ragged edges — llama.cpp Metal does exactly this split (`ggml-metal.metal:10340-10348`). Recovers 4 KB shared and the 32-iteration scalar copy loop.

Evidence for the magnitude: ORT's kernel of exactly this shape gave **2.7x prefill** on Metal over a weaker baseline (Phi-3.5 1K prefill 14.5 s → 5.4 s, PR #23729); MLX runs this shape at 7-9 TF (`quantized.cpp:1059-1064`, `quantized.h:1193-1319`); llama.cpp Metal at 9.0 TF (`ggml-metal-device.cpp:790-796`); llama.cpp WGSL at 71-74% of native. Our baseline is better than ORT's was, so expect somewhat under 2.7x.

### E2. Size the Chrome robustness tax (DO SAME DAY — zero code)
**Expected: information; 14-23% is the measured class. Confidence: high that the number is real; the flag itself is not shippable. Effort: ~1 hour.**

Run the existing bench (`BENCH_QUERY=... npm run bench` / gemm bench harness) in Chrome launched with `--enable-dawn-features=disable_robustness`, and once more adding `enable_integer_range_analysis_in_robustness`. LlamaWeb measured 14-23% average prefill cost for the safety checks (arXiv 2605.20706 §6.3). If the tax is big, restructure indexing so Dawn's integer-range analysis can elide clamps *without* flags: const tile dims, loop bounds derived from uniforms, no runtime-computed offsets into runtime-sized arrays. This number also calibrates every later experiment — it is the part of the gap no kernel work can close for real users.

### E3. Autotune sweep around the E1 shape
**Expected: +10-40%. Confidence: medium-high (LlamaWeb measured +41% average from sweeping vs hand-picked-on-Apple defaults; llama.cpp's tuned Apple config differs from ORT's). Effort: ~1 day given the kernel is already generated from TS.**

Parameterize the generator and sweep, per real shape bucket (M ∈ {64, 256, 512} x the model's K/N set): subgroups/wg {4, 8}, workgroup tile {32x64, 64x64}, per-subgroup fragment grid {2x4, 4x2}, TILE_K {16, 32, 64}, A-staging {shared, direct}, B shared layout {pad-40, llama.cpp 8x8-block swizzle}, store {direct, scratch}. First alternative candidate to try verbatim: llama.cpp WGSL's tuned config — 256 threads / 8 subgroups, 64x64 tile, 4x2 fragments, TILE_K 32 (`ggml-webgpu-shader-lib.hpp:42-49`, PR #22241). Ship a small dispatch table per M-bucket (ORT ships separate prefill/decode kernels and M-gates; PRs #23058, #23908).

### E4. Split-K for underfilled grids
**Expected: large on small-M/small-N dispatches, ~10-30% end-to-end at chunk M=64; ~nil at M=512. Confidence: medium-high (MLX ships it with a 512-threadgroup target). Effort: ~1 day, no subgroup features needed.**

With the 32x64 tile, M=64 x N=2560 yields only 80 workgroups — far under a 30/38-core GPU's appetite; GQA K/V projections (N=1024) are worse. Copy MLX: when `mTiles*nTiles < ~400`, set `split_k = max(1, 512/(mTiles*nTiles))` aligned to 32-wide K-tiles and whole quant groups; partials to an f32 scratch buffer; tiny second dispatch reduces (`quantized.cpp:1131-1155`, `strided_reduce` at :1207-1213). llama.cpp's `ne11_mm_min = 8` also says the tile kernel should win from batch 8 up — check our mv/mm switch point.

### E5. Shared-layout variant: llama.cpp's 8x8-fragment-contiguous swizzle
**Expected: 5-15% over pad-40. Confidence: medium. Effort: hours (as an E3 arm).**
Store staged tiles as contiguous 64-element 8x8 blocks (`sa + 64*ib + 8*ly + lx`) so every fragment load is a dense stride-8, 128-byte block, zero bank conflicts (`ggml-metal.metal:10240-10337`). Mutually exclusive with pad-40; measure both.

### E6. Re-measure the f32-operand config and f16-result config under the new structure
**Expected: 0-15%; mostly diagnostic. Confidence: low-medium. Effort: hours.**
(a) All-f32 operands: hardware is full-rate; if the 5x persists with 8 named accumulators, it is a Dawn/Tint lowering bug — file it (no tracking issue exists; cite philipturner/metal-benchmarks + MLX mma.h as evidence). (b) f16 *results*: ORT and llama.cpp accumulate f16 on Apple; if it measures faster than our f32 results, guard Qwen with a bounded-K drain (fold to f32 every 512-1024 K) — llama.cpp's f16-acc NaN bug on qwen2.5 is real (issue #21602, warned in the shader itself). Our current f16xf16→f32 is the safe default; only move for a measured win.

### E7. Double buffering — LAST, only if plateaued below ~5 TF
**Expected: unknown; no WGSL demonstration exists. Confidence: low. Effort: 1 day.**
Single 2x-size `var<workgroup>` array, alternate halves per K-tile, interleave tile t+1 cooperative loads between tile t's 8 MMA calls (CubeCL's winning recipe, burn.dev/blog/sota-multiplatform-matmul). Every kernel at 66-75% of native ships WITHOUT it. Also per CubeCL: skip producer/consumer subgroup specialization on Apple — it lost even natively.

**Skip entirely:** A-prepack pass (ORT marks Apple 8x8x8 `needsPrepack=false`); separate dequant-then-GEMM pipeline (MLX has no such path at any M — `quantized.cpp:1779-1827`); mining WebLLM/Ratchet/web-rwkv/tinygrad (all behind our kernel tech for prefill).

## 3. (b) Blocked on the platform

- **Robustness clamps for shipped pages** — `disable_robustness` is a local Dawn toggle, not web-exposed. Mitigation: index shapes that Dawn's `enable_integer_range_analysis_in_robustness` can prove in-bounds (Dawn `src/dawn/native/Toggles.cpp`; no public tracking issue cited). Cost class: 14-23% of prefill (LlamaWeb).
- **subgroup-matrix is chromium-experimental** — standardization at https://github.com/gpuweb/gpuweb/issues/4195 (proposal `proposals/subgroup-matrix.md`, Draft 2025-10-02), Chromium tracking https://issues.chromium.org/issues/348702031. Watch for breaking renames; ORT/llama.cpp track Dawn head.
- **Only two Metal configs exist, both 8x8x8** (f16, f32), hardcoded Apple7+ (`PhysicalDeviceMTL.mm:1055-1073`); no other shapes to tune toward.
- **f32-operand fragment slowdown** — if E6 confirms, it's a Dawn/Tint bug to file; nothing we can fix in WGSL.
- **No occupancy/scheduling control from WGSL** (no `maxTotalThreadsPerThreadgroup`, no residency hints) — the only lever is shared-memory footprint per thread, which E1 cuts 4x. MLX's NAX/tensor path and llama.cpp's Metal-4 tensor kernels are M5+/gen-17 only — irrelevant on M2 Max, and NOT part of the 7-9 TF numbers we're chasing.

## 4. Measurement protocol

Every experiment: run the M ∈ {64, 256, 512} x model-shape matrix in the existing gemm bench + end-to-end `BENCH_QUERY="?model=qwen35" npm run bench`, with `macmon` open to catch GPU-clock throttling confounds. Targets: 3.5 TF = gpu.cpp plain-WGSL bar (beat it → dequant path is no longer the limiter); 5.0-6.5 TF = 70-75%-of-native ceiling; 7.2/9.0 TF = native parity (30/38-core), not expected to be reachable. Validate numerics per model against the existing mlx_lm gates (`scripts/validate-model.mjs`); f32-result fragments mean no NaN drain machinery is needed unless E6(b) is adopted.