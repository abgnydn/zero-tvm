<!-- Produced by a four-angle plan + adversarial reconciliation, 2026-08-10.
     Two load-bearing claims were re-verified by hand before committing, both hold:
       - headDim is 128, not 192. o_proj.weight is [2048, 256] in the layer-0
         bundle, so its contracted K is 2048, and engine-core derives
         OPROJ_K_PACKED = qDim/8, pinning qDim = 2048 = 16 x 128. At 192 the
         engine reads a 384-word row against a 256-word one: finite numbers,
         wrong model.
       - widening layerKinds to include 'mla' produces ZERO TypeScript errors
         across its twelve consumers, while silently making allocKVPages return
         a zero-length array and kvBytesPerToken 0. Verified by making the edit
         and running typecheck.
     Items marked [FALSE] correct an individual angle, and are kept because the
     wrong version is the one a reasonable person would have written. -->

# MLA → decode engine: one ordered plan

Everything below was checked against the files. Where an angle asserted something the code does not say, it is corrected here and the correction is marked **[FALSE]**.

---

## Part 0 — Contradictions, resolved

### 0.1 `headDim` = 128, not 192. **[testing angle is FALSE]**

The testing angle sets `headDim: 192` and argues `SM_SCALE = ropeAttnScale(S)/sqrt(S.headDim)` (`engine-core.ts:686`) is then "exactly right". It is right for the scale and wrong for everything else, and the bundle settles it:

`o_proj.weight` is `[2048, 256]` and `o_proj.scales` is `[2048, 32]` (`.weights-local/kernel-refs/dsv2layer0/meta.json`). So o_proj's contracted K is 2048. `engine-core.ts:387-388` derives `OPROJ_K_PACKED = S.qDim/PACK` and `OPROJ_SCALES = S.qDim/QGROUP`, and `variants.ts:278` resolves the o_proj matmul instance from `spec.qDim`. With `headDim: 192`, `qDim = 3072`, `OPROJ_K_PACKED = 384` against a 256-word row — every scale group misread, finite numbers, wrong model.

**Decision: `headDim: 128` ( = `mla.vHeadDim` = `mla.qkNopeHeadDim`), `kvHeads: 16` (what `num_key_value_heads` says), and `SM_SCALE` gets an MLA branch.** Add a hard assertion in `makeModelSpec` — `base.mla && base.mla.vHeadDim !== base.headDim` throws — so this specific mistake becomes a build error rather than a debugging session.

Corollary the angles missed: `add-model.mjs:149` computes `headDim = tc.head_dim ?? d/heads` = 2048/16 = **128** for this config, so a generated spec lands on the right value *by luck*. Pin it with the assertion, don't rely on it.

### 0.2 `rotaryDim` — derive from `mla`, not from `partialRotaryFactor`

The dispatch angle proposes `partialRotaryFactor: 0.5` (→ `rotaryDim = round(128*0.5) = 64`); the loader angle says "set rotaryDim explicitly", which `makeModelSpec` does not permit (`model-spec.ts:420` derives it).

Both land on 64 here, but 0.5 is a coincidence of `vHeadDim == 2 * qkRopeHeadDim`, and `add-model.mjs:283` reads `partial_rotary_factor` from config — DeepSeek's config has none, so a generated spec would silently get `rotaryDim = 128` and a 64-entry yarn table.

**Decision: in `makeModelSpec`, `const rotaryDim = base.mla ? base.mla.qkRopeHeadDim : Math.round(base.headDim * (base.partialRotaryFactor ?? 1))`.** Verified this reproduces the reference: `ropeInvFreqTable` (`model-spec.ts:497-519`) uses `dim = spec.rotaryDim` and emits `spec.halfRotary` entries; with 64/32 that is term-for-term `yarn_inv_freq(ROPE=64)` at `make-dsv2-layer-ref.py:136-151`, and `tests/unit/rope-yarn.test.ts` already pins that table against DeepSeek's own `modeling_deepseek.py` at dim 64.

### 0.3 "Paged" vs "flat" latent cache — not a real disagreement

`mla_scores.wgsl:59` indexes `cache_c[t*L + i]`. A paged layout with per-page stride `pageSize*L` gives `(t/pageSize)*(pageSize*L) + (t%pageSize)*L + i`, which **is** `t*L + i` whenever the page table is the identity — which is what `engine-core.ts:847-852` writes once and nothing ever permutes. `maxPages * pageSize === maxContext` (`model-spec.ts:453`), so the kv angle's buffer and the dispatch/testing angles' buffer are byte-for-byte the same allocation.

**Decision: size it from `maxContext`, keep `maxPages`/`pageSize` as the budget knob and the `MAX_CONTEXT` guards, and do NOT bind `pageIndptr`/`pageValues` to any MLA kernel.** State at the allocation site that the flat indexing is valid *only* while the page table is the identity, and that real paging (eviction, block reuse, a shared prefix pool) would require teaching `mla_scores`/`mla_combine` the page table.

### 0.4 `layerKinds` stays `'attn'`; MLA is a model-level `spec.mla`. **[dispatch + testing angles overreach]**

Full consumer list, verified: `engine-core.ts:95, 315, 331, 517, 970, 1795`; `model-spec.ts:455`; `weight-loader.ts:716`; `chat.ts:46`; `mlx-weights.ts:185`; `tests/unit/kv-budget.test.ts:44`; `tests/kernels/mlx-repack.mjs:234`. Twelve sites.

Widening `ReadonlyArray<'gdn'|'attn'>` to include `'mla'` produces **no TypeScript error** at any of them — `k === 'attn'` against a wider union compiles fine. It silently turns `allocKVPages` into a zero-length array (`:95`), `kvIndex` into all `-1` (`:331`), and `kvBytesPerToken` into `0` (`model-spec.ts:455`, which `chat.ts:216` renders as "~0 MiB"). That is three new silent wrongs bought for nothing, because **no DeepSeek checkpoint mixes MLA with another attention kind in one stack** — `layerKinds` exists for Qwen3.5, which genuinely does mix.

Every branch that must change (`allocKVPages` sizing, `kvBytesPerToken`, `planLayer`, `recordForward`, the bind-group loop) changes under *either* design. Keeping `'attn'` additionally keeps `hybrid` false (`:315`), `computeReuseStart`'s non-hybrid branch (`:836`) and chunked-prefill's gating correct for free — verified: `chunkPrefill` at `:1922` requires `hybrid && !S.moe && !AFFINE`, all three of which fail for DeepSeek.

**Decision: `mla?: MlaDims` on `ModelSpecBase`, `layerKinds[L]` stays `'attn'`.** The one residual risk — forgetting the `allocKVPages` branch is silent (7× VRAM, still-correct output) — is closed by putting the branch *inside* `allocKVPages` (one place; all five call sites at `chat.ts:224`, `loading-ui.ts:272`, `share.ts:193`, `share.ts:577`, `lib/index.ts:224` become correct by construction) and by the kv-budget test row in Step 1.

### 0.5 Loader prep: an `op` on `BufferPlan`, not new `Convert` kinds. **[dispatch angle is FALSE on mechanism]**

The dispatch angle proposes new `Convert` values (`'permute-rows'`, `'dequant-affine-f16-T'`). `Convert` is a **per-part** field (`mlx-weights.ts:138-143`) carrying no parameters, and `planBytes` (`:291-298`) maps it at a fixed byte ratio. A row permutation needs the *concatenated* row count; the kv_b dequant *fuses* three parts that each already have their own `Convert`. Neither is expressible.

**Decision: the loader angle's `op?: PlanOp` on `BufferPlan`, handled inside `buildBuffer`, defaulting to today's concatenate.** Placement inside `buildBuffer` is forced by the cache: `assembleMlx` caches the **built** buffer keyed by plan (`weight-loader-mlx.ts:221-229`), so prep there is paid once and served from OPFS. Prep anywhere later pays 27 layers × 2M-element dequant on every load — including every `npm run dev` page load, where `cacheWrite` is deliberately `undefined` (`weight-loader.ts:566-568`).

### 0.6 ONE kv_b plan holding `[K^T ‖ V]`, two bind regions. **[kv + dispatch angles propose two plans]**

Two plans would fetch the same 1,179,648 B trio and dequantize the same 2,097,152 values twice per layer, on every cold load and every dev-mirror load.

Verified this works: `wk_t` is `np.transpose(Wk, (0,2,1))` (`make-dsv2-layer-ref.py:425`) = `[heads, kvLora, nope]` = `[16,512,128]`, and `wv` = `Wv` = `[heads, vHeadDim, kvLora]` = `[16,128,512]`. Both are 65,536 elements per head; `mla_proj.wgsl:42` computes `wBase = (h*N + n)*K` from the **start of whatever is bound**, so each half is reachable as a region with its own `(N,K)` uniform. The K half is 2,097,152 B — a multiple of 256, so V's offset clears `minStorageBufferOffsetAlignment`. `BindEntry` at `engine-core.ts:63` already accepts `{buffer, offset, size}`; the GDN packed projection at `:1017-1018` is the precedent.

**[dispatch angle is FALSE on the transpose target]**: it writes "TRANSPOSED to `[heads, kvLora, nope]`… no — to `[heads, nope, kvLora]`", self-correcting in the wrong direction. `[heads, kvLora, nope]` is right — that is `wk_t`, and it is what `mla_proj` is dispatched with in the passing test (`real-weights.mjs:903`, `dims(L, NOPE)` → `N=512, K=128`).

**Order is load-bearing and must be stated once: K^T first, V second. The V offset must be computed from `spec.mla`, never hardcoded.**

### 0.7 `position` is an explicit parameter to `recordForward`, not a closure. **[kv angle inverts the risk]**

The kv angle proposes `let curPos` set in `writeStepState` "so no caller passes a stale value". A closure is *precisely* how a stale value survives with no error; a parameter makes a missed call site a `tsc` failure. All four call sites already hold it in a local: `pipelineStep:1442` (`position`), `decodeToken:1475` (`position`), `submitStep:1957` (`position`), `profileStep:2153` (`nextPos`).

**Decision: `recordForward(enc, position)`.** The genuine coupling the kv angle worries about — the grid's `T` and the uniforms' `T` disagreeing — is closed by deriving both from `position` through one shared expression (`T = position + 1`), used in `writeStepState` for the two uniform patches and in `recordForward` for the `mla_scores` grid.

### 0.8 Keep `mla_combine` writing f32; add a narrowing kernel

All four angles agree the f32→f16 seam is real (`mla_combine.wgsl:19` writes f32, `mla_proj.wgsl:27` reads f16; the test bridges it on the CPU at `real-weights.mjs:946`). The kv angle's argument against changing `mla_combine` to emit f16 checks out numerically: `relErr` (`real-weights.mjs:263-269`) normalises by `max|ref|`, f16 gives ~4.9e-4 there, observed error is already 3-4e-4, and the bound is 1e-3. Narrowing inside `mla_combine` would consume nearly the whole margin of an already-passing kernel.

**Decision: a 10-line `mla_narrow.wgsl`. Leave the three verified shaders byte-identical.**

### 0.9 Prep kernels: two files, not one module with two entry points, not three kernels

The dispatch angle wants one `mla_prep.wgsl` with grid `heads+1` (workgroup `heads` doing entirely different work); the kv angle wants `mla_kv_append` + `mla_q_split`; the testing angle wants three.

The dispatch angle's own hazard is the decider: `layout:'auto'` gives each entry point a layout containing only the bindings *it* reads (the class of bug annotated at `engine-core.ts:879-885` and `905-912`), so two entry points in one module cannot share an entry list. Its escape — one entry point with a magic workgroup index — has different bindings *live* on different workgroups, no dispatch saving worth the divergence, and no verification advantage (the bundle dumps `ref_qnope`/`ref_qpe`/`ref_c`/`ref_kpe` separately either way).

**Decision: two files, split by which side of the layer they serve** (the repo's own convention — `gdn_conv` / `gdn_conv_seq` / `gdn_conv_commit` are separate files):

- **`mla_q_split.wgsl`** — q_proj out `[heads, 192]` → `q_nope [heads,128]` + half-split-RoPE'd `q_pe [heads,64]`. Grid `heads`. Binds: q_proj out, inv_freq, `posMap`, q_nope, q_pe, uniform.
- **`mla_kv_write.wgsl`** — kv_a out `[576]` → RMSNorm the first 512 with `kv_a_layernorm.weight` straight into the latent region at `position*512`, RoPE the last 64 into the k_pe region at `position*64`. One 64-lane workgroup (the RMSNorm reduction fits). Binds: kv_a out, gamma, inv_freq, `posMap`, latent region, k_pe region, uniform.

No existing kernel substitutes, verified:
- `rms_norm.wgsl:31,35,57,60` hardwires the row to prelude `D` with a `D/64` loop — binding a 512-wide latent reads 2048 elements, and WGSL clamps out-of-bounds storage reads, so it returns a finite wrong norm.
- `rope.wgsl:67,74` computes `dim_idx = within % HEAD_DIM` and rotates `dim_idx < ROTARY_DIM`, i.e. the **first** 64 of each head; MLA's pe is the **last** 64 of 192, and k_pe is one un-headed 64-wide row.
- `kv_append.wgsl:55-66` writes `page_no*KV_PAGE_STRIDE + head*HEAD_PAGE_STRIDE + slot*HEAD_DIM + dim` and a V region — a head axis and a V that MLA's cache does not have.

Both read `position_map` as a **storage** array exactly as `kv_append.wgsl:52` does — verified `B.posMap` is `makeBuf` (STORAGE) and bound at index 3 in `engine-core.ts:1037`. That keeps the bind groups hoisted; a dynamic bind offset would need a per-token bind group and `t*64*2 = 128 B` k_pe slots are not 256-aligned anyway.

### 0.10 Corrections to smaller claims

- **[FALSE]** dispatch angle: "`S.qkvDim` for this spec is 2304 (qDim 2048 + 2*kvDim 256)". With `kvHeads: 16, headDim: 128`, `kvDim = 2048` and `qkvDim = 6144` (`model-spec.ts:443`) — larger than q_proj's 3072 rows, so the claimed robust-access truncation does not occur. The advice (allocate a dedicated `mlaQ`) is still right, for the reason at `engine-core.ts:530-534`: a stray dense dispatch should fail loudly, not corrupt. Note the *inverse* hazard the number hints at: if anyone sets `kvHeads: 1` to "describe" MLA, `qkvDim` really does become 2304 and the truncation becomes real. Don't invent `kvHeads`; the config says 16.
- **[FALSE]** loader angle hazard: "deriving `rowBytes` as `byteLength/rows` is correct for 4-bit but not for 3-bit". `byteLength/rows` is computed from the actual bytes and is dtype-agnostic; it is right for any uniform row length. What is actually required is exact divisibility (assert it) and that rows are the leading axis (MLX stores `[N, K/8]` — they are).
- **[FALSE]** testing angle: "at the bundle's shipped `tokens: 6`… a page index is always 0, and every page/slot address bug passes." Under the identity page table there is no page/slot arithmetic in the MLA path *at all* — `mla_kv_write` writes `position*L + i` directly. Regenerating at 20 tokens is still worth doing, but for a different reason: it catches a **position-vs-slot** or stale-RoPE-table bug in the append, which shows as error growing with `t` (and `t=0` is RoPE-identity, so 6 tokens is a weak signal). Keep the step, fix the rationale.
- **[UNVERIFIABLE — do not restate as fact]** kv/loader/dispatch angles all state 27 layers, vocab 102400, `maxSeq` 163840, `first_k_dense_replace` 1, 64 experts, `moe_intermediate` 1408, `n_shared_experts` 2. The bundle only proves `d=2048, heads=16, nope=128, rope=64, v=128, kv_lora=512, dense_ffn=10944`. Everything else must be read from the checkpoint's `config.json` when the spec is written. Do not transcribe from these plans.
- Drive-by, not in scope: `SUPPORT_MATRIX` row `constraints.ts:439` still lists yarn under `not:`, but `checkModel` turned yarn **green** at `constraints.ts:242-251` and `add-model.mjs:363-366` emits it. `docs/COMPAT.md` is generated from that row, so it currently documents a limitation that no longer exists. Mentioning, not fixing.
- Drive-by: `allocKVPagesInt8` (`engine-core.ts:112,115`) sizes its arrays from `spec.layers` while `allocKVPages` (`:95`) uses the attention-layer count. Pre-existing, masked because int8 is gated off for hybrid. Do not fix here; do not copy it.
- The kv angle's explicit throws for `S.mla && (fused || int8Mode)` and `S.mla && splitK` guard **unreachable** paths: `AFFINE && fused` already throws at `:368-372`, `int8Mode` requires `fused` at `:307`, and `splitK` only feeds `attnF16BG()`/`bgAttnCombine`, which an MLA layer never dispatches. Skip them (CLAUDE.md: no error handling for impossible scenarios), or reduce to a comment.

---

## Part 1 — Ordered steps

Each step names what verifies it. No step depends on a check that only becomes possible later.

### Step 1 — Spec: `MlaDims`, the MLA derivations, and the budget branch
**Files:** `src/compiler/model-spec.ts` (`ModelSpecBase` 180-252, `ModelSpec` 254-327, `makeModelSpec` 384-466); `tests/unit/kv-budget.test.ts`; a new `tests/unit/mla-spec.test.ts`.

Add `mla?: { kvLoraRank; qLoraRank: number|null; qkNopeHeadDim; qkRopeHeadDim; vHeadDim }`, field-for-field identical to `DetectedMla` (`constraints.ts:73-79`) so `add-model.mjs:240-246` can hand it over verbatim.

New derived fields, **beside** the existing ones — touch none of `kvPageStride`/`headPageStride`/`vPageOffset`/`qkvDim`, because `compile()` builds `kv_append`/`attention`/`qkv_fused` pipelines for **every** spec (`compiler.ts:238-316`) off the prelude consts those feed (`shader-prelude.ts:50-59`):

```
mlaCachePerToken  = kvLoraRank + qkRopeHeadDim        // 576
mlaQProjRows      = heads * (qkNopeHeadDim + qkRopeHeadDim)   // 3072
mlaKvaRows        = kvLoraRank + qkRopeHeadDim        // 576
mlaKvbRows        = heads * (qkNopeHeadDim + vHeadDim)        // 4096
mlaLatentBytes    = maxContext * kvLoraRank * 2       // k_pe region offset
mlaCacheBytes     = maxContext * mlaCachePerToken * 2 // per-layer buffer
```

Three changes to existing derivations, all branched on `base.mla`:
1. `rotaryDim = base.mla ? base.mla.qkRopeHeadDim : <existing>` (§0.2).
2. `kvBytesPerToken` (`:454-455`) → `mla ? mlaCachePerToken * 2 * attnLayers : <existing>`.
3. New assertions: `mla.vHeadDim === headDim` (§0.1); `mla.qLoraRank === null` (V2-Lite; V2-full/V3 split q into `q_a_proj`/`q_b_proj`, a record set this plan does not produce); `mlaLatentBytes % 256 === 0`.

Then `DEEPSEEK_V2_LITE` at the `ADD-MODEL:SPECS` marker (`model-spec.ts:884`), with **layers / vocab / maxSeq / moe dims read from the checkpoint's `config.json`, not from any of these plans**. `maxPages` by the documented rule (`model-spec.ts:191-201`, mirrored at `add-model.mjs:340-349`): `min(1 GiB / (kvBytesPerToken × 16), maxSeq/16)`, floored to a multiple of 64. At 27 layers that is `floor(2^30 / (31104×16)) = 2157 → 2112` pages → **33,792 tokens**, 1002 MiB of latent cache, 37.1 MiB per layer.

**Verify:**
- `npm run test` — new `mla-spec.test.ts` (pattern: `tests/unit/mixed-ffn-stack.test.ts:15-27`) asserts `qDim === 2048`, `rotaryDim === 64`, `halfRotary === 32`, `mlaCachePerToken === 576`, `kvBytesPerToken === 31104`, and that `ropeAttnScale(S) / Math.sqrt(qkNope + qkRope)` equals the bundle's `meta.softmax_scale` (0.1147213868) to 1e-9 — this single assertion is what makes the `sqrt(headDim)` bug impossible to ship.
- kv-budget test: add a DeepSeek row. The line-45 formula and the line-63 `maxPages * kvPageStride * 2 ≤ 128 MiB` assertion both need MLA branches — unbranched, line 63 computes 268 MB and fails. Assert the headline too: MLA is 7.11× under `2*kvHeads*headDim*2*attnLayers` (31,104 vs 221,184 B/token).
- `node tests/kernels/compile-spec.mjs DEEPSEEK_V2_LITE`. **Read this result correctly**: `compile-spec.mjs:51-56` compiles every `.wgsl` in the directory, so `mla_*.wgsl` already compile under every spec — but they read **zero** prelude consts (all dims arrive via `PODArgs`: `mla_scores.wgsl:33-41`, `mla_combine.wgsl:23-27`, `mla_proj.wgsl:30-34`). A green gate proves the *other* shaders survive the new dims. It is not evidence about MLA.

Leave `constraints.ts:340-347` failing. `checkModel` going green is what authorises `add-model` to generate a spec and a registry row.

### Step 2 — Loader prep, byte-verified on CPU, with no checkpoint and no GPU
**Files:** `src/zero-tvm/mlx-weights.ts` (137-150, 163-238, 291-298, 316-360); `src/zero-tvm/weight-loader-mlx.ts` (109-111, 156, 164-182); `src/zero-tvm/weight-loader.ts` (789-853); a new case in/beside `tests/kernels/mlx-repack.mjs`.

**2a. `op?: PlanOp` on `BufferPlan`, handled in `buildBuffer`, defaulting to concatenate.**

```
{ kind: 'permuteRows'; rows: number; src: number[] }
```
Applied to the concatenated bytes. `rowBytes = byteLength / rows` (assert exact divisibility). Dest row `j` takes source row `src[j]` — the direction at `real-weights.mjs:840-845` and `make-dsv2-layer-ref.py:209-215`. Size-preserving. Verified the same `src` map applies unchanged to all three components: q_proj weight 3,145,728/3072 = 1024 B/row, scales 196,608/3072 = 64, biases 64; kv_a 589,824/576 = 1024, 36,864/576 = 64, 64. Affine scale groups run along the **input** axis, which is why rows are independent — that independence is the whole reason DeepSeek's interleaved RoPE is free.

```
{ kind: 'dequantAffineSplit'; bits: 4; group: 64; k; heads; nope; v }
```
Parts are `[weight, scales, biases]`, **fused element-wise** rather than concatenated: `((w[r][i>>3] >> ((i&7)*4)) & 15) * s[r][i>>6] + b[r][i>>6]` — the exact arithmetic of `make-dsv2-layer-ref.py:104-111`, which `--backend both` proved bit-identical to `mx.dequantize`. Emits f16: `heads` blocks of `[kvLora][nope]` (K^T) then `heads` blocks of `[v][kvLora]` (V). Dequantize **row by row and scatter** (V half a row copy, K half a strided column write at stride `nope`) so no `[4096,512]` f16 intermediate is materialised.

The f32→f16 encoder is new (`mlx-weights.ts` has `bf16ToF16` at :96 and `bf16ToF32` at :126, no f32→f16). It **must round f64→f32→f16**, i.e. store through a `Float32Array`, exactly as `tests/kernels/half.mjs:12-14` does, because numpy computes `q*s+b` in f32 then casts (`make-dsv2-layer-ref.py:111`). JS arithmetic is f64; a direct f64→f16 rounding is a different function on ties and will scatter one-bit differences across 2M values that read as a layout bug. Count and throw on overflow past 65504, the way `bf16ToF16` does (`mlx-weights.ts:353-358`).

`planBytes` must become op-aware: `permuteRows` is size-preserving; `dequantAffineSplit` is `heads*(nope+v)*k*2` = **4,194,304 B/layer**, which is **3.556×** the 1,179,648 source bytes. Op-blind, `modelBytes` (`weight-loader-mlx.ts:146-149`), the `mlx-repack.mjs:218-245` residency check and the loading UI all under-report — the failure mode is telling a user a model fits when it does not.

**2b. `planLayer`'s attn arm (`mlx-weights.ts:185-193`) branches on `spec.mla`**, emitting at prefix `model.layers.N` (`mlxPrefix` is `''` — text-only checkpoint):

| plan | source | op |
|---|---|---|
| `mla_q_{w,s,b}` | `self_attn.q_proj` | `permuteRows{rows: 3072, src}` |
| `mla_kva_{w,s,b}` | `self_attn.kv_a_proj_with_mqa` | `permuteRows{rows: 576, src}` |
| `mla_kva_norm` | `self_attn.kv_a_layernorm.weight` | — (`bf16->f16`) |
| `mla_kvb` | `self_attn.kv_b_proj` trio | `dequantAffineSplit` |
| `o_proj_{w,s,b}` | unchanged, the existing line at `:189` | — |

`norm1`/`norm2` unchanged (`:180-183`). Deliberately **no** `c_attn`: MLA has no fused QKV and q_proj is not a q++k++v concatenation. Reusing the `c_attn_*` names would land the weights in `LoadedWeights.qkvWeights`, where the unfused attention branch at `engine-core.ts:1044-1046` finds them and builds a GQA layer against a 3072-row q_proj — bind-group validation passes, the matmul runs, output is nonsense.

`src[j] = j < R/2 ? 2*j : 2*(j - R/2) + 1` for `R = qkRopeHeadDim` — the de-interleave. Emit a **full array of length `rows`**, identity outside the pe rows: for q_proj only `h*192 + 128 + j` move (16 heads × 64 rows of 3072); for kv_a only rows 512..575. A whole-tensor permutation is trivially easy to write and destroys the latent projection.

Note the dtype guard at `mlx-weights.ts:326-335` is load-bearing here: this checkpoint ships `kv_a_layernorm.weight`, all scales and all biases as **F16** (bundle meta), which reaches the pass-through at `:328-329` only because the plan says `'bf16->f16'`. Planning them `'raw'` works today by accident and breaks on any sibling checkpoint that ships bf16.

**2c. `SLOTS` + `LoadedWeights`.** Add `'mla'` to the `Slot` group union (`weight-loader-mlx.ts:156`) — `assembleMlx:243-244` already writes any non-`root`/`layer` group generically — and rows for the seven new plan names. `assembleMlx:220` **throws** on an unplaced plan; that exhaustiveness is the safety net, so add the rows in the same commit as the plans. On `LoadedWeights.layers[]` (`weight-loader.ts:819-848`), `mla?: { qWeights, qScales, qBiases, kvaWeights, kvaScales, kvaBiases, kvaNormGamma, kvbF16 }` beside `gdn?`/`moe?`.

**2d. `planKey` content hash for op-bearing plans only** (`weight-loader-mlx.ts:109-111`). Every existing plan is a pure function of the checkpoint; an op-bearing plan's bytes depend on code under development. A stale OPFS entry survives the fix, and `peer-weights.ts` replicates it to other machines by filename with a SHA that proves transport, not correctness. Append a short hash of `(op.kind + numeric params + part record names)` **only when `plan.op` is present** — `l0.mla_kvb.4a7c` — so every existing key stays byte-identical and the 19.7 GB of Qwen3.6 caches on disk are not invalidated. `opfsKey` (`weight-loader.ts:149-152`) preserves `.` and `-`, so no escaping.

**Verify — this is the strongest check available anywhere in this plan, and it needs no GPU and no 9 GB download:**

The `dsv2layer0` bundle is a flat per-tensor directory (`meta.tensors[name].file`, with `dtype` recorded), and `buildBuffer(plan, readRecord, dtypeOf)` takes both as callbacks. So run the real loader over the bundle:

- **kv_b, byte-exact, zero tolerance.** `make-dsv2-layer-ref.py:111` rounds through `np.float16` **before** widening for the dump, so every value in `wk_t.bin`/`wv.bin` (4,194,304 B each = 16×512×128 f32) is exactly f16-representable. Decode the loader's f16 with `f16BitsToF32` and compare with `===`. Any difference is a real difference — and this is the only check that catches a wrong transpose, because an untransposed Wk still yields a well-formed `[heads,512,128]` matrix that `mla_proj` runs to completion on.
- **Permutation, two checks, at least one going through the bundle's own evidence.** (a) Dequantize the loader's permuted q_proj and assert row `h*192+128+j` equals the raw record's row `h*192+128+src[j]`. (b) Run the affine matvec on `ref_h1` row `query_at` (=5) through the loader's permuted buffers and compare against `ref_qperm` / `ref_kvaperm` at ~1e-3 — which is exactly what `real-weights.mjs:855-880` already does with its own inline permutation. Check (a) alone is circular and passes on a reversed map; `ref_qperm`/`ref_kvaperm` are the only independent evidence in the repo, and they exist because `make-dsv2-layer-ref.py:236-241` asserts at reference-build time that the row permutation reproduces DeepSeek's interleaved rotation.
- **planBytes**: assert `planBytes(mla_kvb) === 4194304`.

Note the DeepSeek case cannot go through `openMlxCheckpoint` like the Qwen3.6 cases at `mlx-repack.mjs:249-317` — that path needs real safetensors shards. Give the new case its own `readRecord`/`dtypeOf` pair over `meta.tensors`, the way `real-weights.mjs:823-827` reads the bundle.

### Step 3 — Compile the five MLA pipelines
**Files:** `src/compiler/compiler.ts` (`Pipelines` 77-182, the literal 238-316); two new shaders + one cast.

`rg` confirms **zero** occurrences of "mla" in `compiler.ts` and all of `src/zero-tvm/*.ts` — nothing is registered today. Add `mlaProj`, `mlaScores`, `mlaCombine`, `mlaQSplit`, `mlaKvWrite`, `mlaNarrow`. All spec-agnostic (every dim in `PODArgs`), so they compile under every spec like the GDN family.

Two hard constraints, both from `layout:'auto'`:
- **`mla_proj` must be ONE pipeline object** serving both directions (`q_nope → q_lat` and `o_lat → o_head`), differing only in the uniform. Two structurally identical pipelines own two distinct layout objects; binding one and dispatching the other is a validation error — the reason `moeMM` is resolved once at `engine-core.ts:411` and the note at `:879-885` exists.
- **`mla_narrow` must read its `n`** for the bounds guard. A uniform the kernel never touches is dropped from its layout and the 3-entry bind group is rejected — the same shape as the sampler note at `engine-core.ts:905-912`.

Write `mla_q_split.wgsl` and `mla_kv_write.wgsl` per §0.9.

**Verify:** `node tests/kernels/compile-spec.mjs DEEPSEEK_V2_LITE` and `PHI3` (the new files must not break any spec). Then, per-stage against the bundle in `real-weights.mjs`, **one dispatch at a time before composing anything**:
- `mla_q_split` fed `ref_qperm` at position 5 → compare `q_nope` against `ref_qnope` and `q_pe` against `ref_qpe`.
- `mla_kv_write` fed `ref_kvaperm` at position 5 → compare the latent it wrote against `ref_c[5]` and the k_pe against `ref_kpe[5]`.
- `mla_narrow` fed `ref_olat` → compare against `f16Array(ref_olat)`.

These three stages are the ones the existing `dsv2Layer` (`real-weights.mjs:816-968`) proves nothing about — it feeds `ref_qnope`/`ref_qpe`/`ref_c`/`ref_kpe` straight from numpy (`:885, 903, 913`). `kv_a_layernorm.weight` is in the bundle and is read by **no GPU test in this repo** today.

### Step 4 — Rewrite `dsv2Layer` to seed only `ref_x`, and take the scale from the spec
**File:** `tests/kernels/real-weights.mjs:816-968`, on the pattern of `qwen36Layer:537-664` / `qwen36Attn:405-512`.

One `runChain` (`:245-259`) over: `rms_norm(ref_x[qi], input_layernorm)` → permuted q_proj (from the **loader**, Step 2, not the test's inline permutation) → permuted kv_a_proj → `mla_q_split` → `mla_kv_write` → `mla_proj(wk_t)` → `mla_scores` → `mla_combine` → `mla_narrow` → `mla_proj(wv)` → o_proj → `add_norm` → gate_up (10944 rows, K=d) → `silu_mul` → down → `add_norm`.

Compare per stage against `ref_h1[qi]`, `ref_qperm`, `ref_kvaperm`, `ref_qnope`, `ref_qpe`, `ref_c[qi]`, `ref_kpe[qi]`, `ref_qlat`, `ref_scores` (normalised both sides as at `:922-943`), `ref_olat`, `ref_oheads`, `ref_attn_out`, `ref_resid1`, `ref_h2`, `ref_ffn_out`, `ref_out`. Every one of those already exists in the bundle. Keep the 1e-3 bound and the reasoning at `:961-962`.

**The scale must come from `ropeAttnScale(DEEPSEEK_V2_LITE) / Math.sqrt(nope + rope)`, not from `meta.softmax_scale`.** Today the test passes `meta.softmax_scale` (`:906-910`), which means an engine that drops yarn's mscale² (1.5896) or divides by `sqrt(headDim)` still passes. Keep `meta.softmax_scale` as a *separate assertion* against the computed value.

Seed the cache for positions 0..qi-1 from `ref_c`/`ref_kpe` at the **engine's addresses** (latent at `t*kvLoraRank`, k_pe at `mlaLatentBytes/2 + t*qkRopeHeadDim`), then let `mla_kv_write` produce position `qi` on the GPU and compare that row. That is the "their cache, our current token" arrangement that made `qwen36Attn` a stronger test than a self-prefilled one (`real-weights.mjs:399-403`), and it is what proves the address math, which nothing currently does.

**Verify:** `npm run test:kernels:real` — all sixteen stages under 1e-3.

### Step 5 — Regenerate the bundle at 20 tokens and build the whole cache ourselves
**Files:** `scripts/make-dsv2-layer-ref.py:63` (`--tokens`, default 6); a second handler in `real-weights.mjs`.

**Copy the bundle first** — the script writes `meta.json` and every `ref_*.bin` in place (`:433-434, 453`). It reads only the `.bin` tensors already on disk (`raw()` at `:87-94`) and generates `x` from the seed, so `--tokens 20` costs no re-pull.

For `t = 0..19`: `rms_norm(ref_x[t])` → kv_a_proj → `mla_kv_write` at position `t`; compare the **whole** cache buffer against `ref_c` and `ref_kpe` **per position**, reporting error per `t` rather than a max. Then run the query at `t=19` against the self-built cache and re-check `ref_scores`/`ref_olat`/`ref_oheads`/`ref_attn_out`.

What this catches that Step 4 cannot: a position-vs-slot bug in the append, or a stale/wrong RoPE table, both of which show as error **growing with `t`** — and `t=0` is RoPE-identity (cos 1, sin 0), so six tokens is a weak signal. (It does **not** test page arithmetic; see §0.10 — under the identity page table there is none.) It is also where `mla_scores`' grid (`blockIdx.x`, `mla_scores.wgsl:50`) and its `T` uniform first have to agree over a range.

**Verify:** `npm run test:kernels:real`, error flat in `t`.

### Step 6 — Engine: allocation, buffers, uniforms, bind groups, dispatch
**File:** `src/zero-tvm/engine-core.ts`.

**6a. `allocKVPages` branches internally** (`:89-99`). When `spec.mla`: one buffer per attention layer of `spec.mlaCacheBytes` (37.1 MiB), labelled `mlaKV_${i}`, return type still `GPUBuffer[]`, `kvIndex` (`:328-332`) unchanged. Two static regions bound as `BindEntry` views (`:63`): latent `{offset: 0, size: maxContext*kvLoraRank*2}`, k_pe `{offset: mlaLatentBytes, size: maxContext*qkRopeHeadDim*2}`. Precedent: `:1017-1018`.

**6b. Buffers** (`:438-485`), all shared across layers — MLA is stateless per layer, exactly like the MoE scratch at `:535-545`:
`mlaQ [mlaQProjRows f16]`, `mlaKva [576 f16]`, `mlaQNope [heads*128 f16]`, `mlaQPe [heads*64 f16]`, `mlaQLat [heads*512 f16]`, `mlaScores [heads*maxContext f32]` (2.06 MiB), `mlaOLat [heads*512 f32]`, `mlaOLat16 [heads*512 f16]`.

`mlaScores` **must** be sized at `maxContext`, not `T` — bind groups are hoisted out of the hot loop (`:854-864`) and cannot be resized per token. Undersized, WGSL clamps the out-of-bounds store and the tail of every long conversation silently attends to zeros.

Reused unchanged: `residual`/`residual2`/`hidden1`/`hidden2`, `posMap`, `ropeFreqs` (`:633-637`, already built on the `!fused` path and already holding the yarn table), and **`attnOut` as the o_head buffer** — `S.qDim*2 = 4096 B` is exactly `heads*vHeadDim*2`. Do **not** reuse `B.qkvOut`.

**6c. Uniforms** (`:554-670`) + the scale branch:
```
SM_SCALE = S.mla
  ? ropeAttnScale(S) / Math.sqrt(S.mla.qkNopeHeadDim + S.mla.qkRopeHeadDim)
  : ropeAttnScale(S) / Math.sqrt(S.headDim)          // engine-core.ts:686
```
Left alone, that is 1/sqrt(128) instead of 1/sqrt(192) — a uniform **1.2247×** on every logit, finite, no NaN, no test failure outside the reference bundle. The Step-1 unit test against `meta.softmax_scale` is what makes it impossible to ship.

`mlaQU`/`mlaKvaU` are ordinary `{K_PACKED, SCALES_PER_ROW, N}` triples (QGROUP 64). `mlaProjKU = {N: kvLoraRank, K: qkNopeHeadDim}`, `mlaProjVU = {N: vHeadDim, K: kvLoraRank}`, `mlaScoresU = {L, R, T, scale f32}` (16 B, `T` at byte offset 8), `mlaCombineU = {L, T}` (`T` at offset 4).

In `writeStepState` (`:1367-1395`), patch both `T` next to the existing `nnzPages` write at `:1377-1380`, reusing one `Uint32Array` scratch. `T = position + 1` — confirmed by the bundle (`query_at 5`, `tokens 6`) and by `make-dsv2-layer-ref.py:218` (`pos = arange(T)`).

**6d. Bind groups** (`:967-1164`). Add `const isMla = !!S.mla` and an `else if (isMla)` branch **after** `if (isGdn)` (`:1010`) and before `hybrid` — so the branch order is `isGdn → isMla → hybrid → !fused → int8 → fused`. Add `mla?: {...}` to `LayerBG` (`:920-955`). Every matmul goes through `withBias` (`:871-875`) — DeepSeek is affine, and a mismatch is a loud "5 entries, expected 6".

The dangerous default this branch prevents: an MLA spec is not hybrid and not fused, so without it, `:1044` builds the ordinary qkv/rope/kv_append/attention chain with q_proj bound as `qkvWeights` — a bind group that validates and a forward pass that produces tokens.

**6e. `recordForward(enc, position)`** (`:1205`, callers `:1442, :1475, :1957, :2153`). Ten dispatches per MLA layer, then fall through to the shared `addNorm1 → FFN/MoE → addNorm2` tail (`:1289-1329`) with no change:

| # | pipeline | grid | label |
|---|---|---|---|
| 1 | `R.matmul` | `mlaQProjRows / R.matmulRowsPerWG` (3072/4 = 768) | `mlaQProj` |
| 2 | `R.matmul` | `mlaKvaRows / R.matmulRowsPerWG` (576/4 = 144) | `mlaKvaProj` |
| 3 | `P.mlaQSplit` | `S.heads` (16) | `mlaQSplit` |
| 4 | `P.mlaKvWrite` | 1 | `mlaKvWrite` |
| 5 | `P.mlaProj` (K uniform) | `ceil(kvLora/64)=8` × y=`heads` | `mlaQLat` |
| 6 | `P.mlaScores` | `position + 1` × y=`heads` | `mlaScores` |
| 7 | `P.mlaCombine` | `S.heads` | `mlaCombine` |
| 8 | `P.mlaNarrow` | `ceil(heads*kvLora/256)` = 32 | `mlaNarrow` |
| 9 | `P.mlaProj` (V uniform) | `ceil(vHeadDim/64)=2` × y=`heads` | `mlaOHead` |
| 10 | `R.matmulOProj` | `S.d / R.matmulRowsPerWG` (2048/4 = 512) | `oproj` |

All row counts divide 1/4/8 exactly, so no `Math.ceil` is load-bearing. Grid **y is the head count** on 5, 6 and 9 — swapping x and y on `mla_scores` still runs, still fills the buffer, and computes scores for the wrong `(t,h)` pairs. Dispatch 6 is the only position-dependent grid in the whole recorder; a static `maxContext × heads` grid is *correct* (the kernel guards `t >= T` at `mla_scores.wgsl:52`) and launches ~540k workgroups per layer per token at full context.

No zeroing of `mlaScores` or the cache is needed and none should be added: both kernels read only `t < T`, and stale bytes past `T` are never touched. Prefix reuse needs nothing new and must not be given anything — `computeReuseStart`'s non-hybrid branch (`:836`) relies on cache writes being idempotent, and rewriting an MLA latent slot with the same latent is equally idempotent.

Also bump `profileStep`'s `CAPACITY` (`:2137`, `2 * (S.layers * 14 + 8)`): an MLA+MoE layer is ~19 dispatches, and past capacity `dispatch()` (`:1181`) silently omits the timestamp writes, so the profile truncates rather than erroring.

**Verify — end to end, with no 9 GB checkpoint and no MoE:**

Build a test-only **3-layer** spec (`mixed-ffn-stack.test.ts:15-27` pattern) with `mla` set, `moe` **unset**, `ffn: 10944`, `maxPages: 4`. Hand-assemble a `LoadedWeights` whose `layers[1]` holds the bundle's layer-0 buffers (via the Step-2 loader), and run `buildDecodeEngine(..., { layerRange: { start: 1, end: 2 } })`.

Three layers, not two, and this is the load-bearing detail: with `layers: 2` and range `{1,2}`, `L1 === S.layers` and the stage builds `bgLmHead`/`bgArgmax`/`samplerU` (`:899-919`) and needs lm_head weights the bundle does not have. With `layers: 3`: `L0 = 1 ≠ 0` → `bgEmbedding` is null (`:887`); `L1 = 2 ≠ 3` → no LM head, no sampler; `bgInitNorm` binds `weights.layers[1].normGamma1` (`:894`); `nextGamma` falls to `lw.normGamma2` (`:976-978`) and never touches `layers[2]`. `pipelineStep` (`:1424-1460`) is then exactly residual-in / residual-out.

Feed `ref_x[t]` for `t = 0..19` (Step 5's regenerated bundle) and compare each returned residual against `ref_out`. That exercises `allocKVPages`' branch, `kvIndex`, the hoisted bind groups, the position threading, the real `SM_SCALE`, the dense FFN at 10944 and both residuals — the entire engine integration, with MLA as the only new variable.

### Step 7 — What must wait for the `dsv2moe` bundle
Only `dsv2layer0` and `dsv2mla` exist under `.weights-local/kernel-refs/`. `real-weights.mjs:970-975` has no `dsv2_moe_layer` key, so the bundle currently prints `SKIP unknown kernel` — which `make-dsv2-layer-ref.py:436-441` says is the correct state. Everything here is blocked on it and must be sequenced last:

- **Mixed dense/MoE FFN in the engine.** `ffnGateUp` is allocated only when `!S.moe` (`:552`) and a MoE spec gets `ffn`/`ffnDown` bind groups `undefined` (`:1153-1160`), so DeepSeek's dense layer 0 would dispatch with an undefined bind group. `S.ffnWidthAt` (`model-spec.ts:426`) exists and is referenced nowhere in `engine-core.ts` — exactly the gap `constraints.ts:360-363` names.
- **The four DeepSeek-MoE differences**, none of which raise an error (`make-dsv2-layer-ref.py:293-317`): unquantized f16 router (`moeRouterLogitsF16` already exists, `compiler.ts:311`); shared expert at 2× a routed expert's width (2816 vs 1408) and therefore **not stackable as index E** — `planLayer:213-218` would concatenate it onto the stack and produce a buffer that is the right total size for nothing; no shared gate at all; `norm_topk_prob: false` with a `routed_scaling_factor`, where renormalising rescales the whole routed branch by `1/mass` (the script prints that factor at `:385-387` so the mistake has a size).
- **The record spelling.** `planLayer:216,224` spells `mlp.shared_expert.*` / `mlp.shared_expert_gate` (Qwen); DeepSeek uses `mlp.shared_experts.*` (`make-dsv2-layer-ref.py:376`). `add-model.mjs:259` already probes both, `planLayer` does not.
- The `dsv2_moe_layer` handler in `real-weights.mjs`, verified with the same `layerRange` single-stage harness from Step 6 against `ref_router_logits` / `ref_topk_ids` / `ref_expert_y` / `ref_shared_out` / `ref_ffn_out`.

### Step 8 — What must wait for the full checkpoint (~9 GB)
- The `mlx-repack.mjs` **loader-replay** and **model-budget** cases (`:218-317`) go through `openMlxCheckpoint` over real shards. The `buildBuffer` half of Step 2 does *not* need them; the replay half does.
- `scripts/mlx-ref.py` + `scripts/validate-model.mjs`: argmax match, logits cosine ≥ 0.999, greedy token-exact for 8+ tokens. Every check before this runs at ONE layer against a reference this repo wrote; the cross-layer failures — wrong per-layer cache indexing, a residual right at layer 1 and drifted by layer 26, context truncating past the first pages — have no signature until the full stack runs.
- `scripts/pipeline-split-test.mjs`: a split moves the latent-cache boundary, the one thing Steps 1-7 verify only within a single stage. Note `share.ts:193` and `:577` pass **no range** to `allocKVPages`, so each stage allocates the whole model's cache (1002 MiB) and uses its slice — pre-existing (already true for Phi-3 at 1.5 GB/stage), and CLAUDE.md's "each stage keeps the KV cache of its own layers" describes *ownership*, not allocation. It is a correctness no-op but it makes the "two machines hold a model neither can" claim quantitatively wrong for the one model where the cache is the thing being economised. Fix it as an optional third `range?` parameter to `allocKVPages` (defaulting to the whole model, so `chat.ts`/`loading-ui.ts`/`lib` need no edit), not before.
- **Only after all of the above**: lift `constraints.ts:339-347`, move MLA out of `SUPPORT_MATRIX:438`'s `not:`, regenerate `docs/COMPAT.md`, add the registry row (`model-registry.ts` markers; `chatTemplateId: 'deepseek'` already renders — `model-select.ts:63-68`, `tests/unit/chat-template-deepseek.test.ts`). Flipping the rule green before then turns `add-model` green for every DeepSeek-shaped checkpoint, which is exactly what the refusal at `constraints.ts:323-337` was written to prevent.

---

## Part 2 — Hazards that survive this plan

1. **`mla_scores`' grid ceiling.** `blockIdx.x` is used raw with no z-fold, unlike the LM head at `:1341-1347`. `maxContext` must stay under `maxComputeWorkgroupsPerDimension` (65535). At 33,792 that is slack, not a bug — but it silently constrains any future `maxPages` raise, and it is undocumented in the kernel.
2. **`mla_combine` is one workgroup per head looping all `T` inside it** (`mla_combine.wgsl:87-93`). At 33k context that is a 33,792-iteration serial loop per head across 16 workgroups. Correctness is unaffected; the "7× less cache" headline will not translate into usable long context without the split-K treatment the repo already demonstrates (`attention_splitk` + `attention_combine`, `compiler.ts:283-285`).
3. **`compile()` builds every pipeline for every spec** (`compiler.ts:238-316`), so `kv_append.wgsl` and `attention.wgsl` compile under an MLA spec with `KV_PAGE_STRIDE`/`V_PAGE_OFFSET`/`HEAD_PAGE_STRIDE` (`shader-prelude.ts:50-52`) describing a cache that does not exist. Harmless only because they are never dispatched. Overloading those derived fields to carry MLA meanings would make them harmful — which is why Step 1 adds fields beside them.
4. **`add-model.mjs:346` computes `maxPages` from the MHA formula** and emits no `mla:` block. An MLA spec generated today would get a ~7× too-small page budget and no MLA dims. Both need branches whenever the constraint is lifted (Step 8).
5. **The OPFS cache is keyed by plan name.** Step 2d's suffix closes it for new op-bearing plans, but only if it lands *the day the permutation convention is written down*. Ship the permutation, then correct it, and every warm profile serves the stale bytes forever with no way to notice — and `peer-weights.ts` replicates them to the room.
6. **`decodeToken`'s overflow message** (`:1466-1469`) reports MB-per-page-block from `S.kvPageStride`, which is meaningless for an MLA spec. Cosmetic, but it is user-facing text that will be wrong by ~7×.
