I have verified every load-bearing claim against source. Here is the reconciled plan.

---

# PAGED KV + PREFIX POOL — one ordered plan

## PART 0 — Fact check of the four angles

### Established facts from the brief: all five hold

| Brief claim | Verdict | Evidence |
|---|---|---|
| Page table is the identity, written once, never permuted | **TRUE** | `engine-core.ts:935-940` fills `pageVals[i]=i` for `S.maxPages`. `writeStepState` (`:1532-1546`) rewrites only `pageIndptr` as `[0, nnzPages]` (`:1539`). No other writer to `B.pageValues` exists. |
| `kv_append` writes `page_no*KV_PAGE_STRIDE + head*HEAD_PAGE_STRIDE + slot*HEAD_DIM + dim`; `attention` reads through the table | **TRUE** | `kv_append.wgsl:55-59`; `attention.wgsl:70-72, 81-88`. |
| A prefix-reuse path already exists for consecutive turns | **TRUE** | `computeReuseStart` `engine-core.ts:919-929`, `getLastPrefill` `:1907`, call site `:2252`. |
| MLA bypasses paging (`position*L` flat) | **TRUE** | `mla_kv_write.wgsl:83-84, 98-100`; comment `:28-33` names the pool as the thing that would undo it; `mla_scores.wgsl:58-59`; `MLA_ENGINE_PLAN.md §0.3`. |
| `peer-weights.ts` replicates an OPFS directory | **TRUE** | `peer-weights.ts:1-38`; sender-computed SHA caveat at `:22-28`. |
| Spec table in `kv-budget.test.ts` | **TRUE** | `tests/unit/kv-budget.test.ts:31-43`, recomputed below. |

### Corrections — claims marked [FALSE], wrong version kept

**[FALSE-1] — structure, step 1(a).** *"Extend the same permutation to `attention_sg`, `attention_splitk`, `attention_splitk_sg` and `attention_int8` so all six readers are pinned under a non-identity table. This is confirmation, not new work."*

Those four are **already pinned, in two independent suites**. Registrations: `run.mjs:1211` (`attention`), `:1215` (`attention_sg`), `:1250-1252` (`attention_splitk` ×3), `:1255` (`attention_splitk_sg`); `compile-qwen.mjs:912` (`attention`), `:913-914` (`splitk`), `:918` (`attention_int8`), `:945` (`attention_sg`), `:946` (`attn_splitk_sg`). The fixtures shuffle: `run.mjs:711` `pageOrder=[2,0,1]`, `run.mjs:882` and `compile-qwen.mjs:490, 767` a rotation. There is **zero** work here. The one real gap is `attention_prefill`, whose only test uses `NUM_PAGES = 1` and `pageVals = new Int32Array([0])` (`compile-qwen35.mjs:1021, 1034`) — identity by construction, one page, no boundary crossing. The `kernels`, `reuse` and `proof` angles all got this right; `structure` did not. This matters because `structure` budgets an afternoon for work that is 90% already done and misses the one kernel that is genuinely unproven.

**[FALSE-2] — kernels, hazard list.** *"`nnz_pages` and `max_num_pages` are dead uniform fields."* — This one is **TRUE** and I am recording it as verified, not false: `rg` over all five decode attention shaders finds them only in the `PODArgs` struct declarations (`attention.wgsl:33-34`, `attention_sg.wgsl:28-29`, `attention_splitk.wgsl:40-41`, `attention_splitk_sg.wgsl:27-28`, `attention_int8.wgsl:31-32`) and at no use site. `writeStepState:1544-1546` writes `nnz_pages` into `attnU` **and** `attnSkU` every single token for nothing. Two wasted `writeBuffer` calls per token. Decide before adding a third.

**[FALSE-3] — kernels, step 8 + hazard.** *"The int8 path pages TWO structures … they must be allocated, freed and copy-on-written together **or the two go out of sync**."* The out-of-sync failure cannot happen through the page table: `attention_int8.wgsl:94` resolves `page_no` **once** and derives both `kv_word_base` (`:99-102`) and `scale_base` (`:107-110`) from it; the writer does the same (`kv_quantize_int8.wgsl:98`, scale at `:102-105`, word at `:143-149`). One `page_no`, two strides. The real constraint is narrower and still worth stating: a COW or a persist must copy **both buffers** for the same physical page, because they are two GPU allocations. That is a bookkeeping requirement, not a synchronization hazard.

**[FALSE-4] — structure, step 1 + hazard.** *"Qwen3-4B needs 160 MiB/layer to hold one 20k prefix plus a 20k live sequence — over the floor."* Arithmetically ~right (2500 pages × 64 KiB = 156.25 MiB) but **irrelevant**: `QWEN3_4B.maxContext` is 7168 (`maxPages: 448`, `model-spec.ts:712`; `maxContext = maxPages*pageSize`, `:540`), and `generatePipelined` refuses a 20k prompt outright at `engine-core.ts:2243-2246`. Qwen3-4B cannot run this workload at any allocation. Using it to argue that the pool needs split per-layer bindings is arguing from a configuration that does not exist.

**[FALSE-5] — reuse, hazard 1.** *"`attention_prefill.wgsl:68` … hardcodes that logical page i is at table slot i. A permuted table therefore breaks chunked prefill only … into fluent wrong output."* Half right. It does read `page_table_values[page_idx]` with no `indptr` and no `page_values_elem_offset` (`attention_prefill.wgsl:23, 27-31, 68`), and `engine-core.ts:2087` binds `B.pageValues` to it directly. But that is exactly right for a **prefix**, which always starts at logical page 0 — the kernel is table-correct, it just cannot express a window that starts elsewhere. It breaks only under eviction-from-the-front or a second sequence's table living at a nonzero offset in the same buffer. The header comment at `:13-16` already says so. Precondition, not bug — but it must be written into the kernel header as a **precondition**, not left as prose about the current engine.

**[FALSE-6] — reuse, step 5 + proof, implicitly.** The claim that the resident-vs-pool ordering is the crux assumes a pool will be consulted on the hot path at all in v1. Under the plan below there is exactly one resident sequence until Phase 3, so the ordering question does not arise until then. Keeping it as a Phase-1 concern buys complexity for nothing.

### Contradictions between angles, resolved

**C1. How do the writers learn the table — bind it (A), or a CPU-computed physical slot map (B)?**
`kernels` argues (B) hard: `position_map` already serves two purposes (RoPE frequency at `qkv_fused.wgsl:170-171`, cache address at `:156-157`; same split in `qk_norm_rope_append.wgsl:62` vs `:126`), so a second region costs no new binding and no new instruction. `proof` argues (A), and separately kills the *uniform* variant of the shortcut by noting `CHUNK_CAP = 64` (`engine-core.ts:1952`) spans up to 5 pages in one `kv_append` dispatch — which is correct and rules out a single page number in a uniform, but does **not** rule out (B)-as-array.

**Pick (A): bind `B.pageValues` to all eight writers.** Reason from the files, and it is decisive: the entire verification strategy is a falsifier that permutes `B.pageValues` and demands bit-identical logits. Under (B), permuting `B.pageValues` alone changes nothing on the write side — the test would have to permute the table *and* the derived slot map, which means it tests the CPU derivation rather than the kernels, and a bug in the derivation is invisible to both halves. (A) has exactly one representation of the mapping, in the same buffer the readers already use. `kernels`' instruction-count argument is real but small; `structure`'s "prefer (B) for decode-path writers, (A) for `kv_append`" is the worst of both — two mechanisms, and the decode path is precisely where a divergence between them is untestable.

Take one thing from `proof`: **append the new binding at the END of each kernel's list.** `bg()` (`engine-core.ts:65-73`) maps array position → binding index sequentially, so a bind-group site that is missed passes N buffers to a pipeline declaring N+1 and fails loudly with a WebGPU validation error, instead of silently shifting every buffer by one.

**C2. Is MLA excluded (structure, proof) or taught the table (kernels) or free (reuse)?**
They are answering different questions and all three are right about their own.
- **Restore/persist for MLA is free today.** `cache_c` rows are `position*L` contiguous (`mla_kv_write.wgsl:84`), so persisting k blocks is one contiguous `writeBuffer` per layer per region. Zero kernel changes. `reuse` is correct.
- **Aliasing for MLA needs three kernels.** `mla_kv_write.wgsl:84, 98`, `mla_scores.wgsl:58-59`, `mla_combine.wgsl:87-91`. `kernels` is correct that it is cheap for `mla_scores` (one i32 load per (t,head) workgroup against a 576-value reduction) and awkward for `mla_combine` (its `t` loop is **innermost**, `:89`, inside the strided `i` loop at `:87`).
- **Excluding it from v1 is right anyway**, but for a reason neither `structure` nor `proof` states: `DEEPSEEK_V2_LITE` is not in `SHIPPED_MODELS` (`model-registry.ts:38-52` — verified, it is absent), so there is no `?model=` URL that reaches it and no user-facing regression from deferring.

**C3. Where does the pool live — VRAM (structure) or OPFS (kernels step 10, reuse)?**
`structure` builds a resident multi-sequence VRAM pool as the primary artifact. `kernels` step 10 and `reuse` both argue OPFS-first on arithmetic. **The arithmetic decides for OPFS-first**, and it is sharper than either stated: Qwen3.5's per-layer KV buffer is **exactly 128.0 MiB** (2048 pages × 65,536 B — I recomputed it), which is both the WebGPU portability floor and the current allocation. A resident pool holding two 20k sequences needs 2500 page-columns = 156.25 MiB/layer. It would *run* (`loading-ui.ts:191-194` and `lib/index.ts:183-186` already raise `maxStorageBufferBindingSize` to the adapter's own value) but it leaves the portable envelope for the one benefit — interleaved conversations — that is third on the agentic priority list. OPFS costs nothing portable and buys cold restart, which is first.

**C4. Does the pool matter for the stated workload at all?**
`proof` says the straight-line loop is **already solved** and any big number measured against `?reuse=0` is a strawman. This is the single most important reconciliation and `proof` is right. Verified: `generatePipelined` calls `computeReuseStart` at `:2252` and prefills only `[startPos, len)`. BENCH.md Prefill round A (`:865-940`) measured Qwen3.5 turn-3 TTFT at **14,340 ms → 194 ms, 922/950 reused, 74×**. The `20k → 50 out → tool → 20.5k` loop is precisely what that path handles.

What is left, ranked by the agentic workload:
1. **Cold restart.** Reload/crash/model-switch drops everything — `absorbed` is closure state (`:897`) and a model switch is a full page reload by design (`model-switcher.ts:6-11`). Today that costs a fresh 20k prefill.
2. **Branching.** `noteAbsorbed:903-904` sets `absorbed.length = position` on any rewind — regenerate, edit-and-resend, or an agent exploring two tool paths **destroys** the discarded branch irrecoverably. This is agentic-specific and nobody's plan led with it.
3. **Interleaved conversations.** Agent + sub-agent, or two tabs. Each switch is a full re-prefill today.
4. Cross-device.

**C5. Which spec?** All four angles circle this; none commits. Committing:

| Spec | maxContext | KiB/token | per-layer buffer | page-column (all layers) | 20k prefix |
|---|---:|---:|---:|---:|---|
| PHI3 | 4,112 | 384.00 | 48.2 MiB | 6.00 MiB | **impossible** |
| QWEN3_4B / QWEN3_4B_MLX | 7,168 | 144.00 | 28.0 MiB | 2.25 MiB | **impossible** |
| **QWEN35_4B** | **32,768** | **32.00** | **128.0 MiB** | **512 KiB** | **625 MiB — fits** |
| QWEN36_35B_A3B(_Q3) | 6,144 | 20.00 | 12.0 MiB | 320 KiB | **impossible** |
| QWEN3_30B_A3B_4BIT | 10,240 | — | — | — | impossible |
| LLAMA_3_2_1B | 32,768 | — | — | — | fits (1B — not an agentic coder) |
| DEEPSEEK_V2_LITE | 33,792 | 30.38 | 37.1 MiB | 486 KiB | 593 MiB — fits, **not shipped** |

(Computed by importing `model-spec.ts` directly, not from the test table.)

**Exactly one shipped, validated, credible agentic-coding spec can hold a 20k prefix: Qwen3.5-4B.** It is also the only spec with chunked prefill (`engine-core.ts:2173-2174` gates on `hybrid && !S.moe && !AFFINE && !partial`), the only one with a measured long-prompt prefill rate (202 tok/s, BENCH.md), the cheapest per token by 4.5×, and the one whose 24 GDN layers hold a recurrent state. **Build for Qwen3.5. Everything else is a bonus or a bystander.** Note the irony and state it in the commit: the hybrid is simultaneously the only viable target and the only one with an unpageable state.

Consequence: `structure`'s step 10 (port chunked prefill to attention-only specs before measuring) drops out of the critical path entirely. The target spec already has it. Keep the port as an independent BENCH item.

---

## PART 1 — The design, in one paragraph each

**The constraint that shapes everything.** K is RoPE'd *before* the cache write on every path — `qkv_fused.wgsl:170-177` computes `rot_lo`/`rot_hi` and writes them at `:186-191`; same in `qk_norm_rope_append.wgsl:126`, in `rope.wgsl` → `kv_append`, and in `mla_kv_write.wgsl:88-99` for `k_pe`. Additionally `gdn_conv.wgsl:57-60` phases its ring by `(pos + j) % RING`. **A cached page is valid only at the absolute positions it was written at.** This is a *prefix* pool, never a vLLM-style relocatable block pool. No shared middle segments, no "cache this tool-definition block wherever it appears", no front eviction, no sliding window. All four angles say this; it deserves to be the first line of the design doc because the failure mode is fluent wrong output, not an error. (One escape hatch, noted and not built: MLA's 512-wide latent carries no position — only the 64-wide `cache_kpe` is rotated, `mla_kv_write.wgsl:88-99`. It is the one spec where relocation is arguable.)

**The safety invariant, stated once.** `KV[0..n) = f(tokens[0..n), spec, resolved variant set, KV dtype)`. Anything the pool keys on must be inside that tuple. Anything outside is a cross-conversation leak. `share.ts` already depends on this — one host engine serves many guests through one `absorbed` record, and it is safe only because the reuse rule is literal token equality.

**Tiers.** Tier 0 = GPU page-columns, per-tab, dies on reload. Tier 1 = the index (entry id, chain hashes, token ids, length, fingerprint, lastUsed, byte location) — tens of KB, wants ordered queries and transactions, so IndexedDB; this would be the repo's first IDB use and it is the right first one. Tier 2 = the bytes, OPFS, gated per-spec. Cross-*tab* sharing is tier 2 only: `GPUBuffer`s are per-device and not transferable, full stop.

**Matching.** Chain hash over `pageSize`-token blocks: `h_0 = H(fingerprint)`, `h_i = H(h_{i-1} ‖ ids[16(i-1)..16i))`. Walk `h_1, h_2, …` until a miss. **128-bit, and the entry stores its token ids and they are compared exactly on attach** — the hash is the index, the ids are the proof, so a collision degrades to a miss rather than serving a stranger's KV inside the same origin. The parent chaining is not optional: without it a block whose 16 ids are `", \"path\": \"src/"` — utterly common in tool-call transcripts — matches across two unrelated conversations at the same offset, the id compare passes, and the graft is silently wrong because V encodes a different attention history. `reuse` is right that this is the one design error that produces a wrong model rather than a slow one.

**Fingerprint.** `spec.id`; weight revision; the **resolved** variant set (`variants.ts:47-75` — `matmul`, `vec4`, `vec4Half`, `vec4Qkv`, `subgroups`/`sgQkv`, `qkvTile`, `qkvTile2`, `fuseQkNorm`, `sgFfn`, `fusePrologue`; these select among five `qkv_fused` siblings at `variants.ts:296-300` and differ in f16 reduction order); `int8KV` (a different *layout*, `allocKVPagesInt8:116-129`); `layerRange`; **the prefill path that wrote it** (`bench-console.ts:388-395` documents that chunked and per-token are not bit-equal); and `adapter.info` vendor/architecture. Excluded, deliberately and with a comment: `splitK` (reorders the attention *output* reduction, touches nothing written to the cache) and **all sampling parameters** — the sampler's counter is already the position (`engine-core.ts:1560-1567`), K/V at position p is a function of `tokens[0..p]` and the weights alone, and adding temperature "to be safe" kills all reuse in any chat with a temperature slider.

**Hybrid entries.** A Qwen3.5 entry = page-columns for the 8 attention layers **plus one fixed-size GDN blob**. Recomputed: `gdnQkvDim = 2·16·128 + 32·128 = 8192`, so conv = `3 × 8192 × 2 B = 48 KiB` (`engine-core.ts:552`) and recur = `32 × (128×128) × 4 B = 2.00 MiB` (`:553`, `gdnStatePerHead` at `model-spec.ts:526`), × 24 GDN layers = **49.15 MiB**, constant whether the prefix is 200 or 20,000 tokens. Against 625 MiB of KV for a 20k prefix that is **7%** — the hybrid is the *cheapest* spec to pool, not the hardest. The blob is snapshot and restored by plain `copyBufferToBuffer` over the 48 state buffers. One snapshot per entry, at the end of the prefix, exact-length attach only. The generalisation that refutes itself: `CHUNK_CAP = 64` (`:1952`), so snapshotting every chunk is 313 blobs × 49 MiB = 15 GiB for one 20k prefix.

---

## PART 2 — The ordered plan

Each step names its verification. Nothing depends on an unverified step.

### PHASE 0 — Measure and pin. No shipping code. (~2 days)

**0.1 — Extract `reuseStart` / `noteAbsorbed` as pure functions and pin them headlessly.**
`computeReuseStart` (`engine-core.ts:919-929`) is eleven lines closed over `absorbed`, `absorbedValid`, `gdnStatePos`, `prefixReuse`, `hybrid`. Its only assertion is `checkReuse()` (`bench-console.ts:395-425`), which needs a browser, a loaded model, 48 decoded tokens and `?chunk=0`. Extract `reuseStart(state, promptIds): number` and `noteAbsorbed(state, position, id)`; have the closures call them. Pin as a table hand-derived **from the source, not from running the new code**: pure attention → `min(lcp, len-1)`; hybrid → `lcp` only when `gdnStatePos === absorbed.length && lcp === absorbed.length && lcp <= len-1`, else 0; empty prompt / `absorbedValid === false` / `prefixReuse === false` → 0. Pin `noteAbsorbed`'s two behaviours separately: rewind truncates (`:903-904`), a gap sets `absorbedValid = false` with no path back except `resetKVTracking()` (`:1911-1916`) — which `chat.ts` **never calls** (only `validate.ts` and `bench-console.ts:318, 354, 400` do).
*Verify:* new `tests/unit/prefix-reuse.test.ts` green against the hand-derived table; `npm run test` + `npm run typecheck` green; `checkReuse()` on `?model=qwen35&chunk=0` still prints `max|Δ| = 0`.
*Why first:* this is the regression net for every later step, and it costs hours.

**0.2 — THE GATE: is the prompt append-only in token ids on a real agentic transcript?**
`buildChatPromptFor` (`model-select.ts:53-76`) takes `ChatMessage[]` — role and content only — and every builder renders the whole transcript to one string and calls `tokenizer.encode` once. `chat.ts:305-308` already stores each assistant turn's exact emitted ids **with a comment saying re-encoding the rendered text is not guaranteed to reproduce them**, and `chat.ts:505` re-encodes anyway. `continueTurn` (`:538`) does use the stored ids; `runTurn` does not. BENCH.md's own numbers show the crack: Qwen3.5 ChatML reuses 922/950 cleanly, but Phi-3 reuses 1086/1105 with a documented boundary merge at `<|assistant|>\n`. Tool-call JSON, digit runs and whitespace are exactly where BPE re-segments.
Freeze a real transcript as a fixture. Real ones exist on this machine: `~/.pi/agent/sessions/--Users-ahmetbarisgunaydin-dev-zero-tvm--/*.jsonl` (pi driving Qwen3.6 against *this* repo) and `~/.claude/projects/**/*.jsonl`. Scrub paths, render through the Qwen3.5 ChatML template, freeze as ids plus per-turn prompt lengths.
*Verify:* a headless test asserting turn N+1's prompt id array has turn N's as an **exact prefix**, and reporting the position of the first divergence when it does not. Plus `encode(decode(ids)) === ids` over the assistant outputs, listing failures rather than asserting none.
*Why this is the gate:* if the id prefix breaks 3k tokens into a 20k transcript, the pool caps at 3k and **the entire feature is worth almost nothing** — and it will still pass every synthetic test. Today a break costs a full prefill and correct output, visible only as a small `reused` in the `[engine] prefill:` log (`:2274-2280`). If this test goes red, widening `buildChatPromptFor` to accept `{role, ids}` segments becomes a hard prerequisite and moves ahead of everything else. Also record here: `llama3DateString` (`tokenizer-bpe.ts:481-482`, used at `:523, 533`) injects today's date into the Llama-3 system block — any Llama-3 pooled prefix expires at midnight ~15 tokens in. Any clock, session id or nonce near the front of an agent system prompt has the same total, silent effect.

> #### RESULT, 2026-08-10 — **GREEN. The gate is passed.**
>
> `scripts/prefix-stability-test.mjs`, over **nine real agent sessions** from
> this machine (Claude Code JSONL, 8–147 MB each), rendered through the repo's
> own template builders and tokenizers:
>
> | | |
> |---|---|
> | transitions compared | **~340** |
> | breaks | **0** |
> | tool calls inside the compared windows | 200+ |
> | `decode(encode(x)) === x` on assistant turns | clean — 1122 turns in the largest |
> | templates | `chatml`, `llama3`, `deepseek` — all clean on the same transcript |
>
> Tool traffic is kept, not stripped: `tool_use` renders as
> `<tool_call>{json}</tool_call>` and `tool_result` as
> `<tool_response>…</tool_response>`, so the JSON punctuation runs, file paths
> and digit runs that would make BPE re-segment are all inside the compared
> text. The script prints the tool-call count for the window and warns when it
> is zero — a PASS over a transcript with the tool traffic stripped is a PASS
> over a plain chat, which was already known to work.
>
> **The falsifier ran first, and it caught a real defect in itself.** `--mutate`
> rewrites turn 1 from turn 3 onward, the way a clock in a system prompt would,
> and must turn the test red. Its first version used a regex that matched
> nothing, so the falsifier *passed* — the exact outcome §0.4 warns about, one
> section below, in this same document. Made unconditional it now reports
> `FAIL 1/5 transitions … first break at turn 3: shared 3 of 133 tokens` with
> the decoded text at the boundary, exits 1 where the control exits 0, and
> refuses to run at all when the mutation did not apply.
>
> **Three limits, and the first is the one that matters.**
>
> 1. **Phi-3 is untested, and it is the template already known to break** —
>    BENCH.md records 1086/1105 with a boundary merge at `<|assistant|>\n`. Its
>    template is SPM and lives in `tokenizer.ts`, which imports `weight-loader`
>    and reads `GPUBufferUsage` at module scope, so this script cannot reach it.
>    A pool keyed per-spec is unaffected; a blanket claim that "prompts are
>    append-only" is not.
> 2. The transcripts are Claude Code's, re-rendered into these templates. The
>    content is genuine agentic content; the exact byte sequence a pi/Qwen agent
>    would send is not identical.
> 3. Windows are capped (`--max`, default 24k) because the full files run to
>    millions of tokens and the comparison is O(n²) re-encodes. The cap is also
>    the workload — Qwen3.5's `maxContext` is 7168.
>
> **Consequence for the plan:** the `{role, ids}` rewrite of
> `buildChatPromptFor` named above as a hard prerequisite **is not needed** for
> byte-level-BPE specs. Phases 0–1 proceed as written.


**0.3 — Three measurements, written into BENCH.md before any threshold is written into code.**
(a) `adapter.limits.maxStorageBufferBindingSize` and `maxBufferSize` on the target machine. The machinery exists (`loading-ui.ts:191-194`) but nobody has recorded what it returns; the number decides whether Phase 3 can hold two sequences in the existing per-layer buffer.
(b) OPFS write and read throughput for 625 MiB through `createSyncAccessHandle` in a worker. Every "seconds vs 99 seconds" claim in all four angles rests on a guessed ~300 MB/s.
(c) GPU→CPU readback throughput for the same 625 MiB in 240 KB pieces. The engine has **never** read the KV cache back; this is the first time.
*Verify:* three numbers in BENCH.md with raw runs, not just medians (BENCH.md:153 records Qwen3.5 drifting downward *within* one invocation).

**0.4 — Build the falsifier and watch it FAIL on HEAD.**
`window.__pageTableCheck(ids, perm, variantFlags)` in `model-smoke.html`, following the existing `__splitCheck` / `__chainCheck` / `__wholeCheck` hooks (`model-smoke.html:52, 76, 123`). It needs one new seam: a `setPageTable(Int32Array)` on `DecodeEngine`, because `B` is closed over and nothing outside can reach `B.pageValues`. Run the same ids twice — identity vs permutation — and require **bit-identical** final-position logits (`max|Δ| === 0`, not a tolerance: same dispatches, same order, same values, only addresses differ). Permutations: identity (control), reversal, rotation by 1, and a shuffle mapping logical page 0 to a high physical page. Prompt ≥ 5 pages (≥80 tokens), a real one from `scripts/mlx-ref.py`. Run under both `['scalar', undefined]` and `['shipped', {sgAttn:true, splitK:8}]`, plus `?fused` and `?kv8=1` separately — those select **different write kernels**.
*Verify:* `node scripts/paging-test.mjs qwen35` prints a large nonzero `max|Δ|` for every non-identity permutation and exactly 0 for identity, **on today's HEAD**, under all four variant/mode combinations. Record that failure output in the commit message.
*Why:* `proof` is exactly right that a test written *after* the kernels change is indistinguishable from a correct implementation when it silently permutes nothing — both report 0. And note the reason nothing catches this today: `debugCompareReuse` (`:1879-1905`) runs both passes under the *same* table, so a permutation bug cancels; `pipeline-split-test.mjs` likewise; the `kv_append` kernel test binds no table because the kernel has none.

---

### PHASE 1 — Persistence. Zero kernel changes. (~1 week) **This is the useful subset.**

The insight that makes this a week rather than a month: **a prefix always restores to its own positions, so it needs no page table at all.** Under the identity table, restoring k blocks is writing the first `k × bytesPerPage` bytes of each layer buffer — one contiguous `writeBuffer` per layer. Everything in Phase 2 and 3 is about *aliasing*, which persistence does not need.

**1.1 — The fingerprint, as a pure function.**
Per PART 1. Print the active fingerprint once at boot so a surprising miss rate is diagnosable.
*Verify:* a test that enumerates every field of `VariantFlags` **reflectively** and asserts flipping each one changes the fingerprint — so adding a field to `VariantFlags` without adding it to the fingerprint fails CI rather than silently corrupting a cache. Plus: two prompts differing only in sampling parameters produce the **same** fingerprint.

**1.2 — The chain-hash index, in memory + IndexedDB.**
Blocks of `spec.pageSize` = 16 (every shipped spec, `model-spec.ts:652, 707, 779, 886, 1005, 1046, 1085, 1114, 1144`). Manifest fully in memory, loaded once at boot (~100 KB for a 20k conversation at 1250 records × ~80 B); only payloads touch storage.
*Verify (no GPU):* chain walk returns the exact page-aligned LCP for prefixes sharing 0 / 16 / 20,000 / 20,005 tokens; an injected forced collision is **rejected** by the id compare; divergence at and inside a block boundary; a tampered parent link is rejected; property test asserting `poolStart(ids) <= lcp(ids, stored)` always, rounded down to a block.

**1.3 — Save path: block-granular, append-only, off the critical path, able to decline.**
Save only the blocks the turn newly filled — in the agentic loop roughly `(tool result + reply)/16` ≈ 35 blocks ≈ 17 MiB on Qwen3.5, not 640 MiB. Stream in fixed pieces, reusing the 240 KB discipline from `peer-weights.ts` rather than allocating a whole-prefix staging buffer. Issue after `generatePipelined` returns, capped per idle slice. Publish a manifest record only after its payload write resolves, so a crash mid-write leaves an unreferenced payload and a miss, never a torn block. Wrap every tier-2 write in a `navigator.locks` lock keyed by entry id — `createWritable` is not exclusive across tabs. Call `navigator.storage.estimate()` first and `persist()`. Gate on resident-weight headroom: BENCH.md:31-33 records the 4-bit 35B MoE at 19.7 GB resident having its **GPU process killed mid-prefill**, and the 3-bit variant collapsing to 11.4 tok/s at 0.2 GB free. On those configurations the save path must decline and log why.
*Verify:* `bench()` decode tok/s with saving on vs off, within session drift; a crash-injection test (abort between payload write and manifest publish) produces a miss on reload, never a partial restore; a two-profile concurrency test (the `peer-weights-e2e.mjs` pattern) asserts exactly one write wins; a quota test asserts the write declines without headroom.

**1.4 — Restore path, per family, including the engine's own state.**
A restore of k blocks must leave the engine exactly where a prefill of `16k` tokens would.
- *Pure attention (Phi-3, Qwen3-4B, MLX dense):* write payloads into `kvPages[kvIndex[L]]` at `page_no * kvPageStride`; set `absorbed = ids[0..16k)`, `absorbedValid = true`. `writeStepState` derives `nnzPages`/`kv_len` from `position` as always (`:1533-1541`) — no uniform changes at all.
- *int8:* two payload regions per block, pages and scales, copied together (see [FALSE-3]).
- *MLA:* **free.** `cache_c` and `cache_kpe` are flat contiguous (`mla_kv_write.wgsl:84, 100`); block i is rows `[16i, 16i+16)`; one `writeBuffer` per layer per region. Ships without touching `mla_scores`/`mla_combine`. (Not user-reachable — `DEEPSEEK_V2_LITE` is absent from `SHIPPED_MODELS`, `model-registry.ts:38-52` — but it costs nothing and makes the MLA engine test a real gate.)
- *Hybrid (Qwen3.5):* KV pages **plus** the 49.15 MiB GDN blob, restored by `copyBufferToBuffer` over `gdnStateBufs`, **plus `gdnStatePos` set to the entry length and checked against the attach position**. Exact-length attach only. This inherits `computeReuseStart`'s existing hybrid rule (`:924-927`) rather than inventing one.
*The failure this step is designed against, stated because no assertion catches it:* restoring KV pages **and** setting `absorbed` **without** the GDN blob satisfies the `gdnStatePos === absorbed.length` guard by construction and then runs 24 GDN layers from a zeroed recurrence against a 16k-token attention history. Fluent, wrong, permanent for the conversation — `clearGdnState` only fires at position 0 (`:1577-1579`). The mirror error (restore KV, forget `absorbed`) is merely wasteful: the next turn re-prefills over pages that already hold the right bytes, so the pool appears to work exactly once.
*Verify:* `debugComparePool` — the direct extension of `debugCompareReuse` (`:1879-1905`, which already proves reuse is bit-exact with expected diff **exactly 0**). Prefill N tokens → persist → tear down the engine → rebuild → restore → `forwardLogits` equals a from-zero prefill of the same sequence, `max|Δ| === 0`, for each of the four families, with `?chunk=0` on hybrid. Plus: a truncated file, a flipped byte, and a fingerprint from a different variant set are each **rejected with a named reason**; a hybrid attach at `length != entry.length` is **refused**, not silently accepted.

**1.5 — Instrumentation, and keeping the benchmark honest.**
Widen `getLastPrefill()` (`:234, :1907-1909`, set at `:2274`) from `{promptLen, reused, chunks}` to also report `restored` and `residentReused` **separately** — otherwise a run that restored 1200 blocks and a run that had them resident report the identical `reused` and no one can tell the TTFT wins apart.
BENCH.md:304-345 records cross-turn reuse silently invalidating **two published A/B pairs** because `bench()` did not call `resetKVTracking()`. A persistent pool makes that strictly worse: it survives the reload, so even "fresh tab, fresh run" is contaminated. Every harness that calls `resetKVTracking()` (`bench-console.ts:318, 354, 400`; `validate.ts`) must additionally clear the pool namespace, and `bench()` must run with the pool **disabled by default**. And the trap runs the *other* way for a pool: a median-of-5 measures a warm pool four times. The A/B unit must be a whole-session replay with an explicit `clearPool()` between replays, distinct from `resetKVTracking()`.
*Verify:* `bench()` asserts `getLastPrefill().reused === 0 && restored === 0` on every run; a fresh-namespace run and a pool-cleared run produce identical TTFT.

**What Phase 1 buys, honestly.** Cold restart on Qwen3.5: a 20k prefix costs ~99 s to re-prefill at the measured 202 tok/s (BENCH.md:425-427) against ~2 s of OPFS read + upload (pending 0.3b). Survives reload, crash and model switch. It does **not** buy interleaved conversations, branching, or cross-device — those need Phase 2 and 3. It requires no kernel change, no allocator, no refcount, and no eviction policy. If the project ships only this, it is still the change that makes a 20k agentic prefix reachable in a fresh tab.

---

### PHASE 2 — The writers learn the table. (~3 days) Correctness first, enabling second.

**2.1 — Close `attention_prefill`'s two holes.**
Widen `compile-qwen35.mjs:1017-1071` from `NUM_PAGES = 1 / pageVals = [0] / T = 8` to `BASE = 20, SEQ = 12` (T = 32, three pages) under a permuted `pageOrder`, still asserting **bit-exact** equality against the per-token `attention.wgsl` reference the test already builds at `:1039-1058` (that reference loop is table-aware, so the two halves diverge if either mishandles the permutation). Separately, write the single-table / logical-page-0 precondition into the kernel header (`attention_prefill.wgsl:13-16`) as a precondition rather than a description of the current engine — it has no `page_values_elem_offset` (`:27-31`) and `engine-core.ts:2087` binds `B.pageValues` raw.
This also closes a gap that exists **today**, independent of paging: the engine runs 64-token chunks crossing 4-5 page boundaries on every real prompt, and no test pins that.
*Verify:* `npm run test:kernels:qwen35` is 19/19 with the widened case, and the new case goes **red** when reverted to a single identity page.

**2.2 — Bind `page_table_values` to the eight writers, twelve sites.**
`kv_append.wgsl:55`; `qk_norm_rope_append.wgsl:63`; `kv_quantize_int8.wgsl:98`; `qkv_fused.wgsl:157, 186`; `qkv_fused_sg.wgsl:130, 153`; `qkv_fused_sg_vec4.wgsl:137, 160`; `qkv_fused_tiled_sg.wgsl:137`; `qkv_fused_tiled2sg.wgsl:150`. Replace `position / PAGE_SIZE` with `page_table_values[position / PAGE_SIZE]`, binding **the same buffer the readers bind** — never a copy. Append at the end of each binding list (see C1). Bind-group sites: `engine-core.ts:1179, 1194, 1199, 1207, 1210, 1219`, plus the chunked path at `:2086`. `qkv_fused_scratch.wgsl` is untouched — verified, it has no `KV_PAGE_STRIDE` reference at all; it writes `kSlot`/`vSlot` and `kv_quantize_int8` pages that path.
*The failure mode is a missed variant.* `?sg=0`, `?qkvtile=1`, `?qkvtile2`, `?vec4qkv=0`, `?kv8=1` each select a different writer (`variants.ts:296-300`; note `vec4Qkv` additionally requires `spec.d % 1024 === 0`, so Qwen3.5's d=2560 never reaches it and Phi-3's d=3072 does). A miss ships green under defaults and produces fluent wrong output only under a URL flag CI never sets.
*Verify, in this order:* (i) with the table still the identity, `npm run test:kernels`, `test:kernels:qwen` (21/21), `test:kernels:qwen35` (19/19), `npm run test`, `npm run test:e2e` are **byte-identical to before** — identity is a special case of the new code, so any red is a regression, not a rebaseline. (ii) Step 0.4's falsifier goes **FAIL → PASS**: `max|Δ| = 0` for every permutation, under all four variant/mode combinations.

**2.3 — The hot-path control.**
2.2 adds an indirect load to `qkv_fused*`, which runs per layer per token on **every** decode step of every workload, including every published tok/s number. Measure steady-state decode tok/s with the change against HEAD, same session, interleaved halves, and report the run-to-run spread alongside the medians — the protocol BENCH.md:302-356 arrived at. Prediction is ~0% (one i32 from a tiny hot buffer), but BENCH.md's negative-results section is full of "should be free".
*Verify:* an A/B table in BENCH.md; regression must be inside measured session drift or 2.2 is reconsidered.

**2.4 — External oracle.**
Every test so far compares this engine against itself. A systematic off-by-one that write and read *agree* on passes all of them. Run `scripts/validate-model.mjs` against `scripts/mlx-ref.py`'s dump after the change — the repo's strongest gate (CLAUDE.md records the 2026-08-06 proof run: cosine 0.999879, greedy token-exact). Use `pipeline-split-test.mjs`'s hard-won rules: **300 tokens, not 8** (it forked at token 248 under shipped variants) and its `TIE = 0.05` top-2-gap rule to separate a real divergence from a coin-flip argmax.
*Verify:* greedy tokens match mlx_lm for 300 tokens, or diverge only where the top-2 gap is under 0.05.

---

### PHASE 3 — The resident allocator. (~2-3 weeks) Interleaved + branching.

Only start this once Phase 1 is measured and 2.2 is green. Everything here is aliasing.

**3.1 — Split pool capacity from per-sequence context.**
`maxContext = maxPages * pageSize` (`model-spec.ts:540`) and `MAX_CONTEXT` (`engine-core.ts:778`) currently mean both "pages the buffer holds" and "longest sequence". Introduce `poolPages >= maxPages`; keep `maxContext = maxPages * pageSize` as the **sequence** ceiling; re-derive the three guards that quote the identity in their error text (`:1602-1603`, `:1638-1641`, `:2243-2246`), which become lies the moment the two decouple. Move `kv-budget.test.ts:59`'s `expect(spec.maxPages * spec.pageSize).toBe(spec.maxContext)` to assert the two numbers separately, and price the pool: `poolPages * pageSize * kvBytesPerToken` against the same ~1 GiB budget (`:66-71`).
*Note this step is deliberately absent from Phase 1* — with one resident sequence the identity still holds and the split is speculative.
*Risk if skipped:* raising `maxPages` without re-deriving `MAX_CONTEXT` silently raises the context ceiling past the trained window and past the VRAM budget; on the MoE (19.7 GB resident, `model-spec.ts:866-873`) it kills the tab mid-prefill.
*Verify:* `kv-budget.test.ts` green with the two ceilings asserted independently for all specs; a prompt one token past the per-sequence ceiling still throws with the right numbers.

**3.2 — `src/zero-tvm/kv-pool.ts`: the allocator. Pure TypeScript, no `GPUDevice`.**
Unit of allocation is a **page-column**: physical index `p` meaning slab `p` in every layer's buffer at once. That is what lets ONE `pageValues` buffer of `maxPages` i32 (`engine-core.ts:476`) serve all layers, so no kernel gains a binding and no bind group is rebuilt. It costs the freedom to evict one layer's page alone — freedom nobody wants, since evicting a prefix's page must evict it in every layer anyway.
State: `free` stack, `cold` stack (owned by an entry at zero live refs), `liveRefs: Int32Array(maxPages)`, `pooled: Uint8Array(maxPages)`, `chainHash: BigUint64Array(maxPages*2)`. Under 40 KB for the largest spec. `alloc()` pops `free`, else reclaims from `cold`, else **returns null** — a typed failure, never a throw from inside a command-encoder build; the caller falls back to the current code path unchanged. A pool that can wedge the engine is worse than no pool.
**Keep `liveRefs` and `pooled` as two separate arrays.** Collapsing them is the information-disclosure bug: both decrement to zero identically, the allocator hands out a column an entry still names, the entry's chain hash still validates on attach, and the next turn reads a stranger's context as its own prefix. Same origin, no error.
*Verify:* seeded property test (the `rng(seed)` mulberry32 at `run.mjs:43` is the house pattern) over ≥10k random `alloc/retain/release/pin/evict` sequences asserting after **every** op: every column is in exactly one of {free, cold, referenced}; refcounts sum to live entries; `evict` never frees a pinned or live column; double-release throws; teardown returns `free.length === maxPages`. Deleting the pinned-page check must go red within a few hundred ops.

**3.3 — Per-sequence page table, written at allocation.**
Replace the one-time identity fill (`:935-940`) with the allocator's ordered `columns[]`. `writeStepState` already writes `pageIndptr = [0, nnzPages]` every token (`:1539`); extend it to write the newly allocated `pageValues` entry when `position % pageSize == 0` — one 4-byte `writeBuffer` per 16 tokens, never a whole-table rewrite. The `nnzPages` computation at `:1533` becomes an assertion (`Math.floor(position/pageSize)+1 === columns.length`) rather than a computation. Split-K composes for free: it partitions the page range by **page** (`attention_splitk.wgsl:84-87`), which is exactly the granularity the pool hands out, and `splitk=8` is the default with sg32.
*Ordering hazard:* `writeBuffer` is queue-ordered against submits (documented at `:1527-1532`) — the table write must sit **inside** that contract, before the submit that first reads the new page. And `generatePipelined` keeps `PIPELINE_DEPTH = 2` steps in flight; its `finally` drains readbacks, not GPU work. Releasing a column and immediately reallocating it is **not** safe. Rule: defer every free behind a generation counter that only advances past a completed submit.
*Verify:* `debugCompareReuse` reports `max|Δ| = 0` on phi3, qwen3, qwen35; a new test allocating pages in reverse order for a >48-token prompt produces token-identical output vs identity allocation.

**3.4 — Attach and the tail-page copy-on-write.**
Full pages shared by refcount, tail page copied. `attach(entry, promptIds)` takes live refs on columns `0..k-1` where `k = floor(entry.length / pageSize)` — no copy — then allocates one fresh column and issues `layers × copyBufferToBuffer` of `bytesPerPage`, one encoder, no kernel, no readback: **512 KiB on Qwen3.5**, 2.25 MiB on Qwen3-4B, 6 MiB on Phi-3 (recomputed). KV buffers are already `STORAGE | COPY_SRC | COPY_DST` (`engine-core.ts:36`), so no usage flags change. Exactly one COW per attach, never more. Two conversations diverging from a shared prefix each COW their own tail; divergence *at* a boundary costs nothing. On extension in place, the `pooled` bit must **move** to the COW'd tail and the original tail goes to `cold`.
Reject the alternative (truncate to a page boundary, re-prefill the ≤15-token remainder): it makes `entry.length` lie about coverage, and on the specs without chunked prefill it is up to 15 full forward passes against one 512 KiB copy.
*Verify:* the load-bearing case, and no other test catches a missing COW — build an entry of a deliberately **non-page-aligned** length (e.g. 20,005, tail page holds 5 of 16 slots), attach twice, continue the two sequences with **different** tokens, assert each one's final-position logits equal its own from-zero prefill at `max|Δ| = 0`.

**3.5 — Eviction: tail-first, whole entries, LRU, never page-granular.**
Truncate entries from the tail in last-attach LRU order; an entry truncated to its first k pages is still a valid entry for a `16k`-token prefix. Never punch a hole: a global page-granular LRU across entries evicts column 600 of a 1250-column entry and leaves something that costs full memory and serves 600 tokens. Plain entry-level LRU is also wrong on its own — a three-column system-prompt entry is touched every turn and a 1250-column conversation prefix is not, so unbiased LRU evicts precisely the expensive thing the pool exists for. Floor: never evict an entry below a few pages.
*Verify:* fill to capacity, attach a live sequence, demand more columns than exist — the live sequence's columns are never reclaimed, `alloc()` returns null rather than throwing, and every surviving entry's column list is a contiguous prefix `0..k`, never a hole. A second test asserts a large entry survives repeated touches of a small one.

**3.6 — The decoy: the only test that asks the pool for a chain it must refuse.**
Prime the pool with conversation D whose tokens match B's **except at a single position deep in the prefix** — token 900 of 1000. Then run B. Assert (i) B's logits equal a fresh full prefill of B bit-exactly, and (ii) B's logits **differ** from a run where the pool was deliberately allowed to serve D past the divergence. Assertion (ii) is what proves the test has teeth: one changed token at position 900 of 1000 produces a small nonzero logit delta, which is the realistic shape of the leak — not a garbage blowup. Run under the **shipped** variant set, not scalar.
Plus an overlap test: two conversations sharing 56 full pages plus a partial — assert the shared columns are physically **aliased** (refcount 2, same index) and the partial is private to each.

---

### PHASE 4 — Deferred, with what breaks meanwhile

**4.1 — MLA aliasing.** `mla_kv_write.wgsl:84, 98` → `pageValues[position/PAGE_SIZE]*(PAGE_SIZE*L) + (position%PAGE_SIZE)*L + i`; `mla_scores.wgsl:58-59` maps its grid-x `t` the same way (one i32 load per (t,head) workgroup against 1152 B of cache reads — 0.35%); `mla_combine.wgsl:87-91` needs a page-outer/slot-inner restructure or its lookup lands in the innermost loop and runs `L/64 = 8×` more often than necessary.
*What breaks meanwhile:* **nothing, if it is asserted rather than assumed.** `allocKVPages` already branches on `spec.mla` (`engine-core.ts:97-99`), and its own comment (`:94-96`) says the branch lives inside the allocator precisely because a forgotten call site "allocates 7× the memory and still produces correct tokens, which is the kind of wrong nobody notices". Follow that discipline: `buildDecodeEngine` with a **resident pool** and `spec.mla` must **throw**, naming the three kernels — not silently fall back. A silent pass-through with a permuted table live puts the latent for position p at row p while the reader looks wherever the table says. Phase 1 persistence for MLA ships regardless (flat contiguous rows, zero kernel changes).
*When it lands, verify:* `mla_scores`/`mla_combine` **bit-identical** to the current kernels under an identity table first — page-outer/slot-inner over the identity IS `t`-ascending, so the f32 accumulation order in `mla_combine` is preserved and the reference bundle keeps meaning something. Then correct under a shuffled table. Then `scripts/mla-engine-test.mjs` reproduces its reference answer at 20 tokens (per `MLA_ENGINE_PLAN.md`'s own reasoning that t=0 is RoPE-identity).

**4.2 — Peer-shared KV: do not ship under the room's trust model.** Mechanically it works — KV entries are flat files in a per-spec OPFS directory, so `peer-weights.ts` replicates them unchanged at the measured ~39 MB/s (674 MiB ≈ 17 s against 99 s of re-prefill). But `peer-weights.ts:22-28` is explicit that the SHA-256 is **sender-computed** and does not defend against a dishonest host. That was defensible for weights: the host could return any tokens it liked anyway, and HuggingFace hashes are independent evidence. A KV cache is verifiable against **nothing** — a malicious peer supplies a cache that makes the receiver's own local model produce attacker-chosen output on a prompt the user typed themselves. That is a prompt-injection channel with no integrity story. It needs a different trust decision, and the narrower version worth considering is: peer KV scoped to **public, independently re-derivable prefixes only** (a published system prompt, a shared agent configuration), signed by whoever authored it. Not "replicate the pool". Separately, different adapters mean different f16 rounding, so imported blocks can never hold the exactly-0 bar the repo has refused to weaken anywhere else.

---

## PART 3 — Scope, honestly

**Full pool: ~5-6 weeks.** Phase 0 (2 days) + Phase 1 (1 week) + Phase 2 (3 days) + Phase 3 (2-3 weeks) + Phase 4.

**The week that buys the most: Phase 0 + Phase 1.** Zero kernel changes, zero allocator, zero refcounts, zero eviction. It buys **cold restart on Qwen3.5** — a 20k prefix goes from ~99 s of re-prefill to (pending 0.3b) a few seconds, and survives reload, crash and model switch. Ranked against the agentic workload it is #1, because the straight-line turn-on-turn loop the brief describes is already solved (BENCH.md: 14,340 → 194 ms, 74×), and a fresh tab is the case where the 20k prefix is currently simply unaffordable.

**Ranked by agentic coding, what each phase buys:**

| Rank | Win | Needs | Why it matters for an agent |
|---|---|---|---|
| 1 | Cold restart | **Phase 1 only** | A reload, a crash, a model switch. Today: full 20k re-prefill. |
| 2 | Branching | Phase 3 (index is Phase 1) | `noteAbsorbed:903-904` **destroys** the discarded branch on any rewind. An agent exploring two tool paths pays full price for the second. Nobody's plan led with this; it is the most agent-specific win in the list. |
| 3 | Interleaved conversations | Phase 2 + 3 | Agent + sub-agent, or two tabs. Every switch is a full re-prefill today. |
| 4 | Cross-device | Phase 4 | Blocked on a trust story, not on engineering. |

**Ship-or-don't thresholds, written before measuring** (BENCH.md's own discipline, after two withdrawn A/B pairs): cold-restore must cost under 1/5 of a cold re-prefill (for Qwen3.5's 20k / 625 MiB, under ~20 s against 99 s); interleaved turn-3 TTFT with the pool must be within ~2× of the consecutive-reuse case and ≥10× better than the no-pool interleaved case; decode regression from 2.2 must be inside measured session drift. Miss the interleaved threshold and the honest outcome is to keep Phase 1 and Phase 2 (a genuine correctness improvement that costs almost nothing) and **not** ship Phase 3.

**Do not publish a straight-line-loop number against `?reuse=0`.** That would be the third withdrawal in BENCH.md's history and it is the easiest number to produce by accident, because `benchTurns()` already exists and already extends turn-on-turn.

---

## PART 4 — Could not verify

1. **`adapter.limits.maxStorageBufferBindingSize` / `maxBufferSize` on the target machine.** Needs a browser. `loading-ui.ts:191-194` and `lib/index.ts:183-186` already request the adapter's own value, so the machinery exists, but the returned number is unrecorded. Decides whether Phase 3 fits in the existing per-layer buffer. Step 0.3(a).
2. **OPFS write/read throughput for 625 MiB.** Every "seconds vs 99 seconds" claim in all four angles is derived from a guessed ~300 MB/s. Step 0.3(b).
3. **GPU→CPU readback throughput and its memory-pressure effect.** The engine has never read the KV cache back. Step 0.3(c).
4. ~~**Whether the real agentic transcript is id-stable across turns.**~~ **RESOLVED 2026-08-10 — GREEN** for byte-level-BPE templates: nine real sessions, ~340 transitions, 0 breaks, tool traffic included (see the RESULT block in §0.2). Still open for **Phi-3**, whose SPM template this script cannot import and which BENCH.md already records breaking at a turn boundary.
5. **Qwen3.6-35B-A3B's actual prefill rate.** Chunked prefill is off for MoE by two independent gates (`engine-core.ts:2173-2174`: `!S.moe` and `!AFFINE`), so it prefills per token at ~340 dispatches. Unmeasured. It is moot for the 20k target (maxContext 6144, `model-spec.ts:887`) but it is the model the user measured as right for agentic coding, and its ceiling is deliberately small for a documented RAM-headroom reason (`model-spec.ts:866-873`: GPU process killed at 0.1 GB free). Whether that ceiling can move at all is a genuine open conflict with this feature, not a tuning detail.
6. **Whether WGSL codegen is bit-stable across Chrome/driver updates on one adapter.** This decides whether `reuse`'s block-0 canary (recompute block 0 through the same path, byte-compare against the stored payload, poison the namespace on mismatch) is belt-and-braces or the load-bearing correctness mechanism, and whether `adapter.info` in the fingerprint is sufficient or an epoch counter is also needed. I have not tested it; the canary costs ~16 token-prefills plus one block readback on a hit and I would include it in Phase 1.5 rather than argue about it.
7. **Whether an entry should persist as f16 or int8.** `?kv8=1` halves tier-2 (`allocKVPagesInt8:116-129`), but int8 is opt-in and explicitly unvalidated on target hardware by its own comment (`:113-115`), has no split-K variant, and a pool keyed on `kvLayout` means f16 and int8 entries never share. Quantizing on the way *out* only (f16 in VRAM, int8 on disk) is tempting and would break the exactly-0 bar for restored entries — which is the repo's only proof mechanism. I would keep f16-only in Phase 1 and revisit with 0.3(b)'s number in hand.