#!/usr/bin/env node
// ADD-MODEL — one command from a HuggingFace MLX checkpoint to a registered,
// compile-gated ModelSpec, or a report naming exactly which kernel is missing.
//
//   node scripts/add-model.mjs mlx-community/Qwen3-4B-4bit --param qwen3mlx
//   node scripts/add-model.mjs <repo> --check-only      # probe + verdict, no writes
//   node scripts/add-model.mjs --compat                 # regenerate docs/COMPAT.md
//
// Pipeline:
//   PROBE    config.json + tokenizer(.config).json + safetensors index and
//            shard headers. MEASURED 11-20 MB, not the 'few hundred KB' this
//            said until 2026-08-10: the config and headers are ranged reads, but
//            tokenizer.json is fetched WHOLE and is 11 MB on Qwen and 17 MB on
//            Llama-3. It matters on a slow link, and it was wrong in the README
//            too. Ranged reads against a
//            multi-GB repo, the same trick weight-loader-mlx.ts uses.
//   DETECT   normalise into constraints.DetectedModel (multimodal text_config
//            handled; qk-norm/shared-expert/biases read from RECORD NAMES, not
//            assumed from the family).
//   CHECK    src/compiler/constraints.ts — red prints {rule, detail, needs}
//            per failure and exits 1; `needs` names the kernel to build.
//   GENERATE a ModelSpec block + registry rows at the ADD-MODEL markers.
//            Everything derived (maxPages from the 1 GiB KV budget rule,
//            sizeLabel from the actual plan bytes, stops from the tokenizer's
//            OWN added_tokens — Qwen3.5 taught us never to assume ids).
//   GATE     tests/kernels/compile-spec.mjs — every WGSL kernel must compile
//            under the new dims before the result is called green.
//
// Scope: MLX-safetensors checkpoints (the format 90% of mlx-community ships).
// MLC q4f16_1 repos are hand-specced — they arrive a couple per year, each
// with format quirks worth human eyes.

import { readFileSync, writeFileSync, existsSync, openSync, readSync, closeSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, resolve, join } from 'node:path'
import { checkModel, SUPPORT_MATRIX, CONFIG_KEYS_READ, CONFIG_KEYS_IGNORED, detectChatTemplate } from '../src/compiler/constraints.ts'
import { makeModelSpec, mlxParamNaming } from '../src/compiler/model-spec.ts'
import { openMlxCheckpoint, planModel } from '../src/zero-tvm/weight-loader-mlx.ts'
import { planBytes } from '../src/zero-tvm/mlx-weights.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const SPEC_FILE = join(ROOT, 'src/compiler/model-spec.ts')
const REGISTRY_FILE = join(ROOT, 'src/zero-tvm/model-registry.ts')
const COMPAT_FILE = join(ROOT, 'docs/COMPAT.md')

// ============================================================
// CLI
// ============================================================

const args = process.argv.slice(2)
const flags = {}
const positional = []
for (let i = 0; i < args.length; i++) {
  if (args[i].startsWith('--')) {
    const key = args[i].slice(2)
    const next = args[i + 1]
    if (next !== undefined && !next.startsWith('--')) { flags[key] = next; i++ }
    else flags[key] = true
  } else positional.push(args[i])
}

if (flags.compat) { writeCompat(); process.exit(0) }
const repo = positional[0]
if (!repo) {
  console.error('usage: node scripts/add-model.mjs <hf-repo> [--param <url-param>] [--id <spec-id>] [--name <display>] [--local-dir <path>] [--check-only]')
  console.error('       node scripts/add-model.mjs --compat')
  process.exit(2)
}

// ============================================================
// PROBE
// ============================================================

const localDir = flags['local-dir'] ? resolve(flags['local-dir']) : null
const base = `https://huggingface.co/${repo}/resolve/main`

/** Thrown for a file the repo genuinely does not have — the only failure the
 *  probe is allowed to interpret. Everything else (429, 503, a dropped
 *  connection) means WE failed, not the model, and must not become evidence. */
class Missing extends Error {}

async function whole(file) {
  if (localDir) {
    try { return new Uint8Array(readFileSync(join(localDir, file))) }
    catch (e) { throw e.code === 'ENOENT' ? new Missing(file) : e }
  }
  // No timeout here once hung the probe forever with no output at all.
  const res = await fetch(`${base}/${file}`, { signal: AbortSignal.timeout(60_000) })
  if (res.status === 404) throw new Missing(file)
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`)
  return new Uint8Array(await res.arrayBuffer())
}
async function range(file, begin, end) {
  if (localDir) {
    // READ THE RANGE, not the file. The old comment here claimed "shards are
    // never read" — but this is exactly how the safetensors header is read, and
    // a single-file MLX checkpoint IS the shard: readFileSync threw
    // "File size (5288196018) is greater than 2 GiB" on Qwen3.6, which is the
    // documented fallback path failing on most models worth probing.
    let fd
    try { fd = openSync(join(localDir, file), 'r') }
    catch (e) { throw e.code === 'ENOENT' ? new Missing(file) : e }
    try {
      const out = new Uint8Array(end - begin)
      readSync(fd, out, 0, out.length, begin)
      return out
    } finally { closeSync(fd) }
  }
  const res = await fetch(`${base}/${file}`, {
    headers: { Range: `bytes=${begin}-${end - 1}` }, signal: AbortSignal.timeout(60_000),
  })
  if (res.status === 404) throw new Missing(file)
  if (res.status !== 206) throw new Error(`${file}: expected 206 for Range, got ${res.status}`)
  return new Uint8Array(await res.arrayBuffer())
}
/** null means the repo HAS NO such file. A transport failure aborts instead:
 *  swallowing one into null let a rate-limited probe report a model as having
 *  no tensors, no qk-norm and no shared expert — a confident wrong answer. */
async function json(file) {
  let raw
  try { raw = new TextDecoder().decode(await whole(file)) }
  catch (e) { if (e instanceof Missing) return null; throw e }
  try { return JSON.parse(raw) } catch { return null }
}
async function text(file) {
  try { return new TextDecoder().decode(await whole(file)) }
  catch (e) { if (e instanceof Missing) return null; throw e }
}

console.log(`probing ${localDir ?? repo} …`)
const cfg = await json('config.json')
if (!cfg) { console.error('config.json unreadable — is the repo name right?'); process.exit(1) }
const tokCfg = await json('tokenizer_config.json') ?? {}
const tok = await json('tokenizer.json')

let checkpoint = null
try { checkpoint = await openMlxCheckpoint({ whole, range }) } catch (e) {
  // An ABSENT index is a finding (the [index] rule reports it). An index we
  // could not fetch is not — continuing would report every record-derived
  // property as false.
  if (!(e instanceof Missing)) {
    console.error(`\nprobe FAILED reading the safetensors index: ${e.message}`)
    console.error('this says nothing about the model — retry, or pass --local-dir')
    process.exit(2)
  }
  console.error(`  (repo has no safetensors index)`)
}
const records = checkpoint ? checkpoint.records : []

// ============================================================
// DETECT
// ============================================================

// Multimodal checkpoints (Qwen3.6-VL-style, Gemma3) nest the text tower's dims
// under text_config and its records under language_model. — merge shallowly,
// text_config winning, and remember which root the records use.
const tc = { ...cfg, ...(cfg.text_config ?? {}) }
const mlxPrefix = records.some((r) => r.startsWith('language_model.')) ? 'language_model.' : ''

// The records the engine would actually load. Multimodal checkpoints ship a
// whole second network — patch embeds, vision blocks, a merger — and asking
// "does this model have X" over ALL records answers for the tower we throw
// away. Under a language_model. root the split is exact; without one, drop the
// known tower roots by name.
const TOWER = /^(vision_tower|visual|vision_model|audio_tower|audio_model|multi_modal_projector)\./
const textRecords = mlxPrefix
  ? records.filter((r) => r.startsWith(mlxPrefix))
  : records.filter((r) => !TOWER.test(r))

// `rope_parameters` is the newer transformers spelling, and it is a MERGE, not
// a rename: it absorbs rope_theta and partial_rotary_factor along with the
// scaling dict. Qwen3.6's config (already on disk here) has NO rope_theta
// under either root — it lives only in text_config.rope_parameters, so the old
// reads returned null. A null theta is not a crash: it is a different RoPE
// table, and the model still generates fluent text.
// (partial_rotary_factor happens to be duplicated at the text_config root in
// that checkpoint, so it survived. A yarn model in the new format would not:
// its whole scaling dict is in here.)
// Normalised into the old shape so every read below stays as it was.
const ropeParams = tc.rope_parameters
if (ropeParams && typeof ropeParams === 'object') {
  if (tc.rope_theta == null) tc.rope_theta = ropeParams.rope_theta
  if (tc.partial_rotary_factor == null) tc.partial_rotary_factor = ropeParams.partial_rotary_factor
  // 'default' means plain RoPE — the dict is present only to carry theta.
  if (tc.rope_scaling == null && ropeParams.rope_type && ropeParams.rope_type !== 'default') {
    tc.rope_scaling = ropeParams
  }
}

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const heads = num(tc.num_attention_heads)
const d = num(tc.hidden_size)
const headDim = num(tc.head_dim) ?? (heads && d ? d / heads : null)
// DeepSeek spells the expert count n_routed_experts; reading only Qwen's name
// made a 64-expert MoE detect as a dense FFN.
const isMoe = num(tc.num_experts) !== null || num(tc.n_routed_experts) !== null
const isMla = num(tc.kv_lora_rank) !== null
const isGdn = num(tc.linear_num_key_heads) !== null

const quantCfg = cfg.quantization ?? cfg.quantization_config ?? null
const overrides = {}
if (quantCfg) {
  for (const [k, v] of Object.entries(quantCfg)) {
    if (v && typeof v === 'object' && 'bits' in v) overrides[k] = { bits: v.bits, groupSize: v.group_size ?? 64 }
    else if (v === false) overrides[k] = { bits: 16, groupSize: 0 }   // explicitly unquantised path
  }
}

// Chat template — from tokenizer_config (string, or a named list on newer
// repos), else the standalone chat_template.jinja file the current HF
// convention ships (e.g. Qwen3-4B-Instruct-2507: no tokenizer_config
// template at all — skipping the jinja file made real ChatML repos red).
const templates = typeof tokCfg.chat_template === 'string' ? [tokCfg.chat_template]
  : Array.isArray(tokCfg.chat_template) ? tokCfg.chat_template.map((t) => t.template ?? '') : []
let templateText = templates.join('\n')
if (!templateText) templateText = (await text('chat_template.jinja')) ?? ''
// Matched on the marker each template is BUILT from. DeepSeek's has no special
// role tokens at all — it writes prose turns — so it is recognised by the two
// literals it interpolates, which is as specific as that family gets.
// bos_token is a bare string in some configs and an AddedToken object in
// others; both spellings are live across the repos this probes.
// bos_token is a bare string in some configs and an AddedToken object in
// others; both spellings are live across the repos this probes.
const bosTokenStr = typeof tokCfg.bos_token === 'string' ? tokCfg.bos_token
  : tokCfg.bos_token && typeof tokCfg.bos_token === 'object' ? (tokCfg.bos_token.content ?? null)
  : null
const chatTemplate = detectChatTemplate(templateText, bosTokenStr)

// Tokenizer pipeline — from tokenizer.json's own structure.
const preTok = JSON.stringify(tok?.pre_tokenizer ?? '')
const tokKind = !tok ? 'unknown'
  : tok.model?.type === 'BPE' && preTok.includes('ByteLevel') ? 'byteLevel'
  : tok.model?.type === 'Unigram' || JSON.stringify(tok.decoder ?? '').includes('▁') ? 'spm'
  : 'unknown'

// Stop ids from the tokenizer's OWN added_tokens, by NAME — Qwen3.5 renumbered
// the ChatML specials for its 248320 vocab and the stale ids mapped to
// ordinary BPE tokens. config.eos_token_id is only the fallback.
const addedId = (content) => tok?.added_tokens?.find((t) => t.content === content)?.id ?? null
const eosIds = [tc.eos_token_id].flat().filter((v) => typeof v === 'number')
const stops = chatTemplate === 'chatml'
  ? [addedId('<|im_end|>'), addedId('<|endoftext|>')].filter((v) => v !== null)
  : chatTemplate === 'llama3'
  ? [addedId('<|eot_id|>'), addedId('<|end_of_text|>')].filter((v) => v !== null)
  : eosIds
if (!stops.length) stops.push(...eosIds)

const detected = {
  repo: localDir ?? repo,
  family: tc.model_type ?? cfg.model_type ?? '',
  multimodal: !!cfg.text_config || mlxPrefix !== '',
  d,
  layers: num(tc.num_hidden_layers),
  heads,
  kvHeads: num(tc.num_key_value_heads),
  headDim,
  ffn: isMoe ? num(tc.moe_intermediate_size) : num(tc.intermediate_size),
  vocab: num(tc.vocab_size),
  ropeTheta: num(tc.rope_theta),
  ropeScalingType: tc.rope_scaling ? (tc.rope_scaling.rope_type ?? tc.rope_scaling.type ?? 'unknown') : null,
  // llama3 remapping params for ropeInvFreqTable() (checked complete by the
  // rope rule); other scaling types carry no params — they are red anyway.
  ropeScaling: (tc.rope_scaling?.rope_type ?? tc.rope_scaling?.type) === 'llama3' ? {
    factor: num(tc.rope_scaling.factor),
    lowFreqFactor: num(tc.rope_scaling.low_freq_factor),
    highFreqFactor: num(tc.rope_scaling.high_freq_factor),
    originalMaxPositionEmbeddings: num(tc.rope_scaling.original_max_position_embeddings),
  } : null,
  ropeYarn: (tc.rope_scaling?.rope_type ?? tc.rope_scaling?.type) === 'yarn' ? {
    factor: num(tc.rope_scaling.factor),
    betaFast: num(tc.rope_scaling.beta_fast),
    betaSlow: num(tc.rope_scaling.beta_slow),
    originalMaxPositionEmbeddings: num(tc.rope_scaling.original_max_position_embeddings),
    mscale: num(tc.rope_scaling.mscale),
    mscaleAllDim: num(tc.rope_scaling.mscale_all_dim),
  } : null,
  rmsEps: num(tc.rms_norm_eps),
  maxSeq: num(tc.max_position_embeddings),
  tied: tc.tie_word_embeddings === true
    || (records.length > 0 && !records.some((r) => r.startsWith(`${mlxPrefix}lm_head.`))),
  qkNorm: records.some((r) => r.endsWith('self_attn.q_norm.weight')),
  // Only the TEXT tower's biases. The engine never loads a vision or audio
  // tower, so a bias there costs it nothing — and counting them made
  // Qwen3.6-35B RED on a kernel it does not need. That model SHIPS here and
  // runs; all 166 of its `.bias` records live under vision_tower.* and its
  // language model has exactly zero. `bias` led the blocker ranking in
  // docs/PORTING.md partly on records like those.
  attnBias: tc.attention_bias === true || textRecords.some((r) => r.endsWith('.bias')),
  // Named in the report. "linear-layer biases" read the same for a q_proj.bias
  // (which really does need the epilogue) and a mamba conv1d.bias (which needs
  // a mamba block, and the model is red for that anyway).
  biasRecords: textRecords.filter((r) => r.endsWith('.bias')).slice(0, 3),
  slidingWindow: num(tc.sliding_window) !== null && tc.use_sliding_window !== false,
  hiddenAct: tc.hidden_act ?? tc.hidden_activation ?? null,
  quant: quantCfg ? { bits: quantCfg.bits ?? 0, groupSize: quantCfg.group_size ?? 0, overrides } : null,
  mlpBias: tc.mlp_bias === true,
  mla: isMla ? {
    kvLoraRank: num(tc.kv_lora_rank),
    qLoraRank: num(tc.q_lora_rank),
    qkNopeHeadDim: num(tc.qk_nope_head_dim) ?? 0,
    qkRopeHeadDim: num(tc.qk_rope_head_dim) ?? 0,
    vHeadDim: num(tc.v_head_dim) ?? 0,
  } : null,
  moe: isMoe ? {
    // Qwen names the dense ones; DeepSeek says how many lead the stack.
    denseLayers: Array.isArray(tc.mlp_only_layers) ? tc.mlp_only_layers
      : Array.from({ length: num(tc.first_k_dense_replace) ?? 0 }, (_, i) => i),
    sparseStep: num(tc.decoder_sparse_step) ?? num(tc.moe_layer_freq) ?? 1,
    scoringFunc: tc.scoring_func ?? null,
    // A quantized router ships .scales beside .weight; DeepSeek's does not.
    routerQuantized: records.some((r) => /\.mlp\.gate\.weight$/.test(r))
      ? records.some((r) => /\.mlp\.gate\.scales$/.test(r))
      : undefined,
    experts: num(tc.num_experts) ?? num(tc.n_routed_experts),
    topK: num(tc.num_experts_per_tok) ?? 0,
    sharedExpert: records.some((r) => r.includes('.shared_expert.') || r.includes('.shared_experts.')),
    sharedFfn: num(tc.shared_expert_intermediate_size),
    normTopkProb: tc.norm_topk_prob === true,
    // The router is 8-bit only when the checkpoint says so as a per-tensor
    // OVERRIDE; with none it sits at the model's base bits. This is not
    // cosmetic — the two widths need different unpacks and the wrong one is
    // silent (Qwen3-30B-A3B: 4-bit router, read as 8-bit, cosine 0.087).
    // A gate with no .scales beside it was never quantized — 16 selects the
    // f16 router kernel, which is a different bind group, not just a width.
    routerBits: records.some((r) => /\.mlp\.gate\.weight$/.test(r)) && !records.some((r) => /\.mlp\.gate\.scales$/.test(r))
      ? 16
      : Object.entries(overrides).find(([p]) => /(^|\.)mlp\.(gate|shared_expert_gate)$/.test(p))?.[1]?.bits
        ?? quantCfg?.bits ?? 4,
  } : null,
  gdn: isGdn ? {
    kHeads: num(tc.linear_num_key_heads),
    vHeads: num(tc.linear_num_value_heads),
    headK: num(tc.linear_key_head_dim),
    headV: num(tc.linear_value_head_dim),
    convK: num(tc.linear_conv_kernel_dim) ?? 0,
    fullAttnInterval: num(tc.full_attention_interval) ?? 0,
  } : null,
  // `attn_output_gate` states it outright, and the engine SUPPORTS it — but the
  // checker never read the key, so a config that carries it went RED on
  // "fields this checker does not read" for a feature that already ships.
  // isGdn stays as the fallback for configs that omit the key; it is a proxy
  // (both shipped GDN families happen to gate attention), not a definition.
  attnGate: tc.attn_output_gate != null ? tc.attn_output_gate === true : isGdn,
  partialRotaryFactor: num(tc.partial_rotary_factor),
  // Verbatim per-layer block claim — the checker refuses unknown kinds (LFM2's
  // 'conv' would otherwise detect as plain attention and fake-green).
  layerTypes: Array.isArray(tc.layer_types) ? tc.layer_types : null,
  // Everything in the config this checker neither reads nor has explicitly
  // dismissed. text_config's own keys count too — a multimodal text tower can
  // carry architecture the root does not.
  unknownConfigKeys: [...new Set([...Object.keys(cfg), ...Object.keys(cfg.text_config ?? {})])]
    .filter((k) => !CONFIG_KEYS_READ.has(k) && !CONFIG_KEYS_IGNORED.has(k))
    .sort(),
  hasIndex: !!checkpoint,
  tokenizer: tokKind,
  chatTemplate,
}

// ============================================================
// CHECK
// ============================================================

const { ok, failures, notes } = checkModel(detected)
const dimStr = detected.d === null ? '?' :
  `d=${detected.d} L=${detected.layers} heads=${detected.heads}/${detected.kvHeads}×${detected.headDim} ffn=${detected.ffn} vocab=${detected.vocab}`
console.log(`\n  family ${detected.family || '?'}${detected.multimodal ? ' (multimodal root)' : ''} | ${dimStr}`)
console.log(`  quant ${detected.quant ? `${detected.quant.bits}-bit g${detected.quant.groupSize} (${Object.keys(overrides).length} overrides)` : 'NONE'}`
  + ` | ${detected.tied ? 'tied' : 'untied'} lm_head | qkNorm ${detected.qkNorm} | ${records.length} records`
  + `${detected.moe ? ` | MoE ${detected.moe.experts}×top${detected.moe.topK}` : ''}${detected.gdn ? ' | GDN hybrid' : ''}`)
// RoPE printed explicitly because its failure mode is silence: a theta read
// from a key the config does not use comes back null, and a spec with no
// scaling is a different model that still generates text.
console.log(`  rope theta ${detected.ropeTheta ?? 'MISSING'}`
  + ` | ${detected.ropeScalingType ?? 'no scaling'}`
  + ` | rotary ${detected.partialRotaryFactor ?? 1}× headDim`
  + (ropeParams ? '  (from rope_parameters)' : ''))
console.log(`  tokenizer ${tokKind} | template ${chatTemplate} | stops [${stops.join(', ')}]`)

for (const n of notes) console.log(`  note: ${n}`)
if (!ok) {
  console.log(`\nRED — ${repo} needs work before it can run here:\n`)
  for (const f of failures) {
    console.log(`  ✗ [${f.rule}] ${f.detail}`)
    console.log(`      needs: ${f.needs}\n`)
  }
  console.log('docs/COMPAT.md has the full support matrix.')
  process.exit(1)
}
console.log(`\nGREEN — every block maps onto the existing kernel set.`)
if (flags['check-only']) process.exit(0)

// ============================================================
// GENERATE
// ============================================================

const tail = repo.split('/').pop()
const id = flags.id ?? tail.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
const param = flags.param ?? id
const exportName = id.toUpperCase().replace(/-/g, '_')
const displayName = flags.name ?? tail.replace(/-(MLX-)?[0-9]bit$/i, '')

const specSrc = readFileSync(SPEC_FILE, 'utf8')
const regSrc = readFileSync(REGISTRY_FILE, 'utf8')
if (specSrc.includes(`id: '${id}'`) || regSrc.includes(`param: '${param}'`)) {
  console.error(`\nid '${id}' or param '${param}' already exists — pass --id/--param, or remove the old block first`)
  process.exit(1)
}

// maxPages: the 1 GiB KV budget rule every hand-written spec follows —
// min(1 GiB / (kvBytesPerToken × pageSize), maxSeq / pageSize), floored to a
// multiple of 64 pages. (Phi-3 predates the rule and keeps its trained window.)
const attnLayers = detected.gdn
  ? Math.floor(detected.layers / detected.gdn.fullAttnInterval)
  : detected.layers
const kvBytesPerToken = 2 * detected.kvHeads * detected.headDim * 2 * attnLayers
const budgetPages = Math.floor(2 ** 30 / (kvBytesPerToken * 16))
const maxPages = Math.max(64, Math.floor(Math.min(budgetPages, detected.maxSeq / 16) / 64) * 64)

const lines = [
  `export const ${exportName}: ModelSpec = makeModelSpec({`,
  `  id: '${id}',`,
  `  d: ${detected.d},`,
  `  layers: ${detected.layers},`,
  `  heads: ${detected.heads},`,
  `  kvHeads: ${detected.kvHeads},`,
  `  headDim: ${detected.headDim},`,
  `  ffn: ${detected.ffn},${detected.moe ? '   // moe_intermediate_size (per-expert width)' : ''}`,
  `  vocab: ${detected.vocab},`,
  `  pageSize: 16,`,
  `  // ${maxPages * 16} tokens on the 1 GiB KV budget rule (${(kvBytesPerToken / 1024).toFixed(0)} KiB/token × ${attnLayers} attn layers).`,
  `  maxPages: ${maxPages},`,
  `  maxSeq: ${detected.maxSeq},`,
  `  ropeTheta: ${detected.ropeTheta},`,
  // yarn as well as llama3 — the checker turned yarn GREEN, and a generated
  // spec that omitted it would run plain RoPE past the trained context with no
  // error anywhere. The checker refuses a yarn config missing any of these.
  ...(detected.ropeYarn ? [
    `  ropeScaling: { ropeType: 'yarn', factor: ${detected.ropeYarn.factor}, betaFast: ${detected.ropeYarn.betaFast}, betaSlow: ${detected.ropeYarn.betaSlow}, originalMaxPositionEmbeddings: ${detected.ropeYarn.originalMaxPositionEmbeddings}, mscale: ${detected.ropeYarn.mscale ?? 1}, mscaleAllDim: ${detected.ropeYarn.mscaleAllDim ?? 0} },`,
  ] : []),
  ...(detected.ropeScaling ? [
    `  // llama3 frequency remapping — ropeInvFreqTable() precomputes the table.`,
    `  ropeScaling: { ropeType: 'llama3', factor: ${detected.ropeScaling.factor}, lowFreqFactor: ${detected.ropeScaling.lowFreqFactor}, highFreqFactor: ${detected.ropeScaling.highFreqFactor}, originalMaxPositionEmbeddings: ${detected.ropeScaling.originalMaxPositionEmbeddings} },`,
  ] : []),
  `  rmsEps: ${detected.rmsEps},`,
  `  tiedEmbeddings: ${detected.tied},`,
  `  qkNorm: ${detected.qkNorm},`,
  `  // Resolved from THIS repo's tokenizer.json added_tokens by name, not assumed.`,
  `  stops: [${stops.join(', ')}],`,
  `  chatTemplateId: '${chatTemplate}',`,
  `  tokenizerKind: '${tokKind}',`,
  `  hfRepo: '${repo}',`,
  `  manifestName: 'model.safetensors.index.json',`,
  `  weightFormat: 'mlx-safetensors',`,
  ...(mlxPrefix ? [`  mlxPrefix: '${mlxPrefix}',`] : []),
  ...(detected.gdn ? [
    `  fullAttnInterval: ${detected.gdn.fullAttnInterval},`,
    `  gdn: { kHeads: ${detected.gdn.kHeads}, vHeads: ${detected.gdn.vHeads}, headK: ${detected.gdn.headK}, headV: ${detected.gdn.headV}, convK: ${detected.gdn.convK} },`,
    `  attnGate: true,`,
  ] : []),
  ...(detected.partialRotaryFactor && detected.partialRotaryFactor !== 1
    ? [`  partialRotaryFactor: ${detected.partialRotaryFactor},`] : []),
  ...(detected.moe ? [
    `  moe: { experts: ${detected.moe.experts}, topK: ${detected.moe.topK}, normTopkProb: ${detected.moe.normTopkProb}, sharedExpert: ${detected.moe.sharedExpert}, routerBits: ${detected.moe.routerBits} },`,
  ] : []),
  `  paramNaming: mlxParamNaming(${JSON.stringify(mlxPrefix)}),`,
  `})`,
]
const specBlock = `\n// ${repo} — generated by scripts/add-model.mjs (${new Date().toISOString().slice(0, 10)})\n${lines.join('\n')}\n`

// Build the same spec IN MEMORY to (a) fail here if makeModelSpec throws and
// (b) price the download from the actual plan bytes.
const spec = makeModelSpec({
  id, d: detected.d, layers: detected.layers, heads: detected.heads, kvHeads: detected.kvHeads,
  headDim: detected.headDim, ffn: detected.ffn, vocab: detected.vocab, pageSize: 16, maxPages,
  maxSeq: detected.maxSeq, ropeTheta: detected.ropeTheta, rmsEps: detected.rmsEps,
  ...(detected.ropeScaling ? { ropeScaling: { ropeType: 'llama3', ...detected.ropeScaling } } : {}),
  ...(detected.ropeYarn ? { ropeScaling: {
    ropeType: 'yarn', factor: detected.ropeYarn.factor, betaFast: detected.ropeYarn.betaFast,
    betaSlow: detected.ropeYarn.betaSlow,
    originalMaxPositionEmbeddings: detected.ropeYarn.originalMaxPositionEmbeddings,
    mscale: detected.ropeYarn.mscale ?? 1, mscaleAllDim: detected.ropeYarn.mscaleAllDim ?? 0,
  } } : {}),
  tiedEmbeddings: detected.tied, qkNorm: detected.qkNorm, stops, chatTemplateId: chatTemplate,
  tokenizerKind: tokKind, hfRepo: repo, manifestName: 'model.safetensors.index.json',
  weightFormat: 'mlx-safetensors', mlxPrefix,
  ...(detected.gdn ? {
    fullAttnInterval: detected.gdn.fullAttnInterval,
    gdn: { kHeads: detected.gdn.kHeads, vHeads: detected.gdn.vHeads, headK: detected.gdn.headK, headV: detected.gdn.headV, convK: detected.gdn.convK },
    attnGate: true,
  } : {}),
  ...(detected.partialRotaryFactor && detected.partialRotaryFactor !== 1 ? { partialRotaryFactor: detected.partialRotaryFactor } : {}),
  ...(detected.moe ? { moe: { experts: detected.moe.experts, topK: detected.moe.topK, normTopkProb: detected.moe.normTopkProb, sharedExpert: detected.moe.sharedExpert, routerBits: detected.moe.routerBits } } : {}),
  paramNaming: mlxParamNaming(mlxPrefix),
})

let gb = null
try {
  const src = (r) => { const l = checkpoint.locate(r); return l.end - l.begin }
  const bytes = planModel(spec).reduce((a, { plan }) => a + planBytes(plan, src), 0)
  gb = bytes / 2 ** 30
} catch (e) {
  console.error(`\nplan does not resolve against the checkpoint: ${e.message}`)
  console.error('(a record the loader expects is missing — this is a format quirk worth human eyes)')
  process.exit(1)
}
const sizeLabel = `~${gb.toFixed(1)} GB`
const ramNote = gb > 8 ? `, ramNote: 'needs ~${Math.ceil(gb * 1.25)} GB free RAM'` : ''
const archLine = detected.moe ? 'MoE' : detected.gdn ? 'hybrid (DeltaNet)' : 'dense'
const bMatch = tail.match(/(\d+(?:\.\d+)?)B/i)
const paramsLine = flags.params ?? (bMatch ? `${bMatch[1]}B ${archLine}` : archLine)

// Insert at the markers.
writeFileSync(SPEC_FILE, specSrc.replace(/\n\/\/ ADD-MODEL:SPECS\n/,
  `\n// ADD-MODEL:SPECS\n${specBlock}`))
writeFileSync(REGISTRY_FILE, regSrc
  .replace('// ADD-MODEL:IMPORTS', `import { ${exportName} } from '../compiler/model-spec.ts'\n// ADD-MODEL:IMPORTS`)
  .replace('  // ADD-MODEL:MODELS', `  { param: '${param}', spec: ${exportName} },\n  // ADD-MODEL:MODELS`)
  .replace('  // ADD-MODEL:BRANDINGS', `  [${exportName}.id]: { name: '${displayName}', params: '${paramsLine}', sizeLabel: '${sizeLabel}', rateLabel: ''${ramNote} },\n  // ADD-MODEL:BRANDINGS`))
writeCompat()
console.log(`\nwrote ${exportName} (${sizeLabel}) → model-spec.ts + model-registry.ts (param '${param}')`)

// ============================================================
// GATE
// ============================================================

console.log(`\ncompile gate: every kernel under the new dims …`)
const gate = spawnSync('node', ['tests/kernels/compile-spec.mjs', exportName], { cwd: ROOT, stdio: 'inherit' })
if (gate.status !== 0) {
  console.error(`\ncompile gate FAILED — the generated spec stays in place for debugging;`)
  console.error(`revert with: git checkout -- src/compiler/model-spec.ts src/zero-tvm/model-registry.ts`)
  process.exit(1)
}

console.log(`\ndone. Next:`)
console.log(`  npm run typecheck && npm run dev`)
console.log(`  validate.html?model=${param}   # logits vs mlx_lm before trusting it`)
console.log(`  zero-tvm.html?model=${param}   # chat (weights stream from ${repo})`)
console.log(`  registry branding (rateLabel) stays '' until a measured number exists — BENCH.md protocol.`)

// ============================================================
// docs/COMPAT.md
// ============================================================

function writeCompat() {
  const rows = SUPPORT_MATRIX.map((r) =>
    `| ${r.area} | ${r.supported} | ${r.not} | ${r.needs || '—'} |`).join('\n')
  writeFileSync(COMPAT_FILE, `# Model compatibility

<!-- GENERATED from src/compiler/constraints.ts (SUPPORT_MATRIX) by
     scripts/add-model.mjs --compat. Edit the matrix there, not this file. -->

\`scripts/add-model.mjs <hf-repo>\` checks a checkpoint against this matrix and
either generates a registered ModelSpec (green) or names the missing kernel per
failure (red). The checks themselves live in \`src/compiler/constraints.ts\` —
same file as this table's source, so they cannot drift far.

| Area | Supported | Not supported | Lifting it needs |
|---|---|---|---|
${rows}

Dimension rules (all enforced by the checker): heads divisible by kvHeads;
headDim %32 and ≤256; d, qkvDim, kvDim %256; every matmul K (d, qDim, ffn,
gdnVDim) %64 — the scale-group width, since the general int4 matmul strides K
and stops on a bound rather than a trip count; vocab %4.
`)
  console.log(`wrote docs/COMPAT.md (${SUPPORT_MATRIX.length} rows)`)
}
