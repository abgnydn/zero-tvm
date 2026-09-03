/**
 * TOKENIZER-BPE — byte-level BPE tokenizer for Qwen3 (GPT-2 style).
 *
 * Implements the EXACT pipeline shipped in mlc-ai/Qwen3-4B-q4f16_1-MLC's
 * tokenizer.json, which is structurally different from Phi-3's
 * SentencePiece-as-BPE (src/zero-tvm/tokenizer.ts — untouched by this file):
 *
 *   normalizer:     NFC
 *   pre_tokenizer:  Sequence[
 *                     Split(regex, behavior: Isolated),   ← GPT-2-style word split
 *                     ByteLevel(add_prefix_space: false, use_regex: false)
 *                   ]
 *   model:          BPE (byte_fallback: false, unk_token: null,
 *                        ignore_merges: false, fuse_unk: false)
 *   post_processor: ByteLevel (adds NO tokens — no BOS/EOS wrapping)
 *   decoder:        ByteLevel
 *
 * Encode pipeline, in order (each step pinned by tests/tokenizer/
 * tokenizer-bpe.test.ts against HF-generated fixtures):
 *
 *   1. Split around added tokens (all 26 entries, special AND non-special —
 *      <think>, <tool_call>, ... are extracted too). They are declared
 *      normalized: false, so this split runs on the RAW text, before NFC.
 *      lstrip/rstrip are all false: no whitespace swallowing (unlike Phi-3).
 *   2. Per text section: NFC-normalize, then apply the Split regex. Behavior
 *      "Isolated" keeps every match as its own piece (the pattern covers all
 *      text, but gaps are preserved as pieces too, for exactness).
 *   3. ByteLevel: map each piece's UTF-8 bytes to visible stand-in chars via
 *      the GPT-2 byte→unicode table (space → Ġ, newline → Ċ, ...).
 *   4. Rank-based BPE over the mapped chars, per piece. Every single byte
 *      char is in the vocab, so there is no byte-fallback and no <unk>.
 *
 * Decode mirrors the ByteLevel decoder: concat token strings, reverse the
 * byte↔unicode map, UTF-8-decode with replacement on invalid sequences.
 * Special (special: true) added tokens are stripped from output — matching
 * HF decode(..., skip_special_tokens: true) — while non-special added tokens
 * (<think>, </think>, <tool_call>, ...) are emitted literally, also matching
 * HF, so the chat layer can render thinking blocks.
 *
 * Chat template + stop tokens (ground truth: tokenizer_config.json
 * chat_template and mlc-chat-config.json conv_template "qwen3"):
 *
 *   <|im_start|>system\n{system}<|im_end|>\n
 *   <|im_start|>user\n{user}<|im_end|>\n
 *   <|im_start|>assistant\n
 *
 *   eos/stop:  <|im_end|> (151645) and <|endoftext|> (151643)
 *   bos:       none (tokenizer_config bos_token: null; MLC re-uses
 *              <|endoftext|> as bos_token_id/pad_token_id)
 *   default system message (MLC): "You are a helpful assistant."
 *
 * Thinking mode: Qwen3 thinks by DEFAULT. The HF template only appends an
 * empty `<think>\n\n</think>\n\n` block after the generation prompt when
 * `enable_thinking is false` is passed explicitly; MLC's "qwen3" template
 * matches the default (thinking) path. buildChatPrompt() below defaults to
 * thinking enabled and takes `{ thinking: false }` to emit the exact
 * non-thinking suffix from the template.
 *
 * Correctness is pinned by tests/tokenizer/tokenizer-bpe.test.ts against
 * fixtures generated with the HuggingFace reference implementation
 * (scripts/gen-tokenizer-fixtures-qwen.mjs).
 */

import type { Tokenizer } from './tokenizer.js'

// ============================================================
// tokenizer.json types (subset we need)
// ============================================================

export interface ByteLevelTokenizerJSON {
  /** Qwen3 ships {type:'NFC'}; Llama-3 ships null (no normalizer). */
  normalizer?: { type: string } | null
  /** Sequence[Split(regex), ByteLevel] — the Split pattern differs per family
   *  (Qwen3 splits single digits, Llama-3 digit runs of ≤3), so encode()
   *  translates the repo's OWN regex instead of assuming Qwen3's. */
  pre_tokenizer?: {
    type: string
    pretokenizers?: Array<{ type: string; pattern?: { Regex?: string } }>
    pattern?: { Regex?: string }
  } | null
  model: {
    type: string
    vocab: Record<string, number>
    // HF ships two serializations: legacy "left right" strings and the
    // current ["left", "right"] pairs (Qwen3 uses pairs). Support both.
    merges: Array<string | [string, string]>
  }
  added_tokens: Array<{
    id: number
    content: string
    single_word: boolean
    lstrip: boolean
    rstrip: boolean
    normalized: boolean
    special: boolean
  }>
}

// ============================================================
// Fetch (with cache fallback — same helper pattern as weight-loader)
// ============================================================

const QWEN3_MODEL_BASE = 'https://huggingface.co/mlc-ai/Qwen3-4B-q4f16_1-MLC/resolve/main/'

async function fetchText(url: string): Promise<string> {
  try {
    const cacheNames = await caches.keys()
    for (const name of cacheNames) {
      const store = await caches.open(name)
      const resp = await store.match(url)
      if (resp) return resp.text()
    }
  } catch { /* no Cache API */ }

  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching tokenizer.json`)
  return resp.text()
}

// ============================================================
// Byte-level BPE implementation
// ============================================================

export interface ByteLevelTokenizer extends Tokenizer {
  /** Generation stop set: [<|im_end|>, <|endoftext|>] per mlc-chat-config. */
  stopIds: number[]
}

/**
 * The Split-pretokenizer regex from Qwen3's tokenizer.json, translated to a
 * JS RegExp. Original (oniguruma syntax):
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 *
 * The only construct JS lacks is the inline case-insensitive group `(?i:...)`
 * (a global `i` flag would wrongly affect nothing else here, but per-branch
 * expansion is exact): each contraction letter becomes a [xX] class. Note
 * \p{N} matches ONE digit — Qwen3 splits every digit into its own pre-token
 * (not runs of up to 3 like Llama-3). The trailing-whitespace lookahead
 * `\s+(?!\S)` and `\s*[\r\n]+` translate verbatim.
 */
const PRETOKENIZE_RE =
  /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu

/**
 * Split-pretokenizer regex from the repo's OWN tokenizer.json. Llama-3's
 * pattern differs from Qwen3's in exactly one branch — `\p{N}{1,3}` (digit
 * runs of up to 3) where Qwen3 has `\p{N}` — so assuming the hardcoded
 * PRETOKENIZE_RE would silently mistokenize every number in a Llama prompt.
 *
 * The only oniguruma construct JS lacks is the inline case-insensitive group
 * `(?i:...)`, and both families ship the identical contraction group, so the
 * translation is one exact-string replacement (same per-branch expansion the
 * hardcoded regex documents); everything else (`\p{L}`, `\p{N}{1,3}`, the
 * `(?!\S)` lookahead) is valid JS with /u. Falls back to the Qwen3 regex when
 * the JSON has no Split pattern — for Qwen3 itself the translation and the
 * fallback are the same regex (pinned by tests/tokenizer fixtures).
 */
function pretokenizeRegexFrom(json: ByteLevelTokenizerJSON): RegExp {
  const pt = json.pre_tokenizer
  const parts = pt?.pretokenizers ?? (pt ? [pt] : [])
  const pattern = parts.find((p) => p.type === 'Split')?.pattern?.Regex
  if (typeof pattern !== 'string') return PRETOKENIZE_RE
  const js = pattern.replace(
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    "'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])",
  )
  try {
    return new RegExp(js, 'gu')
  } catch {
    return PRETOKENIZE_RE
  }
}

/**
 * GPT-2 byte↔unicode table: printable latin-1 bytes map to themselves,
 * everything else (controls, space, DEL, ...) maps to U+0100+n so every
 * byte-level token is a string of visible, single-code-unit chars.
 */
function buildByteUnicodeMaps(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const byteToChar = new Array<string>(256)
  const charToByte = new Map<string, number>()
  let n = 0
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff)
    const ch = String.fromCharCode(printable ? b : 0x100 + n++)
    byteToChar[b] = ch
    charToByte.set(ch, b)
  }
  return { byteToChar, charToByte }
}

const utf8Encoder = new TextEncoder()

/**
 * Build a ByteLevelTokenizer from a parsed tokenizer.json. Pure and
 * synchronous so unit tests can construct it from a committed vocab file;
 * the browser path (loadByteLevelTokenizer) fetches the same JSON and
 * delegates here. Same interface shape as createTokenizer() in tokenizer.ts
 * so the engine can consume either.
 */
export function createByteLevelTokenizer(json: ByteLevelTokenizerJSON): Tokenizer {
  const vocab = json.model.vocab   // token_str → id
  const { byteToChar, charToByte } = buildByteUnicodeMaps()
  const pretokRe = pretokenizeRegexFrom(json)
  // Qwen3 normalizes NFC; Llama-3 ships normalizer: null (raw text). The
  // field predates this file supporting Llama, so treat "absent" as Qwen3's
  // NFC to keep the committed fixtures' contract.
  const nfc = json.normalizer === undefined || json.normalizer !== null

  // Build reverse vocab: id → token_str. (No Math.max(...values) here —
  // spreading 151k vocab ids overflows the call stack.)
  let maxId = 0
  for (const id of Object.values(vocab)) if (id > maxId) maxId = id
  const idToToken = new Array<string>(maxId + 1)
  for (const [tok, id] of Object.entries(vocab)) idToToken[id] = tok

  // Build merge priority map: "left right" → rank (pairs never contain a raw
  // space — byte-level tokens encode space as Ġ — so " " is a safe joiner).
  const mergeRank = new Map<string, number>()
  const merges = json.model.merges
  for (let i = 0; i < merges.length; i++) {
    const m = merges[i]
    mergeRank.set(typeof m === 'string' ? m : m[0] + ' ' + m[1], i)
  }

  // Added tokens: ALL of them split the input at encode time (special flag
  // only governs decode stripping). None have lstrip/rstrip set.
  const addedTokens = new Map<string, number>()
  const specialIds = new Set<number>()       // special: true  → stripped in decode
  const addedContentById = new Map<number, string>()  // special: false → literal in decode
  for (const at of json.added_tokens ?? []) {
    addedTokens.set(at.content, at.id)
    if (at.special) specialIds.add(at.id)
    else addedContentById.set(at.id, at.content)
  }

  // Special-token split regex, built once. Longest match first so e.g.
  // </tool_response> beats </tool_call>-style prefixes.
  const specialPattern = [...addedTokens.keys()]
    .sort((a, b) => b.length - a.length)
    .map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|')
  const specialSplitRe = specialPattern ? new RegExp(`(${specialPattern})`) : null

  // Qwen3 has no BOS token (tokenizer_config.json bos_token: null); MLC's
  // config re-uses <|endoftext|> as bos_token_id, so mirror that here to
  // keep the Tokenizer interface shape. Llama-3 ships neither ChatML name —
  // fall through to its own specials (<|eot_id|>/<|begin_of_text|>). Nothing
  // engine-side consumes these (spec.stops drives the decode loops); they
  // exist for the Tokenizer interface shape and the unit tests.
  const eosId = addedTokens.get('<|im_end|>') ?? addedTokens.get('<|eot_id|>') ?? 151645
  const bosId = addedTokens.get('<|endoftext|>') ?? addedTokens.get('<|begin_of_text|>') ?? 151643
  // ChatML: [<|im_end|>, <|endoftext|>]; Llama-3: [<|eot_id|>, <|end_of_text|>].
  const stopIds = [eosId, addedTokens.get('<|end_of_text|>') ?? bosId]

  // --------------------------------------------------------
  // BPE core: same greedy lowest-rank merge loop as tokenizer.ts, plus a
  // word cache — byte-level BPE re-tokenizes the same small pre-token
  // pieces (" the", " a", ...) constantly.
  // --------------------------------------------------------
  const bpeCache = new Map<string, string[]>()

  function bpe(word: string): string[] {
    const cached = bpeCache.get(word)
    if (cached !== undefined) return cached

    let symbols = [...word]
    while (symbols.length > 1) {
      let bestRank = Infinity
      let bestIdx = -1

      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = mergeRank.get(symbols[i] + ' ' + symbols[i + 1])
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank
          bestIdx = i
        }
      }

      if (bestIdx === -1) break

      // Merge symbols[bestIdx] and symbols[bestIdx+1]
      const merged = symbols[bestIdx] + symbols[bestIdx + 1]
      symbols = [...symbols.slice(0, bestIdx), merged, ...symbols.slice(bestIdx + 2)]
    }

    bpeCache.set(word, symbols)
    return symbols
  }

  // --------------------------------------------------------
  // Pre-tokenizer: Split(regex, Isolated) — every regex match is its own
  // piece. The pattern covers all possible text, but preserve gaps as
  // pieces anyway so behavior is exactly HF's Isolated split.
  // --------------------------------------------------------
  function pretokenize(text: string): string[] {
    const pieces: string[] = []
    let last = 0
    for (const m of text.matchAll(pretokRe)) {
      if (m.index > last) pieces.push(text.slice(last, m.index))
      pieces.push(m[0])
      last = m.index + m[0].length
    }
    if (last < text.length) pieces.push(text.slice(last))
    return pieces
  }

  // --------------------------------------------------------
  // Encode: text → token IDs (see the pipeline description at the top of
  // this file; every step below is pinned by the fixture tests).
  // --------------------------------------------------------
  function encode(text: string): number[] {
    if (text.length === 0) return []

    // Step 1: with a capture group, split() alternates text and separator
    // parts: even indices are plain text, odd indices are added tokens.
    // Added tokens are normalized:false, so this runs before NFC.
    const parts = specialSplitRe ? text.split(specialSplitRe) : [text]

    const result: number[] = []
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]
      if (!part) continue

      if (i % 2 === 1) {
        const addedId = addedTokens.get(part)
        if (addedId !== undefined) {
          result.push(addedId)
          continue
        }
      }

      // Steps 2-4: normalize (NFC when the JSON declares it — Llama-3 ships
      // normalizer: null), regex-split, byte→unicode map, BPE.
      for (const piece of pretokenize(nfc ? part.normalize('NFC') : part)) {
        let mapped = ''
        for (const byte of utf8Encoder.encode(piece)) mapped += byteToChar[byte]

        for (const tok of bpe(mapped)) {
          const id = vocab[tok]
          // All 256 byte chars are in the vocab, so unmerged single chars
          // always resolve; HF drops unknowns (unk_token: null) — mirror it.
          if (id !== undefined) result.push(id)
        }
      }
    }

    return result
  }

  // --------------------------------------------------------
  // Decode: token IDs → text
  //
  // ByteLevel decoder: concat token strings, reverse the byte↔unicode map,
  // UTF-8 decode. A codepoint's bytes can span token boundaries, so bytes
  // are buffered and flushed as one UTF-8 string (with U+FFFD replacement
  // on invalid sequences — same as HF).
  //
  // `idToBytes` is precomputed so the hot loop (streaming chat re-decodes
  // the full id array per generated token) is one array lookup per id.
  // Special tokens are stripped; non-special added tokens (<think>, ...)
  // are emitted literally — both matching HF skip_special_tokens: true.
  // --------------------------------------------------------
  const idToBytes = new Array<Uint8Array | null>(idToToken.length).fill(null)
  for (let id = 0; id < idToToken.length; id++) {
    const tok = idToToken[id]
    if (!tok) continue
    const bytes = new Uint8Array(tok.length)
    for (let i = 0; i < tok.length; i++) bytes[i] = charToByte.get(tok[i]) ?? 0
    idToBytes[id] = bytes
  }

  const utf8Decoder = new TextDecoder('utf-8', { fatal: false })
  function decode(ids: number[] | Int32Array): string {
    let text = ''
    let pendingBytes: number[] = []
    const flush = () => {
      if (pendingBytes.length === 0) return
      text += utf8Decoder.decode(Uint8Array.from(pendingBytes))
      pendingBytes = []
    }
    for (const id of ids) {
      if (id < 0 || specialIds.has(id)) { flush(); continue }
      const added = addedContentById.get(id)
      if (added !== undefined) { flush(); text += added; continue }
      const bytes = idToBytes[id]
      if (!bytes) continue
      for (const b of bytes) pendingBytes.push(b)
    }
    flush()
    return text
  }

  return { encode, decode, bosId, eosId, stopIds }
}

export async function loadByteLevelTokenizer(
  onProgress?: (msg: string) => void,
  baseUrl: string = QWEN3_MODEL_BASE,
): Promise<Tokenizer> {
  onProgress?.('Loading tokenizer.json...')
  const url = baseUrl + 'tokenizer.json'
  const json: ByteLevelTokenizerJSON = JSON.parse(await fetchText(url))
  const tokenizer = createByteLevelTokenizer(json)
  onProgress?.('Tokenizer ready')
  return tokenizer
}

// ============================================================
// Chat template for Qwen3 (ChatML)
//
// <|im_start|>system\n{system}<|im_end|>\n
// <|im_start|>user\n{user}<|im_end|>\n
// <|im_start|>assistant\n            ← generation prompt
//
// Ground truth: tokenizer_config.json chat_template (non-tools path) and
// mlc-chat-config.json conv_template "qwen3". Callers wanting the model's
// default persona should prepend a system message with
// QWEN3_DEFAULT_SYSTEM_MESSAGE — neither template injects one into the
// token stream automatically.
//
// Thinking: enabled by default (no suffix). `{ thinking: false }` appends
// the template's `enable_thinking is false` branch verbatim:
// "<think>\n\n</think>\n\n".
// ============================================================

/**
 * Which ChatML generation's rule for the past-turn <think> block applies.
 *   'qwen3'  — Qwen3-4B, Qwen3-30B-A3B: only a TRAILING assistant turn (their
 *              extra `loop.last or reasoning_content` gate, and we carry no
 *              reasoning_content).
 *   'qwen35' — Qwen3.5-4B/9B, Qwen3.6-35B-A3B: every assistant turn in the
 *              current round (`loop.index0 > ns.last_query_index`).
 *   'qwen38' — Qwen3.8-27B: EVERY assistant turn. Its condition opens with
 *              `preserve_thinking is undefined or ...`, and nothing defines
 *              preserve_thinking, so the branch is always taken. Deliberate,
 *              not a missing guard — and confirmed by rendering the template
 *              through transformers three ways (undefined / false / true).
 *
 * Verified in each checkpoint's own tokenizer_config.json or
 * chat_template.jinja; scripts/render-diff.py diffs our output against them.
 */
export type ChatMLGeneration = 'qwen3' | 'qwen35' | 'qwen38'

/** MLC conv_template "qwen3" default system message. */
export const QWEN3_DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'

/**
 * Index of the last REAL user query — jinja's `ns.last_query_index`.
 *
 * The templates walk the messages backwards and stop at the first user turn
 * that is not entirely a <tool_response> block, because a tool result rides in
 * a user turn and is not a new question. Everything after that index is the
 * CURRENT tool-calling round.
 *
 * Defaults to the last index when there is no real query, matching jinja's
 * initial value — under which no turn satisfies `index > last_query_index` and
 * no think block is emitted. (jinja raises there instead; a chat surface
 * rendering a partial conversation should not.)
 */
function lastQueryIndex(messages: ReadonlyArray<{ role: string; content: string }>): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role !== 'user') continue
    const c = messages[i].content.trim()
    if (c.startsWith('<tool_response>') && c.endsWith('</tool_response>')) continue
    return i
  }
  return messages.length - 1
}

/** jinja `x.lstrip('\n')` / `rstrip` / `strip` — NEWLINES ONLY, not whitespace.
 *  Distinct from `|trim`, which strips all whitespace; the templates use both
 *  and they are not interchangeable. */
const lstripNl = (t: string) => t.replace(/^\n+/, '')
const rstripNl = (t: string) => t.replace(/\n+$/, '')

/**
 * The templates' own handling of an assistant turn that already contains a
 * `</think>` block: the reasoning is split back out of the content, so it can
 * be re-emitted (or dropped) according to the past-turn rule rather than
 * passed through inline.
 *
 * We pass no `reasoning_content` field, so only the `'</think>' in content`
 * branch can fire — and it fires on ordinary text. The chat page stores the
 * model's RAW output as history (chat-flow.ts), and a room host renders a
 * remote GUEST's history verbatim (room-host.ts), so a turn containing the
 * literal `</think>` is reachable without anything unusual happening.
 */
function splitReasoning(content: string): { content: string; reasoning: string } {
  if (!content.includes('</think>')) return { content, reasoning: '' }
  const parts = content.split('</think>')
  const before = parts[0]
  return {
    content: lstripNl(parts[parts.length - 1]),
    reasoning: lstripNl(rstripNl(before).split('<think>').pop() ?? ''),
  }
}

export function buildChatPrompt(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  // Only encode() is needed, so the SPM Tokenizer shape (no stopIds) is
  // accepted too — the model-select factory routes either kind here.
  tokenizer: Pick<Tokenizer, 'encode'>,
  opts?: {
    thinking?: boolean
    generation?: ChatMLGeneration
    /** Out-param: receives the token index where the generation prompt starts.
     *  An out-param rather than a changed return type because five call sites
     *  and four sibling builders return `number[]`, and only the callers that
     *  drive multi-turn conversations need this. */
    split?: { genStart: number }
  },
): number[] {
  // Which past assistant turns carry the empty <think> block. This was
  // UNCONDITIONAL — every assistant turn got one — and that is not what any
  // Qwen template does. Measured on Qwen3.6 at ~24k tokens: 286 spurious blocks
  // through the history, 5,434 characters the model was never trained to see,
  // and a tool-calling loop that read the right files, computed the right
  // answer and then replied in prose instead of calling the tool. Both rules
  // below are read off the checkpoints' own tokenizer_config.json.
  const gen = opts?.generation ?? 'qwen3'
  const q = lastQueryIndex(messages)
  // Qwen3.5 and later run every message's content through jinja's `|trim`;
  // the Qwen3-era template interpolates `message.content` verbatim. Trimming
  // on the wrong one moves whitespace at a turn boundary, which no test would
  // see and the model answers slightly off distribution for.
  const trims = gen !== 'qwen3'
  let text = ''
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    // The block belongs to the CURRENT round only. It is not gated on thinking
    // mode: the templates emit it from the index rule alone, and `enable_thinking`
    // reaches only the generation prompt below.
    //
    // Qwen3-era additionally requires `loop.last or reasoning_content`, and we
    // carry no reasoning_content, so in practice a past turn there gets nothing
    // — only a trailing assistant turn does. Qwen3.5 and later dropped that
    // gate, which is why the two generations are separate ids.
    const raw = trims ? msg.content.trim() : msg.content
    if (msg.role !== 'assistant') {
      text += `<|im_start|>${msg.role}\n${raw}<|im_end|>\n`
      continue
    }
    // Qwen3.8 has NO split: its assistant branch only reads a `reasoning_content`
    // FIELD, which we never send, so a `</think>` inside the text passes
    // through whole. Qwen3 and Qwen3.5/3.6 split it back out. Splitting on 3.8
    // cuts the reply at the tag and re-emits half of it as reasoning.
    const split = gen === 'qwen38' ? { content: raw, reasoning: '' } : splitReasoning(raw)
    const inRound = i > q
    // Qwen3 needs `loop.last or reasoning_content`; 3.5/3.6 dropped that gate;
    // 3.8 opens its condition with `preserve_thinking is undefined`, which
    // nothing defines, so it always fires.
    const think = gen === 'qwen38'
      || (inRound && (gen === 'qwen35' || i === messages.length - 1 || !!split.reasoning))
    // Qwen3 emits `reasoning_content.strip('\n')` + `content.lstrip('\n')`;
    // 3.5+ emits a `|trim`-ed reasoning and the content as-is. Both already
    // stripped above, so these only differ on which stripper ran.
    const reasoning = gen === 'qwen3' ? lstripNl(rstripNl(split.reasoning)) : split.reasoning.trim()
    const body = think
      ? `<think>\n${reasoning}\n</think>\n\n${gen === 'qwen3' ? lstripNl(split.content) : split.content}`
      : split.content
    text += `<|im_start|>${msg.role}\n${body}<|im_end|>\n`
  }
  // Encode the history and the generation prompt SEPARATELY, so the boundary
  // between them is a known token index. That is exact rather than an estimate:
  // the generation prompt opens with `<|im_start|>`, an ADDED token, and byte-
  // level BPE never merges across one — so the concatenation is identical to
  // encoding the whole string. (Verified in tokenizer-bpe.test.ts against the
  // single-encode path for every template this builder serves.)
  let genPrompt = '<|im_start|>assistant\n'
  if (opts?.thinking === false) genPrompt += '<think>\n\n</think>\n\n'
  const history = tokenizer.encode(text)
  // Where the NEXT turn will diverge: it re-renders this turn's reply in place
  // of this generation prompt. The engine takes its GDN rewind snapshot here.
  if (opts?.split) opts.split.genStart = history.length
  return [...history, ...tokenizer.encode(genPrompt)]
}

// ============================================================
// Chat template for Llama-3 (header style)
//
// <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
// Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n{system}<|eot_id|>
// <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
// <|start_header_id|>assistant<|end_header_id|>\n\n     ← generation prompt
//
// Ground truth: tokenizer_config.json chat_template of
// mlx-community/Llama-3.2-1B-Instruct-4bit (the no-tools path), pinned
// TOKEN-exact against mlx_lm's apply_chat_template by
// tests/tokenizer/tokenizer-llama3.test.ts. Two behaviors that are easy to
// get wrong: the template ALWAYS emits a system block — the knowledge-cutoff
// line plus today's date (strftime_now "%d %b %Y"), with the caller's system
// message (if any) appended after the blank line — and every message content
// is |trim'ed. Stop ids: <|eot_id|> (128009), <|end_of_text|> (128001).
// ============================================================

const LLAMA3_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** Python strftime("%d %b %Y") in the C locale — what the template's
 *  strftime_now emits, e.g. "06 Aug 2026" (day zero-padded). */
export function llama3DateString(d: Date = new Date()): string {
  return `${String(d.getDate()).padStart(2, '0')} ${LLAMA3_MONTHS[d.getMonth()]} ${d.getFullYear()}`
}

/**
 * DeepSeek-V2's chat template. Plain prose turns rather than header tokens:
 *
 *   <bos>{system}\n\nUser: {content}\n\nAssistant: {content}<eos>Assistant:
 *
 * Two details that are easy to get wrong and impossible to see afterwards: a
 * system message is emitted with NO label at all (the template writes only its
 * content), and the trailing generation prompt is `Assistant:` with no space —
 * the model has learned to produce the leading space itself, so adding one
 * puts it off-distribution from its first token.
 *
 * Unlike the Llama-3 builder, content is NOT trimmed: the template interpolates
 * `message['content']` verbatim.
 */
export function buildDeepSeekChatPrompt(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  tokenizer: Pick<Tokenizer, 'encode'>,
  opts?: { bosToken?: string; eosToken?: string },
): number[] {
  const bos = opts?.bosToken ?? '<｜begin▁of▁sentence｜>'
  const eos = opts?.eosToken ?? '<｜end▁of▁sentence｜>'
  let text = bos
  for (const msg of messages) {
    if (msg.role === 'user') text += `User: ${msg.content}\n\n`
    else if (msg.role === 'assistant') text += `Assistant: ${msg.content}${eos}`
    else if (msg.role === 'system') text += `${msg.content}\n\n`
  }
  text += 'Assistant:'
  return tokenizer.encode(text)
}

export function buildLlama3ChatPrompt(
  // 'ipython' is a real Llama-3 role, not a leak: renderToolResults returns
  // tool results in ipython turns and the loop below renders any role as its
  // own header, so this only widens the type to what the body already did.
  messages: Array<{ role: 'system' | 'user' | 'assistant' | 'ipython'; content: string }>,
  tokenizer: Pick<Tokenizer, 'encode'>,
  // dateString exists so tests can pin the fixture date; live chat renders
  // today's, exactly like mlx_lm running the template with strftime_now.
  opts?: { dateString?: string },
): number[] {
  const date = opts?.dateString ?? llama3DateString()
  // The template extracts a leading system message into the fixed system
  // block; everything else renders as header turns.
  let rest = messages
  let system = ''
  if (rest[0]?.role === 'system') {
    system = rest[0].content.trim()
    rest = rest.slice(1)
  }
  let text = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
  text += `Cutting Knowledge Date: December 2023\nToday Date: ${date}\n\n`
  text += `${system}<|eot_id|>`
  for (const msg of rest) {
    text += `<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content.trim()}<|eot_id|>`
  }
  text += '<|start_header_id|>assistant<|end_header_id|>\n\n'
  return tokenizer.encode(text)
}

// ============================================================
// Chat template for the Mistral / Llama-2 `[INST]` family
//
// ONE family, THREE published spacings. Copying any of them onto the others
// moves a space at every turn boundary — invisible in the transcript, and the
// model answers slightly off distribution forever:
//
//   Mistral-7B-Instruct-v0.3   <s> [INST] u [/INST] a </s> [INST] ...
//   Mistral-Nemo-2407          <s>[INST] u[/INST] a</s>[INST] ...
//   Ministral-8B-2410          <s>[INST]u[/INST] a</s>[INST] ...
//
// Ground truth: the chat_template in each repo's own tokenizer_config.json
// (mlx-community/{Mistral-7B-Instruct-v0.3,Mistral-Nemo-Instruct-2407,
// Ministral-8B-Instruct-2410}-4bit; v0.3 ships a NAMED list and the "default"
// entry is the one without tools). Every case is pinned in
// tests/unit/chat-template-mistral.test.ts.
//
// None of the three emits a generation prompt: the rendered string ends after
// `[/INST]` and the model continues from there. `add_generation_prompt` does
// not appear anywhere in these templates — there is no assistant header to
// append, unlike every other builder in this file.
//
// A system message is not a turn here; it is folded into ONE user turn's
// content, joined with "\n\n". WHICH turn differs, and it decides whether
// cross-turn KV reuse works at all:
//
//   - v1 folds into the FIRST user turn, so past turns re-render identically
//     and each prompt extends the last (the property tokenizer-bpe.ts:446
//     claims for ChatML).
//   - 2407/2410 fold into the LAST, so every new turn MOVES the system prompt
//     and the re-rendered transcript stops being a prefix of the previous one:
//     the shared prefix collapses to `<s>[INST] `. That is the vendor's own
//     template, so it is what gets emitted, and chat.ts always sends a system
//     message — those two models pay full prefill every turn. Measured in the
//     test rather than left as a comment.
//
// v0.3's jinja has no system branch at all — it calls raise_exception. The
// fold-into-first comes from Mistral's own tokenizer instead: mistral_common
// InstructTokenizerV1.encode_user_message does
// `if is_first and system_prompt: content = system_prompt + "\n\n" + content`.
//
// The alternation guard (raise_exception when roles do not alternate) is not
// ported — chat.ts cannot produce a non-alternating history.
// ============================================================

export type MistralTemplateVariant = 'v1' | 'nemo' | 'ministral'

export function buildMistralChatPrompt(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  tokenizer: Pick<Tokenizer, 'encode'>,
  opts?: { variant?: MistralTemplateVariant; bosToken?: string; eosToken?: string },
): number[] {
  const variant = opts?.variant ?? 'v1'
  const bos = opts?.bosToken ?? '<s>'
  const eos = opts?.eosToken ?? '</s>'

  let rest = messages
  let system = ''
  if (rest[0]?.role === 'system') {
    system = rest[0].content
    rest = rest.slice(1)
  }
  // jinja's `is_first` (v1) vs `loop.last` (2407/2410). The index is resolved
  // once here; the fold only lands if that message is a user turn, which is
  // also what the templates do (the branch sits inside their user case).
  const foldAt = system === '' ? -1
    : variant === 'v1' ? rest.findIndex((m) => m.role === 'user')
    : rest.length - 1

  let text = bos
  for (let i = 0; i < rest.length; i++) {
    const msg = rest[i]
    if (msg.role === 'assistant') {
      // 2410 is the only one that trims the assistant content; and only v1
      // puts a space between it and the eos token.
      const reply = variant === 'ministral' ? msg.content.trim() : msg.content
      text += variant === 'v1' ? ` ${reply} ${eos}` : ` ${reply}${eos}`
      continue
    }
    const content = i === foldAt ? `${system}\n\n${msg.content}` : msg.content
    text += variant === 'v1' ? ` [INST] ${content} [/INST]`
      : variant === 'nemo' ? `[INST] ${content}[/INST]`
      : `[INST]${content}[/INST]`
  }
  return tokenizer.encode(text)
}

// ============================================================
// Chat template for the Tulu-3 lineage (OLMo-2, OLMoE, Falcon3)
//
//   {bos}<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n
//
// Ground truth: the chat_template in mlx-community/OLMo-2-1124-7B-Instruct-4bit
// and mlx-community/Falcon3-7B-Instruct-4bit tokenizer_config.json. The two are
// character-identical except that OLMo-2 (and OLMoE) open with
// `{{ bos_token }}` and Falcon3 does not — its bos_token is null — which is
// why `bosToken` exists and defaults to none. Pinned in
// tests/unit/chat-template-tulu.test.ts.
//
// Details the template hides:
//
//   - the eos token is INSIDE the transcript, after each assistant turn, and a
//     newline follows it — except when that turn is the LAST message, where
//     the template drops the newline (`{% if not loop.last %}`).
//   - the generation prompt is emitted from INSIDE the loop
//     (`loop.last and add_generation_prompt`), so an empty message list renders
//     as bare bos with no `<|assistant|>` at all. Mirrored by emitting it on
//     the last iteration rather than after the loop.
//   - content is never trimmed (Llama-3's template trims; this one does not).
//   - eos is not constant across the family: OLMo-2 and Falcon3 use
//     `<|endoftext|>`, but OLMoE-1B-7B-0125 uses `|||IP_ADDRESS|||` — the OLMo
//     tokenizer's PII placeholder, doubling as bos and eos. Hardcoding
//     `<|endoftext|>` would put a literal, unrelated token in every OLMoE
//     transcript.
//
// Unlike the Mistral family this one is append-only with or without a system
// message: the system turn is rendered in place and never moves.
// ============================================================

export function buildTuluChatPrompt(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  tokenizer: Pick<Tokenizer, 'encode'>,
  opts?: { bosToken?: string; eosToken?: string },
): number[] {
  const eos = opts?.eosToken ?? '<|endoftext|>'
  let text = opts?.bosToken ?? ''
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    const last = i === messages.length - 1
    text += msg.role === 'assistant'
      ? `<|assistant|>\n${msg.content}${eos}${last ? '' : '\n'}`
      : `<|${msg.role}|>\n${msg.content}\n`
    if (last) text += '<|assistant|>\n'
  }
  return tokenizer.encode(text)
}
