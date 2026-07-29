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

// ============================================================
// tokenizer.json types (subset we need)
// ============================================================

export interface ByteLevelTokenizerJSON {
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

export interface ByteLevelTokenizer {
  encode(text: string): number[]
  decode(ids: number[] | Int32Array): string
  eosId: number
  bosId: number
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
export function createByteLevelTokenizer(json: ByteLevelTokenizerJSON): ByteLevelTokenizer {
  const vocab = json.model.vocab   // token_str → id
  const { byteToChar, charToByte } = buildByteUnicodeMaps()

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
  // keep the Tokenizer interface shape.
  const eosId = addedTokens.get('<|im_end|>') ?? 151645
  const bosId = addedTokens.get('<|endoftext|>') ?? 151643
  const stopIds = [eosId, bosId]

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
    for (const m of text.matchAll(PRETOKENIZE_RE)) {
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

      // Steps 2-4: NFC normalize, regex-split, byte→unicode map, BPE.
      for (const piece of pretokenize(part.normalize('NFC'))) {
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
): Promise<ByteLevelTokenizer> {
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

/** MLC conv_template "qwen3" default system message. */
export const QWEN3_DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'

export function buildChatPrompt(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  // Only encode() is needed, so the SPM Tokenizer shape (no stopIds) is
  // accepted too — the model-select factory routes either kind here.
  tokenizer: Pick<ByteLevelTokenizer, 'encode'>,
  opts?: { thinking?: boolean }
): number[] {
  let text = ''
  for (const msg of messages) {
    // Non-thinking mode: past assistant turns are re-rendered WITH the empty
    // <think> block. The generation actually produced those tokens (the
    // suffix below was part of the generation prompt the model continued
    // from), so this is the faithful transcript — and it makes each turn's
    // prompt an exact token-level extension of the previous turn's absorbed
    // sequence, which is what the engine's cross-turn prefix reuse matches
    // against (engine-core.ts computeReuseStart).
    const content = opts?.thinking === false && msg.role === 'assistant'
      ? `<think>\n\n</think>\n\n${msg.content}`
      : msg.content
    text += `<|im_start|>${msg.role}\n${content}<|im_end|>\n`
  }
  text += '<|im_start|>assistant\n'
  if (opts?.thinking === false) text += '<think>\n\n</think>\n\n'
  return tokenizer.encode(text)
}
