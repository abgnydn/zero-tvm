/**
 * TOKENIZER — Load and run BPE tokenizer from tokenizer.json.
 *
 * Implements BPE encode + decode for Phi-3 Mini (SentencePiece-style).
 * Loads tokenizer.json from browser cache or HuggingFace directly.
 *
 * Phi-3 Mini ships a Llama-style SentencePiece-as-BPE tokenizer with
 * `legacy: false` in tokenizer_config.json. The reference pipeline
 * (HuggingFace LlamaTokenizer, non-legacy path) is:
 *
 *   1. Prepend a literal "▁" to the whole raw text, after replacing any
 *      pre-existing "▁" characters with " ".
 *   2. Split around added (special) tokens. Tokens with rstrip=true (`</s>`
 *      and every <|...|> marker) swallow ALL whitespace that follows them —
 *      this is why the chat template's "\n" after <|user|> etc. does NOT
 *      produce a <0x0A> token.
 *   3. Per text section: replace every " " with "▁" (Metaspace,
 *      prepend_scheme "first" — only section 0 gets a prefix, and step 1
 *      already provided it). No word splitting — BPE runs over the whole
 *      section, so "    " can merge into "▁▁▁▁"-style tokens and \n / \t
 *      survive intact.
 *   4. BPE with byte_fallback: symbols with no vocab entry are emitted as
 *      <0xHH> byte tokens (newline = <0x0A>, emoji = 4 byte tokens, ...).
 *   5. If the first token came out as a bare "▁" and the second is an added
 *      token, drop the "▁" (upstream legacy=false quirk).
 *
 * Decode mirrors the shipped decoder chain: Replace "▁" → " ", ByteFallback,
 * Fuse, then Strip exactly ONE leading space.
 *
 * Correctness is pinned by tests/tokenizer/tokenizer.test.ts against
 * fixtures generated with the HuggingFace reference implementation
 * (scripts/gen-tokenizer-fixtures.mjs).
 */

import { PHI3_MODEL_BASE } from './weight-loader.js'

// ============================================================
// tokenizer.json types (subset we need)
// ============================================================

export interface TokenizerJSON {
  model: {
    type: string
    vocab: Record<string, number>
    merges: string[]
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
  pre_tokenizer?: {
    type: string
    prepend_scheme?: string
    replacement?: string
  } | null
}

// ============================================================
// Fetch (with cache fallback — same helper pattern as weight-loader)
// ============================================================

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
// BPE implementation
// ============================================================

export interface Tokenizer {
  encode(text: string): number[]
  decode(ids: number[] | Int32Array): string
  eosId: number
  bosId: number
}

// The metaspace character (▁, U+2581) is used as space prefix in SentencePiece
const METASPACE = '▁'
// Hoisted so decode() doesn't build a RegExp per call (streaming chat decodes
// the full id array per generated token).
const METASPACE_RE = new RegExp(METASPACE, 'g')
const BYTE_FALLBACK_RE = /^<0x([0-9A-Fa-f]{2})>$/
const utf8Encoder = new TextEncoder()

/**
 * Build a Tokenizer from a parsed tokenizer.json. Pure and synchronous so
 * unit tests can construct it from a committed vocab file; the browser path
 * (loadTokenizer) fetches the same JSON the app ships and delegates here.
 */
export function createTokenizer(json: TokenizerJSON): Tokenizer {
  const vocab = json.model.vocab   // token_str → id
  const merges = json.model.merges  // ["tok1 tok2", ...]

  // Build reverse vocab: id → token_str
  const idToToken = new Array<string>(Math.max(...Object.values(vocab)) + 1)
  for (const [tok, id] of Object.entries(vocab)) idToToken[id] = tok

  // Override with added_tokens (special tokens)
  for (const at of json.added_tokens ?? []) idToToken[at.id] = at.content

  // Build merge priority map: "tok1 tok2" → rank
  const mergeRank = new Map<string, number>()
  for (let i = 0; i < merges.length; i++) mergeRank.set(merges[i], i)

  // Added tokens: content → {id, lstrip, rstrip} for encode-time splitting.
  const addedTokens = new Map<string, { id: number; lstrip: boolean; rstrip: boolean }>()
  const addedTokenIds = new Set<number>()
  for (const at of json.added_tokens ?? []) {
    addedTokens.set(at.content, { id: at.id, lstrip: at.lstrip, rstrip: at.rstrip })
    addedTokenIds.add(at.id)
  }

  // Special-token split regex, built ONCE (used to be rebuilt per encode()
  // call). Longest match first so e.g. <|placeholder1|> beats <|user|>.
  const specialPattern = [...addedTokens.keys()]
    .sort((a, b) => b.length - a.length)
    .map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    .join('|')
  const specialSplitRe = specialPattern ? new RegExp(`(${specialPattern})`) : null

  const bosId = vocab['<s>'] ?? 1
  const eosId = vocab['</s>'] ?? 2

  // Byte-fallback lookup: byte value → <0xHH> token id.
  const unkId = vocab['<unk>'] ?? 0
  const byteToId = new Int32Array(256).fill(unkId)
  for (let b = 0; b < 256; b++) {
    const id = vocab[`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`]
    if (id !== undefined) byteToId[b] = id
  }

  // Precomputed decode tables — built once at load time so the streaming
  // decode() hot loop is branchless per id:
  //   byteFallback[id] = byte value  (0..255) for <0xHH> tokens, else undefined
  //   controlIds       = ids whose surface form is a chat-template marker
  //                      (`<s>`, `</s>`, `<pad>`, `<|...|>`), filtered from output.
  const byteFallback = new Map<number, number>()
  const controlIds = new Set<number>()
  for (let id = 0; id < idToToken.length; id++) {
    const tok = idToToken[id]
    if (!tok) continue
    const m = BYTE_FALLBACK_RE.exec(tok)
    if (m) { byteFallback.set(id, parseInt(m[1], 16)); continue }
    if (tok === '<s>' || tok === '</s>' || tok === '<pad>') { controlIds.add(id); continue }
    if (tok.startsWith('<|') && tok.endsWith('|>')) controlIds.add(id)
  }

  // --------------------------------------------------------
  // BPE core: given a list of symbols, apply merges until no more apply
  // --------------------------------------------------------
  function bpe(symbols: string[]): string[] {
    if (symbols.length <= 1) return symbols

    while (true) {
      let bestRank = Infinity
      let bestIdx = -1

      for (let i = 0; i < symbols.length - 1; i++) {
        const pair = symbols[i] + ' ' + symbols[i + 1]
        const rank = mergeRank.get(pair)
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

    return symbols
  }

  // --------------------------------------------------------
  // Encode: text → token IDs (see the pipeline description at the top of
  // this file; every step below is pinned by the fixture tests).
  // --------------------------------------------------------
  const metaspaceId = vocab[METASPACE]

  function encode(text: string): number[] {
    if (text.length === 0) return []

    // Step 1: SPM non-legacy prefix — literal ▁ on the raw text, existing
    // ▁ chars downgraded to plain spaces.
    const full = METASPACE + text.replaceAll(METASPACE, ' ')

    // Step 2: with a capture group, split() alternates text and separator
    // parts: even indices are plain text, odd indices are special tokens.
    const parts = specialSplitRe ? full.split(specialSplitRe) : [full]

    // lstrip/rstrip: special tokens eat whitespace off their neighbors.
    for (let i = 1; i < parts.length; i += 2) {
      const at = addedTokens.get(parts[i])
      if (!at) continue
      if (at.lstrip && i > 0) parts[i - 1] = parts[i - 1].trimEnd()
      if (at.rstrip && i < parts.length - 1) parts[i + 1] = parts[i + 1].trimStart()
    }

    const result: number[] = []
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]
      if (!part) continue

      if (i % 2 === 1) {
        const special = addedTokens.get(part)
        if (special !== undefined) {
          result.push(special.id)
          continue
        }
      }

      // Step 3: Metaspace — replace spaces only. Newlines/tabs are NOT
      // spaces; they pass through to byte-fallback.
      const normalized = part.replaceAll(' ', METASPACE)

      // Step 4: BPE over the whole section (Unicode-aware char split),
      // then map symbols → ids with byte fallback.
      for (const tok of bpe([...normalized])) {
        const id = vocab[tok]
        if (id !== undefined) {
          result.push(id)
        } else {
          // Byte fallback: emit <0xHH> ids for each UTF-8 byte. Unmerged
          // symbols are always single chars, so this only ever sees chars
          // that have no vocab entry (\n, \t, exotic codepoints, ...).
          for (const byte of utf8Encoder.encode(tok)) {
            result.push(byteToId[byte])
          }
        }
      }
    }

    // Step 5: drop the lone leading ▁ the prefix created when the text
    // begins with a special token ("▁<|user|>..." → "<|user|>...").
    if (result.length > 1 && result[0] === metaspaceId && addedTokenIds.has(result[1])) {
      result.shift()
    }

    return result
  }

  // --------------------------------------------------------
  // Decode: token IDs → text
  //
  // Phi-3 (SentencePiece) uses byte-fallback tokens of the form <0xHH> for
  // raw bytes (newline 0x0A, UTF-8 continuation bytes for characters that
  // weren't learned as merges, etc.). We buffer consecutive byte tokens and
  // flush them as a UTF-8 string — emitting them individually would produce
  // literal `<0x0A>` in the output and corrupt multi-byte codepoints.
  //
  // `byteFallback` and `controlIds` are precomputed above so the hot loop is
  // two map lookups per id (no regex, no string prefix checks).
  //
  // The tail matches the reference decoder chain: Replace ▁ → " ", then
  // Strip exactly ONE leading space (the one the normalizer prepended) —
  // NOT trimStart(), which would destroy intentional leading whitespace.
  // --------------------------------------------------------
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
      if (id < 0 || controlIds.has(id)) { flush(); continue }
      const byte = byteFallback.get(id)
      if (byte !== undefined) { pendingBytes.push(byte); continue }
      const tok = idToToken[id]
      if (!tok) continue
      flush()
      text += tok
    }
    flush()
    text = text.replace(METASPACE_RE, ' ')
    return text.startsWith(' ') ? text.slice(1) : text
  }

  return { encode, decode, bosId, eosId }
}

export async function loadTokenizer(onProgress?: (msg: string) => void): Promise<Tokenizer> {
  onProgress?.('Loading tokenizer.json...')
  const url = PHI3_MODEL_BASE + 'tokenizer.json'
  const json: TokenizerJSON = JSON.parse(await fetchText(url))
  const tokenizer = createTokenizer(json)
  onProgress?.('Tokenizer ready')
  return tokenizer
}

// ============================================================
// Chat template for Phi-3
//
// <|system|>...<|end|>\n<|user|>...<|end|>\n<|assistant|>\n
// ============================================================

export function buildChatPrompt(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  tokenizer: Tokenizer
): number[] {
  let text = ''
  for (const msg of messages) {
    if (msg.role === 'system') {
      text += `<|system|>\n${msg.content}<|end|>\n`
    } else if (msg.role === 'user') {
      text += `<|user|>\n${msg.content}<|end|>\n`
    } else {
      text += `<|assistant|>\n${msg.content}<|end|>\n`
    }
  }
  text += '<|assistant|>\n'
  return tokenizer.encode(text)
}
