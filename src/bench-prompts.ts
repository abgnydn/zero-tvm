/**
 * Shared bench prompt material.
 *
 * `BENCH_PROMPT` is the one-line prompt every baseline page and the engine's
 * own `window.bench()` already used; it lives here now only so the TTFT sweep
 * can reuse the identical string.
 *
 * `buildPromptOfLength()` is the new piece: it grows a filler user message
 * until the FULLY TEMPLATED prompt lands within `tolerance` tokens of a
 * target length. Both halves of the TTFT-vs-prompt-length experiment
 * (src/zero-tvm/bench-console.ts and src/tjs-bench/main.ts) call it with
 * their own encode/template functions, so the two engines see the same text
 * at the same measured length instead of two hand-rolled approximations.
 *
 * The filler is deliberately repetitive prose. Prefill cost is a function of
 * sequence length, not of content, so repetition is harmless here — and it
 * makes the corpus deterministic and short enough to read. Each block is
 * numbered so the text is not literally identical block-to-block.
 */

export const BENCH_PROMPT = 'Write a four-sentence explanation of how photosynthesis works.'

const FILLER_BLOCK =
  'Section %d. The transport of water through a plant begins at the root hairs, ' +
  'where dissolved minerals enter by active transport and water follows by osmosis. ' +
  'Cohesion between water molecules and adhesion to the xylem walls maintain a ' +
  'continuous column that is pulled upward as vapour escapes from the stomata. ' +
  'The rate of that escape depends on humidity, temperature, and wind, so a plant ' +
  'that closes its stomata to conserve water also limits the carbon dioxide ' +
  'available to its chloroplasts. '

/** Deterministic filler text of `n` numbered blocks. */
export function filler(n: number): string {
  let out = ''
  for (let i = 1; i <= n; i++) out += FILLER_BLOCK.replace('%d', String(i))
  return out
}

/**
 * Build a chat prompt whose templated token count is close to `targetTokens`.
 *
 * `encodeTemplated` must apply the model's own chat template to the given user
 * message and return the resulting token ids — i.e. exactly what the engine
 * would prefill. That keeps the template overhead inside the measured length
 * rather than beside it, which matters most at the short end (a 64-token
 * target is mostly template).
 *
 * Returns the user message plus the length actually achieved; callers must
 * report `actualTokens`, not the target, because the two differ by a few
 * tokens and a TTFT curve plotted against a nominal length is not a
 * measurement of anything.
 */
export function buildPromptOfLength(
  targetTokens: number,
  encodeTemplated: (userMessage: string) => number[],
  tolerance = 4,
): { userMessage: string; actualTokens: number; blocks: number } {
  // One measurement to get tokens-per-block on this tokenizer, then a bounded
  // hill-climb. No binary search: the relationship is close to linear, so this
  // converges in a couple of steps and stays easy to reason about.
  const probeBlocks = 4
  const probeLen = encodeTemplated(filler(probeBlocks)).length
  const emptyLen = encodeTemplated('').length
  const perBlock = Math.max(1, (probeLen - emptyLen) / probeBlocks)

  let blocks = Math.max(0, Math.round((targetTokens - emptyLen) / perBlock))
  let ids = encodeTemplated(filler(blocks))
  // Walk one block at a time toward the target, then trim words for the
  // remainder. 200 iterations is far more than the ~5 this needs; it exists
  // so a pathological tokenizer cannot hang the bench.
  for (let i = 0; i < 200 && Math.abs(ids.length - targetTokens) > tolerance; i++) {
    if (ids.length > targetTokens && blocks === 0) break
    blocks += ids.length < targetTokens ? 1 : -1
    ids = encodeTemplated(filler(blocks))
    if (blocks === 0) break
  }

  // Fine-tune by trimming or repeating whole words, which moves the count by
  // ~1 token at a time.
  let message = filler(blocks)
  let words = message.split(' ')
  for (let i = 0; i < 400 && ids.length > targetTokens + tolerance && words.length > 1; i++) {
    words = words.slice(0, -1)
    message = words.join(' ')
    ids = encodeTemplated(message)
  }
  for (let i = 0; i < 400 && ids.length < targetTokens - tolerance; i++) {
    message += ' water'
    ids = encodeTemplated(message)
  }

  return { userMessage: message, actualTokens: ids.length, blocks }
}

/** Prompt lengths swept by the TTFT-vs-prompt-length experiment. */
export const TTFT_PROMPT_LENGTHS = [64, 256, 1024, 2048]
