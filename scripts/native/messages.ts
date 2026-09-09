/**
 * OpenAI request messages → the turns a chat template renders.
 *
 * Its own module because this is where the tool-call bugs live, and inside
 * agent-native.mjs nothing could reach it without booting an engine on a GPU.
 * The one that shipped: `m.role === 'tool' ? 'user'` with the tool's output as
 * the turn's whole content. The role is right and the content was not — the
 * template wraps a result in <tool_response> markers, foldToolResults does
 * that, and it had tests and no callers. The model read every tool result as
 * something the user had typed.
 *
 * `surface` is dist-lib's hostSurface(), injected rather than imported so a
 * test can pass the pure functions directly.
 */

export interface RequestMessage {
  role: string
  content: unknown
  tool_calls?: Array<{ function?: { name?: string; arguments?: string } }>
  /** Carried by every OpenAI `tool` message. Unused here — the templates
   *  address results by position, not by id — but a type that rejected it
   *  would reject a real request. */
  tool_call_id?: string
}
/** The slice of dist-lib's hostSurface() this needs. Typed off tool-calls.ts
 *  rather than restated, so a dialect that is not a real dialect is a compile
 *  error here and not a silently-wrong render at runtime. */
type Surface = Pick<typeof import('../../src/zero-tvm/tool-calls.ts'),
  'renderAssistantCalls' | 'foldToolResults'>
type Dialect = Parameters<Surface['renderAssistantCalls']>[0]

/** A turn the templates understand — `tool` included, since the fold below
 *  is what turns it into one they render. */
export type Turn = Parameters<Surface['foldToolResults']>[1][number]

const flattenContent = (c: unknown): string => typeof c === 'string' ? c
  : Array.isArray(c) ? c.map((p) => typeof p === 'string' ? p : (p as { type?: string; text?: string })?.type === 'text' ? ((p as { text?: string }).text ?? '') : '').join('') : ''

/**
 * VERBATIM ASSISTANT TURNS — what the model wrote, not a reconstruction.
 *
 * A client hands assistant turns back as STRUCTURE (content + tool_calls),
 * because that is what the OpenAI shape carries. Re-rendering that structure
 * produces text that is *equivalent* but rarely byte-identical to what the
 * model emitted — different whitespace inside a <tool_call> block is enough.
 * Different text means different tokens, which means the new prompt stops
 * matching the KV cache at the FIRST assistant turn, and the engine has to
 * re-prefill the entire conversation. Measured on a real Cline session: a
 * 16,454-token prompt re-prefilled in full, 98s to first token, when only
 * 4,733 tokens were new.
 *
 * So the raw output is kept, keyed by the structure the client will send
 * back, and substituted on the next turn. Bounded, and a miss simply falls
 * back to re-rendering — the old behaviour, never an error.
 */
const RAW_CACHE_MAX = 64
const rawKey = (content: string, calls: Array<{ name: string; arguments: Record<string, unknown> }>) => JSON.stringify([content ?? '',
  (calls ?? []).map((c) => [c.name, JSON.stringify(c.arguments ?? {})])])

export function makeNormalizer(surface: Surface, dialect: Dialect) {
  const RAW_CACHE = new Map<string, string>()
  const rememberRaw = (text: string, calls: Array<{ name: string; arguments: Record<string, unknown> }>, raw: string) => {
    const k = rawKey(text, calls)
    RAW_CACHE.delete(k)
    RAW_CACHE.set(k, raw)
    while (RAW_CACHE.size > RAW_CACHE_MAX) RAW_CACHE.delete(RAW_CACHE.keys().next().value as string)
  }

  const normalize = (messages: RequestMessage[]): Turn[] => {
    const out: Turn[] = []
    for (const m of messages ?? []) {
      // 'tool' stays 'tool' here and is folded at the end. It used to become
      // 'user' on the spot, which threw away the one fact the template needs:
      // the model then read every tool's output as something the user typed.
      const role = m.role === 'developer' ? 'system' : m.role
      if (role !== 'system' && role !== 'user' && role !== 'assistant' && role !== 'tool') continue
      let content = flattenContent(m.content)
      if (m.role === 'assistant' && m.tool_calls?.length) {
        const calls = m.tool_calls.map((c) => {
          let a: Record<string, unknown> = {}; try { a = JSON.parse(c.function?.arguments || '{}') } catch { /* keep {} */ }
          return { name: c.function?.name ?? '', arguments: a }
        })
        const raw = RAW_CACHE.get(rawKey(content, calls))
        content = raw ?? surface.renderAssistantCalls(dialect, content ?? '', calls)
      }
      out.push({ role, content })
    }
    return surface.foldToolResults(dialect, out)
  }

  return { normalize, rememberRaw }
}

