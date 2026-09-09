// The native host's request → prompt-turns step.
//
// tool-calls.test.ts already pins the RENDERERS against vendor jinja. This
// pins that the host actually CALLS them, which is the failure that shipped:
// renderToolResults wrapped a result in <tool_response> correctly, and had
// tests, and had no callers — both hosts mapped role 'tool' to 'user' and sent
// the tool's raw output as the turn's whole content. Nothing errored. The model
// read every tool result as something the user had typed, coped with one or
// two, and stopped coping once a conversation was deep enough to need the
// frame: at ~24k tokens it read three files, computed the answer, and replied
// in prose instead of calling attempt_completion.
//
// So these assert on the host's OUTPUT, not on the renderers.

import { describe, expect, it } from 'vitest'
import { makeNormalizer } from '../../scripts/native/messages.ts'
import { renderAssistantCalls, foldToolResults } from '../../src/zero-tvm/tool-calls.ts'

const surface = { renderAssistantCalls, foldToolResults }
const host = (dialect: 'chatml-xml' | 'chatml-json' | 'llama3' = 'chatml-xml') =>
  makeNormalizer(surface, dialect)

/** One round of the conversation the agentic eval fails on. */
const ROUND = [
  { role: 'user', content: 'What number does capacity() return?' },
  {
    role: 'assistant',
    content: null,
    tool_calls: [{
      id: 'c1', type: 'function',
      function: { name: 'read_file', arguments: '{"path":"src/dims.ts"}' },
    }],
  },
  { role: 'tool', tool_call_id: 'c1', content: 'export const WIDTH = 12' },
]

describe('the native host normalizer', () => {
  it('wraps a tool result in <tool_response> — it used to send it bare', () => {
    const { normalize } = host()
    const out = normalize(ROUND)
    const last = out[out.length - 1]
    expect(last.role).toBe('user')
    expect(last.content).toBe('<tool_response>\nexport const WIDTH = 12\n</tool_response>')
    // The exact shipped output, named so a regression is unmistakable.
    expect(last.content).not.toBe('export const WIDTH = 12')
  })

  it('renders the assistant tool call rather than erasing it', () => {
    // content is null on a call-only turn; flattening it to '' would delete the
    // call from the transcript and the model would not see what it had done.
    const out = host().normalize(ROUND)
    expect(out[1].content).toContain('<function=read_file>')
    expect(out[1].content).toContain('src/dims.ts')
  })

  it('batches a parallel round into ONE user turn', () => {
    const out = host().normalize([
      { role: 'tool', tool_call_id: 'a', content: 'first' },
      { role: 'tool', tool_call_id: 'b', content: 'second' },
    ])
    expect(out).toHaveLength(1)
    expect(out[0].content).toBe(
      '<tool_response>\nfirst\n</tool_response>\n<tool_response>\nsecond\n</tool_response>')
  })

  it('keeps separate rounds separate', () => {
    const out = host().normalize([
      { role: 'tool', tool_call_id: 'a', content: 'first' },
      { role: 'assistant', content: 'thinking' },
      { role: 'tool', tool_call_id: 'b', content: 'second' },
    ])
    expect(out.map((m: { role: string }) => m.role)).toEqual(['user', 'assistant', 'user'])
  })

  it('folds developer to system and drops roles no template has', () => {
    const out = host().normalize([
      { role: 'developer', content: 'be terse' },
      { role: 'function', content: 'legacy role' },
      { role: 'user', content: 'hi' },
    ])
    expect(out).toEqual([
      { role: 'system', content: 'be terse' },
      { role: 'user', content: 'hi' },
    ])
  })

  it('flattens multi-part content arrays', () => {
    const out = host().normalize([
      { role: 'user', content: [{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }] },
    ])
    expect(out[0].content).toBe('ab')
  })

  it('substitutes the model\'s VERBATIM text for a call it emitted', () => {
    // Re-rendering is equivalent but rarely byte-identical, and a byte that
    // moves costs the whole KV prefix on the next turn.
    const { normalize, rememberRaw } = host()
    const raw = '<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/dims.ts\n</parameter>\n</function>\n</tool_call>   '
    rememberRaw('', [{ name: 'read_file', arguments: { path: 'src/dims.ts' } }], raw)
    expect(normalize(ROUND)[1].content).toBe(raw)
  })

  it('gives Llama-3 results their own ipython turns', () => {
    const out = host('llama3').normalize(ROUND)
    expect(out[out.length - 1]).toEqual({ role: 'ipython', content: '"export const WIDTH = 12"' })
  })
})
