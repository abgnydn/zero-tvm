// TOOL CALLS — the render half against the vendor jinja, the parse half
// against the output a real stream produces.
//
// The RENDER fixtures below are the vendor `chat_template` from three
// checkpoints in .weights-local (Qwen3-4B-4bit, Qwen3.6-35B-A3B-MLX-4bit,
// Llama-3.2-1B-Instruct-4bit) rendered through jinja2 the way transformers
// renders it: ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks) with
// `tojson` bound to json.dumps(ensure_ascii=False). So these tests pin the
// builders to the templates rather than to my reading of them — a tools block
// that is subtly wrong does not error, the model just quietly stops calling
// tools.
//
// The render assertions go through the SHIPPED template builders
// (tokenizer-bpe.ts) with a tokenizer stub that hands back the string, because
// what withTools() produces is only correct in terms of what those builders do
// with it.
//
// The PARSE assertions are hand-written streams, including the ones that are
// supposed to be broken: malformed JSON in a closed block, an unterminated
// block, braces inside a string argument.

import { describe, expect, it } from 'vitest'
import {
  parseToolCalls,
  renderAssistantCalls,
  renderToolResults,
  foldToolResults,
  toolDialectFor,
  withTools,
  type ToolCall,
  type ToolChatMessage,
  type ToolDef,
} from '../../src/zero-tvm/tool-calls.ts'
import { buildChatPrompt, buildLlama3ChatPrompt } from '../../src/zero-tvm/tokenizer-bpe.ts'

// ============================================================
// Harness
// ============================================================

/** Hands back the text instead of ids, so the assertion reads as the prompt. */
const capture = { encode: (t: string) => [t] as unknown as number[] }

/** buildChatPrompt's message type does not know the `ipython` role yet — see
 *  the note in tool-calls.ts. Widening model-select.ts's ChatMessage is what
 *  removes this cast. */
type TemplateMessages = Array<{ role: 'system' | 'user' | 'assistant'; content: string }>
const asTemplate = (m: ToolChatMessage[]) => m as TemplateMessages

const chatml = (m: ToolChatMessage[]) =>
  (buildChatPrompt(asTemplate(m), capture, { thinking: false }) as unknown as string[])[0]
/** Qwen3.5+ rule: every assistant turn in the CURRENT round keeps its <think>
 *  block. Identical to `chatml` on any conversation whose last turn is a real
 *  user query, which is why the pre-existing fixtures do not distinguish them. */
const chatml35 = (m: ToolChatMessage[]) =>
  (buildChatPrompt(asTemplate(m), capture, { thinking: false, generation: 'qwen35' }) as unknown as string[])[0]
const llama3 = (m: ToolChatMessage[]) =>
  (buildLlama3ChatPrompt(asTemplate(m), capture, { dateString: '07 Aug 2026' }) as unknown as string[])[0]

const TOOLS: ToolDef[] = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the weather for a city',
      parameters: {
        type: 'object',
        properties: { city: { type: 'string', description: 'City name' } },
        required: ['city'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_python',
      description: 'Execute python',
      parameters: {
        type: 'object',
        properties: { code: { type: 'string' } },
        required: ['code'],
      },
    },
  },
]

// ============================================================
// Vendor fixtures (jinja output, verbatim)
// ============================================================

const QWEN3_TOOLS_BLOCK = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"City name\"}}, \"required\": [\"city\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"run_python\", \"description\": \"Execute python\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\"}}, \"required\": [\"code\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"

const QWEN36_TOOLS_BLOCK = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"City name\"}}, \"required\": [\"city\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"run_python\", \"description\": \"Execute python\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\"}}, \"required\": [\"code\"]}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

const LLAMA3_TOOLS_BLOCK = "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n{\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"get_weather\",\n        \"description\": \"Get the weather for a city\",\n        \"parameters\": {\n            \"type\": \"object\",\n            \"properties\": {\n                \"city\": {\n                    \"type\": \"string\",\n                    \"description\": \"City name\"\n                }\n            },\n            \"required\": [\n                \"city\"\n            ]\n        }\n    }\n}\n\n{\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"run_python\",\n        \"description\": \"Execute python\",\n        \"parameters\": {\n            \"type\": \"object\",\n            \"properties\": {\n                \"code\": {\n                    \"type\": \"string\"\n                }\n            },\n            \"required\": [\n                \"code\"\n            ]\n        }\n    }\n}\n\n"

// Rendered from the same templates with `tools` omitted — the tools block is
// pinned above and would otherwise swamp these.
const QWEN3_CALL_AND_RESULT = '<|im_start|>user\nweather in Istanbul?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "İstanbul", "days": 2}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C, clear\n</tool_response><|im_end|>\n<|im_start|>user\nthanks<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
const QWEN3_TWO_CALLS = '<|im_start|>user\nboth?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "A"}}\n</tool_call>\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "B"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C\n</tool_response>\n<tool_response>\n21C\n</tool_response><|im_end|>\n<|im_start|>user\nthanks<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
const QWEN3_CALL_WITH_PROSE = '<|im_start|>user\nweather?<|im_end|>\n<|im_start|>assistant\nChecking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "A"}}\n</tool_call><|im_end|>\n<|im_start|>user\nthanks<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

const QWEN36_CALL_AND_RESULT = '<|im_start|>user\nweather in Istanbul?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nİstanbul\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C, clear\n</tool_response><|im_end|>\n<|im_start|>user\nthanks<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
const QWEN36_TWO_CALLS = '<|im_start|>user\nboth?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nB\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C\n</tool_response>\n<tool_response>\n21C\n</tool_response><|im_end|>\n<|im_start|>user\nthanks<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
const QWEN36_CALL_WITH_PROSE = '<|im_start|>user\nweather?<|im_end|>\n<|im_start|>assistant\nChecking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nA\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\nthanks<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

const LLAMA3_CALL_AND_RESULT = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 07 Aug 2026\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nweather in Istanbul?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{"name": "get_weather", "parameters": {"city": "İstanbul", "days": 2}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n"18C, clear"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nthanks<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

// ============================================================
// Render — the tools block
// ============================================================

describe('withTools', () => {
  it('folds the Qwen3 tools block into the system turn, after the system text', () => {
    const out = chatml(withTools('chatml-json', [
      { role: 'system', content: 'You are terse.' },
      { role: 'user', content: 'weather in Istanbul?' },
    ], TOOLS))
    expect(out).toBe(
      `<|im_start|>system\nYou are terse.\n\n${QWEN3_TOOLS_BLOCK}<|im_end|>\n` +
      '<|im_start|>user\nweather in Istanbul?<|im_end|>\n' +
      '<|im_start|>assistant\n<think>\n\n</think>\n\n',
    )
  })

  it('creates the Qwen3 system turn when the conversation has none', () => {
    // The vendor emits a system turn whenever tools are present, even with no
    // system message — skipping it would drop the tools entirely.
    const out = chatml(withTools('chatml-json', [{ role: 'user', content: 'weather in Istanbul?' }], TOOLS))
    expect(out).toBe(
      `<|im_start|>system\n${QWEN3_TOOLS_BLOCK}<|im_end|>\n` +
      '<|im_start|>user\nweather in Istanbul?<|im_end|>\n' +
      '<|im_start|>assistant\n<think>\n\n</think>\n\n',
    )
  })

  it('puts Qwen3.6 system text AFTER the tools block, not before', () => {
    // Qwen3.6 reversed the order its predecessor used, and trims the system
    // text. Copying the Qwen3 branch would look right and be wrong.
    const out = chatml(withTools('chatml-xml', [
      { role: 'system', content: '  You are terse.  ' },
      { role: 'user', content: 'weather in Istanbul?' },
    ], TOOLS))
    expect(out).toBe(
      `<|im_start|>system\n${QWEN36_TOOLS_BLOCK}\n\nYou are terse.<|im_end|>\n` +
      '<|im_start|>user\nweather in Istanbul?<|im_end|>\n' +
      '<|im_start|>assistant\n<think>\n\n</think>\n\n',
    )
  })

  it('drops the Qwen3.6 separator when there is no system text', () => {
    const out = chatml(withTools('chatml-xml', [{ role: 'user', content: 'weather in Istanbul?' }], TOOLS))
    expect(out).toBe(
      `<|im_start|>system\n${QWEN36_TOOLS_BLOCK}<|im_end|>\n` +
      '<|im_start|>user\nweather in Istanbul?<|im_end|>\n' +
      '<|im_start|>assistant\n<think>\n\n</think>\n\n',
    )
  })

  it('puts the Llama-3 schemas in the FIRST USER message, indent=4', () => {
    // Everything except one line: the vendor also prepends "Environment:
    // ipython\n" to the system block, ahead of the knowledge-cutoff line, and
    // buildLlama3ChatPrompt hardcodes that block's order. Removing it from the
    // fixture is how this test records the gap instead of hiding it.
    const out = llama3(withTools('llama3', [
      { role: 'system', content: 'You are terse.' },
      { role: 'user', content: 'weather in Istanbul?' },
    ], TOOLS))
    expect(out).toBe(
      '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' +
      'Cutting Knowledge Date: December 2023\nToday Date: 07 Aug 2026\n\nYou are terse.<|eot_id|>' +
      `<|start_header_id|>user<|end_header_id|>\n\n${LLAMA3_TOOLS_BLOCK}weather in Istanbul?<|eot_id|>` +
      '<|start_header_id|>assistant<|end_header_id|>\n\n',
    )
  })

  it('keeps the vendor typo in the Llama-3 preamble', () => {
    // "...its value}.Do not use variables." — two jinja literals concatenated
    // with no space. It reads like a bug and it is what the model was tuned on.
    expect(LLAMA3_TOOLS_BLOCK).toContain('its value}.Do not use variables.')
  })

  it('refuses Llama-3 tools with no user message to put them in', () => {
    expect(() => withTools('llama3', [{ role: 'system', content: 'x' }], TOOLS)).toThrow(/user message/)
  })

  it('is a no-op with an empty tool list', () => {
    const msgs: ToolChatMessage[] = [{ role: 'user', content: 'hi' }]
    expect(withTools('chatml-json', msgs, [])).toEqual(msgs)
  })

  it('serializes schemas with python separators, not JSON.stringify\'s', () => {
    // `tojson` is json.dumps(ensure_ascii=False): ", " and ": ". Every byte is
    // prompt, so "tidying" this to JSON.stringify would retokenize every schema
    // the model reads.
    const out = chatml(withTools('chatml-json', [{ role: 'user', content: 'x' }], TOOLS))
    expect(out).toContain('{"type": "function", "function": {"name": "get_weather"')
    expect(out).not.toContain('{"type":"function"')
  })

  it('leaves the caller\'s messages untouched', () => {
    const msgs: ToolChatMessage[] = [{ role: 'user', content: 'hi' }]
    withTools('chatml-json', msgs, TOOLS)
    expect(msgs).toEqual([{ role: 'user', content: 'hi' }])
  })
})

// ============================================================
// Render — calls and results back into the transcript
// ============================================================

describe('renderAssistantCalls', () => {
  const call: ToolCall = { name: 'get_weather', arguments: { city: 'İstanbul', days: 2 } }

  it('renders a Qwen3 call the way its template writes one', () => {
    const content = renderAssistantCalls('chatml-json', '', [call])
    expect(content).toBe('<tool_call>\n{"name": "get_weather", "arguments": {"city": "İstanbul", "days": 2}}\n</tool_call>')
    expect(QWEN3_CALL_AND_RESULT).toContain(`<|im_start|>assistant\n${content}<|im_end|>`)
  })

  it('renders a Qwen3.6 call as nested XML with raw string values', () => {
    const content = renderAssistantCalls('chatml-xml', '', [call])
    expect(content).toBe('<tool_call>\n<function=get_weather>\n<parameter=city>\nİstanbul\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>')
    expect(QWEN36_CALL_AND_RESULT).toContain(`<|im_start|>assistant\n${content}<|im_end|>`)
  })

  it('renders a Llama-3 call as a bare object keyed "parameters"', () => {
    const content = renderAssistantCalls('llama3', '', [call])
    expect(content).toBe('{"name": "get_weather", "parameters": {"city": "İstanbul", "days": 2}}')
    expect(LLAMA3_CALL_AND_RESULT).toContain(content)
  })

  it('separates two Qwen calls with one newline', () => {
    const a: ToolCall = { name: 'get_weather', arguments: { city: 'A' } }
    const b: ToolCall = { name: 'get_weather', arguments: { city: 'B' } }
    expect(QWEN3_TWO_CALLS).toContain(`<|im_start|>assistant\n${renderAssistantCalls('chatml-json', '', [a, b])}<|im_end|>`)
    expect(QWEN36_TWO_CALLS).toContain(`<|im_start|>assistant\n${renderAssistantCalls('chatml-xml', '', [a, b])}<|im_end|>`)
  })

  it('joins prose to the first call with \\n on Qwen3 and \\n\\n on Qwen3.6', () => {
    const a: ToolCall = { name: 'get_weather', arguments: { city: 'A' } }
    expect(renderAssistantCalls('chatml-json', 'Checking.', [a])).toBe(
      'Checking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "A"}}\n</tool_call>')
    expect(QWEN3_CALL_WITH_PROSE).toContain(`<|im_start|>assistant\n${renderAssistantCalls('chatml-json', 'Checking.', [a])}<|im_end|>`)
    expect(QWEN36_CALL_WITH_PROSE).toContain(`<|im_start|>assistant\n${renderAssistantCalls('chatml-xml', 'Checking.', [a])}<|im_end|>`)
  })

  it('drops the prose on Llama-3, because its template does', () => {
    const a: ToolCall = { name: 'get_weather', arguments: { city: 'A' } }
    expect(renderAssistantCalls('llama3', 'Checking.', [a])).toBe('{"name": "get_weather", "parameters": {"city": "A"}}')
  })

  it('refuses two calls in one Llama-3 message', () => {
    const a: ToolCall = { name: 'x', arguments: {} }
    expect(() => renderAssistantCalls('llama3', '', [a, a])).toThrow(/single tool call/)
  })

  it('returns the prose unchanged when there were no calls', () => {
    expect(renderAssistantCalls('chatml-json', 'just talking', [])).toBe('just talking')
  })
})

describe('renderToolResults', () => {
  it('batches every ChatML result of a round into ONE user turn', () => {
    const [turn] = renderToolResults('chatml-json', ['18C', '21C'])
    expect(turn.role).toBe('user')
    expect(QWEN3_TWO_CALLS).toContain(`<|im_start|>user\n${turn.content}<|im_end|>`)
    expect(QWEN36_TWO_CALLS).toContain(`<|im_start|>user\n${turn.content}<|im_end|>`)
  })

  it('gives each Llama-3 result its own ipython turn — and QUOTES it', () => {
    // The template renders content with tojson when it `is mapping or is
    // iterable`, and jinja counts a string as iterable, so a plain string
    // result reaches the model quoted. Reproduced on purpose.
    const turns = renderToolResults('llama3', ['18C, clear'])
    expect(turns).toEqual([{ role: 'ipython', content: '"18C, clear"' }])
    expect(LLAMA3_CALL_AND_RESULT).toContain(`<|start_header_id|>ipython<|end_header_id|>\n\n${turns[0].content}<|eot_id|>`)
  })

  it('renders a whole Llama-3 tool round exactly', () => {
    // Nothing diverges on this path, so the assertion can be the whole prompt.
    const out = llama3([
      { role: 'user', content: 'weather in Istanbul?' },
      { role: 'assistant', content: renderAssistantCalls('llama3', '', [{ name: 'get_weather', arguments: { city: 'İstanbul', days: 2 } }]) },
      ...renderToolResults('llama3', ['18C, clear']),
      { role: 'user', content: 'thanks' },
    ])
    expect(out).toBe(LLAMA3_CALL_AND_RESULT)
  })
})

// A conversation with a COMPLETED round before the current one. Every other
// fixture here ends with a real user query, so no assistant turn is ever inside
// the current round and both Qwen generations render identically — which is how
// an unconditional <think> block survived in buildChatPrompt. Generated by
// scripts/gen-toolcall-fixtures.py from the checkpoints' own templates.
const QWEN3_MULTI_ROUND = '<|im_start|>user\nfirst question<|im_end|>\n<|im_start|>assistant\nfirst answer<|im_end|>\n<|im_start|>user\nweather in Istanbul?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "İstanbul"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C, clear\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
const QWEN36_MULTI_ROUND = '<|im_start|>user\nfirst question<|im_end|>\n<|im_start|>assistant\nfirst answer<|im_end|>\n<|im_start|>user\nweather in Istanbul?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nİstanbul\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C, clear\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

describe('toolDialectFor', () => {
  // Was a prefix match on spec.id, duplicated in both hosts. It caught Qwen3.8
  // only after it shipped wrong, and still sent BOTH Qwen3.5 builds the
  // Qwen3-era JSON dialect — which renders a tools block they cannot follow and
  // parses calls they never make.
  it('every Qwen3.5-and-later template speaks XML', () => {
    expect(toolDialectFor('chatml-q35')).toBe('chatml-xml')
    expect(toolDialectFor('chatml-q38')).toBe('chatml-xml')
  })

  it('the Qwen3-era template speaks JSON', () => {
    expect(toolDialectFor('chatml')).toBe('chatml-json')
  })

  it('llama3 is its own dialect, and anything unread falls back to JSON', () => {
    expect(toolDialectFor('llama3')).toBe('llama3')
    expect(toolDialectFor('phi3')).toBe('chatml-json')
  })
})

describe('the past-turn <think> block', () => {
  // What shipped: EVERY assistant turn got one. On Qwen3.6 at ~24k tokens that
  // was 286 spurious blocks and 5,434 characters of prompt the model was never
  // trained to see; it read the right files, computed the right answer, and
  // replied in prose instead of calling the tool. mlx_lm on the same checkpoint
  // and the same conversation emitted the call at every depth.
  const round = (calls: string) => [
    { role: 'user' as const, content: 'first question' },
    { role: 'assistant' as const, content: 'first answer' },
    { role: 'user' as const, content: 'weather in Istanbul?' },
    { role: 'assistant' as const, content: calls },
    { role: 'tool' as const, content: '18C, clear' },
  ]
  const CALL: ToolCall = { name: 'get_weather', arguments: { city: 'İstanbul' } }

  it('Qwen3.5+: on the current round, never on a finished one', () => {
    expect(chatml35(foldToolResults('chatml-xml', round(renderAssistantCalls('chatml-xml', '', [CALL])))))
      .toBe(QWEN36_MULTI_ROUND)
  })

  it('Qwen3: not even on the current round — their extra gate', () => {
    // Qwen3 also needs `loop.last or reasoning_content`, and we carry no
    // reasoning_content, so a mid-conversation turn gets nothing.
    expect(chatml(foldToolResults('chatml-json', round(renderAssistantCalls('chatml-json', '', [CALL])))))
      .toBe(QWEN3_MULTI_ROUND)
  })

  it('a finished round keeps NO block under either rule', () => {
    // The one both generations agree on, and the one the old code broke on
    // every turn of a long session.
    const done = [
      { role: 'user' as const, content: 'q1' },
      { role: 'assistant' as const, content: 'a1' },
      { role: 'user' as const, content: 'q2' },
    ]
    for (const render of [chatml, chatml35]) {
      expect(render(done)).toBe(
        '<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\na1<|im_end|>\n' +
        '<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    }
  })

  it('a tool result does not count as a new query — the round continues', () => {
    // <tool_response> turns are skipped when locating the last real query. If
    // they counted, the round would end at the first tool result and the rest
    // of a multi-step loop would lose its blocks.
    const out = chatml35(foldToolResults('chatml-xml', [
      { role: 'user', content: 'go' },
      { role: 'assistant', content: renderAssistantCalls('chatml-xml', '', [CALL]) },
      { role: 'tool', content: 'a' },
      { role: 'assistant', content: renderAssistantCalls('chatml-xml', '', [CALL]) },
      { role: 'tool', content: 'b' },
    ]))
    expect(out.match(/<think>/g)).toHaveLength(3)   // two round turns + the generation prompt
  })

  it('Qwen3 DOES emit one on a trailing assistant turn (loop.last)', () => {
    expect(chatml([
      { role: 'user', content: 'q' },
      { role: 'assistant', content: 'partial' },
    ])).toBe('<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\npartial<|im_end|>\n'
      + '<|im_start|>assistant\n<think>\n\n</think>\n\n')
  })
})

describe('foldToolResults — the step both hosts skipped', () => {
  // The shipped behaviour was `role === 'tool' ? 'user'` with the tool's output
  // as the whole content. These assert against the vendor fixtures, so they
  // fail on that: an unwrapped result cannot produce the template's text.

  it('wraps a Qwen3.6 result in <tool_response> — the whole round matches vendor', () => {
    const call: ToolCall = { name: 'get_weather', arguments: { city: 'İstanbul', days: 2 } }
    const out = chatml35(foldToolResults('chatml-xml', [
      { role: 'user', content: 'weather in Istanbul?' },
      { role: 'assistant', content: renderAssistantCalls('chatml-xml', '', [call]) },
      { role: 'tool', content: '18C, clear' },
      { role: 'user', content: 'thanks' },
    ]))
    expect(out).toBe(QWEN36_CALL_AND_RESULT)
    // Name the failure the old code produced, so a regression reads clearly.
    expect(out).toContain('<|im_start|>user\n<tool_response>\n18C, clear\n</tool_response><|im_end|>')
    expect(out).not.toContain('<|im_start|>user\n18C, clear<|im_end|>')
  })

  it('does the same for the Qwen3-era JSON dialect', () => {
    const call: ToolCall = { name: 'get_weather', arguments: { city: 'İstanbul', days: 2 } }
    expect(chatml(foldToolResults('chatml-json', [
      { role: 'user', content: 'weather in Istanbul?' },
      { role: 'assistant', content: renderAssistantCalls('chatml-json', '', [call]) },
      { role: 'tool', content: '18C, clear' },
      { role: 'user', content: 'thanks' },
    ]))).toBe(QWEN3_CALL_AND_RESULT)
  })

  it('collapses CONSECUTIVE results into one turn, as the template does', () => {
    const a: ToolCall = { name: 'get_weather', arguments: { city: 'A' } }
    const b: ToolCall = { name: 'get_weather', arguments: { city: 'B' } }
    const folded = foldToolResults('chatml-xml', [
      { role: 'user', content: 'both?' },
      { role: 'assistant', content: renderAssistantCalls('chatml-xml', '', [a, b]) },
      { role: 'tool', content: '18C' },
      { role: 'tool', content: '21C' },
      { role: 'user', content: 'thanks' },
    ])
    // Four turns, not five: the two results share one <|im_start|>user block.
    expect(folded).toHaveLength(4)
    expect(chatml35(folded)).toBe(QWEN36_TWO_CALLS)
  })

  it('a run broken by another role starts a new turn', () => {
    // Two separate rounds are two separate turns. Collapsing them would put a
    // later round's output inside an earlier round's block.
    const folded = foldToolResults('chatml-xml', [
      { role: 'tool', content: 'first' },
      { role: 'user', content: 'and now?' },
      { role: 'tool', content: 'second' },
    ])
    expect(folded.map((m) => m.role)).toEqual(['user', 'user', 'user'])
    expect(folded[0].content).toBe('<tool_response>\nfirst\n</tool_response>')
    expect(folded[1].content).toBe('and now?')
    expect(folded[2].content).toBe('<tool_response>\nsecond\n</tool_response>')
  })

  it('gives each Llama-3 result its own ipython turn, quoted', () => {
    const out = llama3(foldToolResults('llama3', [
      { role: 'user', content: 'weather in Istanbul?' },
      { role: 'assistant', content: renderAssistantCalls('llama3', '', [{ name: 'get_weather', arguments: { city: 'İstanbul', days: 2 } }]) },
      { role: 'tool', content: '18C, clear' },
      { role: 'user', content: 'thanks' },
    ]))
    expect(out).toBe(LLAMA3_CALL_AND_RESULT)
  })

  it('leaves a conversation with no tool turns byte-identical', () => {
    const msgs: ToolChatMessage[] = [
      { role: 'system', content: 'You are terse.' },
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ]
    expect(foldToolResults('chatml-xml', msgs)).toEqual(msgs)
  })

  it('survives a conversation that is nothing but results', () => {
    expect(foldToolResults('chatml-xml', [{ role: 'tool', content: 'x' }]))
      .toEqual([{ role: 'user', content: '<tool_response>\nx\n</tool_response>' }])
    expect(foldToolResults('chatml-xml', [])).toEqual([])
  })
})

// ============================================================
// Parse
// ============================================================

const CLEAN = {
  'chatml-json': '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Istanbul"}}\n</tool_call>',
  'chatml-xml': '<tool_call>\n<function=get_weather>\n<parameter=city>\nIstanbul\n</parameter>\n</function>\n</tool_call>',
  llama3: '{"name": "get_weather", "parameters": {"city": "Istanbul"}}',
} as const

describe('parseToolCalls — the happy path', () => {
  for (const dialect of ['chatml-json', 'chatml-xml', 'llama3'] as const) {
    it(`${dialect}: a clean call`, () => {
      const r = parseToolCalls(dialect, CLEAN[dialect])
      expect(r.partial).toBe(false)
      expect(r.text).toBe('')
      expect(r.calls).toEqual([
        { name: 'get_weather', arguments: { city: 'Istanbul' }, raw: expect.any(String) },
      ])
    })

    it(`${dialect}: prose before and after a call`, () => {
      const r = parseToolCalls(dialect, `Let me check.\n${CLEAN[dialect]}\nDone.`)
      expect(r.calls).toHaveLength(1)
      expect(r.calls[0].name).toBe('get_weather')
      expect(r.text).toBe('Let me check.\n\nDone.')
      expect(r.partial).toBe(false)
    })

    it(`${dialect}: survives a round trip through the renderer`, () => {
      const args = { city: 'İstanbul', note: 'ok', nested: { a: 1 } }
      const content = renderAssistantCalls(dialect, '', [{ name: 'get_weather', arguments: args }])
      const r = parseToolCalls(dialect, content)
      expect(r.calls).toHaveLength(1)
      expect(r.calls[0].error).toBeUndefined()
      expect(r.calls[0].name).toBe('get_weather')
      expect(r.calls[0].arguments).toEqual(args)
    })
  }

  it('chatml: two calls in one turn', () => {
    const r = parseToolCalls('chatml-json',
      '<tool_call>\n{"name": "a", "arguments": {"i": 1}}\n</tool_call>\n' +
      '<tool_call>\n{"name": "b", "arguments": {"i": 2}}\n</tool_call>')
    expect(r.calls.map((c) => c.name)).toEqual(['a', 'b'])
    expect(r.calls.map((c) => c.arguments)).toEqual([{ i: 1 }, { i: 2 }])
    expect(r.text).toBe('\n')
    expect(r.partial).toBe(false)
  })

  it('llama3: two calls separated by the format\'s semicolon', () => {
    const r = parseToolCalls('llama3',
      '{"name": "a", "parameters": {"i": 1}}; {"name": "b", "parameters": {"i": 2}}')
    expect(r.calls.map((c) => c.name)).toEqual(['a', 'b'])
    // The separator stays in the text. Deciding it is punctuation needs the
    // NEXT object, and text already emitted cannot be taken back mid-stream —
    // see the incremental-safety sweep at the bottom of this file.
    expect(r.text).toBe('; ')
  })

  it('llama3: strips a <|python_tag|> that survived decoding', () => {
    const r = parseToolCalls('llama3', '<|python_tag|>{"name": "a", "parameters": {}}')
    expect(r.calls.map((c) => c.name)).toEqual(['a'])
    expect(r.text).toBe('')
  })

  it('chatml: accepts arguments emitted as an embedded JSON string', () => {
    // Qwen's own template branches on `arguments is string`, so this is inside
    // the format, not outside it.
    const r = parseToolCalls('chatml-json', '<tool_call>\n{"name": "a", "arguments": "{\\"city\\": \\"X\\"}"}\n</tool_call>')
    expect(r.calls[0].error).toBeUndefined()
    expect(r.calls[0].arguments).toEqual({ city: 'X' })
  })

  it('chatml: accepts the llama3 key, and a call with no arguments at all', () => {
    const r = parseToolCalls('chatml-json',
      '<tool_call>\n{"name": "a", "parameters": {"i": 1}}\n</tool_call>' +
      '<tool_call>\n{"name": "now"}\n</tool_call>')
    expect(r.calls[0].arguments).toEqual({ i: 1 })
    expect(r.calls[1]).toEqual({ name: 'now', arguments: {}, raw: expect.any(String) })
  })

  it('chatml-xml: reads numbers as numbers and leaves quoted text quoted', () => {
    const r = parseToolCalls('chatml-xml',
      '<tool_call>\n<function=f>\n' +
      '<parameter=n>\n42\n</parameter>\n' +
      '<parameter=flag>\ntrue\n</parameter>\n' +
      '<parameter=obj>\n{"a": [1, 2]}\n</parameter>\n' +
      '<parameter=s>\n"quoted"\n</parameter>\n' +
      '<parameter=code>\nline one\n\nline three\n</parameter>\n' +
      '</function>\n</tool_call>')
    expect(r.calls[0].arguments).toEqual({
      n: 42,
      flag: true,
      obj: { a: [1, 2] },
      // A JSON reading would drop the quotes the model actually wrote.
      s: '"quoted"',
      // Only the template's own framing newline comes off each end.
      code: 'line one\n\nline three',
    })
  })
})

describe('parseToolCalls — braces and unicode', () => {
  it('chatml: braces and escaped quotes inside a string argument', () => {
    // Where a regex parser breaks. The `}` inside the code string must not end
    // the object, and `\"` must not end the string.
    const code = 'fn main() { if x { println!("}{\\"}"); } }'
    const body = JSON.stringify({ name: 'run', arguments: { code } })
    const r = parseToolCalls('chatml-json', `<tool_call>\n${body}\n</tool_call>`)
    expect(r.calls[0].error).toBeUndefined()
    expect(r.calls[0].arguments).toEqual({ code })
  })

  it('llama3: an unmarked call ends at ITS closing brace, not the first one', () => {
    const patch = 'func f() {\n  return map[string]int{"a": 1}\n}'
    const obj = JSON.stringify({ name: 'edit', parameters: { patch } })
    const r = parseToolCalls('llama3', `Applying it.\n${obj}\nand then some prose {with braces}.`)
    expect(r.calls).toHaveLength(1)
    expect(r.calls[0].arguments).toEqual({ patch })
    expect(r.text).toBe('Applying it.\n\nand then some prose {with braces}.')
  })

  it('carries unicode through unharmed, escaped or raw', () => {
    const r = parseToolCalls('chatml-json',
      '<tool_call>\n{"name": "note", "arguments": {"city": "İstanbul", "sky": "🌤", "esc": "\\u00e7ay"}}\n</tool_call>')
    expect(r.calls[0].arguments).toEqual({ city: 'İstanbul', sky: '🌤', esc: 'çay' })
  })
})

describe('parseToolCalls — malformed and truncated', () => {
  it('chatml: malformed JSON in a CLOSED block is an error call, not a drop', () => {
    // The tags prove the model meant a call. Dropping it silently makes the
    // model repeat itself forever; throwing kills the turn. The caller answers
    // with a tool_response describing the failure, which needs `raw`.
    const body = '\n{"name": "get_weather", "arguments": {"city": }}\n'
    const r = parseToolCalls('chatml-json', `<tool_call>${body}</tool_call>`)
    expect(r.partial).toBe(false)
    expect(r.calls).toHaveLength(1)
    expect(r.calls[0].error).toMatch(/invalid JSON/)
    expect(r.calls[0].raw).toBe(body)
    expect(r.calls[0].arguments).toEqual({})
  })

  it('chatml: braces that never balance inside a CLOSED block are an error', () => {
    const r = parseToolCalls('chatml-json', '<tool_call>\n{"name": "a", "arguments": {"x": 1}\n</tool_call>tail')
    expect(r.partial).toBe(false)
    expect(r.calls[0].error).toMatch(/no JSON object/)
    // The stream moved on: the tail after the block is still prose.
    expect(r.text).toBe('tail')
  })

  it('chatml: a call whose name is missing is an error call', () => {
    const r = parseToolCalls('chatml-json', '<tool_call>\n{"arguments": {"city": "X"}}\n</tool_call>')
    expect(r.calls[0].error).toMatch(/no "name"/)
  })

  it('chatml: arguments that are not an object are an error, but the name survives', () => {
    const r = parseToolCalls('chatml-json', '<tool_call>\n{"name": "a", "arguments": [1, 2]}\n</tool_call>')
    expect(r.calls[0].name).toBe('a')
    expect(r.calls[0].error).toMatch(/not an object/)
  })

  it('chatml-xml: a block with no <function=...> is an error call', () => {
    const r = parseToolCalls('chatml-xml', '<tool_call>\n{"name": "a"}\n</tool_call>')
    expect(r.calls[0].error).toMatch(/no <function=/)
  })

  it('chatml-xml: an unterminated <function> inside a closed block is an error', () => {
    const r = parseToolCalls('chatml-xml', '<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</tool_call>')
    expect(r.partial).toBe(false)
    expect(r.calls[0].name).toBe('f')
    expect(r.calls[0].error).toMatch(/unterminated/)
  })

  it('chatml: an UNTERMINATED block is partial — no call, nothing rendered', () => {
    const r = parseToolCalls('chatml-json', 'Checking.\n<tool_call>\n{"name": "get_weather", "argu')
    expect(r.partial).toBe(true)
    expect(r.calls).toEqual([])
    // The half-written block is withheld: rendering it would flash on screen
    // and vanish when the closing tag lands.
    expect(r.text).toBe('Checking.\n')
  })

  it('chatml: withholds a trailing PREFIX of the open marker', () => {
    const r = parseToolCalls('chatml-json', 'Checking.\n<tool_c')
    expect(r.partial).toBe(true)
    expect(r.text).toBe('Checking.\n')
  })

  it('chatml: a trailing "<" is held back but is not called partial', () => {
    // A lone `<` is not evidence a call is coming: flagging it would make a
    // finished reply that happens to end in `<` look truncated. It is still
    // withheld, because `text` must never shrink between tokens.
    const r = parseToolCalls('chatml-json', 'compare a < b, or <div>, or a <')
    expect(r.partial).toBe(false)
    expect(r.text).toBe('compare a < b, or <div>, or a ')
  })

  it('llama3: an unbalanced object at the tail is partial', () => {
    const r = parseToolCalls('llama3', 'Sure.\n{"name": "get_weather", "parameters": {"city"')
    expect(r.partial).toBe(true)
    expect(r.calls).toEqual([])
    expect(r.text).toBe('Sure.\n')
  })

  it('llama3: an object that is not call-shaped stays PROSE', () => {
    // No markers, so nothing proves intent — the asymmetry with ChatML is
    // deliberate. A JSON answer the user asked for must not vanish into a
    // failed call.
    const r = parseToolCalls('llama3', 'Here is the JSON you asked for: {"city": "X", "temp": 18}')
    expect(r.calls).toEqual([])
    expect(r.text).toBe('Here is the JSON you asked for: {"city": "X", "temp": 18}')
    expect(r.partial).toBe(false)
  })

  it('llama3: unparseable braces stay prose and do not spin the scanner', () => {
    const r = parseToolCalls('llama3', 'a {not json} b {"name": "f", "parameters": {}} c')
    expect(r.calls.map((c) => c.name)).toEqual(['f'])
    expect(r.text).toBe('a {not json} b  c')
  })
})

describe('parseToolCalls — incremental safety', () => {
  // What chat.ts actually does: re-decode every id and re-parse the whole
  // string on every token. So parse runs on every prefix of the output, and a
  // call it has already reported must never change under it.
  // The third element is the call list each stream is SUPPOSED to end with,
  // read off the literal above it. It is the anchor, not decoration: every
  // assertion in the sweep below is stated relative to `full`, so a parser that
  // found nothing at all would satisfy all of them vacuously (0 <= 0, an empty
  // inner loop, and '' is a prefix of everything). Pinning the names is what
  // makes the sweep able to fail.
  const STREAMS: Array<[string, 'chatml-json' | 'chatml-xml' | 'llama3', string[]]> = [
    ['Let me look that up.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "İstanbul", "note": "a } in a string"}}\n</tool_call>\n<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>\nDone.',
      'chatml-json', ['get_weather', 'b']],
    ['Reasoning first.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nİstanbul\n</parameter>\n</function>\n</tool_call>',
      'chatml-xml', ['get_weather']],
    ['Sure.\n{"name": "get_weather", "parameters": {"city": "İstanbul"}}; {"name": "b", "parameters": {}}\nDone.',
      'llama3', ['get_weather', 'b']],
  ]

  for (const [stream, dialect, names] of STREAMS) {
    it(`${dialect}: no prefix claims a call it has not finished reading`, () => {
      const full = parseToolCalls(dialect, stream)
      expect(full.partial).toBe(false)
      expect(full.calls.map((c) => c.name)).toEqual(names)
      expect(full.calls.every((c) => c.error === undefined)).toBe(true)
      // n runs to stream.length inclusive, so the final iteration IS the whole
      // output — there is no separate endpoint case to check.
      for (let n = 1; n <= stream.length; n++) {
        const r = parseToolCalls(dialect, stream.slice(0, n))
        // Never more calls than the finished output has, and every call
        // already reported is byte-identical to the final one — a call that
        // could still change is a call that must not have been reported.
        expect(r.calls.length).toBeLessThanOrEqual(full.calls.length)
        for (let i = 0; i < r.calls.length; i++) expect(r.calls[i]).toEqual(full.calls[i])
        // Text only grows, and only toward the final text.
        expect(full.text.startsWith(r.text)).toBe(true)
      }
    })
  }
})

describe('chatml-xml: values that contain the closing tag', () => {
  it('keeps a parameter value that itself contains </parameter>', () => {
    // A file body is an ordinary argument, and one quoting this dialect used
    // to truncate at the first inner close and hand the tool a partial file.
    const body = '<function=write>\n<parameter=content>\n<parameter=x>oops</parameter>\n</parameter>\n</function>'
    const { calls } = parseToolCalls('chatml-xml', `<tool_call>${body}</tool_call>`)
    expect(calls).toHaveLength(1)
    expect(calls[0].error).toBeUndefined()
    expect(calls[0].arguments.content).toBe('<parameter=x>oops</parameter>')
  })

  it('still splits two ordinary parameters correctly', () => {
    const body = '<function=write>\n<parameter=path>\na.ts\n</parameter>\n<parameter=content>\nconst a = 1\n</parameter>\n</function>'
    const { calls } = parseToolCalls('chatml-xml', `<tool_call>${body}</tool_call>`)
    expect(calls[0].arguments).toEqual({ path: 'a.ts', content: 'const a = 1' })
  })

  it('flags a truncated call rather than returning a usable-looking one', () => {
    // What the token budget produces when it cuts the model off mid-call. The
    // host must not forward this: the client fails the tool and retries the
    // identical prompt for the identical result.
    const body = '<function=write>\n<parameter=content>\nconst a = 1'
    const { calls } = parseToolCalls('chatml-xml', `<tool_call>${body}`)
    if (calls.length) expect(calls[0].error).toBeTruthy()
  })
})
