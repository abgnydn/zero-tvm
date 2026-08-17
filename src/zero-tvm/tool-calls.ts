/**
 * TOOL CALLS — the pure half of tool use: rendering tools INTO a prompt, and
 * parsing a model's raw output back OUT into { text, calls }.
 *
 * No engine, no execution, no I/O. Every function here is a function of its
 * arguments, so the whole surface is pinned headlessly by
 * tests/unit/tool-calls.test.ts. The decode loop that actually runs a call and
 * feeds the result back is a separate module.
 *
 * THREE dialects, not two, and the third was a surprise. `chatTemplateId` does
 * NOT discriminate them: qwen3-4b, qwen3-30b-a3b, qwen35-4b and qwen36-35b-a3b
 * all declare 'chatml' in model-spec.ts and they do not agree on the call
 * format.
 *
 *   'chatml-json'  Qwen3-4B, Qwen3-30B-A3B (the Qwen3-era ChatML tunes): a
 *                  `# Tools` section folded into the system turn holding one
 *                  `tool | tojson` per line inside <tools></tools>; a call
 *                  comes back as
 *                    <tool_call>\n{"name": "x", "arguments": {..}}\n</tool_call>
 *   'chatml-xml'   Qwen3.6-35B-A3B: the same <tools> block (different prose),
 *                  but the CALL is nested XML —
 *                    <tool_call>\n<function=x>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>
 *                  — and parameter values arrive as RAW TEXT, not JSON.
 *   'llama3'       Llama-3.1/3.2: no markers at all. `Environment: ipython` in
 *                  the system block, the JSON schemas inside the FIRST USER
 *                  message, and a call is a bare
 *                    {"name": "x", "parameters": {..}}
 *                  object in the assistant turn — "parameters", not
 *                  "arguments", and one call per message maximum.
 *
 * Ground truth for all three is the `chat_template` in each checkpoint's own
 * tokenizer_config.json, read out of .weights-local — not from memory of what
 * these formats look like. A tool prompt that is subtly wrong does not error:
 * the model just stops calling tools, or calls them in a shape the parser never
 * matches, and nothing points back here. qwen35-4b is the one shipped spec
 * whose template was NOT read (no local checkout): read its own
 * tokenizer_config before assuming it is 'chatml-json' — Qwen3.6 is proof that
 * the family name does not settle it.
 *
 * The tool schema itself is NOT ours. It is the OpenAI/HF function shape,
 * `{type:"function", function:{name, description, parameters}}`, because that
 * object is interpolated verbatim by `tool | tojson` in every one of those
 * templates — whatever the caller puts in it is what the model reads.
 *
 * INCREMENTAL SAFETY. chat.ts re-decodes the whole id array per token and hands
 * the renderer the full string, so parseToolCalls() runs on a growing PREFIX of
 * the output dozens of times a second. It therefore never reports a call it has
 * not finished reading, and `partial` says the withheld tail is the start of
 * one. What the caller does with a partial depends on whether the stream is
 * still open:
 *
 *   still generating — execute nothing, and render only `text`. The withheld
 *     tail is a half-written call; rendering it makes the block flash on screen
 *     and then vanish when the closing tag lands.
 *   generation ENDED — the model ran out of token budget mid-call. That is an
 *     error, not a pending call: there is nothing to execute and nothing to
 *     wait for, so surface the truncation instead of guessing at the arguments.
 *
 * Two properties the streaming caller can lean on, swept over every prefix of a
 * real output by the test: `text` never shrinks, and a call once reported never
 * changes. Both are why nothing here is retroactive.
 *
 * Malformed JSON inside a well-formed <tool_call> block is the common failure,
 * not a theoretical one — see the `error` field on ToolCall.
 */

// ============================================================
// Types
// ============================================================

/** OpenAI/HF function-calling shape — serialized into the prompt as-is. */
export interface ToolDef {
  type: 'function'
  function: {
    name: string
    description?: string
    /** JSON Schema object. Passed through untouched; we never read it. */
    parameters?: Record<string, unknown>
  }
}

export type ToolDialect = 'chatml-json' | 'chatml-xml' | 'llama3'

/**
 * One call. The render side needs only `name` + `arguments`; the parse side
 * fills the other two.
 */
export interface ToolCall {
  name: string
  arguments: Record<string, unknown>
  /** Parse side: the block body exactly as the model wrote it. This is what an
   *  error turn echoes back — the model cannot fix a call it cannot see. */
  raw?: string
  /** Parse side: set when the block did NOT yield a runnable call. `arguments`
   *  is then whatever survived (often {}). Callers must check this before
   *  dispatching: the answer to a malformed call is a tool_response describing
   *  the failure, which is the only thing that gets a model back on track.
   *  Dropping the call silently instead just makes it repeat itself. */
  error?: string
}

export interface ParsedOutput {
  /** The output verbatim, minus the call blocks (and minus the withheld tail
   *  when `partial`). Prose before and after a call survives. */
  text: string
  calls: ToolCall[]
  /** The tail is the beginning of a call that has not finished arriving. */
  partial: boolean
}

/**
 * A chat message, widened with the one role the tool turns need: Llama-3 puts
 * a tool result in an `ipython` turn (Qwen puts it in a `user` turn, so the
 * ChatML dialects need no new role). model-select.ts's ChatMessage must widen
 * to this before these messages can be handed to buildChatPromptFor.
 */
export interface ToolChatMessage {
  role: 'system' | 'user' | 'assistant' | 'ipython'
  content: string
}

// ============================================================
// JSON serialization that matches the templates' `tojson`
// ============================================================

/**
 * `json.dumps(v, ensure_ascii=False)` — which is what HF's jinja `tojson`
 * filter is bound to, NOT what JSON.stringify emits.
 *
 * The only difference that matters is separators: python defaults to `", "`
 * and `": "` where JSON.stringify writes `,` and `:`. Every byte of the tools
 * block is prompt, so those spaces change the token ids of every schema the
 * model reads — this repo pins templates token-exact against mlx_lm elsewhere
 * (buildLlama3ChatPrompt) and there is no reason to be looser here.
 *
 * Two known and accepted differences: floats that are integral render as `1`
 * where python renders `1.0`, and JS reorders integer-like object keys where
 * python preserves insertion order. Neither occurs in a JSON Schema written by
 * hand; both would need a serializer far larger than the problem.
 */
function tojson(v: unknown): string {
  if (v === null || v === undefined) return 'null'
  if (Array.isArray(v)) return `[${v.map(tojson).join(', ')}]`
  if (typeof v === 'object') {
    const o = v as Record<string, unknown>
    const parts: string[] = []
    // json.dumps keeps a None value; JSON.stringify drops an undefined one, so
    // skip undefined rather than emit a key python would never have seen.
    for (const k of Object.keys(o)) {
      if (o[k] === undefined) continue
      parts.push(`${JSON.stringify(k)}: ${tojson(o[k])}`)
    }
    return `{${parts.join(', ')}}`
  }
  // Strings, numbers, booleans: identical to python with ensure_ascii=False —
  // same short escapes, same \uXXXX for the rest, non-ASCII left raw.
  return JSON.stringify(v)
}

// ============================================================
// Render — tools into the prompt
// ============================================================

// Qwen3 / Qwen3-30B-A3B, verbatim from their chat_template.
const CHATML_JSON_HEAD =
  '# Tools\n\nYou may call one or more functions to assist with the user query.' +
  '\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>'
const CHATML_JSON_TAIL =
  '\n</tools>\n\nFor each function call, return a json object with function name' +
  ' and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n' +
  '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'

// Qwen3.6-35B-A3B, verbatim from its chat_template.
const CHATML_XML_HEAD = '# Tools\n\nYou have access to the following functions:\n\n<tools>'
const CHATML_XML_TAIL =
  '\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:' +
  '\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>' +
  '\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span' +
  '\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:' +
  '\n- Function calls MUST follow the specified format: an inner <function=...></function> block' +
  ' must be nested within <tool_call></tool_call> XML tags' +
  '\n- Required parameters MUST be specified' +
  '\n- You may provide optional reasoning for your function call in natural language BEFORE the' +
  ' function call, but NOT after' +
  '\n- If there is no function call available, answer the question like normal with your current' +
  ' knowledge and do not tell the user about function calls\n</IMPORTANT>'

// Llama-3.1/3.2, verbatim — including the missing space after "value}." The
// template concatenates two literals there with nothing between them; it reads
// like a typo and it is what the model was tuned on, so it stays.
const LLAMA3_TOOLS_PREAMBLE =
  'Given the following functions, please respond with a JSON for a function call ' +
  'with its proper arguments that best answers the given prompt.\n\n' +
  'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' +
  'Do not use variables.\n\n'

/**
 * Fold `tools` into `messages` so that rendering the RESULT through the
 * existing per-family chat-template builders reproduces the vendor's own
 * tools template. Nothing here emits control tokens — that stays in
 * tokenizer-bpe.ts, which is the only place that knows them.
 *
 * Per dialect:
 *  - chatml-json: the tools block goes into the system turn AFTER the caller's
 *    system text (a system message is created when there is none, because the
 *    vendor emits the system turn whenever tools are present).
 *  - chatml-xml: same turn, but the tools block comes FIRST and the caller's
 *    system text is appended after a blank line — Qwen3.6 reversed the order.
 *  - llama3: the schemas ride in the FIRST USER message (the template's
 *    `tools_in_user_message` default), not the system turn.
 *
 * KNOWN GAP (llama3): the vendor also prepends `Environment: ipython\n` to the
 * system block, ahead of the `Cutting Knowledge Date` line. That line cannot be
 * injected through a message — buildLlama3ChatPrompt hardcodes the order — so
 * it needs an option there. Reported, not worked around: faking it by prefixing
 * the system message would put it AFTER the date, which is not the prompt the
 * model was tuned on either.
 */
export function withTools(
  dialect: ToolDialect,
  messages: ToolChatMessage[],
  tools: ToolDef[],
): ToolChatMessage[] {
  if (tools.length === 0) return messages.slice()
  const schemas = tools.map((t) => `\n${tojson(t)}`).join('')

  if (dialect === 'llama3') {
    const out = messages.slice()
    const first = out.findIndex((m) => m.role === 'user')
    // The vendor template raise_exception()s here. A tools turn with nothing
    // for the model to answer is a caller bug, and a silent no-op would show
    // up much later as "the model ignores my tools".
    if (first < 0) throw new Error('Cannot put tools in the first user message when there is no user message')
    const each = tools.map((t) => `${JSON.stringify(t, null, 4)}\n\n`).join('')
    // json.dumps(indent=4) and JSON.stringify(v, null, 4) agree byte for byte —
    // with an indent python switches its item separator to a bare ',', so the
    // `", "` correction above does not apply on this path.
    out[first] = { role: 'user', content: LLAMA3_TOOLS_PREAMBLE + each + out[first].content.trim() }
    return out
  }

  const block = dialect === 'chatml-xml'
    ? CHATML_XML_HEAD + schemas + CHATML_XML_TAIL
    : CHATML_JSON_HEAD + schemas + CHATML_JSON_TAIL
  const out = messages.slice()
  const sys = out[0]?.role === 'system' ? out[0].content : null
  let content: string
  if (dialect === 'chatml-xml') {
    // Qwen3.6 trims the system content and drops the separator entirely when
    // it is empty.
    const trimmed = (sys ?? '').trim()
    content = trimmed ? `${block}\n\n${trimmed}` : block
  } else {
    content = sys ? `${sys}\n\n${block}` : block
  }
  if (sys === null) out.unshift({ role: 'system', content })
  else out[0] = { role: 'system', content }
  return out
}

/**
 * The assistant turn's CONTENT for a reply that made calls — what goes back
 * into the transcript on the next turn, so the model sees its own call in the
 * form its template writes rather than the form it happened to emit.
 *
 * `content` is the prose the model wrote alongside the calls (usually empty).
 */
export function renderAssistantCalls(
  dialect: ToolDialect,
  content: string,
  calls: ToolCall[],
): string {
  if (calls.length === 0) return content

  if (dialect === 'llama3') {
    // The vendor template raise_exception()s on more than one. Rendering only
    // the first would drop a call the engine already ran, and the transcript
    // would disagree with what happened.
    if (calls.length > 1) throw new Error('llama3 renders a single tool call per assistant message')
    return `{"name": "${calls[0].name}", "parameters": ${tojson(calls[0].arguments)}}`
  }

  if (dialect === 'chatml-xml') {
    let out = content.trim()
    calls.forEach((c, i) => {
      out += i === 0 ? (out ? '\n\n' : '') : '\n'
      out += `<tool_call>\n<function=${c.name}>\n`
      for (const [k, v] of Object.entries(c.arguments)) {
        // Strings go in raw, everything else as JSON — the asymmetry is the
        // template's, and it is what parseToolCalls has to undo.
        out += `<parameter=${k}>\n${typeof v === 'string' ? v : tojson(v)}\n</parameter>\n`
      }
      out += '</function>\n</tool_call>'
    })
    return out
  }

  let out = content
  calls.forEach((c, i) => {
    if (i > 0 || content) out += '\n'
    out += `<tool_call>\n{"name": "${c.name}", "arguments": ${tojson(c.arguments)}}\n</tool_call>`
  })
  return out
}

/**
 * The turn(s) carrying tool results back to the model.
 *
 * ChatML batches every result of one round into a SINGLE user turn (the
 * template opens `<|im_start|>user` once and closes it after the last
 * consecutive tool message); Llama-3 gives each its own `ipython` turn.
 *
 * The Llama-3 quirk is real and looks like a bug: its template renders the
 * result with `tojson` whenever the content `is mapping or is iterable`, and
 * jinja counts a string as iterable — so a plain string result reaches the
 * model QUOTED. Reproduced deliberately; that is the prompt Meta tuned on.
 */
export function renderToolResults(dialect: ToolDialect, results: string[]): ToolChatMessage[] {
  if (dialect === 'llama3') {
    return results.map((r) => ({ role: 'ipython' as const, content: tojson(r) }))
  }
  return [{
    role: 'user',
    content: results.map((r) => `<tool_response>\n${r}\n</tool_response>`).join('\n'),
  }]
}

// ============================================================
// Parse
// ============================================================

const OPEN = '<tool_call>'
const CLOSE = '</tool_call>'

/**
 * The first balanced JSON object at or after `from`.
 *
 * `end` is exclusive; `end === -1` means the object is still open at the end of
 * `s` — a PREFIX, which the streaming caller must not mistake for a syntax
 * error. Returns null when there is no `{` at all.
 *
 * String-aware, including `\"` escapes. This scanner exists precisely because
 * the naive regex (`/\{.*\}/s` or a brace counter that ignores strings) breaks
 * on the arguments agentic tools actually take: a `code` or `patch` string full
 * of braces and quoted braces.
 */
function scanObject(s: string, from: number): { start: number; end: number } | null {
  const start = s.indexOf('{', from)
  if (start < 0) return null
  let depth = 0
  let inStr = false
  let esc = false
  for (let i = start; i < s.length; i++) {
    const ch = s[i]
    if (inStr) {
      if (esc) esc = false
      else if (ch === '\\') esc = true
      else if (ch === '"') inStr = false
      continue
    }
    if (ch === '"') inStr = true
    else if (ch === '{') depth++
    else if (ch === '}' && --depth === 0) return { start, end: i + 1 }
  }
  return { start, end: -1 }
}

/**
 * Length of the longest suffix of `s` (starting at or after `from`) that is a
 * proper prefix of `marker`, else 0. `<tool_call>` is one added token for a
 * Qwen tokenizer so it arrives whole — but a ChatML tune on a tokenizer that
 * lacks the added token (Llama-3's has neither) emits it piecewise, and a chat
 * UI that renders `<tool_c` for one frame and then swallows it looks broken.
 */
function trailingPrefix(s: string, from: number, marker: string): number {
  const max = Math.min(marker.length - 1, s.length - from)
  for (let k = max; k > 0; k--) {
    if (s.endsWith(marker.slice(0, k))) return k
  }
  return 0
}

/** `{name, arguments}` out of an already-parsed JSON value, or an error. */
function callFromJson(value: unknown, raw: string): ToolCall {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return { name: '', arguments: {}, raw, error: 'tool call is not a JSON object' }
  }
  const o = value as Record<string, unknown>
  if (typeof o.name !== 'string' || o.name === '') {
    return { name: '', arguments: {}, raw, error: 'tool call has no "name"' }
  }
  // The two vendor templates disagree on the key — Qwen writes "arguments",
  // Llama-3 writes "parameters" — and models cross-contaminate, so accept
  // either in either dialect rather than reject a call we can plainly read.
  let args = o.arguments ?? o.parameters
  if (typeof args === 'string') {
    // Qwen's own template branches on `tool_call.arguments is string`, so a
    // model emitting the arguments as an embedded JSON string is inside the
    // format, not outside it.
    try { args = JSON.parse(args) } catch { /* falls into the shape check */ }
  }
  // A zero-argument call may legitimately omit the key entirely.
  if (args === undefined || args === null) return { name: o.name, arguments: {}, raw }
  if (typeof args !== 'object' || Array.isArray(args)) {
    return { name: o.name, arguments: {}, raw, error: 'tool call arguments are not an object' }
  }
  return { name: o.name, arguments: args as Record<string, unknown>, raw }
}

/** Body of one <tool_call> block in the JSON dialect. */
function parseJsonBody(body: string): ToolCall {
  const scan = scanObject(body, 0)
  // Braces that never balance INSIDE a closed block are malformed, not
  // partial — the model already wrote </tool_call>, so nothing more is coming.
  if (!scan || scan.end < 0) return { name: '', arguments: {}, raw: body, error: 'no JSON object in <tool_call> block' }
  const slice = body.slice(scan.start, scan.end)
  let value: unknown
  try {
    value = JSON.parse(slice)
  } catch (e) {
    return { name: '', arguments: {}, raw: body, error: `invalid JSON in <tool_call> block: ${(e as Error).message}` }
  }
  return callFromJson(value, body)
}

/** Body of one <tool_call> block in the Qwen3.6 nested-XML dialect. */
function parseXmlBody(body: string): ToolCall {
  const open = body.indexOf('<function=')
  const nameEnd = open < 0 ? -1 : body.indexOf('>', open)
  if (open < 0 || nameEnd < 0) {
    return { name: '', arguments: {}, raw: body, error: 'no <function=...> in <tool_call> block' }
  }
  const name = body.slice(open + '<function='.length, nameEnd)
  const close = body.indexOf('</function>', nameEnd)
  if (close < 0) {
    return { name, arguments: {}, raw: body, error: 'unterminated <function=...> block' }
  }
  const args: Record<string, unknown> = {}
  const inner = body.slice(nameEnd + 1, close)
  let i = 0
  for (;;) {
    const p = inner.indexOf('<parameter=', i)
    if (p < 0) break
    const keyEnd = inner.indexOf('>', p)
    if (keyEnd < 0) return { name, arguments: args, raw: body, error: 'unterminated <parameter=...> tag' }
    // Find the CLOSING tag, not merely the first one. A file body is an
    // ordinary argument, and one quoting this dialect (`</parameter>` in the
    // text) used to truncate the value there and hand the tool a partial file.
    // The format has no escaping, so the disambiguator is what FOLLOWS: a real
    // close is the last thing before the next `<parameter=` or the end of the
    // block. Anything else is part of the value.
    let end = -1
    for (let q = inner.indexOf('</parameter>', keyEnd); q >= 0; q = inner.indexOf('</parameter>', q + 1)) {
      const after = inner.slice(q + '</parameter>'.length).trim()
      if (after === '' || after.startsWith('<parameter=')) { end = q; break }
    }
    if (end < 0) return { name, arguments: args, raw: body, error: 'unterminated <parameter> value' }
    const key = inner.slice(p + '<parameter='.length, keyEnd)
    // The template writes `<parameter=k>\n` + value + `\n</parameter>` — strip
    // exactly the one newline it added on each side, not all whitespace: a
    // value that is a code block ends in meaningful blank lines.
    let raw = inner.slice(keyEnd + 1, end)
    if (raw.startsWith('\n')) raw = raw.slice(1)
    if (raw.endsWith('\n')) raw = raw.slice(0, -1)
    args[key] = coerceXmlValue(raw)
    i = end + '</parameter>'.length
  }
  return { name, arguments: args, raw: body }
}

/**
 * Undo the XML dialect's write rule (strings raw, everything else `tojson`) as
 * far as it can be undone. Accept a JSON reading only when it is NOT a string,
 * so `42` → 42 and `{"a":1}` → an object, while `"quoted"` stays the literal
 * text the model wrote instead of losing its quotes.
 *
 * The format is genuinely lossy here: a string argument whose text is `42` is
 * indistinguishable from the number. Nothing in the block says which, so the
 * ambiguity is the template's, not ours — a tool whose schema wants a string
 * gets a number here and must coerce.
 */
function coerceXmlValue(raw: string): unknown {
  try {
    const v: unknown = JSON.parse(raw)
    return typeof v === 'string' ? raw : v
  } catch {
    return raw
  }
}

/**
 * Split a model's raw output into prose and calls. Safe on a PREFIX of the
 * output — see the incremental-safety note at the top of this file for what to
 * do with `partial`.
 */
export function parseToolCalls(dialect: ToolDialect, out: string): ParsedOutput {
  return dialect === 'llama3' ? parseLlama3(out) : parseChatML(dialect, out)
}

function parseChatML(dialect: ToolDialect, out: string): ParsedOutput {
  const calls: ToolCall[] = []
  let text = ''
  let partial = false
  let i = 0
  while (i < out.length) {
    const o = out.indexOf(OPEN, i)
    if (o < 0) {
      const held = trailingPrefix(out, i, OPEN)
      text += out.slice(i, out.length - held)
      // Withholding and flagging are separate thresholds on purpose. ANY
      // marker prefix is withheld, so `text` only ever grows — withholding at
      // two characters and not at one just moves the flicker to `<`. But a
      // lone `<` is not evidence that a call is coming, and flagging it would
      // make a finished reply that happens to end in `<` look truncated.
      partial = held >= 2
      break
    }
    text += out.slice(i, o)
    const bodyStart = o + OPEN.length
    const c = out.indexOf(CLOSE, bodyStart)
    if (c < 0) {
      // The block is still being written: hold back everything from the open
      // tag, including the tag itself.
      partial = true
      break
    }
    const body = out.slice(bodyStart, c)
    calls.push(dialect === 'chatml-xml' ? parseXmlBody(body) : parseJsonBody(body))
    i = c + CLOSE.length
  }
  return { text, calls, partial }
}

/**
 * Llama-3 has no call markers, so intent has to be inferred from shape: a
 * balanced object carrying a string `name` is a call, and anything else is
 * prose that happens to contain braces. That asymmetry with the ChatML
 * dialects is deliberate — there, the tags prove the model meant a call, so a
 * body that will not parse is an ERROR; here, nothing proves it, so an object
 * we cannot read stays in the text where the user can see it.
 *
 * Llama-3.1 separates consecutive calls with `;`, and that `;` stays in `text`.
 * Suppressing it would mean deciding, after the fact, that text already handed
 * to the caller was punctuation — and whether a trailing `; ` is a separator or
 * the start of a sentence is only knowable once the NEXT object closes. A
 * parser that revises what it already emitted breaks the one property the
 * streaming caller relies on (tests/unit/tool-calls.test.ts sweeps every
 * prefix); a stray semicolon between two calls does not.
 */
function parseLlama3(out: string): ParsedOutput {
  const calls: ToolCall[] = []
  let text = ''
  let partial = false
  let i = 0
  while (i < out.length) {
    const scan = scanObject(out, i)
    if (!scan) { text += out.slice(i); break }
    let gap = out.slice(i, scan.start)
    // <|python_tag|> is special:true, so our decoder already strips it — this
    // is for streams that did not come through tokenizer-bpe.ts's decode.
    if (gap.endsWith('<|python_tag|>')) gap = gap.slice(0, -'<|python_tag|>'.length)
    if (scan.end < 0) {
      text += gap
      partial = true
      break
    }
    const slice = out.slice(scan.start, scan.end)
    let call: ToolCall | null = null
    try {
      const parsed = callFromJson(JSON.parse(slice), slice)
      if (!parsed.error) call = parsed
    } catch { /* not JSON at all — prose with braces in it */ }
    // Either way, step past this object so the scan cannot loop on the same `{`.
    text += call ? gap : out.slice(i, scan.end)
    if (call) calls.push(call)
    i = scan.end
  }
  return { text, calls, partial }
}
