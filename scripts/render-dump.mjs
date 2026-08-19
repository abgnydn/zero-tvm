/**
 * Render a conversation through the HOST's own path and print the text.
 *
 * The node half of scripts/render-diff.py, which diffs this against the
 * checkpoint's jinja. Kept separate because the two halves cannot run in one
 * process: jinja needs transformers, and the host path is TypeScript.
 *
 * Imports the same modules the host does — messages.ts for normalize+fold,
 * tool-calls.ts for the tools block, tokenizer-bpe.ts for the template. Not
 * buildChatPromptFor, only because model-select.ts pulls in weight-loader,
 * which reads GPUBufferUsage at module scope and cannot load under plain Node.
 * The registry lookup below asserts the dispatch it would have made.
 *
 *   node scripts/render-dump.mjs <case.json>
 */
import { readFileSync } from 'node:fs'
import { makeNormalizer } from './native/messages.ts'
import { renderAssistantCalls, foldToolResults, withTools, toolDialectFor } from '../src/zero-tvm/tool-calls.ts'
import { buildChatPrompt } from '../src/zero-tvm/tokenizer-bpe.ts'
import { specForParam } from '../src/zero-tvm/model-registry.ts'

const { messages, tools, param } = JSON.parse(readFileSync(process.argv[2], 'utf8'))
const spec = specForParam(param)
const GENERATION = { chatml: 'qwen3', 'chatml-q35': 'qwen35', 'chatml-q38': 'qwen38' }
const generation = GENERATION[spec.chatTemplateId]
if (!generation) {
  throw new Error(`${param} renders with '${spec.chatTemplateId}' — this dump only covers the chatml family`)
}
// The same derivation both hosts use.
const dialect = toolDialectFor(spec.chatTemplateId)

const { normalize } = makeNormalizer({ renderAssistantCalls, foldToolResults }, dialect)
let turns = normalize(messages)
if (tools?.length) turns = withTools(dialect, turns, tools)

// Hand back the text instead of ids — what is being compared is the prompt.
const capture = { encode: (t) => [t] }
process.stdout.write(buildChatPrompt(turns, capture, { thinking: false, generation })[0])
