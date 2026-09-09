// THE CHECK REGISTRY — one list, read by the CLI and the UI alike.
//
// It was two lists. release-check.mjs held thirteen rows with `needs` labels,
// and package.json held nineteen test scripts; neither knew about the other,
// four release-check rows were display templates carrying a literal `<param>`
// that cannot be executed, and nothing anywhere recorded what a check actually
// COVERS. That last omission is the expensive one: `gdn_chunk_chain` was green
// at six tokens against a shipped cap of 1024 — a true assertion about almost
// nothing — and no artifact in the repo could show that.
//
// So a row here carries three things a pass/fail list does not:
//
//   covers   the box the check runs in — spec, tokens, cap, gemm. An axis
//            that is ABSENT is uncovered, never "any". This is what makes a
//            green square at six tokens legible as the hole it is.
//   proves   the sentence the check entitles you to say.
//   args     the parameters it takes, so the UI can offer them instead of
//            printing `<param>` and leaving you to guess.
//
// Adding a row here is the only place a new check needs to be declared.

import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')

/** Models with weights on this disk, offered by the UI's param picker. */
export const PARAMS = [
  'qwen38', 'qwen36', 'qwen36q3', 'qwen35', 'qwen35mlx',
  'qwen3', 'qwen3mlx', 'qwen30b', 'llama32', 'embed', 'phi3',
]

/** `needs`
 *   'none'  runnable anywhere — gate on it.
 *   'gpu'   needs a real adapter.
 *   'model' needs a checkpoint on disk as well as a GPU.
 *
 *  `covers` — the axes this check exercises. Omit an axis it does not vary;
 *  an omitted axis is UNCOVERED. Never write 'any'.
 *
 *  `args` — [{ name, kind: 'param'|'int'|'flag', default, why }]. The UI
 *  renders these; `cmd` is a function of the chosen values.
 */
export const CHECKS = [
  // ── runnable anywhere ────────────────────────────────────────────────────
  {
    id: 'typecheck', name: 'typecheck', needs: 'none', group: 'static',
    cmd: () => ['npm', 'run', 'typecheck'],
    proves: 'the types the tests and the site compile against agree',
    covers: {},
  },
  {
    id: 'unit', name: 'unit + tokenizer', needs: 'none', group: 'static',
    cmd: () => ['npm', 'run', 'test:unit'],
    proves: 'every unit assertion holds at the scale the unit tests use',
    covers: { tokens: '≤ a few hundred' },
  },
  {
    id: 'build-site', name: 'build (site)', needs: 'none', group: 'static',
    cmd: () => ['npm', 'run', 'build'],
    proves: 'the shipped pages bundle',
    covers: {},
  },
  {
    id: 'build-lib', name: 'build (lib)', needs: 'none', group: 'static',
    cmd: () => ['npm', 'run', 'build:lib'],
    proves: 'the library entry bundles',
    covers: {},
  },
  {
    id: 'mutation-gate', name: 'mutation gate', needs: 'none', group: 'static',
    cmd: () => ['node', 'scripts/mutation-gate.mjs'],
    proves: 'each shipped defect is caught BY THE TEST WHOSE NAME CLAIMS IT — '
      + 'not merely that the suite goes red somewhere',
    covers: { defects: '10 that actually shipped' },
    warn: 'mutates src/ in place and is not concurrency-safe — never run two at once',
  },
  {
    id: 'facts', name: 'facts registry', needs: 'none', group: 'static',
    cmd: () => ['node', 'bin/check-facts.mjs'],
    cwd: join(ROOT, '..', 'sites-shared'),
    proves: 'no published number is unregistered or drifted',
    covers: {},
  },

  // ── needs a GPU ──────────────────────────────────────────────────────────
  {
    id: 'kernels', name: 'kernel numerics (all)', needs: 'gpu', group: 'kernels',
    cmd: () => ['npm', 'run', 'test:kernels'],
    proves: 'every kernel matches its CPU reference at the sizes the suite uses',
    covers: { tokens: '5–6', spec: 'synthetic dims' },
    gap: 'five tokens against a shipped cap of 1024 — the scale gap that let qwen38 ship',
  },
  {
    id: 'kernels-qwen35', name: 'kernel numerics (qwen35)', needs: 'gpu', group: 'kernels',
    cmd: () => ['npm', 'run', 'test:kernels:qwen35'],
    proves: 'the hybrid GDN + int8 pack at head-dim 256',
    covers: { spec: 'qwen35', tokens: '6, plus one 1024 arm', gemm: 'int4_matmul_batched_dyn' },
    gap: 'the 1024 arm runs batched_dyn; cap 1024 only happens when subgroup-matrix '
      + 'exists, which is exactly when the engine picks E5 — arm and shipped config never coincide',
  },
  {
    id: 'kernels-mlx', name: 'kernel numerics (mlx loader)', needs: 'none', group: 'kernels',
    cmd: () => ['npm', 'run', 'test:kernels:mlx'],
    proves: 'the repack is byte-identical to the mlx-validated bundle',
    covers: { spec: 'qwen36', tokens: 'n/a — byte comparison' },
    note: 'no GPU needed; needs the 19 GB checkpoint or it skips',
  },
  {
    id: 'e2e', name: 'e2e (browser)', needs: 'gpu', group: 'kernels',
    cmd: () => ['npm', 'run', 'test:e2e'],
    proves: 'the shipped pages answer a five-prompt battery in a real browser',
    covers: { spec: '4 of 11', tokens: 'short' },
  },

  // ── evals: catch "fluent but wrong" ──────────────────────────────────────
  {
    id: 'chunk-identity', name: 'chunked prefill identity', needs: 'model', group: 'eval',
    args: [
      { name: 'param', kind: 'param', default: 'qwen35', why: 'which spec' },
      { name: 'PROMPT', kind: 'int', default: 16000, why: 'prompt tokens — the whole point is depth' },
      { name: 'CAP', kind: 'int', default: 1024, why: 'chunk cap under test' },
    ],
    cmd: (a) => ['node', 'scripts/chunk-prefill-test.mjs', a.param],
    env: (a) => ({ PROMPT: String(a.PROMPT), CAP: String(a.CAP) }),
    proves: 'chunked prefill emits the same tokens as the per-token path',
    covers: { spec: 'chosen', tokens: 'chosen', cap: 'chosen' },
    why: 'chunking changes arithmetic ORDER; token identity is the only honest check',
  },
  {
    id: 'render-diff', name: 'render vs vendor template', needs: 'model', group: 'eval',
    args: [
      { name: 'param', kind: 'param', default: 'qwen38', why: 'our spec' },
      { name: 'model', kind: 'text', default: '.weights-local/Qwen3.8-27B-4bit', why: 'checkpoint dir' },
      { name: 'shapes', kind: 'flag', default: true, why: 'the awkward-conversation battery' },
      { name: 'depth', kind: 'int', default: 0, why: '0 = structural; raise to check length changes nothing' },
    ],
    cmd: (a) => ['uv', 'run', 'python', join(ROOT, 'scripts/render-diff.py'),
      '--model', a.model.startsWith('/') ? a.model : join(ROOT, a.model),
      '--param', a.param,
      ...(a.shapes ? ['--shapes'] : []),
      ...(Number(a.depth) > 0 ? ['--depth', String(a.depth)] : [])],
    cwd: join(ROOT, '..', 'ml-research'),
    proves: 'our rendered prompt equals the checkpoint\'s own jinja',
    covers: { spec: 'chosen', shapes: '9 conversation shapes', tokens: 'chosen depth' },
    gap: 'covers ONLY the chatml family — llama3, phi3 and deepseek have no template check at all',
    warn: 'omitting --param renders the DEFAULT spec against the target\'s jinja and prints spurious DIFFERS',
  },
  {
    id: 'fidelity-depth-ref', name: 'fidelity at depth · 1. reference', needs: 'model', group: 'eval',
    args: [
      { name: 'model', kind: 'text', default: '.weights-local/Qwen3.8-27B-4bit', why: 'checkpoint dir' },
      { name: 'depth', kind: 'int', default: 16000, why: 'padding tokens before the question' },
      { name: 'out', kind: 'text', default: '/tmp/ref-qwen38-16k.json', why: 'reference dir' },
    ],
    cmd: (a) => ['uv', 'run', 'python', '-u', join(ROOT, 'scripts/mlx-ref.py'),
      '--model', a.model.startsWith('/') ? a.model : join(ROOT, a.model),
      '--depth', String(a.depth), '--out', a.out],
    cwd: join(ROOT, '..', 'ml-research'),
    env: () => ({ PYTHONUNBUFFERED: '1' }),
    proves: 'what mlx_lm answers at this depth — the oracle the engine is compared to',
    covers: { spec: 'chosen', tokens: 'chosen depth' },
    note: 'a 27B model at 16k took 24 min and ~15 GB here; close other apps first',
  },
  {
    id: 'fidelity-depth-engine', name: 'fidelity at depth · 2. engine', needs: 'model', group: 'eval',
    args: [
      { name: 'param', kind: 'param', default: 'qwen38', why: 'our spec' },
      { name: 'ref', kind: 'text', default: '/tmp/ref-qwen38-16k.json', why: 'reference from step 1' },
      { name: 'ZTVM_PROTOCOL_MIN', kind: 'int', default: 60, why: 'puppeteer protocol timeout, minutes — '
        + '10 is not enough for a 20k prefill and it fails with the engine working' },
    ],
    cmd: (a) => ['node', 'scripts/validate-model.mjs', a.param, '--ref', a.ref],
    env: (a) => ({ ZTVM_PROTOCOL_MIN: String(a.ZTVM_PROTOCOL_MIN), ZTVM_PAGE_LOG: '1' }),
    proves: 'the engine agrees with mlx_lm at depth, not just at ~20 tokens',
    covers: { spec: 'chosen', tokens: 'the reference\'s depth' },
    why: 'THE gap that let qwen38 ship broken — validate-model alone checks ~20 tokens',
  },
  {
    id: 'agentic', name: 'agentic loop', needs: 'model', group: 'eval',
    args: [
      { name: 'param', kind: 'param', default: 'qwen38', why: 'which spec' },
      { name: 'PAD', kind: 'int', default: 24000, why: 'context padding — short AND deep both matter' },
    ],
    cmd: (a) => ['node', 'scripts/agentic-eval.mjs', a.param],
    env: (a) => ({ PAD: String(a.PAD) }),
    proves: 'the model still calls tools correctly with a long history',
    covers: { spec: 'chosen', tokens: 'chosen PAD' },
    gap: 'ends at console.log and exits 0 whatever it finds',
  },
  {
    id: 'needle', name: 'long-context retrieval', needs: 'model', group: 'eval',
    args: [{ name: 'param', kind: 'param', default: 'qwen38', why: 'which spec' }],
    cmd: (a) => ['node', 'scripts/needle-test.mjs', a.param],
    proves: 'it can still see the start of its context from the end',
    covers: { spec: 'chosen' },
    gap: 'ends at console.log and exits 0 whatever it finds',
  },
]

export const byId = (id) => CHECKS.find((c) => c.id === id)

/** Resolve a check + arg values into what will actually be executed. */
export function resolve(check, values = {}) {
  const a = {}
  for (const arg of check.args ?? []) a[arg.name] = values[arg.name] ?? arg.default
  return {
    cmd: check.cmd(a),
    cwd: check.cwd ?? ROOT,
    env: check.env ? check.env(a) : {},
    values: a,
  }
}
