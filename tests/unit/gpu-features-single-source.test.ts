// Every inference surface must request the SAME GPU features.
//
// This is a source-text check, which is unusual here and deliberate: the thing
// being asserted is that no *call site* invents its own list, and there is no
// runtime value that can express that. A unit test on the constant would pass
// while five callers ignored it — which is exactly what happened.
//
// What happened: `optionalFeatures` was a parameter of bootEngine, and the six
// callers asked for six different things. The repo therefore shipped six
// engines while its comments said it shipped one.
//
//   chat-flow    subgroups + subgroup-matrix (+ timestamp-query)  cap 1024, E5
//   share host   subgroups only                                   cap 64, batched_dyn
//   share stage  subgroups only                                   cap 64, batched_dyn
//   agent-host   subgroups + subgroup-matrix                      cap 1024, E5
//   validate     NOTHING                                          scalar everything
//   lib          subgroups + subgroup-matrix                      cap 1024, E5
//
// Three live consequences, none of which raised anything: share.ts claimed the
// two hosting surfaces "cannot drift" while running a different prefill GEMM at
// a different chunk cap; validate.html refused every MoE model at 3%, because
// the MoE probe needs a `subgroups` feature the page never requested; and
// timestamp-query — suspected of serialising Metal command execution — was
// requested on the browser path and not the native one, so BENCH.md compared
// measurements taken under different conditions.

import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { ENGINE_GPU_FEATURES, PROFILE_GPU_FEATURES } from '../../src/zero-tvm/variants.ts'

const ROOT = join(import.meta.dirname, '../..')
const read = (rel: string): string => readFileSync(join(ROOT, rel), 'utf8')

/** Every module that boots an engine or builds a device. A new one belongs
 *  here; if it is missing, this file cannot see it, which is the failure mode
 *  the list itself is guarding against. */
const BOOT_SITES = [
  'src/zero-tvm/chat-flow.ts',
  'src/zero-tvm/share.ts',
  'src/zero-tvm/agent-host.ts',
  'src/zero-tvm/validate.ts',
  'src/lib/index.ts',
]

describe('the shared feature list', () => {
  it('names the features the engine ladder actually gates on', () => {
    expect(ENGINE_GPU_FEATURES).toContain('subgroups')
    expect(ENGINE_GPU_FEATURES, 'the matrix unit gates sgmat/E5 and, through sgmatAvail, CHUNK_CAP 1024')
      .toContain('chromium-experimental-subgroup-matrix')
  })

  it('keeps timestamp-query OUT of the default list', () => {
    // It is suspected of serialising Metal command execution. A surface that
    // requests it is not measuring the same machine as one that does not, so
    // it is opt-in per run and never shared.
    expect(ENGINE_GPU_FEATURES).not.toContain('timestamp-query')
    expect(PROFILE_GPU_FEATURES).toContain('timestamp-query')
  })
})

describe('no boot site invents its own feature list', () => {
  it.each(BOOT_SITES)('%s references the shared constant', (rel) => {
    const src = read(rel)
    expect(src, `${rel} boots an engine without ENGINE_GPU_FEATURES`)
      .toContain('ENGINE_GPU_FEATURES')
  })

  it.each(BOOT_SITES)('%s passes the constant, not a literal array', (rel) => {
    // Every `optionalFeatures:` must be the shared constant or a spread of it.
    // A literal array is a caller enumerating features itself, which is how the
    // six drifted apart. Written as a direct check on the call site rather than
    // a filtered grep: the first version of this test filtered its own matches
    // away and passed with share.ts reverted — vacuous, which is the fault this
    // whole file is about.
    const src = read(rel)
    const bad = [...src.matchAll(/optionalFeatures:\s*(\[[^\]]*\]|[A-Za-z_$][\w$]*)/g)]
      .map((m) => m[1].replace(/\s+/g, ' ').trim())
      .filter((v) => !(v === 'ENGINE_GPU_FEATURES' || v.startsWith('[...ENGINE_GPU_FEATURES')))
    expect(bad, `${rel} builds its own feature list: ${bad.join(' | ')}`).toEqual([])
  })

  it('lib/index.ts pushes from the constant rather than named literals', () => {
    // It builds its device by hand rather than through bootEngine, so the
    // check above cannot see it.
    const src = read('src/lib/index.ts')
    const pushed = [...src.matchAll(/features\.push\(\s*'([^']+)'/g)].map((m) => m[1])
    expect(pushed.filter((f) => f !== 'shader-f16'),
      'shader-f16 is required and named here; every optional feature comes from the constant')
      .toEqual([])
  })

  it('variants.ts still defines the constants this file imports', () => {
    // Named for what it does. It was called "the only module that spells the
    // feature names", which it does not check — it passed while share.ts was
    // reverted to a literal list. The it.each cases above do that work; this is
    // only a guard against the constants being renamed or moved, which would
    // make those cases pass vacuously.
    const v = read('src/zero-tvm/variants.ts')
    expect(v).toContain("'chromium-experimental-subgroup-matrix'")
    expect(v).toContain('export const ENGINE_GPU_FEATURES')
    expect(v).toContain('export const PROFILE_GPU_FEATURES')
  })
})
