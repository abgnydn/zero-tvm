// The station's translation from a load request to the engine's argv.
//
// Worth pinning because the failures in this translation are SILENT. The one
// that shipped: `pool` means expert slots, it was passed as `--pool`, and
// `--pool` means the KV prefix cache on disk — so the default build sent
// `--pool 0` and turned off the cache that lets a prefill survive a restart.
// Every request still answered. Reloads just went back to costing minutes.
//
// Each case below states the flag's DEFAULT as well as its opt-out, because a
// flag that is silently always-on and a flag that is silently always-off look
// identical from the outside.

import { describe, expect, it } from 'vitest'
import { engineArgs, type LoadRequest } from '../../scripts/native/engine-args.ts'

const args = (o: Partial<LoadRequest> = {}) => engineArgs({ param: 'qwen36q3', port: 8019, ...o })

describe('engineArgs', () => {
  it('a bare request names the model and the port, and nothing else', () => {
    expect(args()).toEqual(['qwen36q3', '--port', '8019'])
  })

  it('expert slots go to --experts, never --pool', () => {
    // --pool is the KV disk cache. Sending slots there is the shipped bug.
    expect(args({ pool: 12 })).toEqual(['qwen36q3', '--port', '8019', '--experts', '12'])
    expect(args({ pool: 12 })).not.toContain('--pool')
  })

  it('zero expert slots emits no flag at all — the default build', () => {
    expect(args({ pool: 0 })).toEqual(['qwen36q3', '--port', '8019'])
  })

  it('int8 KV is ON unless explicitly refused', () => {
    expect(args()).not.toContain('--kv8')
    expect(args({ kv8: true })).not.toContain('--kv8')
    expect(args({ kv8: false }).join(' ')).toContain('--kv8 0')
  })

  it('prefix reuse is ON unless explicitly refused', () => {
    // The diagnostic arm: every turn prefills from zero. Absent means reuse,
    // which is what every multi-turn run on this surface has been doing.
    expect(args()).not.toContain('--reuse')
    expect(args({ reuse: true })).not.toContain('--reuse')
    expect(args({ reuse: false }).join(' ')).toContain('--reuse 0')
  })

  it('chunked prefill is ON unless explicitly refused', () => {
    expect(args()).not.toContain('--chunk')
    expect(args({ chunk: true })).not.toContain('--chunk')
    expect(args({ chunk: false }).join(' ')).toContain('--chunk 0')
  })

  it('the three length-dependent subsystems can be refused independently', () => {
    // int8 KV, chunked prefill and cross-turn reuse are the only things whose
    // behaviour changes with prompt LENGTH, so a depth-dependent defect is
    // bisected by turning them off one at a time. That only works if each flag
    // is separate — one combined "safe mode" would prove nothing about which.
    expect(args({ kv8: false }).join(' ')).not.toContain('--chunk')
    expect(args({ chunk: false }).join(' ')).not.toContain('--kv8')
    expect(args({ reuse: false }).join(' ')).not.toContain('--chunk')
    expect(args({ kv8: false, chunk: false, reuse: false }).join(' '))
      .toContain('--kv8 0 --reuse 0 --chunk 0')
  })

  it('carries a chunk cap only when one was chosen', () => {
    // Varying the cap changes how many chunk BOUNDARIES a prompt crosses while
    // staying on the chunked path, which is the cheap way to test a
    // boundary-dependent defect: seconds per arm instead of the 46 minutes a
    // per-token run costs at 16k.
    expect(args()).not.toContain('--cap')
    expect(args({ cap: 0 })).not.toContain('--cap')
    expect(args({ cap: 4096 }).join(' ')).toContain('--cap 4096')
  })

  it('carries ctx only when one was chosen', () => {
    expect(args({ ctx: 32768 }).join(' ')).toContain('--ctx 32768')
    expect(args({ ctx: 0 })).not.toContain('--ctx')
  })

  it('composes the flags a diagnostic run actually asks for', () => {
    expect(args({ ctx: 32768, kv8: false, reuse: false, chunk: false })).toEqual([
      'qwen36q3', '--port', '8019', '--ctx', '32768', '--kv8', '0', '--reuse', '0', '--chunk', '0',
    ])
  })
})
