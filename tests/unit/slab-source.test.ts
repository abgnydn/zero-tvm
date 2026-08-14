// SLAB SOURCE — offset math against a synthetic checkpoint.
//
// The failure this guards is not a crash. An expert slab read at the wrong
// offset is still 2.5 MiB of real quantized weights, so the model keeps
// generating fluent text and nothing anywhere reports an error. The only place
// the mistake is visible is here, against a fixture whose every byte is known.
//
// So the fixture is a REAL safetensors container — 8-byte header length, JSON
// header with dtypes/shapes/data_offsets, data section — parsed by the loader's
// own `openMlxCheckpoint`, and the OPFS side is written by the loader's own
// `buildPlan`. Nothing about the layout is restated in this file; a fixture
// that agreed with a hand-written copy of the layout would prove nothing.
//
// Dims are tiny (N=2, K=2 words) but the expert COUNT is the spec's real 256,
// because the stack depth is exactly what the offsets multiply.

import { describe, expect, it, beforeAll, afterAll } from 'vitest'
import { closeSync, openSync, readFileSync, readSync, statSync } from 'node:fs'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DEEPSEEK_V2_LITE, QWEN36_35B_A3B } from '../../src/compiler/model-spec.ts'
import { planLayer } from '../../src/zero-tvm/mlx-weights.ts'
import { buildPlan, openMlxCheckpoint, planKey, type MlxSource } from '../../src/zero-tvm/weight-loader-mlx.ts'
import {
  fetchSlabSource, opfsSlabSource,
  type SlabDirectory, type SlabFileHandle, type SlabKind, type SlabProj,
} from '../../src/zero-tvm/slab-source.ts'

const SPEC = QWEN36_35B_A3B          // 256 routed experts + a shared one
const EXPERTS = SPEC.moe!.experts
const SHARED = EXPERTS               // the shared expert's slab index
const LAYERS = [0, 1]
const N = 2                          // rows per expert
const K = 2                          // u32 words per row
const PROJS: SlabProj[] = ['gate', 'up', 'down']
const KINDS: SlabKind[] = ['w', 's', 'b']

const dtypeOf = (kind: SlabKind) => (kind === 'w' ? 'U32' : 'BF16')
const shapeOf = (kind: SlabKind) => (kind === 'w' ? [N, K] : [N, 1])
const strideOf = (kind: SlabKind) => (kind === 'w' ? N * K * 4 : N * 1 * 2)

// ── fixture content ────────────────────────────────────────────────────────
// Every slab is filled from (tag, expert) so a slab read one stride early, one
// row short, or from the wrong layer is a different byte string.
function slabContent(tag: number, expert: number, kind: SlabKind): Uint8Array {
  const bytes = new Uint8Array(strideOf(kind))
  if (kind === 'w') {
    for (let i = 0; i < bytes.length; i++) bytes[i] = (tag * 31 + expert * 7 + i * 13) & 0xff
    return bytes
  }
  // bf16 patterns held in a range f16 represents exactly, so the loader's
  // bf16->f16 conversion is lossless and cannot throw on overflow.
  const u16 = new Uint16Array(bytes.buffer)
  for (let i = 0; i < u16.length; i++) {
    u16[i] = ((0x74 + ((tag + expert + i) % 8)) << 7) | ((expert * 5 + i + tag) & 0x7f)
  }
  return bytes
}

/** Serialize one safetensors shard. `shrink` under-declares every tensor's
 *  byte range, which is the header self-inconsistency the source must catch. */
function shard(tensors: { name: string; dtype: string; shape: number[]; bytes: Uint8Array }[], shrink = 0): Uint8Array {
  const header: Record<string, unknown> = {}
  let off = 0
  for (const t of tensors) {
    header[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + t.bytes.length - shrink] }
    off += t.bytes.length
  }
  let json = JSON.stringify(header)
  while ((json.length + 8) % 8 !== 0) json += ' '        // safetensors pads the header
  const body = new Uint8Array(8 + json.length + off)
  new DataView(body.buffer).setUint32(0, json.length, true)
  body.set(new TextEncoder().encode(json), 8)
  let at = 8 + json.length
  for (const t of tensors) { body.set(t.bytes, at); at += t.bytes.length }
  return body
}

/**
 * Write a checkpoint whose expert tensors follow `planLayer`'s own record
 * names, with the stacked tensors in one shard and the shared expert's in
 * ANOTHER — the case the fetch path has to address across files.
 */
async function writeCheckpoint(
  dir: string,
  opts: { experts?: number; shrink?: number } = {},
): Promise<void> {
  const experts = opts.experts ?? EXPERTS
  const stacked: { name: string; dtype: string; shape: number[]; bytes: Uint8Array }[] = []
  const sharedT: typeof stacked = []
  const map: Record<string, string> = {}
  let tag = 0
  for (const L of LAYERS) {
    const plans = planLayer(SPEC, L)
    for (const proj of PROJS) {
      for (const kind of KINDS) {
        const plan = plans.find((p) => p.name === `moe_${proj}_proj_${kind}`)!
        tag++
        const stack = new Uint8Array(experts * strideOf(kind))
        for (let e = 0; e < experts; e++) stack.set(slabContent(tag, e, kind), e * strideOf(kind))
        stacked.push({ name: plan.parts[0].record, dtype: dtypeOf(kind), shape: [experts, ...shapeOf(kind)], bytes: stack })
        map[plan.parts[0].record] = 'shard-0.safetensors'
        sharedT.push({
          name: plan.parts[1].record, dtype: dtypeOf(kind), shape: shapeOf(kind),
          bytes: slabContent(tag, SHARED, kind),
        })
        map[plan.parts[1].record] = 'shard-1.safetensors'
      }
    }
  }
  await writeFile(join(dir, 'shard-0.safetensors'), shard(stacked, opts.shrink ?? 0))
  await writeFile(join(dir, 'shard-1.safetensors'), shard(sharedT))
  await writeFile(join(dir, 'model.safetensors.index.json'), JSON.stringify({ metadata: {}, weight_map: map }))
}

/** The loader's MlxSource over local files. `short` truncates every ranged
 *  read by one byte — a server or a disk handing back less than was asked. */
function fileSource(dir: string, short = false): MlxSource {
  const read = (f: string) => readFileSync(join(dir, f))
  return {
    whole: async (f) => new Uint8Array(read(f)),
    range: async (f, begin, end) => {
      const b = read(f)
      return new Uint8Array(b.buffer.slice(b.byteOffset + begin, b.byteOffset + end - (short ? 1 : 0)))
    },
  }
}

/** An OPFS directory backed by a real one. `sync: false` hides
 *  createSyncAccessHandle so the getFile() fallback runs; `shortRead` keeps
 *  the file's size honest but hands back one byte less than asked. */
function nodeDir(root: string, opts: { sync?: boolean; shortRead?: boolean } = {}): SlabDirectory {
  return {
    async getFileHandle(name: string): Promise<SlabFileHandle> {
      const path = join(root, name)
      statSync(path)                                   // OPFS rejects a missing file
      const handle: SlabFileHandle = {
        async getFile() {
          const data = readFileSync(path)
          return {
            size: data.byteLength,
            slice: (b, e) => ({
              arrayBuffer: async () => data.buffer.slice(data.byteOffset + b, data.byteOffset + e),
            }),
          }
        },
      }
      if (opts.sync !== false) {
        handle.createSyncAccessHandle = async () => {
          const fd = openSync(path, 'r')
          return {
            getSize: () => statSync(path).size,
            read: (buf, { at }) => readSync(fd, buf, 0, buf.byteLength - (opts.shortRead ? 1 : 0), at),
            close: () => closeSync(fd),
          }
        }
      }
      return handle
    },
  }
}

// ── the good checkpoint, shared by most cases ──────────────────────────────
let dir: string
let cache: string
let locate: Awaited<ReturnType<typeof openMlxCheckpoint>>['locate']

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), 'slab-source-'))
  cache = join(dir, 'opfs')
  await mkdir(cache)
  await writeCheckpoint(dir)
  const src = fileSource(dir)
  locate = (await openMlxCheckpoint(src)).locate
  // The OPFS side is what the LOADER writes: the built buffer, keyed by plan.
  for (const L of LAYERS) {
    for (const plan of planLayer(SPEC, L).filter((p) => p.name.startsWith('moe_'))) {
      await writeFile(join(cache, planKey(plan, L)), await buildPlan(plan, locate, src))
    }
  }
})

afterAll(async () => { await rm(dir, { recursive: true, force: true }) })

describe('slab geometry', () => {
  it('derives the stride from the header, not from the spec', () => {
    const s = fetchSlabSource(SPEC, locate, fileSource(dir))
    expect(s.slabBytes(0, 'gate', 'w')).toBe(N * K * 4)     // U32
    expect(s.slabBytes(0, 'gate', 's')).toBe(N * 2)         // BF16, one group per row
    expect(s.slabBytes(0, 'down', 'b')).toBe(N * 2)
  })

  it('rejects a header whose byte range and shape disagree', async () => {
    const bad = await mkdtemp(join(tmpdir(), 'slab-shrunk-'))
    await writeCheckpoint(bad, { shrink: 4 })
    const l = (await openMlxCheckpoint(fileSource(bad))).locate
    const s = fetchSlabSource(SPEC, l, fileSource(bad))
    expect(() => s.slabBytes(0, 'gate', 'w')).toThrow(/declares .* but .* is \d+ B/)
    await rm(bad, { recursive: true, force: true })
  })

  it('rejects a checkpoint stacked to a different depth than the spec claims', async () => {
    const bad = await mkdtemp(join(tmpdir(), 'slab-8e-'))
    await writeCheckpoint(bad, { experts: 8 })
    const l = (await openMlxCheckpoint(fileSource(bad))).locate
    const s = fetchSlabSource(SPEC, l, fileSource(bad))
    expect(() => s.slabBytes(0, 'gate', 'w')).toThrow(/256 experts but .* stacked 8 deep/)
    await rm(bad, { recursive: true, force: true })
  })

  it('refuses a layer that runs a dense FFN', () => {
    // DeepSeek-V2's layer 0 is dense while every other layer is MoE. There is
    // no expert stack to slice, and the dense FFN's records are a different
    // width — reading one as an expert slab is the silent-garbage case.
    const s = fetchSlabSource(DEEPSEEK_V2_LITE, locate, fileSource(dir))
    expect(() => s.slabBytes(0, 'gate', 'w')).toThrow(/no 'moe_gate_proj_w' buffer — not a MoE layer/)
  })
})

describe('fetchSlabSource', () => {
  it("returns exactly the i-th stride of the stacked tensor", async () => {
    const s = fetchSlabSource(SPEC, locate, fileSource(dir))
    let tag = 0
    for (const L of LAYERS) {
      for (const proj of PROJS) {
        for (const kind of KINDS) {
          tag++
          for (const e of [0, 1, 127, EXPERTS - 1]) {
            const got = await s.read(L, proj, kind, e)
            expect(got.byteLength).toBe(strideOf(kind))
            // 'w' is raw; 's'/'b' come back converted to f16, so compare against
            // the loader's own built buffer rather than the on-disk bf16.
            const want = kind === 'w'
              ? slabContent(tag, e, kind)
              : builtSlab(L, proj, kind, e)
            expect([...got]).toEqual([...want])
          }
        }
      }
    }
  })

  it('resolves the shared expert at index E, from its own shard', async () => {
    const s = fetchSlabSource(SPEC, locate, fileSource(dir))
    const rec = planLayer(SPEC, 0).find((p) => p.name === 'moe_up_proj_w')!.parts[1].record
    expect(locate(rec).file).toBe('shard-1.safetensors')     // a DIFFERENT shard
    const got = await s.read(0, 'up', 'w', SHARED)
    expect([...got]).toEqual([...builtSlab(0, 'up', 'w', SHARED)])
    // and it is not the last routed expert, which is the neighbouring slab
    expect([...got]).not.toEqual([...await s.read(0, 'up', 'w', EXPERTS - 1)])
  })

  it('throws on a short ranged read instead of uploading it', async () => {
    const s = fetchSlabSource(SPEC, locate, fileSource(dir, true))
    await expect(s.read(0, 'gate', 'w', 5)).rejects.toThrow(/returned 15 of 16 B/)
  })

  it('refuses an expert index outside the stack', async () => {
    const s = fetchSlabSource(SPEC, locate, fileSource(dir))
    await expect(s.read(0, 'gate', 'w', SHARED + 1)).rejects.toThrow(/outside 0\.\.256/)
    await expect(s.read(0, 'gate', 'w', -1)).rejects.toThrow(/outside 0\.\.256/)
    await expect(s.read(0, 'gate', 'w', 1.5)).rejects.toThrow(/outside 0\.\.256/)
  })
})

describe('opfsSlabSource', () => {
  it('reads the same bytes as the fetch path, through a sync access handle', async () => {
    const o = opfsSlabSource(SPEC, locate, nodeDir(cache))
    const f = fetchSlabSource(SPEC, locate, fileSource(dir))
    for (const L of LAYERS) {
      for (const proj of PROJS) {
        for (const kind of KINDS) {
          for (const e of [0, 3, EXPERTS - 1, SHARED]) {
            expect([...await o.read(L, proj, kind, e)]).toEqual([...await f.read(L, proj, kind, e)])
          }
        }
      }
    }
    o.close()
  })

  it('falls back to getFile() when the runtime has no sync access handle', async () => {
    const o = opfsSlabSource(SPEC, locate, nodeDir(cache, { sync: false }))
    expect([...await o.read(1, 'down', 's', 9)]).toEqual([...builtSlab(1, 'down', 's', 9)])
    expect([...await o.read(1, 'down', 's', SHARED)]).toEqual([...builtSlab(1, 'down', 's', SHARED)])
    o.close()
  })

  it('addresses each layer separately', async () => {
    const o = opfsSlabSource(SPEC, locate, nodeDir(cache))
    expect([...await o.read(0, 'gate', 'w', 3)]).not.toEqual([...await o.read(1, 'gate', 'w', 3)])
    o.close()
  })

  it('throws when the cached buffer is the wrong length', async () => {
    const torn = join(dir, 'torn')
    await mkdir(torn, { recursive: true })
    const key = planKey(planLayer(SPEC, 0).find((p) => p.name === 'moe_gate_proj_w')!, 0)
    const full = readFileSync(join(cache, key))
    await writeFile(join(torn, key), full.subarray(0, full.length - 1))
    const o = opfsSlabSource(SPEC, locate, nodeDir(torn))
    await expect(o.read(0, 'gate', 'w', 0)).rejects.toThrow(/is 4111 B, expected 4112 B/)
    o.close()
  })

  it('throws when the handle hands back a short read', async () => {
    const o = opfsSlabSource(SPEC, locate, nodeDir(cache, { shortRead: true }))
    await expect(o.read(0, 'gate', 'w', 2)).rejects.toThrow(/read 15 of 16 B at 32/)
    o.close()
  })

  it('names the missing entry when the cache has no such buffer', async () => {
    const empty = join(dir, 'empty')
    await mkdir(empty, { recursive: true })
    const o = opfsSlabSource(SPEC, locate, nodeDir(empty))
    await expect(o.read(0, 'gate', 'w', 0)).rejects.toThrow(/no cached buffer 'l0\.moe_gate_proj_w'/)
    o.close()
  })
})

/** Expert `e`'s slab as it sits in the buffer the LOADER built — the ground
 *  truth both sources have to reproduce. */
function builtSlab(layer: number, proj: SlabProj, kind: SlabKind, expert: number): Uint8Array {
  const plan = planLayer(SPEC, layer).find((p) => p.name === `moe_${proj}_proj_${kind}`)!
  const built = readFileSync(join(cache, planKey(plan, layer)))
  const stride = strideOf(kind)
  return new Uint8Array(built.subarray(expert * stride, (expert + 1) * stride))
}
