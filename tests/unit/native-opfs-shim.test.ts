import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir, homedir } from 'node:os'
import { join } from 'node:path'

/**
 * The native host's OPFS stand-in, exercised the way kv-pool uses it.
 *
 * This exists because of a silent data-loss bug: `write()` fell through to
 * `new Uint8Array(chunk)` for anything that was not an ArrayBuffer or a view,
 * and a STRING there is read as a LENGTH — `new Uint8Array('{"tokens":1}')`
 * is Uint8Array(0). kv-pool writes entry.bin as buffers (fine) and meta.json
 * as JSON text (silently empty), so every saved KV prefix had a payload and
 * no commit record. poolTryRestore read that as "no entry for this
 * fingerprint" and re-prefilled from scratch, forever, without an error.
 *
 * The browser's FileSystemWritableFileStream accepts strings, so no browser
 * test could have caught it — only the shim diverged.
 */

// installShims writes under ~/.zerotvm/opfs; keep the real one untouched.
const HOME = homedir()
let tmp: string

beforeAll(() => { tmp = mkdtempSync(join(tmpdir(), 'zt-shim-')) })
afterAll(() => { rmSync(tmp, { recursive: true, force: true }) })

async function opfsRoot() {
  process.env.HOME = tmp
  const { installShims } = await import('../../scripts/native/shims.mjs?fresh=' + Math.random())
  await installShims({ unsafe: false })
  process.env.HOME = HOME
  return await (navigator as unknown as {
    storage: { getDirectory: () => Promise<FileSystemDirectoryHandle> }
  }).storage.getDirectory()
}

describe('native OPFS shim', () => {
  it('writes a STRING as its utf-8 bytes, not as a length', async () => {
    const root = await opfsRoot()
    const dir = await root.getDirectoryHandle('pool', { create: true })
    const meta = JSON.stringify({ format: 1, tokens: 43702, ids: [1, 2, 3] })

    const w = await (await dir.getFileHandle('meta.json', { create: true })).createWritable()
    await w.write(meta)
    await w.close()

    const back = await (await (await dir.getFileHandle('meta.json')).getFile()).text()
    // The regression: this used to be '' and JSON.parse threw, which kv-pool
    // caught and reported as a cache miss.
    expect(back).toBe(meta)
    expect(JSON.parse(back).tokens).toBe(43702)
  })

  it('round-trips binary chunks and concatenates them in order', async () => {
    const root = await opfsRoot()
    const dir = await root.getDirectoryHandle('pool', { create: true })
    const a = new Uint8Array([1, 2, 3])
    const b = new Uint8Array([4, 5])

    const w = await (await dir.getFileHandle('entry.bin', { create: true })).createWritable()
    await w.write(a.buffer)
    await w.write(b)
    await w.close()

    const ab = await (await (await dir.getFileHandle('entry.bin')).getFile()).arrayBuffer()
    expect([...new Uint8Array(ab)]).toEqual([1, 2, 3, 4, 5])
  })

  it('THROWS on a chunk type it cannot encode instead of writing nothing', async () => {
    const root = await opfsRoot()
    const dir = await root.getDirectoryHandle('pool', { create: true })
    const w = await (await dir.getFileHandle('odd.bin', { create: true })).createWritable()
    // A silent zero-byte write is how the original bug hid for weeks.
    await expect(w.write({ type: 'write', data: 'x' } as unknown as ArrayBuffer)).rejects.toThrow(/unsupported chunk/)
    await w.close()
  })
})
