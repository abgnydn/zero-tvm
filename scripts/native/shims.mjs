// NODE SHIMS — the exact set that lets the browser engine run on dawn.node.
//
// Installed BEFORE any src/ module is imported: weight-loader reads
// GPUBufferUsage at module scope, so the globals must exist at import time or
// the import itself throws (the wall every earlier headless attempt hit).
//
// Three shims, each replacing one browser facility:
//   1. WebGPU globals + navigator.gpu     — from dawn.node ('webgpu' npm)
//   2. navigator.storage (OPFS)           — fs-backed, ~/.zerotvm/opfs
//   3. fetch of root-relative URLs        — rewritten to the vite dev server,
//      which serves /local-weights/* from .weights-local (the same mirror the
//      browser uses; weight bytes never leave the machine)
//
// Deliberately NOT shimmed: document/DOM. Anything that touches it
// (loading-ui's progress bars, chat.ts) has no business in the host, and a
// loud crash beats a silent fake.

import { mkdirSync, existsSync, openSync, readSync, closeSync, fstatSync } from 'node:fs'
import { open, readFile, rm } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

export async function installShims({ viteBase = 'http://localhost:5173', unsafe = true } = {}) {
  // ---- 1. WebGPU ----------------------------------------------------------
  const { create, globals } = await import('webgpu')
  Object.assign(globalThis, globals)
  const gpu = create(unsafe
    // The same flags ztvm's Chrome runs with — measured +22% prefill
    // end-to-end (BENCH.md, E2). This is a local trusted host; there is no
    // untrusted page to protect.
    ? ['enable-dawn-features=allow_unsafe_apis,disable_robustness']
    : ['enable-dawn-features=allow_unsafe_apis'])

  // ---- 2. OPFS over fs ----------------------------------------------------
  // Only what weight-loader and kv-pool actually call: getDirectory →
  // getDirectoryHandle / getFileHandle / removeEntry, file handles with
  // createWritable / getFile. Files live under ~/.zerotvm/opfs so the cache
  // (and the KV pool) survives across runs exactly like browser OPFS does.
  const ROOT = join(homedir(), '.zerotvm', 'opfs')
  mkdirSync(ROOT, { recursive: true })

  const dirHandle = (abs) => ({
    kind: 'directory',
    async getDirectoryHandle(name, opts = {}) {
      const p = join(abs, name)
      if (!existsSync(p)) {
        if (!opts.create) throw Object.assign(new Error(`NotFound: ${name}`), { name: 'NotFoundError' })
        mkdirSync(p, { recursive: true })
      }
      return dirHandle(p)
    },
    async getFileHandle(name, opts = {}) {
      const p = join(abs, name)
      if (!existsSync(p) && !opts.create) {
        throw Object.assign(new Error(`NotFound: ${name}`), { name: 'NotFoundError' })
      }
      return {
        kind: 'file',
        async createWritable() {
          mkdirSync(abs, { recursive: true })
          const fh = await open(p, 'w')
          return {
            async write(chunk) {
              // STRINGS FIRST, and the catch-all THROWS. `new Uint8Array(str)`
              // reads the string as a length, gets NaN, and writes zero bytes
              // — so kv-pool's meta.json (its commit record, written as JSON
              // text) came out empty on every save while entry.bin looked
              // perfect. Restore then read "no entry" forever: 683 MB of
              // payload on disk that could never be loaded, and not one error.
              // The browser's FileSystemWritableFileStream takes strings, so
              // this only ever broke under the native host.
              const u8 = typeof chunk === 'string' ? new TextEncoder().encode(chunk)
                : chunk instanceof ArrayBuffer ? new Uint8Array(chunk)
                : ArrayBuffer.isView(chunk) ? new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength)
                : null
              if (!u8) throw new TypeError(`OPFS write: unsupported chunk ${Object.prototype.toString.call(chunk)}`)
              await fh.write(u8)
            },
            async close() { await fh.close() },
          }
        },
        // The path slab-source.ts prefers, and the only one that can read ONE
        // expert out of a multi-GB built buffer. Without it the loader fell
        // back to getFile(), which here returns a plain object with no
        // slice() — and which reads the entire file into memory to hand back
        // 2.5 MiB of it.
        async createSyncAccessHandle() {
          const fd = openSync(p, 'r')
          return {
            getSize() { return fstatSync(fd).size },
            read(into, opts = {}) {
              return readSync(fd, into, 0, into.byteLength, opts.at ?? 0)
            },
            close() { closeSync(fd) },
          }
        },
        async getFile() {
          const buf = await readFile(p)
          const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
          return {
            size: buf.byteLength,
            async arrayBuffer() { return ab },
            // Blob.slice, so a caller that wants a window does not have to
            // materialise the whole file. The browser's File has this; the
            // stand-in must too, or code that works in a tab throws here.
            slice(begin = 0, end = buf.byteLength) {
              const part = ab.slice(begin, end)
              return { size: part.byteLength, async arrayBuffer() { return part } }
            },
            async text() { return new TextDecoder().decode(ab) },
          }
        },
      }
    },
    async removeEntry(name, _opts = {}) {
      await rm(join(abs, name), { recursive: true, force: true })
    },
    async *entries() {
      const { readdirSync, statSync } = await import('node:fs')
      for (const e of readdirSync(abs)) {
        const p = join(abs, e)
        yield [e, statSync(p).isDirectory() ? dirHandle(p) : { kind: 'file', name: e }]
      }
    },
  })

  // Node 21+ ships a read-only `navigator` getter — assignment throws. Define
  // over it instead; the property is configurable even though it has no setter.
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu, storage: { getDirectory: async () => dirHandle(ROOT) } },
    configurable: true,
  })

  // ---- 3. root-relative fetch --------------------------------------------
  // weight-loader's dev mirror uses URLs like /local-weights/<repo>/file; in
  // a browser they resolve against the page origin. Here they resolve against
  // the vite server, which serves .weights-local — so weights come off local
  // disk at localhost speed, not from HF.
  const realFetch = globalThis.fetch
  // TWO rewrites, and the second is the one that matters. weight-loader gates
  // its local mirror on import.meta.env.DEV, which does not exist under Node —
  // so the loader innocently goes to huggingface.co, and the first native boot
  // pulled weights at 0.9 MB/s over the user's slow line for ten minutes
  // before being killed. Every HF resolve URL is rewritten to the vite mirror
  // (which serves .weights-local); a mirror 404 falls back to the real URL so
  // an unprimed checkpoint still works.
  globalThis.fetch = async (input, init) => {
    if (typeof input === 'string') {
      if (input.startsWith('/')) input = viteBase + input
      const m = /^https:\/\/huggingface\.co\/[^/]+\/([^/]+)\/resolve\/[^/]+\/(.+)$/.exec(input)
      if (m) {
        const mirrored = `${viteBase}/local-weights/${m[1]}/${m[2]}`
        const r = await realFetch(mirrored, init)
        if (r.ok || r.status === 206) return r
        // fall through to HF for files the mirror does not hold
      }
    }
    return realFetch(input, init)
  }
  return { gpu }
}
