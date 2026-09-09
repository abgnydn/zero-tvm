/* eslint-env serviceworker */
/**
 * weights-cache-sw.js — Phi-3 weight cache, shared by any page that loads those weights.
 *
 * Both zero-tvm's hand-written loader and WebLLM fetch the same files from the
 * HF mirror at
 * `huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC/resolve/main/*`.
 * Without this SW each cache their own copy, so visiting both pages downloads
 * 1.8 GB twice. This SW intercepts those URLs and serves from a shared OPFS
 * directory (`zero-tvm-weights/`, the same dir zero-tvm's own loader uses for
 * its fast path) — populating it on first download from either side.
 *
 * Result: whichever page the visitor opens first pays the network cost once.
 * Subsequent visits to either page hit OPFS instantly.
 */

// KEEP IN SYNC with WEIGHTS_OPFS_DIR in src/zero-tvm/weight-loader.ts — this
// SW is plain JS and cannot import the exported constant, so the two
// definitions are paired by hand.
const SHARED_DIR = "zero-tvm-weights";
const HF_PHI3_HOST = "huggingface.co";
const HF_PHI3_PATH_RE = /^\/mlc-ai\/Phi-3-mini-4k-instruct-q4f16_1-MLC\/resolve\/[^/]+\//;

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  let url;
  try { url = new URL(req.url); } catch { return; }
  if (url.host !== HF_PHI3_HOST || !HF_PHI3_PATH_RE.test(url.pathname)) return;

  event.respondWith(handle(req, url));
});

async function handle(req, url) {
  // Flatten the path-suffix into an OPFS-safe filename, matching the
  // `opfsKey()` convention in zero-tvm's weight-loader.ts byte-for-byte
  // (so files written by the SW are reused by zero-tvm's direct loader,
  // and vice versa).
  const dataPath = url.pathname.replace(HF_PHI3_PATH_RE, "");
  const key = dataPath.replace(/[^A-Za-z0-9._-]/g, "_");

  // 1. Try OPFS hit first.
  const cached = await opfsRead(key).catch(() => null);
  if (cached) {
    return new Response(cached, {
      status: 200,
      headers: {
        "content-type": guessType(key),
        "content-length": String(cached.byteLength),
        "x-zero-tvm-cache": "opfs-hit",
      },
    });
  }

  // 2. Network fall-through. Stream into OPFS so subsequent visits hit cache.
  const netResp = await fetch(req);
  // Only cache complete 200 bodies: a 206 (the weight-loader resumes dead
  // transfers with Range requests) buffered under the full key would poison
  // the shared cache with a truncated shard. Pass everything else through.
  if (netResp.status !== 200 || !netResp.body) return netResp;

  const buf = await netResp.clone().arrayBuffer();
  // Best-effort write — failures (quota, no OPFS) are fine; just no caching.
  opfsWrite(key, buf).catch(() => {});

  return new Response(buf, {
    status: netResp.status,
    headers: {
      "content-type": netResp.headers.get("content-type") || guessType(key),
      "content-length": String(buf.byteLength),
      "x-zero-tvm-cache": "network",
    },
  });
}

async function getDir() {
  if (!self.navigator?.storage?.getDirectory) throw new Error("no OPFS in SW");
  const root = await self.navigator.storage.getDirectory();
  return root.getDirectoryHandle(SHARED_DIR, { create: true });
}

async function opfsRead(key) {
  const dir = await getDir();
  const fh = await dir.getFileHandle(key);
  const file = await fh.getFile();
  return await file.arrayBuffer();
}

async function opfsWrite(key, buf) {
  const dir = await getDir();
  const fh = await dir.getFileHandle(key, { create: true });
  const writer = await fh.createWritable();
  await writer.write(buf);
  await writer.close();
}

function guessType(key) {
  if (key.endsWith(".json")) return "application/json";
  if (key.endsWith(".wasm")) return "application/wasm";
  return "application/octet-stream";
}
