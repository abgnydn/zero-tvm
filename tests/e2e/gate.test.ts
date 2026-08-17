/**
 * E2E TESTS — Download gate + shared weight-cache Service Worker.
 *
 * These exercise the new boot path without needing a real 1.8 GB Phi-3
 * download:
 *
 *   1. Fresh OPFS  → #start-btn is injected (or kept) and visible
 *   2. Pre-seeded ndarray-cache.json sentinel → gate is skipped
 *   3. Service worker registers and is the active controller
 *   4. SW intercepts the Phi-3 mirror URL pattern and tags the response
 *      with `x-zero-tvm-cache` (proving the shared-cache plumbing is wired)
 */

import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { BASE, newPage, startHarness, stopHarness } from "./harness.js";

beforeAll(async () => {
  await startHarness();
}, 60_000);

afterAll(async () => {
  await stopHarness();
});

async function wipeOpfs(page: import("puppeteer").Page): Promise<void> {
  await page.evaluate(async () => {
    if (!navigator.storage?.getDirectory) return;
    const root = await navigator.storage.getDirectory();
    try { await root.removeEntry("zero-tvm-weights", { recursive: true }); } catch { /* dir didn't exist — fine */ }
    // Chat history persists per model since 2026-08-15 and the suite shares
    // one Chrome profile — a restored conversation would change what later
    // assertions see.
    try { localStorage.clear(); } catch { /* private mode */ }
  });
}

/**
 * A sentinel the CURRENT cache-probe accepts as "fully cached".
 *
 * The old one wrote `{"records":[]}`, which cache-probe.ts now rejects on
 * purpose: an empty record list means the manifest exists but no shard does,
 * which is exactly the interrupted-download state the gate must still appear
 * for. So the sentinel names one shard and writes that shard's file too —
 * the probe's real contract (manifest + every shard it names).
 */
async function seedOpfsSentinel(page: import("puppeteer").Page): Promise<void> {
  await page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle("zero-tvm-weights", { create: true });
    type Writable = { write(data: BlobPart): Promise<void>; close(): Promise<void> };
    const put = async (name: string, body: string) => {
      const fh = await dir.getFileHandle(name, { create: true });
      const w = await (fh as unknown as { createWritable: () => Promise<Writable> }).createWritable();
      await w.write(new TextEncoder().encode(body));
      await w.close();
    };
    await put("ndarray-cache.json", '{"records":[{"dataPath":"params_shard_0.bin"}]}');
    await put("params_shard_0.bin", "x");   // opfsKey('params_shard_0.bin') is itself
  });
}

describe("zero-tvm.html download gate", () => {
  test("fresh OPFS → gate is shown (#start-btn visible)", async () => {
    const page = await newPage("/zero-tvm.html");
    // Make sure we're starting clean. wipeOpfs may run after the page already
    // probed cache, but the assertion is that on the FIRST visit (where wipe
    // is a no-op) the gate is present.
    await wipeOpfs(page);
    // Reload so the boot probe runs against the wiped state.
    await page.goto(`${BASE}/zero-tvm.html`, { waitUntil: "domcontentloaded" });
    const btn = await page.waitForSelector("#start-btn", { timeout: 15_000 });
    expect(btn).not.toBeNull();
    const text = await page.$eval("#start-btn", (el) => el.textContent || "");
    expect(text.toLowerCase()).toContain("download");
    await page.close();
  }, 30_000);

  test("seeded ndarray-cache.json sentinel → gate is skipped", async () => {
    const page = await newPage("/zero-tvm.html");
    await seedOpfsSentinel(page);
    await page.goto(`${BASE}/zero-tvm.html`, { waitUntil: "domcontentloaded" });
    // Give boot()'s probe + main() time to run. main() will fail to actually
    // load weights (the seeded sentinel is empty), but the test only cares
    // that no gate was shown.
    await new Promise((r) => setTimeout(r, 4_000));
    const gate = await page.$("#start-btn");
    expect(gate).toBeNull();
    // Cleanup so subsequent tests start clean.
    await wipeOpfs(page);
    await page.close();
  }, 30_000);
});

describe("weight-cache Service Worker", () => {
  test("SW registers and becomes active controller", async () => {
    const page = await newPage("/zero-tvm.html");
    const swActive = await page.evaluate(async () => {
      if (!("serviceWorker" in navigator)) return "no-sw-api";
      const reg = await navigator.serviceWorker.ready;
      return reg.active?.scriptURL ?? "no-active";
    });
    expect(swActive).toContain("/weights-cache-sw.js");
    await page.close();
  }, 30_000);

  test("SW intercepts the Phi-3 mirror URL and tags the response", async () => {
    const page = await newPage("/zero-tvm.html");
    await page.evaluate(async () => {
      await navigator.serviceWorker.ready;
    });
    // Probe a tiny endpoint (config.json) on the Phi-3 mirror — small enough
    // to download repeatedly during tests, real enough that the SW must be
    // intercepting it for our header to appear.
    const result = await page.evaluate(async () => {
      const url =
        "https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC/resolve/main/mlc-chat-config.json";
      try {
        const r = await fetch(url, { mode: "cors", cache: "no-store" });
        return {
          ok: r.ok,
          status: r.status,
          tag: r.headers.get("x-zero-tvm-cache"),
        };
      } catch (e) {
        return { ok: false, status: 0, tag: null, err: String(e) };
      }
    });
    expect(result.ok).toBe(true);
    // First fetch should be `network` (or `opfs-hit` if a previous test seeded
    // it). Either proves the SW is on the request path.
    expect(["network", "opfs-hit"]).toContain(result.tag);
    await page.close();
  }, 60_000);
});
