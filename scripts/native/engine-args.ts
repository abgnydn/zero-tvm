/**
 * The engine's argv, as a pure function of a load request.
 *
 * Its own module — not a closure inside station.mjs — for two reasons.
 * station.mjs binds :8017 at import, so nothing can read this without starting
 * a server; and every flag here has been mistranslated at least once. `pool`
 * is EXPERT SLOTS, a memory build, and this passed it as `--pool`, which is the
 * KV prefix cache on disk — so the DEFAULT build (slots 0) sent `--pool 0` and
 * silently turned off the thing that lets a prefill survive a restart. Nothing
 * errored. Reloads just got expensive again.
 */
export interface LoadRequest {
  param: string
  port: number
  ctx?: number
  /** EXPERT SLOTS per MoE layer. Not the KV disk pool — see above. */
  pool?: number
  kv8?: boolean
  reuse?: boolean
  chunk?: boolean
}

export function engineArgs({ param, port, ctx, pool, kv8, reuse, chunk }: LoadRequest): string[] {
  const argv = [param, '--port', String(port)]
  if (ctx) argv.push('--ctx', String(ctx))
  if (pool) argv.push('--experts', String(pool))
  if (kv8 === false) argv.push('--kv8', '0')   // int8 is the default; this is the opt-out
  // Diagnostic only (no UI control): the arm where every turn prefills from zero.
  if (reuse === false) argv.push('--reuse', '0')
  if (chunk === false) argv.push('--chunk', '0')
  return argv
}
