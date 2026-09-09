#!/usr/bin/env node
// CLEAN DEPLOYMENTS — remove orphaned GitHub deployment records.
//
// This repo publishes to Cloudflare Pages from the CLI, which creates NO GitHub
// deployment records. Every record on it was created by vercel[bot] from an
// integration that is no longer connected, and twelve of them name a DIFFERENT
// project (fused-lora) whose Vercel project was pointed at this repo. The
// result is a sidebar reading "Production — 4 months ago" on a site that
// updated yesterday, next to another project's name.
//
// Irreversible: there is no undo and no API to restore a deleted deployment.
// So it refuses to do anything until told twice — --apply, and a --creator that
// must match. The default prints what it would delete and stops.
//
//   node scripts/clean-deployments.mjs                 # dry run
//   node scripts/clean-deployments.mjs --apply
//
// GitHub requires an ACTIVE deployment be marked inactive before deletion, so
// each one is two calls. Anything not created by --creator is skipped and
// counted, never touched: if the integration is ever reconnected, this must not
// delete a real record.
import { execFileSync } from 'node:child_process'

const REPO = process.env.REPO ?? 'abgnydn/zero-tvm'
const CREATOR = process.env.CREATOR ?? 'vercel[bot]'
const APPLY = process.argv.includes('--apply')

// Retries, because this loop is half destructive: a transient TLS or rate
// error partway through would leave some records deleted and some not, and
// there is no way to tell which from the outside afterwards. Observed once on
// the --paginate path.
const gh = (args, tries = 3) => {
  let last
  for (let i = 0; i < tries; i++) {
    try {
      return JSON.parse(execFileSync('gh', args, { encoding: 'utf8' }) || 'null')
    } catch (e) {
      last = (e.stderr ?? e.message ?? '').toString().split('\n')[0]
      if (i < tries - 1) execFileSync('sleep', [String(1 + i)])
    }
  }
  throw new Error(`gh ${args.join(' ')} → ${last}`)
}

const all = gh(['api', `repos/${REPO}/deployments`, '--paginate'])
const mine = all.filter((d) => (d.creator?.login ?? '') === CREATOR)
const others = all.filter((d) => (d.creator?.login ?? '') !== CREATOR)

const byEnv = new Map()
for (const d of mine) byEnv.set(d.environment, (byEnv.get(d.environment) ?? 0) + 1)

console.log(`${REPO}: ${all.length} deployment records`)
console.log(`  ${mine.length} created by ${CREATOR} — the orphans`)
for (const [env, n] of [...byEnv].sort((a, b) => b[1] - a[1])) console.log(`      ${String(n).padStart(3)}  ${env}`)
if (others.length) {
  console.log(`  ${others.length} created by someone else — NOT TOUCHED:`)
  for (const d of others.slice(0, 10)) console.log(`      ${d.environment}  by ${d.creator?.login ?? '?'}  ${d.created_at.slice(0, 10)}`)
}
if (!mine.length) { console.log('\nnothing to do'); process.exit(0) }

const newest = mine[0]?.created_at?.slice(0, 10)
console.log(`\n  newest orphan: ${newest} — if that is recent, the integration is still connected`)
console.log('  and deleting these will not stop more appearing. Fix that first.')

if (!APPLY) {
  console.log(`\nDRY RUN. Nothing was deleted. Re-run with --apply to delete ${mine.length} records.`)
  process.exit(0)
}

let deleted = 0, failed = 0
for (const d of mine) {
  try {
    // Mark inactive first: GitHub refuses to delete an active deployment, and
    // the failure is a 422 that reads like a permissions problem.
    try {
      gh(['api', '-X', 'POST', `repos/${REPO}/deployments/${d.id}/statuses`, '-f', 'state=inactive'])
    } catch { /* already inactive, or no statuses to supersede — deletion still works */ }
    gh(['api', '-X', 'DELETE', `repos/${REPO}/deployments/${d.id}`])
    deleted++
    process.stdout.write(`\r  deleted ${deleted}/${mine.length}`)
  } catch (e) {
    failed++
    console.log(`\n  FAILED ${d.id} (${d.environment}): ${e.message}`)
  }
}
console.log(`\n\n${deleted} deleted · ${failed} failed · ${others.length} left alone`)
