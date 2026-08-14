#!/usr/bin/env node
// KERNEL-AB — our WGSL int4 matmul against MLX's, at matched shapes.
//
//   node --experimental-strip-types scripts/kernel-ab.mjs
//
// BENCH.md's LM Studio table is a RUNTIME result: two whole pipelines, neither
// isolating a kernel. It cannot answer "is our GEMM faster than MLX's". This
// can, because it times one kernel per side with nothing else in the loop.
//
// CROSS-PROCESS, which every other A/B here avoids. MLX is Python and we are
// Node, so one process cannot hold both arms; the compromise is to alternate
// the two processes round by round and PAIR them — the same structure
// lmstudio-ab.mjs uses, and the same reason decode-kernel-ab.mjs pairs: an
// M2 Max drifts ~20% across a run, and a median of absolutes then reports
// where in the drift each arm's samples landed rather than which is faster.
//
// Both arms rotate over distinct weight sets so every read is cold, both use
// 4-bit / group 64 / affine, and both run the same iteration count — that last
// one is not cosmetic: at 24 iterations fixed overhead dominated and OUR arm
// read 2x slow, which is exactly the kind of setup error that would have
// produced a confident wrong answer.
//
// VOID on battery: `pmset -g batt` must say charged/charging first.

import { execFileSync } from 'node:child_process'

const ROUNDS = Number(process.env.ROUNDS) || 3
const ITERS = Number(process.env.ITERS) || 200
const MS = process.env.MS || '1,256'
const ML_RESEARCH = process.env.ML_RESEARCH || `${process.env.HOME}/dev/ml-research`
const HERE = new URL('..', import.meta.url).pathname

const run = (cmd, args, opts) => JSON.parse(execFileSync(cmd, args, {
  encoding: 'utf8', maxBuffer: 1 << 24, ...opts,
}).trim().split('\n').at(-1))

const ours = () => run('node', ['--experimental-strip-types', `${HERE}scripts/kernel-ab-ours.mjs`, '--json'],
  { cwd: HERE, env: { ...process.env, ITERS: String(ITERS), MS } })
const mlx = () => run('uv', ['run', 'python', `${HERE}scripts/mlx-kernel-ab.py`, '--json',
  '--iters', String(ITERS), '--ms', MS], { cwd: ML_RESEARCH })

const key = (r) => `${r.name}|${r.M}`
const rounds = []
for (let r = 0; r < ROUNDS; r++) {
  // Order flips each round so neither arm always runs on the warmer machine.
  const [a, b] = r % 2 === 0 ? [ours(), mlx()] : [mlx(), ours()]
  const o = a.engine === 'zero-tvm' ? a : b
  const m = a.engine === 'mlx' ? a : b
  const byKey = Object.fromEntries(m.results.map((x) => [key(x), x]))
  rounds.push(Object.fromEntries(o.results.map((x) => [key(x), { ours: x, mlx: byKey[key(x)] }])))
  console.log(`round ${r} done`)
}

const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]
console.log(`\nint4 matmul, 4-bit group 64 affine, ${ITERS} iters x ${ROUNDS} rounds, paired per round`)
console.log(`(Qwen3.5-9B shapes; M=1 is decode, M=256 a prefill chunk)\n`)
console.log(`  ${'shape'.padEnd(13)} ${'M'.padStart(4)} ${'ours'.padStart(9)} ${'mlx'.padStart(9)} `
  + `${'mlx is'.padStart(8)}  ${'unit'.padEnd(8)} ${'ours'.padStart(7)} ${'mlx'.padStart(7)}`)
for (const k of Object.keys(rounds[0])) {
  const rs = rounds.map((r) => r[k])
  const oMs = med(rs.map((x) => x.ours.ms)), mMs = med(rs.map((x) => x.mlx.ms))
  const M = rs[0].ours.M
  const unit = M === 1 ? 'GB/s' : 'GFLOP/s'
  const oV = med(rs.map((x) => (M === 1 ? x.ours.gbPerS : x.ours.gflops)))
  const mV = med(rs.map((x) => (M === 1 ? x.mlx.gbPerS : x.mlx.gflops)))
  console.log(`  ${rs[0].ours.name.padEnd(13)} ${String(M).padStart(4)} ${oMs.toFixed(3).padStart(9)} `
    + `${mMs.toFixed(3).padStart(9)} ${`${(oMs / mMs).toFixed(2)}x`.padStart(8)}  ${unit.padEnd(8)} `
    + `${oV.toFixed(0).padStart(7)} ${mV.toFixed(0).padStart(7)}`)
}
console.log('\nRatio > 1.00x means MLX is faster. This measures KERNELS, not engines —')
console.log('BENCH.md has the runtime comparison, and the two do not agree in sign.')
