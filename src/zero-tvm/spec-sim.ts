/**
 * Prompt-lookup speculative-decoding acceptance simulator.
 *
 * Runs CPU-only over an already-generated token sequence and measures what
 * fraction of tokens would have been "free" under the PLD algorithm:
 *   At decode step t, take the last N tokens (the key). Search the history
 *   [0..t-1] for that exact N-gram. If found, the K tokens following the
 *   match become the speculation. The longest prefix of the speculation that
 *   matches actual[t..] is accepted; the rest is rejected.
 *
 * Throughput model (from measured batched-matmul 2× ceiling on Apple):
 *   M=1 forward: 1.0 unit of GPU time, yields 1 token → 1 tok/unit
 *   M=K+1 forward: ~(K+1)/2 units of GPU time (2× weight amortization),
 *                  yields 1 + accepted tokens → (1+accepted)/((K+1)/2) tok/unit
 *
 * Effective speedup at mean acceptance rate α over K speculated tokens:
 *   speedup = (1 + α·K) / ((K+1) / 2) = 2·(1 + α·K) / (K+1)
 *
 * For K=3: speedup > 1 requires α > 0.67. For K=2: α > 0.5. For K=1: α > 0.5.
 * So we need ≥50-67% acceptance on realistic prompts to win.
 */

export interface SimResult {
  name: string
  N: number        // lookup n-gram size
  K: number        // max speculated tokens per step
  totalSteps: number
  totalAccepted: number
  meanAcceptedPerStep: number  // α · K
  acceptanceRate: number        // α ∈ [0, 1]
  theoreticalSpeedup: number    // 2·(1 + α·K) / (K+1)
  hitRate: number               // fraction of steps with any n-gram match
}

/**
 * Simulate PLD decoding over an existing token sequence.
 *
 * The sequence includes prompt and generated tokens concatenated; the
 * simulator reads from index `promptLen` onward (the generated portion),
 * using the full sequence (prompt + prior generations) as the lookup table.
 */
export function simulatePLD(
  sequence: number[],
  promptLen: number,
  N: number = 3,
  K: number = 3,
): { totalSteps: number; totalAccepted: number; totalHits: number } {
  let t = promptLen
  let totalSteps = 0
  let totalAccepted = 0
  let totalHits = 0

  while (t < sequence.length) {
    totalSteps++

    // Need at least N tokens of context to form a lookup key.
    if (t < N) { t++; continue }

    // Key = last N tokens ending at t-1.
    const keyStart = t - N
    // Search history [0..keyStart-1] for an N-gram match (rightmost match wins,
    // per the PLD paper — more recent matches tend to be better).
    let matchIdx = -1
    for (let i = keyStart - 1; i >= 0; i--) {
      // Match if sequence[i..i+N-1] === sequence[keyStart..keyStart+N-1]
      if (i + N > keyStart) continue  // overlap with key itself
      let ok = true
      for (let j = 0; j < N; j++) {
        if (sequence[i + j] !== sequence[keyStart + j]) { ok = false; break }
      }
      if (ok) { matchIdx = i; break }
    }

    if (matchIdx < 0) {
      // No match → one M=1 forward produces sequence[t]. No speculation benefit.
      t++
      continue
    }

    totalHits++
    // Speculation = sequence[matchIdx + N .. matchIdx + N + K - 1]
    // Accept tokens where speculation matches actual[t..].
    let accepted = 0
    for (let k = 0; k < K; k++) {
      const specIdx = matchIdx + N + k
      const actIdx = t + k
      if (specIdx >= sequence.length) break
      if (actIdx >= sequence.length) break
      if (sequence[specIdx] !== sequence[actIdx]) break
      accepted++
    }
    totalAccepted += accepted
    // One M=K+1 forward produces the "sure" next token (sequence[t] which we
    // don't need to re-predict) PLUS accepted tokens. In the simulator we just
    // advance by (1 + accepted) since each forward commits at least 1 token.
    t += 1 + accepted
  }

  return { totalSteps, totalAccepted, totalHits }
}

export function summarize(
  name: string,
  sequence: number[],
  promptLen: number,
  N: number = 3,
  K: number = 3,
): SimResult {
  const { totalSteps, totalAccepted, totalHits } = simulatePLD(sequence, promptLen, N, K)
  const meanAcceptedPerStep = totalSteps > 0 ? totalAccepted / totalSteps : 0
  const acceptanceRate = K > 0 ? meanAcceptedPerStep / K : 0
  const theoreticalSpeedup = (2 * (1 + meanAcceptedPerStep)) / (K + 1)
  const hitRate = totalSteps > 0 ? totalHits / totalSteps : 0
  return {
    name, N, K,
    totalSteps, totalAccepted,
    meanAcceptedPerStep, acceptanceRate,
    theoreticalSpeedup, hitRate,
  }
}
