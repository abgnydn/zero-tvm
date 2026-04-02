/**
 * WebLLM Engine Wrapper
 *
 * Thin typed wrapper around @mlc-ai/web-llm.
 * Handles model loading, streaming chat, and performance tracking.
 */

import * as webllm from '@mlc-ai/web-llm'

/** Supported model IDs */
export const MODELS = {
  PHI3_MINI_Q4: 'Phi-3-mini-4k-instruct-q4f16_1-MLC',
  PHI3_MINI_Q8: 'Phi-3-mini-4k-instruct-q4f32_1-MLC',
  LLAMA_3_8B: 'Llama-3.1-8B-Instruct-q4f16_1-MLC',
  QWEN_0_5B: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  SMOLLM_360M: 'SmolLM2-360M-Instruct-q4f16_1-MLC',
} as const

export type ModelId = typeof MODELS[keyof typeof MODELS]

/** Load progress callback */
export type ProgressFn = (msg: string, pct: number) => void

/** Chat message */
export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
}

/** Token callback for streaming */
export type TokenFn = (token: string) => void

/** Performance stats from a generation */
export interface GenStats {
  prefillTokens: number
  prefillMs: number
  decodeTokens: number
  decodeMs: number
  tokPerSec: number
}

export class LLMEngine {
  private engine: webllm.MLCEngine | null = null
  private history: Message[] = []

  /** Load a model. Calls progressFn with download/init updates. */
  async load(modelId: ModelId, progressFn?: ProgressFn): Promise<void> {
    this.engine = await webllm.CreateMLCEngine(modelId, {
      initProgressCallback: (report) => {
        const pct = report.progress ?? 0
        progressFn?.(report.text, pct)
      },
    })
    this.history = []
  }

  /** Check if model is loaded */
  get ready(): boolean { return this.engine !== null }

  /** Reset conversation history */
  resetHistory(): void { this.history = [] }

  /** Set system prompt */
  setSystem(prompt: string): void {
    this.history = [{ role: 'system', content: prompt }]
  }

  /** Generate a response, streaming tokens via callback */
  async chat(userMessage: string, onToken: TokenFn): Promise<GenStats> {
    if (!this.engine) throw new Error('Model not loaded')

    this.history.push({ role: 'user', content: userMessage })

    const t0 = performance.now()
    let fullResponse = ''
    let tokenCount = 0

    const chunks = await this.engine.chat.completions.create({
      messages: this.history.map(m => ({
        role: m.role as 'user' | 'assistant' | 'system',
        content: m.content,
      })),
      stream: true,
      temperature: 0,
      max_tokens: 500,
    })

    for await (const chunk of chunks) {
      const delta = chunk.choices[0]?.delta?.content ?? ''
      if (delta) {
        fullResponse += delta
        tokenCount++
        onToken(delta)
      }
    }

    this.history.push({ role: 'assistant', content: fullResponse })

    const elapsed = performance.now() - t0

    return {
      prefillTokens: 0,
      prefillMs: 0,
      decodeTokens: tokenCount,
      decodeMs: elapsed,
      tokPerSec: tokenCount / (elapsed / 1000),
    }
  }

  /** Get runtime stats from the last completion's usage field */
  getStats(): string {
    return '' // stats now come from our profiler instead of deprecated runtimeStatsText
  }

  /** Reset conversation (keep model loaded) */
  async reset(): Promise<void> {
    if (this.engine) {
      await this.engine.resetChat()
    }
  }

  /** Cleanup */
  async destroy(): Promise<void> {
    if (this.engine) {
      await this.engine.unload()
      this.engine = null
    }
  }
}
