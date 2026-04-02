/**
 * SHADER DUMP UTILITIES
 *
 * Exports functions to dump captured WGSL shader source code
 * from the runtime-captured shader modules. Shaders are captured
 * by the profiler's monkey-patched createShaderModule and are
 * only available after model load in the browser.
 */

import { captured } from './profiler.js'
import type { CapturedShader, CapturedPipeline } from './profiler.js'

/** Decode layer shader indices (unique set from the repeating pattern) */
const DECODE_LAYER_INDICES: readonly number[] = [6, 40, 26, 29, 46, 22, 47, 19, 27] as const

/** Embedding shader index */
const EMBEDDING_INDEX = 77

/** Initial norm shader index */
const INITIAL_NORM_INDEX = 66

/** All shader indices relevant to a full decode layer (including embedding + initial norm) */
const ALL_LAYER_INDICES: readonly number[] = [
  ...DECODE_LAYER_INDICES,
  EMBEDDING_INDEX,
  INITIAL_NORM_INDEX,
] as const

function findPipelineForShader(shaderIndex: number): CapturedPipeline | undefined {
  for (const [, pipeline] of captured.pipelines) {
    if (pipeline.shaderIndex === shaderIndex) {
      return pipeline
    }
  }
  return undefined
}

function formatShaderHeader(shader: CapturedShader): string {
  const pipeline = findPipelineForShader(shader.index)
  const lines: string[] = [
    `// ${'='.repeat(70)}`,
    `// SHADER INDEX:  ${shader.index}`,
    `// LABEL:         ${shader.label || 'unlabeled'}`,
    `// ENTRY POINT:   ${pipeline?.entryPoint ?? 'unknown'}`,
    `// PIPELINE:      ${pipeline?.label ?? 'unmapped'}`,
    `// LINES:         ${shader.lineCount}`,
    `// BINDINGS:      ${shader.hasBindings}`,
    `// ${'='.repeat(70)}`,
  ]
  return lines.join('\n')
}

/**
 * Dump the WGSL source of a single shader by index.
 * Returns the source with a metadata header comment block.
 */
export function dumpShader(index: number): string {
  const shader = captured.shaders[index]
  if (!shader) {
    return `// ERROR: shader[${index}] not found (${captured.shaders.length} shaders captured)`
  }
  return `${formatShaderHeader(shader)}\n\n${shader.code}`
}

/**
 * Dump the WGSL source of the 11 shaders used in the decode layer pattern.
 *
 * Includes the 9 unique shaders from the repeating layer pattern:
 *   [6, 40, 26, 29, 46, 22, 47, 19, 27]
 * Plus embedding [77] and initial norm [66].
 */
export function dumpLayerShaders(): string {
  const sections: string[] = [
    `// DECODE LAYER SHADER DUMP`,
    `// Generated at: ${new Date().toISOString()}`,
    `// Shaders captured: ${captured.shaders.length}`,
    `// Pipelines captured: ${captured.pipelines.size}`,
    `// Layer pattern indices: [${DECODE_LAYER_INDICES.join(', ')}]`,
    `// Embedding index: ${EMBEDDING_INDEX}`,
    `// Initial norm index: ${INITIAL_NORM_INDEX}`,
    '',
  ]

  for (const idx of ALL_LAYER_INDICES) {
    sections.push(dumpShader(idx))
    sections.push('')
  }

  return sections.join('\n')
}

/**
 * Dump ALL captured shader sources (all 85 modules) with headers.
 */
export function dumpAllShaders(): string {
  const sections: string[] = [
    `// FULL SHADER DUMP — ALL ${captured.shaders.length} MODULES`,
    `// Generated at: ${new Date().toISOString()}`,
    `// Pipelines: ${captured.pipelines.size}`,
    '',
  ]

  for (const shader of captured.shaders) {
    sections.push(formatShaderHeader(shader))
    sections.push('')
    sections.push(shader.code)
    sections.push('')
  }

  return sections.join('\n')
}
