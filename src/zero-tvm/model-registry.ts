/**
 * MODEL REGISTRY — the single table every "which models exist" surface renders
 * from: the landing page's model cards, the chat header's switcher, and the
 * gate dialog's branding.
 *
 * This exists so the landing page CANNOT go stale: adding a spec here is the
 * whole act of shipping it to every surface. (The old landing hand-wrote its
 * model list and stats into the HTML; after Qwen3.6 shipped, twelve separate
 * claims on that page were wrong.)
 *
 * DELIBERATELY LIGHT: imports only model-spec.ts, which is dependency-free.
 * The landing page runs this in browsers WITHOUT WebGPU (where it must still
 * render cards and say "needs WebGPU") — and weight-loader.ts, the previous
 * home of modelBranding's neighbours, reads `GPUBufferUsage` at module scope,
 * which throws at import time on such browsers.
 */

import {
  PHI3, QWEN3_4B, QWEN35_4B, QWEN36_35B_A3B, QWEN36_35B_A3B_Q3,
  type ModelSpec,
} from '../compiler/model-spec.js'

/** URL `?model=` value for each spec; Phi-3 is the no-flag default. */
export const SHIPPED_MODELS: ReadonlyArray<{ param: string; spec: ModelSpec }> = [
  { param: '', spec: PHI3 },
  { param: 'qwen3', spec: QWEN3_4B },
  { param: 'qwen35', spec: QWEN35_4B },
  // The 35B MoE ships twice: 3-bit experts first (the build most machines can
  // actually run — ~20 GB free RAM), full 4-bit for the boxes that can.
  { param: 'qwen36q3', spec: QWEN36_35B_A3B_Q3 },
  { param: 'qwen36', spec: QWEN36_35B_A3B },
]

export interface ModelBrand {
  name: string
  /** Parameter-count line for cards ("4B hybrid", "35B-A3B MoE"). */
  params: string
  sizeLabel: string
  /** Measured decode/total rate, or '' when no honest number exists. */
  rateLabel: string
  /** Present when the model needs more memory than a typical machine has. */
  ramNote?: string
}

/** Page-facing copy for a model (gate dialog, header, landing cards). */
export function modelBranding(spec: ModelSpec): ModelBrand {
  // rateLabels are measured numbers only — see BENCH.md for the protocol each
  // one came from; '' means no protocol-quality number exists yet.
  switch (spec.id) {
    case QWEN36_35B_A3B_Q3.id:
      // ~55 t/s measured by the machine owner on a quiet 32 GB M2 Max
      // (2026-08-05, single session — not yet a protocol round).
      return {
        name: 'Qwen3.6-35B-A3B', params: '35B-A3B MoE · 3-bit experts',
        sizeLabel: '~16.4 GB', rateLabel: '~55 t/s',
        ramNote: 'needs ~20 GB free RAM',
      }
    case QWEN36_35B_A3B.id:
      return {
        name: 'Qwen3.6-35B-A3B', params: '35B-A3B MoE · full 4-bit',
        sizeLabel: '~19.5 GB', rateLabel: '',
        ramNote: 'needs ~24 GB free RAM (64 GB Mac recommended)',
      }
    case QWEN35_4B.id:
      return { name: 'Qwen3.5-4B', params: '4B hybrid (DeltaNet)', sizeLabel: '~2.6 GB', rateLabel: '~65 t/s' }  // 65.28 total, M2 Max (BENCH.md 2026-07-30)
    case QWEN3_4B.id:
      return { name: 'Qwen3-4B', params: '4B dense', sizeLabel: '~2.3 GB', rateLabel: '~60 t/s' }  // 59.85 total, M2 Max (BENCH.md 2026-07-30)
    default:
      return { name: 'Phi-3-mini', params: '3.8B dense', sizeLabel: '~2 GB', rateLabel: '~70 t/s' }  // 69.55 total, M2 Max (BENCH.md 2026-07-30)
  }
}
