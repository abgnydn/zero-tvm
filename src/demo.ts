/**
 * Demo page — interactive visualizations + live chat demo.
 */

// @ts-ignore
import embeddingWGSL from './compiler/shaders/embedding.wgsl?raw'
// @ts-ignore
import rmsNormWGSL from './compiler/shaders/rms_norm.wgsl?raw'
// @ts-ignore
import qkvFusedWGSL from './compiler/shaders/qkv_fused.wgsl?raw'
// @ts-ignore
import attentionWGSL from './compiler/shaders/attention.wgsl?raw'
import { int4MatmulWGSL as genInt4Matmul } from './compiler/shaders/int4_matmul.gen'
// The int4_matmul family is generated; render the scalar variant for display.
const int4MatmulWGSL = genInt4Matmul()
// @ts-ignore
import fusedFFNWGSL from './compiler/shaders/fused_ffn.wgsl?raw'
// @ts-ignore
import addNormWGSL from './compiler/shaders/add_norm.wgsl?raw'
// @ts-ignore
import kvAppendWGSL from './compiler/shaders/kv_append.wgsl?raw'
// @ts-ignore
import ropeWGSL from './compiler/shaders/rope.wgsl?raw'
// @ts-ignore
import argmaxWGSL from './compiler/shaders/argmax.wgsl?raw'

// ============================================================
// Dispatch Timeline Visualization
// ============================================================

const COLORS: Record<string, string> = {
  matmul: '#10b981', attention: '#ff9f43', norm: '#2ecc71',
  activation: '#a855f7', kv: '#ffd54f', sampling: '#ff6b6b', embed: '#10b981',
}

interface DispatchInfo {
  name: string; category: string; workgroups: number; layer?: number
}

function buildDispatchMap(): DispatchInfo[] {
  const dispatches: DispatchInfo[] = []

  dispatches.push({ name: 'Embedding lookup', category: 'embed', workgroups: 12 })
  dispatches.push({ name: 'Initial RMSNorm', category: 'norm', workgroups: 1 })

  const layerSteps: Array<{ name: string; cat: string; wg: number }> = [
    { name: 'QKV projection', cat: 'matmul', wg: 9216 },
    { name: 'RoPE', cat: 'activation', wg: 36 },
    { name: 'KV cache append', cat: 'kv', wg: 12 },
    { name: 'Attention', cat: 'attention', wg: 32 },
    { name: 'Output projection', cat: 'matmul', wg: 3072 },
    { name: 'Residual + RMSNorm', cat: 'norm', wg: 1 },
    { name: 'FFN gate+up', cat: 'matmul', wg: 16384 },
    { name: 'SiLU activation', cat: 'activation', wg: 32 },
    { name: 'FFN down', cat: 'matmul', wg: 3072 },
    { name: 'Residual + RMSNorm', cat: 'norm', wg: 1 },
  ]

  for (let L = 0; L < 32; L++) {
    for (const s of layerSteps) {
      dispatches.push({ name: `L${L} ${s.name}`, category: s.cat, workgroups: s.wg, layer: L })
    }
  }

  for (let i = 0; i < 20; i++) {
    const name = i === 0 ? 'LM head' : `Sampling ${i}`
    dispatches.push({ name, category: 'sampling', workgroups: i === 0 ? 32064 : 126 })
  }

  return dispatches
}

function renderTimeline(canvas: HTMLCanvasElement, dispatches: DispatchInfo[]): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const dpr = window.devicePixelRatio || 1
  const rect = canvas.getBoundingClientRect()
  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr
  ctx.scale(dpr, dpr)

  const W = rect.width
  const H = rect.height
  const barW = Math.max(1, (W - 20) / dispatches.length)
  const maxWG = Math.max(...dispatches.map(d => Math.log2(d.workgroups + 1)))

  ctx.fillStyle = '#0a0a0a'
  ctx.fillRect(0, 0, W, H)

  for (let i = 0; i < dispatches.length; i++) {
    const d = dispatches[i]
    const x = 10 + i * barW
    const barH = (Math.log2(d.workgroups + 1) / maxWG) * (H - 30)
    const y = H - 15 - barH

    ctx.fillStyle = COLORS[d.category] || '#666'
    ctx.fillRect(x, y, Math.max(1, barW - 0.5), barH)

    // Layer separators
    if (d.layer !== undefined && d.layer > 0 && i > 2 && dispatches[i - 1]?.layer !== d.layer) {
      ctx.strokeStyle = '#333'
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, H)
      ctx.stroke()
    }
  }

  // Labels
  ctx.fillStyle = '#666'
  ctx.font = '10px -apple-system, sans-serif'
  ctx.fillText('Preamble', 10, 12)
  ctx.fillText('Layer 0', 10 + 2 * barW, 12)
  ctx.fillText('Layer 31', 10 + 322 * barW, 12)
  ctx.fillText('Tail', 10 + 322 * barW + 10, 12)

  // Tooltip on hover
  canvas.onmousemove = (e) => {
    const tooltip = document.getElementById('timeline-tooltip')!
    const idx = Math.floor((e.offsetX - 10) / barW)
    if (idx >= 0 && idx < dispatches.length) {
      const d = dispatches[idx]
      tooltip.style.display = 'block'
      tooltip.style.left = `${e.pageX + 12}px`
      tooltip.style.top = `${e.pageY - 30}px`
      tooltip.innerHTML = `<strong>${d.name}</strong><br>${d.workgroups.toLocaleString()} workgroups`
    } else {
      tooltip.style.display = 'none'
    }
  }
  canvas.onmouseleave = () => { document.getElementById('timeline-tooltip')!.style.display = 'none' }
}

// ============================================================
// Shader Gallery
// ============================================================

function renderShaderGallery(): void {
  const layerShaders = [
    { name: 'QKV Projection', entry: 'fused_dequantize1_NT_matmul10', lines: 117, cat: 'matmul' },
    { name: 'RoPE', entry: 'fused_rope_kernel', lines: 65, cat: 'activation' },
    { name: 'KV Cache Append', entry: 'tir_kv_cache_transpose_append', lines: 36, cat: 'kv' },
    { name: 'Paged Attention', entry: 'batch_decode_paged_kv', lines: 192, cat: 'attention' },
    { name: 'Output Projection', entry: 'fused_dequantize2_NT_matmul11', lines: 117, cat: 'matmul' },
    { name: 'Residual + RMSNorm', entry: 'fuse_add_norm_decode', lines: 120, cat: 'norm' },
    { name: 'FFN Gate+Up', entry: 'fused_dequantize3_NT_matmul12', lines: 117, cat: 'matmul' },
    { name: 'SiLU Activation', entry: 'fused_split2_silu2_multiply2', lines: 25, cat: 'activation' },
    { name: 'FFN Down', entry: 'fused_dequantize4_NT_matmul13', lines: 117, cat: 'matmul' },
  ]

  const customShaders = [
    { name: 'Embedding', lines: embeddingWGSL.split('\n').length, cat: 'custom', code: embeddingWGSL, desc: 'Token-id → f16 row lookup from the 32064 × 3072 embedding matrix.' },
    { name: 'RMSNorm', lines: rmsNormWGSL.split('\n').length, cat: 'custom', code: rmsNormWGSL, desc: 'Single-workgroup parallel reduction for the 3072-dim hidden state.' },
    { name: 'QKV + RoPE + KV-append (fused)', lines: qkvFusedWGSL.split('\n').length, cat: 'custom', code: qkvFusedWGSL, desc: 'Decode-path fusion. Replaces 3 dispatches per layer (int4 matmul + RoPE + KV append) with 1. Writes K/V directly into the paged cache.' },
    { name: 'Paged Attention', lines: attentionWGSL.split('\n').length, cat: 'custom', code: attentionWGSL, desc: 'Decode attention against a paged KV cache. One workgroup per head, online-softmax reduction in shared memory.' },
    { name: 'int4 Matmul (output + LM head)', lines: int4MatmulWGSL.split('\n').length, cat: 'custom', code: int4MatmulWGSL, desc: 'Dequantize-on-the-fly int4 × f16 matmul. Used for the output projection and the 32064-vocab LM head.' },
    { name: 'Fused FFN (gate+up+SiLU+mul+down)', lines: fusedFFNWGSL.split('\n').length, cat: 'custom', code: fusedFFNWGSL, desc: 'SwiGLU FFN in a single dispatch: gate proj + up proj + SiLU(gate) * up + down proj, hidden state kept in shared memory across the two matmul stages.' },
    { name: 'Add + RMSNorm (fused)', lines: addNormWGSL.split('\n').length, cat: 'custom', code: addNormWGSL, desc: 'Residual add + RMSNorm fused — same structure as TVM\'s fuse_add_norm_decode.' },
    { name: 'KV Append (prefill)', lines: kvAppendWGSL.split('\n').length, cat: 'custom', code: kvAppendWGSL, desc: 'Standalone KV-cache write used on the prefill path (decode path folds this into qkv_fused).' },
    { name: 'RoPE (prefill)', lines: ropeWGSL.split('\n').length, cat: 'custom', code: ropeWGSL, desc: 'Standalone rotary embedding for the prefill path. Decode path fuses RoPE into qkv_fused.' },
    { name: 'Argmax Sampler', lines: argmaxWGSL.split('\n').length, cat: 'custom', code: argmaxWGSL, desc: 'Parallel-reduction argmax over the 32064 logits. Replaces TVM\'s ~20-dispatch sampling chain with one dispatch (greedy only).' },
  ]

  const badgeClass: Record<string, string> = {
    matmul: 'badge-matmul', attention: 'badge-attn', norm: 'badge-norm',
    activation: 'badge-act', custom: 'badge-custom', kv: 'badge-sample',
  }

  const layerContainer = document.getElementById('layer-shaders')!
  for (const s of layerShaders) {
    const card = document.createElement('div')
    card.className = 'shader-card'
    card.innerHTML = `
      <div class="name">${s.name}</div>
      <div class="meta">${s.entry} — ${s.lines} lines WGSL</div>
      <span class="badge ${badgeClass[s.cat] || ''}">${s.cat.toUpperCase()}</span>
    `
    layerContainer.appendChild(card)
  }

  const customContainer = document.getElementById('custom-shaders')!
  for (const s of customShaders) {
    const card = document.createElement('div')
    card.className = 'shader-card'
    card.innerHTML = `
      <div class="name">${s.name}</div>
      <div class="meta">${s.lines} lines WGSL — ${s.desc}</div>
      <span class="badge badge-custom">CUSTOM FUSION</span>
      <details style="margin-top:8px"><summary style="font-size:12px;color:#666;padding:4px 0">View WGSL source</summary><pre style="font-size:10px;max-height:300px;overflow-y:auto">${s.code}</pre></details>
    `
    customContainer.appendChild(card)
  }
}

// ============================================================
// Buffer Treemap
// ============================================================

function renderBufferTreemap(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const dpr = window.devicePixelRatio || 1
  const rect = canvas.getBoundingClientRect()
  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr
  ctx.scale(dpr, dpr)

  const W = rect.width
  const H = rect.height

  const categories = [
    { name: 'Weights', bytes: 3_870_000_000, color: '#10b981', count: 270 },
    { name: 'KV Cache', bytes: 19_000_000, color: '#ffd54f', count: 37 },
    { name: 'Activations', bytes: 424_000, color: '#2ecc71', count: 71 },
    { name: 'Uniforms', bytes: 7_708, color: '#a855f7', count: 350 },
  ]
  const total = categories.reduce((s, c) => s + c.bytes, 0)

  // Simple horizontal bar layout (proportional to log of bytes for visibility)
  const logTotal = categories.reduce((s, c) => s + Math.log10(c.bytes + 1), 0)
  let x = 0
  for (const cat of categories) {
    const w = (Math.log10(cat.bytes + 1) / logTotal) * W
    ctx.fillStyle = cat.color + '33'
    ctx.fillRect(x, 0, w, H)
    ctx.strokeStyle = cat.color
    ctx.strokeRect(x, 0, w, H)

    // Label
    ctx.fillStyle = cat.color
    ctx.font = 'bold 16px -apple-system, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(cat.name, x + w / 2, H / 2 - 10)
    ctx.font = '13px -apple-system, sans-serif'
    ctx.fillStyle = '#aaa'
    ctx.fillText(`${cat.count} buffers`, x + w / 2, H / 2 + 10)
    ctx.fillText(formatBytes(cat.bytes), x + w / 2, H / 2 + 28)

    x += w
  }

  void total  // used conceptually
}

function formatBytes(b: number): string {
  if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB'
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB'
  if (b >= 1e3) return (b / 1e3).toFixed(1) + ' KB'
  return b + ' B'
}

// ============================================================
// Init
// ============================================================

function main(): void {
  // Dispatch timeline
  const timelineCanvas = document.getElementById('timeline-canvas') as HTMLCanvasElement
  const dispatches = buildDispatchMap()
  renderTimeline(timelineCanvas, dispatches)
  window.addEventListener('resize', () => renderTimeline(timelineCanvas, dispatches))

  // Shader gallery
  renderShaderGallery()

  // Buffer treemap
  const bufferCanvas = document.getElementById('buffer-canvas') as HTMLCanvasElement
  renderBufferTreemap(bufferCanvas)
  window.addEventListener('resize', () => renderBufferTreemap(bufferCanvas))

}

main()
