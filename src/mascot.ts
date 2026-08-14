/**
 * MASCOTS — one chibi character per model, raymarched and cel-shaded in WGSL.
 *
 * Anime reads through four things, and all four are reachable in a distance
 * field: chibi proportions (the head is nearly half the figure), flat colour in
 * hard-terminator bands rather than a smooth ramp, an ink outline, and very
 * large eyes carrying a bright catchlight. What is NOT reachable this way is a
 * detailed anime figure — strand-level hair, cloth folds, a drawn mouth. Those
 * want 2D art, and 2D art per model breaks the rule below.
 *
 * No meshes, no image assets. Every feature is read from the model's own spec,
 * so a model added by scripts/add-model.mjs draws itself and the picture cannot
 * disagree with the table.
 *
 *   log2(d)         → head size (the figure's whole scale)
 *   layers          → hair spikes, by polar domain repetition
 *   GDN fraction    → a tail; recurrent layers are the part that trails
 *   experts / topK  → orbiting sprites, topK lit at a time, rotating = routing
 *   sharedExpert    → one sprite fused to the head, always lit
 *   kvHeads         → ear tufts
 *   architecture    → hair colour lane
 *   bits + group    → cel band count and outline weight: a 3-bit build reads
 *                     posterised and heavier-inked than a 4-bit one, and the
 *                     quant group (32 MLC-symmetric vs 64 MLX-affine) shifts
 *                     the band split, so the two Qwen3-4B builds differ
 *   embeddingOnly   → eyes closed. It does not answer; it returns a vector.
 */

import type { ModelSpec } from './compiler/model-spec.js'

const WGSL = /* wgsl */ `
struct U {
  res: vec2<f32>, time: f32, hover: f32,
  headR: f32, spikes: f32, coil: f32, kvh: f32,
  experts: f32, topk: f32, sharedFused: f32, bands: f32,
  ink: f32, eyesOpen: f32, bandShift: f32, _p0: f32,
  hair: vec3<f32>, _p1: f32,
  iris: vec3<f32>, cached: f32,
};
@group(0) @binding(0) var<uniform> u: U;

fn rot(a: f32) -> mat2x2<f32> { let c = cos(a); let s = sin(a); return mat2x2<f32>(c, -s, s, c); }
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}
fn sdEllip(p: vec3<f32>, r: vec3<f32>) -> f32 {
  let k0 = length(p / r);
  let k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / k1;
}
fn sdCap(p: vec3<f32>, h: f32, r: f32) -> f32 {
  return length(vec3<f32>(p.x, p.y - clamp(p.y, -h, h), p.z)) - r;
}

// x = distance, y = material
//   0 skin   1 hair   2 eye white   3 iris   4 sprite
fn map2(pIn: vec3<f32>) -> vec2<f32> {
  var p = pIn;
  p.y = p.y - sin(u.time * 1.2) * 0.018;            // idle bob
  let m = rot(sin(u.time * 0.4) * 0.42);            // sway, never a full spin
  let xz = m * vec2<f32>(p.x, p.z);
  p = vec3<f32>(xz.x, p.y, xz.y);

  let R = u.headR;
  let hy = R * 0.62;                                 // head centre
  var q = p; q.x = abs(q.x);                         // opSymX — free symmetry

  // CHIBI PROPORTION. The head is the character; the body is a hint under it.
  var d = sdEllip(p - vec3<f32>(0.0, hy, 0.0), vec3<f32>(R, R * 1.02, R * 0.96));
  var mat = 0.0;

  // Body + arms, deliberately small.
  let body = sdCap(p - vec3<f32>(0.0, -R * 0.72, 0.0), R * 0.30, R * 0.40);
  d = smin(d, body, R * 0.18);
  let arm = sdCap(q - vec3<f32>(R * 0.46, -R * 0.72, 0.0), R * 0.16, R * 0.14);
  d = smin(d, arm, R * 0.06);

  // HAIR: a cap over the skull, plus spikes by polar repetition so the count
  // follows the layer count instead of being a fixed loop.
  var hair = sdEllip(p - vec3<f32>(0.0, hy + R * 0.10, -R * 0.02), vec3<f32>(R * 1.06, R * 1.05, R * 1.04));
  // FRINGE. Cut hair only where it would cover the face: below the fringe line
  // AND on the front side. A flat horizontal cut put the fringe at eye level
  // and the eyes ended up inside the hair.
  // The fringe is POINTED, not a flat line — a horizontal cut reads as a
  // swim cap. Modulating the cut height by angle gives it locks.
  let fringeY = hy + R * 0.10 - abs(sin(atan2(p.x, p.z) * 3.0)) * R * 0.20;
  hair = max(hair, -max(p.y - fringeY, -p.z));
  {
    let n = max(u.spikes, 3.0);
    let sect = 6.28318 / n;
    let a = atan2(p.z, p.x);
    let i = round(a / sect);
    let ang = i * sect;
    let dir = vec3<f32>(cos(ang), 0.0, sin(ang));
    let tip = vec3<f32>(0.0, hy + R * 0.42, 0.0) + dir * R * 0.92;
    let spike = sdCap((p - tip) * vec3<f32>(1.0, 0.70, 1.0), R * 0.26, R * 0.115);
    hair = smin(hair, spike, R * 0.16);
  }
  // Ear tufts, from the KV head count.
  {
    let t = sdEllip(q - vec3<f32>(R * 0.88, hy + R * 0.30, 0.0),
                    vec3<f32>(R * 0.16, R * 0.30 * clamp(u.kvh / 16.0, 0.4, 1.4), R * 0.16));
    hair = smin(hair, t, R * 0.05);
  }
  if (hair < d) { d = hair; mat = 1.0; }

  // Recurrent tail.
  if (u.coil > 0.01) {
    for (var i = 0.0; i < 5.0; i = i + 1.0) {
      let t = i / 4.0;
      let a = t * 5.0 + u.time * 0.6;
      let rad = R * (0.30 + t * 0.42);
      let c = p - vec3<f32>(cos(a) * rad, -R * (1.02 + t * 0.34), sin(a) * rad);
      let seg = length(c) - R * 0.13 * (1.0 - t * 0.45);
      if (seg < d) { d = seg; mat = 1.0; }
    }
  }

  // EYES. Large, flattened, sitting proud of the face — additive, never carved.
  // A concave dimple cannot hold a catchlight, and the catchlight is the whole
  // reason an anime eye reads as alive.
  if (u.eyesOpen > 0.5) {
    let ec = vec3<f32>(R * 0.38, hy - R * 0.22, R * 0.74);
    let white = sdEllip(q - ec, vec3<f32>(R * 0.30, R * 0.38, R * 0.20));
    if (white < d) { d = white; mat = 2.0; }
    let iris = sdEllip(q - (ec + vec3<f32>(0.0, -R * 0.04, R * 0.11)),
                       vec3<f32>(R * 0.23, R * 0.30, R * 0.17));
    if (iris < d) { d = iris; mat = 3.0; }
  } else {
    // Closed: a lash line, so the face still has a feature.
    let lash = sdEllip(q - vec3<f32>(R * 0.38, hy - R * 0.20, R * 0.80),
                       vec3<f32>(R * 0.28, R * 0.05, R * 0.10));
    if (lash < d) { d = lash; mat = 3.0; }
  }

  // A small mouth. Anime faces are eyes plus almost nothing, but "almost
  // nothing" is not "nothing".
  if (u.eyesOpen > 0.5) {
    let mo = sdEllip(p - vec3<f32>(0.0, hy - R * 0.62, R * 0.80),
                     vec3<f32>(R * 0.11, R * 0.055, R * 0.08));
    if (mo < d) { d = mo; mat = 5.0; }
  }

  // Expert sprites in orbit.
  if (u.experts > 0.5) {
    let n = min(u.experts, 24.0);
    let sect = 6.28318 / n;
    let a = atan2(p.z, p.x) + u.time * 0.5;
    let i = round(a / sect);
    let ang = i * sect - u.time * 0.5;
    let c = p - vec3<f32>(cos(ang) * R * 1.55, hy + sin(i * 2.0) * R * 0.35, sin(ang) * R * 1.55);
    let s = length(c) - R * 0.075;
    if (s < d) { d = s; mat = 4.0 + i; }
  }
  return vec2<f32>(d, mat);
}

fn map(p: vec3<f32>) -> f32 { return map2(p).x; }

fn normalAt(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(0.0012, 0.0);
  return normalize(vec3<f32>(
    map(p + e.xyy) - map(p - e.xyy),
    map(p + e.yxy) - map(p - e.yxy),
    map(p + e.yyx) - map(p - e.yyx)));
}

// CEL SHADING. Anime light is banded with a hard terminator, not a ramp. Band
// COUNT comes from the checkpoint's bit width and the split from its quant
// group, so a coarser build is visibly more posterised.
fn cel(x: f32, bands: f32, shift: f32) -> f32 {
  let t = clamp(x, 0.0, 1.0);
  let s = floor(t * bands + shift) / bands;
  let f = fract(t * bands + shift);
  return clamp(s + smoothstep(0.46, 0.54, f) / bands, 0.42, 1.0);
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = (frag.xy - 0.5 * u.res) / u.res.y;
  let ro = vec3<f32>(0.0, 0.05, 3.05);
  let rd = normalize(vec3<f32>(uv.x, -uv.y, -1.5));

  var t = 0.0; var hit = false; var mat = 0.0;
  for (var i = 0; i < 92; i = i + 1) {
    let r = map2(ro + rd * t);
    if (r.x < 0.0012) { hit = true; mat = r.y; break; }
    t = t + r.x * 0.85;
    if (t > 6.0) { break; }
  }

  var shadowA = 0.0;
  if (rd.y < -0.0001) {
    let ft = (-0.92 - ro.y) / rd.y;
    if (ft > 0.0 && (!hit || ft < t)) {
      let fp = ro + rd * ft;
      shadowA = exp(-dot(fp.xz, fp.xz) * 6.0) * 0.42;
    }
  }
  if (!hit) { return vec4<f32>(0.0, 0.0, 0.0, shadowA); }

  let p = ro + rd * t;
  let nrm = normalAt(p);
  let key = normalize(vec3<f32>(0.45, 0.70, 0.70));
  let diff = clamp(dot(nrm, key) * 0.5 + 0.5, 0.0, 1.0);
  let lit = cel(diff, u.bands, u.bandShift);
  let fres = pow(1.0 - clamp(dot(nrm, -rd), 0.0, 1.0), 1.8);

  let SKIN = vec3<f32>(0.98, 0.86, 0.80);
  var col: vec3<f32>;

  if (mat >= 5.0) {
    col = vec3<f32>(0.42, 0.22, 0.24);
  } else if (mat >= 4.0) {
    let id = mat - 4.0;
    let n = min(u.experts, 24.0);
    let m = max(n / max(u.topk, 1.0), 1.0);
    let s = id + floor(u.time * 1.6);
    let hot = select(0.0, 1.0, (s - m * floor(s / m)) < 1.0);
    let fused = select(0.0, 1.0, u.sharedFused > 0.5 && abs(id) < 0.5);
    col = mix(u.hair * 0.45, vec3<f32>(1.0, 0.82, 0.55), max(hot, fused));
  } else if (mat >= 3.0) {
    // Iris: dark base, bright rim low, and one hard catchlight upper-left. The
    // eye must hold both the darkest mass and the brightest pixel in frame.
    let cl = pow(clamp(dot(reflect(-key, nrm), -rd), 0.0, 1.0), 60.0);
    col = u.iris * (0.30 + 0.45 * lit)
        + u.iris * 1.8 * pow(clamp(-nrm.y, 0.0, 1.0), 2.0) * 0.5
        + vec3<f32>(1.0) * step(0.55, cl) * 1.6;
  } else if (mat >= 2.0) {
    col = vec3<f32>(0.97, 0.97, 1.0) * (0.82 + 0.18 * lit);
  } else if (mat >= 1.0) {
    col = u.hair * lit;
  } else {
    col = SKIN * lit;
    // A blush band, low on the cheek — cheap, and it is a strong anime cue.
    let cheek = smoothstep(0.30, 0.0, length(vec2<f32>(abs(p.x) - u.headR * 0.52, p.y - u.headR * 0.24)));
    col = mix(col, vec3<f32>(1.0, 0.62, 0.62), cheek * 0.30);
  }

  // INK OUTLINE. Weight follows the bit width: a coarser build is drawn with a
  // heavier pen.
  col = mix(col, vec3<f32>(0.04, 0.04, 0.07), smoothstep(1.0 - u.ink, 0.995, fres));
  // ALREADY ON THIS DEVICE. A cached model is lit — it can start now, where an
  // uncached one costs a download first. The line under the card says so in
  // words; this says it before the words are read.
  if (u.cached > 0.5) {
    col = col + u.hair * fres * 0.55 + u.hair * 0.06;
  }
  col = col * (0.94 + 0.14 * u.hover);
  col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.05));
  return vec4<f32>(col, 1.0);
}
`

export interface MascotHandle {
  setSpec: (spec: ModelSpec, cached?: boolean) => void
  setHover: (on: boolean) => void
  /** One frame as a PNG data URL, for surfaces that want the character
   *  without a GPU pass of their own — a message avatar appears once per
   *  reply and would otherwise be a live canvas each. Awaits the queue: read
   *  before the frame lands and the canvas is still empty, which shows up as
   *  a blank avatar rather than an error. */
  snapshot: () => Promise<string>
  destroy: () => void
}

let devicePromise: Promise<GPUDevice | null> | null = null
function getDevice(): Promise<GPUDevice | null> {
  if (!devicePromise) {
    devicePromise = (async () => {
      if (!('gpu' in navigator)) return null
      try {
        const adapter = await navigator.gpu.requestAdapter()
        return (await adapter?.requestDevice()) ?? null
      } catch { return null }
    })()
  }
  return devicePromise
}

// Hair lanes. Saturated enough to be characters, held to four hues so eight
// figures on one page still read as a set.
const HAIR = {
  dense: [0.42, 0.62, 0.95],
  hybrid: [0.30, 0.84, 0.70],
  moe: [0.96, 0.60, 0.36],
  mla: [0.72, 0.54, 0.95],
  embed: [0.62, 0.66, 0.74],
}
const IRIS = {
  dense: [0.30, 0.52, 0.92],
  hybrid: [0.16, 0.72, 0.60],
  moe: [0.94, 0.52, 0.24],
  mla: [0.60, 0.40, 0.92],
  embed: [0.45, 0.50, 0.58],
}

/** The lane a model's character belongs to, as CSS. The chat page themes
 *  itself from this, so the accent on screen is the same colour as the
 *  creature that was on the loading gate — one model, one colour, and both
 *  derived from the architecture rather than picked per page. */
export function mascotPalette(spec: ModelSpec): { accent: string; accentHi: string } {
  const gdn = spec.layerKinds.filter((k) => k === 'gdn').length
  const coil = gdn / Math.max(spec.layers, 1)
  const lane = spec.embeddingOnly === true ? 'embed'
    : spec.moe != null ? 'moe'
    : spec.mla != null ? 'mla'
    : coil > 0.1 ? 'hybrid' : 'dense'
  const hex = (c: number[]): string =>
    '#' + c.map((v) => Math.round(Math.min(1, v) * 255).toString(16).padStart(2, '0')).join('')
  const base = HAIR[lane as keyof typeof HAIR]
  return { accent: hex(base), accentHi: hex(base.map((v) => v * 0.55 + 0.45)) }
}

/** Spec → the numbers the shader draws. Nothing is authored per model. */
export function mascotParams(spec: ModelSpec, cached = false): Float32Array<ArrayBuffer> {
  const gdn = spec.layerKinds.filter((k) => k === 'gdn').length
  const coil = gdn / Math.max(spec.layers, 1)
  const isMoe = spec.moe != null
  const isMla = spec.mla != null
  const isEmb = spec.embeddingOnly === true
  const lane = isEmb ? 'embed' : isMoe ? 'moe' : isMla ? 'mla' : coil > 0.1 ? 'hybrid' : 'dense'
  const hair = HAIR[lane as keyof typeof HAIR]
  const iris = IRIS[lane as keyof typeof IRIS]

  // Bit width and quant group drive the DRAWING, not the anatomy: a 3-bit build
  // is posterised into fewer bands and inked heavier. The group (32 for
  // MLC-symmetric, 64 for MLX-affine) shifts where the bands split, which is
  // what makes the two Qwen3-4B builds different pictures — keying off
  // `moe.bits` alone made them byte-identical.
  const bits = spec.moe?.bits ?? 4
  const group = spec.weightFormat === 'mlx-safetensors' ? 64 : 32

  const a: Float32Array<ArrayBuffer> = new Float32Array(new ArrayBuffer(112))
  a.set([
    0, 0, 0, 0,
    0.30 + Math.log2(spec.d) * 0.016,
    Math.max(4, Math.min(14, Math.round(spec.layers / 3))),
    coil,
    spec.kvHeads,
    isMoe ? spec.moe!.experts : 0,
    isMoe ? spec.moe!.topK : 0,
    spec.sharedExpertIndex >= 0 ? 1 : 0,
    bits <= 3 ? 2 : 3,
    bits <= 3 ? 0.46 : 0.34,
    isEmb ? 0 : 1,
    group === 64 ? 0.5 : 0.0,
    0,
    hair[0], hair[1], hair[2], 0,
    iris[0], iris[1], iris[2], cached ? 1 : 0,
  ])
  return a
}

/** Mount one continuously-animating character. The handle lets the browser
 *  re-point it at another model without rebuilding the pipeline. */
export async function mountMascot(
  canvas: HTMLCanvasElement, spec: ModelSpec,
): Promise<MascotHandle | null> {
  const device = await getDevice()
  const ctx = device ? canvas.getContext('webgpu') : null
  if (!device || !ctx) return null

  const format = navigator.gpu.getPreferredCanvasFormat()
  ctx.configure({ device, format, alphaMode: 'premultiplied' })
  const module = device.createShaderModule({ code: WGSL })
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: {
      module, entryPoint: 'fs',
      targets: [{
        format,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        },
      }],
    },
  })

  let params = mascotParams(spec)
  const ubo = device.createBuffer({
    size: params.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: ubo } }],
  })

  const still = matchMedia('(prefers-reduced-motion: reduce)').matches
  let hover = 0, t = 0, raf = 0, dead = false

  const frame = (): void => {
    if (dead) return
    const dpr = Math.min(devicePixelRatio || 1, 2)
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr))
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr))
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }
    params[0] = canvas.width
    params[1] = canvas.height
    params[2] = t
    params[3] = hover
    device.queue.writeBuffer(ubo, 0, params)
    const enc = device.createCommandEncoder()
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear', storeOp: 'store',
      }],
    })
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.draw(3); pass.end()
    device.queue.submit([enc.finish()])
    if (!still) { t += 1 / 60; raf = requestAnimationFrame(frame) }
  }
  frame()

  return {
    setSpec(next, cached = false) {
      const p = mascotParams(next, cached)
      p[2] = t
      params = p
      if (still) frame()
    },
    setHover(on) { hover = on ? 1 : 0 },
    async snapshot() {
      await device.queue.onSubmittedWorkDone()
      return canvas.toDataURL('image/png')
    },
    destroy() { dead = true; cancelAnimationFrame(raf) },
  }
}
