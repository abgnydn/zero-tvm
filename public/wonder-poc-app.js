import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------- field (three.js, dumb InstancedMesh) ----------
const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
view.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101014);
scene.fog = new THREE.Fog(0x101014, 30, 70);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
camera.position.set(17, 14, 21);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(7.5, 2, 7.5);
scene.add(new THREE.HemisphereLight(0xdde8ff, 0x3a2e22, 0.9));
const key = new THREE.DirectionalLight(0xffe3b3, 1.6); key.position.set(8, 14, 6);
key.castShadow = true; key.shadow.mapSize.set(1024, 1024); scene.add(key);
const rim = new THREE.DirectionalLight(0x7cb7ff, 0.5); rim.position.set(-8, 6, -6); scene.add(rim);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(60, 60),
  new THREE.MeshStandardMaterial({ color: 0x17171c, roughness: 1 }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
const grid = new THREE.GridHelper(16, 16, 0x3a3a44, 0x26262e);
grid.position.set(7.5, 0.01, 7.5); scene.add(grid);

const MAX = 320;
const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.92, 0.92, 0.92),
  new THREE.MeshStandardMaterial({ roughness: 0.7 }), MAX);
mesh.castShadow = true; mesh.count = 0; scene.add(mesh);
const labelLayer = new THREE.Group(); scene.add(labelLayer);
const dummy = new THREE.Object3D(); const tmpColor = new THREE.Color();
function resize() {
  const r = view.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1); camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view); resize();
(function tick() { requestAnimationFrame(tick); controls.update(); renderer.render(scene, camera); })();

function renderBlocks(blocks, labels) {
  mesh.count = Math.min(blocks.length, MAX);
  blocks.forEach((b, i) => {
    dummy.position.set(b.at[0] + 0.5, b.at[1] + 0.5, b.at[2] + 0.5);
    const s = b.pop ?? 1; dummy.scale.setScalar(Math.min(s, 1));
    dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix);
    mesh.setColorAt(i, tmpColor.set(b.color || '#e8955a'));
  });
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  labelLayer.clear();
  labels.slice(0, 8).forEach((l) => {
    const c = document.createElement('canvas'); c.width = 256; c.height = 64;
    const g = c.getContext('2d');
    g.fillStyle = 'rgba(0,0,0,.55)'; g.fillRect(0, 0, 256, 64);
    g.fillStyle = '#fff'; g.font = '28px system-ui'; g.textAlign = 'center';
    g.fillText(String(l.text).slice(0, 22), 128, 42);
    const tex = new THREE.CanvasTexture(c);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
    sp.position.set(l.at[0] + 0.5, l.at[1] + 1.4, l.at[2] + 0.5);
    sp.scale.set(2, 0.5, 1); labelLayer.add(sp);
  });
}

// ---------- repairer (30 lines that matter) ----------
const live = { blocks: [], labels: [] };
// pop animation
setInterval(() => {
  let dirty = false;
  for (const b of live.blocks) if ((b.pop ?? 1) < 1) { b.pop = Math.min(1, (b.pop ?? 0.2) + 0.12); dirty = true; }
  if (dirty) renderBlocks(live.blocks, live.labels);
}, 50);

const PALETTE = { butter: '#ffd16b', milk: '#f5efe0', protein: '#e58b82', heat: '#ff5a1f', water: '#5aa9ff', leaf: '#6fce7d', wood: '#b08954', stone: '#9a958c' };
function repair(ops) {
  const seen = new Set(); const blocks = []; const labels = [];
  for (const o of ops) {
    if (!o || typeof o !== 'object') continue;
    if (o.op === 'label' && o.text && Array.isArray(o.at)) { labels.push(o); continue; }
    if (o.op === 'box') {
      const [x1, y1, z1] = o.from ?? [0, 0, 0]; const [x2, y2, z2] = o.to ?? [0, 0, 0];
      for (let x = Math.min(x1, x2); x <= Math.max(x1, x2); x++)
        for (let y = Math.min(y1, y2); y <= Math.max(y1, y2); y++)
          for (let z = Math.min(z1, z2); z <= Math.max(z1, z2); z++)
            ops.push({ op: 'block', at: [x, y, z], color: o.color });
      continue;
    }
    if (o.op !== 'block' || !Array.isArray(o.at)) continue;
    let [x, y, z] = o.at.map(Number);
    if (![x, y, z].every(Number.isFinite)) continue;
    x = Math.max(0, Math.min(15, Math.round(x))); y = Math.max(0, Math.min(9, Math.round(y))); z = Math.max(0, Math.min(15, Math.round(z)));
    const k = x + ',' + y + ',' + z; if (seen.has(k)) continue; seen.add(k);
    if (y > 0 && !seen.has(x + ',' + (y - 1) + ',' + z)) continue; // no floating
    if (blocks.length >= 300) break;
    blocks.push({ at: [x, y, z], color: PALETTE[o.color] || o.color || '#e8955a', pop: 0.2 });
  }
  return { blocks, labels };
}
function fallback(question) {
  const ops = [
    { op: 'box', from: [2, 0, 6], to: [5, 3, 9], color: 'protein' },
    { op: 'box', from: [10, 0, 6], to: [13, 1, 9], color: 'butter' },
    { op: 'label', text: 'before', at: [3, 4, 7] },
    { op: 'label', text: 'after heat', at: [11, 3, 7] },
  ];
  return { ops, note: 'fallback template for: ' + question };
}

// ---------- mock brain (so field is demonstrable with no weights) ----------
function mockBrain(question) {
  const q = question.toLowerCase(); const ops = [];
  if (q.includes('sun') || q.includes('star')) {
    ops.push({ op: 'box', from: [6, 0, 6], to: [9, 3, 9], color: 'heat' });
    ops.push({ op: 'block', at: [7, 4, 7], color: 'butter' }, { op: 'block', at: [8, 5, 8], color: 'butter' });
    ops.push({ op: 'label', text: 'pressure fuses H', at: [7, 6, 7] });
  } else if (q.includes('diamond') || q.includes('graphite') || q.includes('pencil')) {
    ops.push({ op: 'box', from: [2, 0, 6], to: [5, 3, 9], color: '#bfe3ff' });
    ops.push({ op: 'box', from: [10, 0, 5], to: [13, 0, 5], color: 'stone' });
    ops.push({ op: 'box', from: [10, 0, 7], to: [13, 0, 7], color: 'stone' });
    ops.push({ op: 'box', from: [10, 0, 9], to: [13, 0, 9], color: 'stone' });
    ops.push({ op: 'label', text: 'diamond: locked', at: [3, 4, 7] });
    ops.push({ op: 'label', text: 'graphite: slides', at: [11, 2, 7] });
  } else { // melt archetype
    ops.push({ op: 'box', from: [2, 0, 6], to: [5, 3, 9], color: 'protein' });
    ops.push({ op: 'box', from: [10, 0, 6], to: [13, 1, 9], color: 'butter' });
    ops.push({ op: 'label', text: 'tangled = solid', at: [3, 4, 7] });
    ops.push({ op: 'label', text: 'loose = melted', at: [11, 3, 7] });
  }
  return ops;
}

// ---------- real brain (zero-tvm, optional) ----------
const $ = (id) => document.getElementById(id);
const log = (m) => { const el = $('log'); el.textContent += m + '\n'; el.scrollTop = el.scrollHeight; };
let brain = null; // { spec, tokenizer, engine }
const SYS = `You build tiny science dioramas with boxes. Reply with JSON ONLY: a flat list of ops, no prose, no fences.
Allowed ops: {"op":"block","at":[x,y,z],"color":C}, {"op":"box","from":[x1,y1,z1],"to":[x2,y2,z2],"color":C}, {"op":"label","text":T,"at":[x,y,z]}.
Grid x,z 0-15, y 0-9. No floating: every block needs support below. Colors: protein,butter,heat,water,leaf,wood,stone. Max 40 ops.
Vary the build to the QUESTION. Comparisons go left pile (x 2-5) vs right pile (x 10-13).
Example Q "why does ice melt?" A [{"op":"box","from":[2,0,6],"to":[4,2,8],"color":"water"},{"op":"label","text":"neat grid = ice","at":[3,3,7]},{"op":"box","from":[10,0,6],"to":[13,0,9],"color":"water"},{"op":"label","text":"spread = water","at":[11,2,7]}]`;
function extractJSONArray(text) {
  const clean = String(text).replace(/```[a-z]*\n?/gi, '```').replace(/```/g, '');
  const start = clean.indexOf('[');
  if (start < 0) throw new Error('no JSON in reply');
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < clean.length; i++) {
    const c = clean[i];
    if (inStr) { if (esc) esc = false; else if (c === '\\') esc = true; else if (c === '"') inStr = false; }
    else if (c === '"') inStr = true;
    else if (c === '[') depth++;
    else if (c === ']') { depth--; if (depth === 0) return JSON.parse(clean.slice(start, i + 1)); }
  }
  throw new Error('truncated JSON in reply');
}
async function bootBrain() {
  log('wonder-poc v0.2');
  try {
    const [{ specFromSearch }, { bootChatEngine }, { buildChatPromptFor }] =
      await Promise.all([
        import('/src/zero-tvm/model-select.ts'),
        import('/src/zero-tvm/chat-flow.ts'),
        import('/src/zero-tvm/model-select.ts'),
      ]);
    const spec = specFromSearch(location.search);
    log('booting ' + spec.id + ' … (weights download on first run)');
    const res = await bootChatEngine({ spec });
    if (!res.ok) throw new Error(res.reason);
    // NOTE: BootedEngine carries device/tokenizer/weights/engine but NOT spec
    // (loading-ui.ts) — keep our own spec or buildChatPromptFor crashes on
    // spec.chatTemplateId.
    brain = { ...res, spec, buildChatPromptFor };
    $('badge').textContent = 'live: ' + spec.id; $('badge').className = 'ready';
    log('brain ready: ' + spec.id);
  } catch (e) {
    $('badge').textContent = 'mock mode (no weights/gpu)'; $('badge').className = 'error';
    log('mock mode: ' + e.message);
  }
  $('go').disabled = false;
}
async function askBrain(question) {
  if (!brain) return { ops: mockBrain(question), source: 'mock' };
  const { spec, tokenizer, engine, buildChatPromptFor } = brain;
  const ids = buildChatPromptFor(spec, [
    { role: 'system', content: SYS },
    { role: 'user', content: question + ' Reply with JSON ONLY.' },
  ], tokenizer);
  const out = []; const budget = Math.min(512, engine.maxContext - ids.length);
  await engine.generatePipelined(ids, budget, (id) => out.push(id));
  const text = tokenizer.decode(out);
  $('out').textContent = text.slice(0, 2000);
  return { ops: extractJSONArray(text), source: spec.id };
}
// headless-test hook: window.__poc = { live, run } (also lets the console drive it)
window.__poc = { live, run: (...a) => run(...a) };

// ---------- wire up ----------
const demos = ['why does cheese melt?', 'diamond vs pencil?', 'how does the sun stay on?', 'why oil and water separate?'];
$('chips').append(...demos.map((d) => {
  const b = document.createElement('button'); b.textContent = d;
  b.onclick = () => { $('q').value = d; run(); }; return b;
}));
async function run() {
  const q = $('q').value.trim() || 'why does cheese melt?';
  $('go').disabled = true; log('ask: ' + q);
  try {
    const { ops, source } = await askBrain(q);
    let r = repair(Array.isArray(ops) ? ops : []);
    if (!r.blocks.length) { const f = fallback(q); r = repair(f.ops); log(f.note); }
    live.blocks = r.blocks; live.labels = r.labels;
    renderBlocks(live.blocks, live.labels);
    log(`built ${r.blocks.length} blocks via ${source}`);
    if (source === 'mock') $('out').textContent = JSON.stringify(ops.slice(0, 12), null, 1) + '\n…mock template (boot real brain with ?model=llama32)';
  } catch (e) { const f = fallback(q); const r = repair(f.ops); live.blocks = r.blocks; live.labels = r.labels; renderBlocks(live.blocks, live.labels); log('fallback: ' + e.message); }
  $('go').disabled = false;
}
$('go').onclick = run;
bootBrain().then(() => run());
