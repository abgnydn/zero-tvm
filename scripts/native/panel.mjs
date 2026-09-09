/**
 * THE PANEL — a one-page control surface for the native host, served by the
 * host itself at `/` to anything asking for HTML.
 *
 * What it is for: driving this server from Cline (or pi) gave no feedback at
 * all — a request came back or it did not. This shows what LM Studio's log
 * shows, plus the things that decide whether a model FITS: resident weights,
 * KV cost of the chosen window, and how much of that window a conversation
 * has actually eaten.
 *
 * Deliberately dependency-free and build-free: one HTML string, fetch on a
 * timer, no framework. It is served over the tailnet like the API, so the
 * machine driving the model can watch it without switching machines.
 *
 * Every number here comes from the host's own /health record — the same
 * measured values the terminal prints — or from the registry. Nothing is
 * estimated in this file.
 */

export function panelHtml() {
  return `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>zero-tvm — native host</title>
<style>
  :root {
    --bg:#0a0f1c; --panel:#0e1522; --line:#1c2740; --text:#e8edf7;
    --dim:#7d8aa3; --accent:#f0a860; --ok:#54d18c; --warn:#d9a441;
    --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  .wrap { max-width:980px; margin:0 auto; padding:28px 20px 60px }
  header { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;
           padding-bottom:16px; border-bottom:1px solid var(--line) }
  h1 { font-size:1.05rem; margin:0; font-weight:600; letter-spacing:-0.01em }
  .brand { color:var(--accent) }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--dim);
         display:inline-block; margin-right:6px }
  .dot.on { background:var(--ok); box-shadow:0 0 8px var(--ok) }
  .dot.busy { background:var(--warn); box-shadow:0 0 8px var(--warn) }
  .status { font-family:var(--mono); font-size:.72rem; letter-spacing:.08em;
            text-transform:uppercase; color:var(--dim); margin-left:auto }
  h2 { font-size:.68rem; font-family:var(--mono); letter-spacing:.16em;
       text-transform:uppercase; color:var(--dim); margin:28px 0 10px; font-weight:500 }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1px;
          background:var(--line); border:1px solid var(--line); border-radius:6px; overflow:hidden }
  .cell { background:var(--panel); padding:14px 16px }
  .cell .v { font-family:var(--mono); font-size:1.3rem; font-variant-numeric:tabular-nums }
  .cell .k { font-size:.66rem; font-family:var(--mono); letter-spacing:.1em;
             text-transform:uppercase; color:var(--dim); margin-top:4px }
  .cell .v small { font-size:.6em; color:var(--dim); margin-left:3px }
  table { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:.76rem }
  th { text-align:right; font-weight:400; color:var(--dim); font-size:.64rem;
       letter-spacing:.1em; text-transform:uppercase; padding:6px 10px; border-bottom:1px solid var(--line) }
  th:first-child, td:first-child { text-align:left }
  td { padding:6px 10px; border-bottom:1px solid #131c2e; text-align:right;
       font-variant-numeric:tabular-nums }
  tr:last-child td { border-bottom:0 }
  .bar { height:4px; background:#16203400; border-radius:2px; background:#162034; overflow:hidden; margin-top:8px }
  .bar i { display:block; height:100%; background:linear-gradient(90deg,#f0a86088,var(--accent)) }
  code { font-family:var(--mono); font-size:.76rem; background:var(--panel);
         border:1px solid var(--line); border-radius:4px; padding:2px 6px; color:var(--text) }
  .endpoint { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:8px }
  button { font:inherit; font-size:.7rem; font-family:var(--mono); cursor:pointer;
           background:var(--panel); color:var(--dim); border:1px solid var(--line);
           border-radius:4px; padding:4px 9px }
  button:hover { color:var(--text); border-color:var(--dim) }
  .empty { color:var(--dim); font-family:var(--mono); font-size:.76rem; padding:10px 0 }
  .note { color:var(--dim); font-size:.74rem; margin-top:10px; line-height:1.6 }
</style></head><body><div class="wrap">

<header>
  <h1>zero<span class="brand">-tvm</span> · native host</h1>
  <span id="model" style="font-family:var(--mono);font-size:.8rem;color:var(--dim)"></span>
  <span class="status"><span class="dot" id="dot"></span><span id="state">connecting</span></span>
</header>

<h2>Last request</h2>
<div class="grid">
  <div class="cell"><div class="v" id="pf">—</div><div class="k">prefill tok/s</div></div>
  <div class="cell"><div class="v" id="dec">—</div><div class="k">decode tok/s</div></div>
  <div class="cell"><div class="v" id="ttft">—</div><div class="k">time to first token</div></div>
  <div class="cell"><div class="v" id="pt">—</div><div class="k">prompt tokens</div></div>
  <div class="cell"><div class="v" id="gt">—</div><div class="k">generated</div></div>
</div>

<h2>Context</h2>
<div class="cell" style="border:1px solid var(--line);border-radius:6px">
  <div class="v" id="ctx">—</div>
  <div class="k">used of window</div>
  <div class="bar"><i id="ctxbar" style="width:0%"></i></div>
</div>

<h2>Recent requests</h2>
<table><thead><tr>
  <th>time</th><th>prompt</th><th>prefill</th><th>ttft</th><th>gen</th><th>decode</th>
</tr></thead><tbody id="rows"></tbody></table>
<div class="empty" id="none">no requests yet — point a client at the endpoint below</div>

<h2>Endpoint</h2>
<div class="endpoint">
  <code id="base"></code><button onclick="cp('base')">copy</button>
  <code id="mid"></code><button onclick="cp('mid')">copy</button>
</div>
<div class="note">
  OpenAI-compatible. In Cline choose <b>OpenAI Compatible</b>, paste the base URL,
  put anything in the API-key field (this server has no auth), and set the context
  window to the number above — clients default it low and silently truncate.
  One request at a time: a single KV cache, so parallel calls queue.
</div>

<h2>Reading these numbers</h2>
<div class="note">
  The first token splits a request in two: everything before it is prefill
  (prompt processing), everything after is decode. Both are wall clock over real
  token counts, the same definitions BENCH.md uses. A follow-up turn in the same
  conversation shows a huge prefill rate because the prefix cache re-reads almost
  nothing — the honest prefill number is the first message of a task.
</div>

<script>
const $ = (id) => document.getElementById(id)
const fmt = (n, d = 0) => n == null ? '—' : n.toLocaleString(undefined, { maximumFractionDigits: d })
function cp(id) { navigator.clipboard.writeText($(id).textContent) }
const seen = []
async function tick() {
  let h
  try { h = await (await fetch('/health', { cache: 'no-store' })).json() }
  catch { $('state').textContent = 'offline'; $('dot').className = 'dot'; return }
  $('dot').className = 'dot ' + (h.busy ? 'busy' : 'on')
  $('state').textContent = h.busy ? 'generating' : 'ready'
  $('model').textContent = h.hosting + ' · ' + fmt(h.ctx) + ' ctx'
  $('base').textContent = location.origin + '/v1'
  $('mid').textContent = h.hosting
  const l = h.last
  if (l) {
    $('pf').innerHTML = fmt(l.prefillTokPerSec)
    $('dec').innerHTML = fmt(l.decodeTokPerSec, 1)
    $('ttft').innerHTML = (l.ttftMs / 1000).toFixed(2) + '<small>s</small>'
    $('pt').textContent = fmt(l.promptTokens)
    $('gt').textContent = fmt(l.genTokens)
    const pct = h.ctx ? (l.contextUsed / h.ctx) * 100 : 0
    $('ctx').innerHTML = fmt(l.contextUsed) + '<small>/ ' + fmt(h.ctx) + ' · ' + pct.toFixed(1) + '%</small>'
    $('ctxbar').style.width = Math.min(100, pct) + '%'
    if (!seen.length || seen[0].at !== l.at) {
      seen.unshift(l); seen.length = Math.min(seen.length, 12)
      $('none').style.display = 'none'
      $('rows').innerHTML = seen.map((r) => '<tr><td>' + new Date(r.at).toLocaleTimeString()
        + '</td><td>' + fmt(r.promptTokens) + '</td><td>' + fmt(r.prefillTokPerSec)
        + '</td><td>' + (r.ttftMs / 1000).toFixed(2) + 's</td><td>' + fmt(r.genTokens)
        + '</td><td>' + fmt(r.decodeTokPerSec, 1) + '</td></tr>').join('')
    }
  }
}
tick(); setInterval(tick, 1000)
</script>
</div></body></html>`
}
