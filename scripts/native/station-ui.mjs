/**
 * STATION UI — the page scripts/station.mjs serves at `/`.
 *
 * One self-contained HTML string: no framework, no build step, no CDN (the
 * machine serving this may be offline by design). It polls /api/state on a
 * timer and renders three things — what is loaded and how fast it is running,
 * the catalogue with each model's real configuration, and the endpoint.
 *
 * Numbers policy: sizes, context ceilings and KV costs come from the spec;
 * rates are BENCH.md's measured labels and are simply ABSENT for models never
 * measured; live throughput is the engine's own record. The memory figure is
 * arithmetic on the first two (weights + ctx x KV/token) and is labelled as
 * an estimate, because allocator overhead is not modelled.
 */

export function stationUi() {
  return `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>zero-tvm station</title>
<style>
  :root{--bg:#0a0f1c;--panel:#0e1522;--panel2:#111a2b;--line:#1c2740;--text:#e8edf7;
        --dim:#7d8aa3;--accent:#f0a860;--ok:#54d18c;--warn:#d9a441;--err:#e2685f;
        --mono:ui-monospace,SFMono-Regular,Menlo,monospace}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:24px 20px 64px}
  header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;
         padding-bottom:14px;border-bottom:1px solid var(--line)}
  h1{font-size:1.02rem;margin:0;font-weight:600}
  .brand{color:var(--accent)}
  .pill{margin-left:auto;font-family:var(--mono);font-size:.68rem;letter-spacing:.09em;
        text-transform:uppercase;color:var(--dim);display:flex;align-items:center;gap:7px}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--dim)}
  .dot.ready{background:var(--ok);box-shadow:0 0 8px var(--ok)}
  .dot.loading{background:var(--warn);box-shadow:0 0 8px var(--warn);animation:p 1.2s infinite}
  .dot.busy{background:var(--accent);box-shadow:0 0 8px var(--accent);animation:p 1s infinite}
  .dot.failed{background:var(--err);box-shadow:0 0 8px var(--err)}
  @keyframes p{50%{opacity:.35}}
  h2{font-size:.66rem;font-family:var(--mono);letter-spacing:.16em;text-transform:uppercase;
     color:var(--dim);margin:26px 0 10px;font-weight:500}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
        background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden}
  .cell{background:var(--panel);padding:13px 15px}
  .cell .v{font-family:var(--mono);font-size:1.25rem;font-variant-numeric:tabular-nums}
  .cell .v small{font-size:.58em;color:var(--dim);margin-left:3px}
  .cell .k{font-size:.63rem;font-family:var(--mono);letter-spacing:.1em;text-transform:uppercase;
           color:var(--dim);margin-top:4px}
  .bar{height:4px;background:#162034;border-radius:2px;overflow:hidden;margin-top:9px}
  .bar i{display:block;height:100%;background:linear-gradient(90deg,#f0a86088,var(--accent));transition:width .4s}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:7px;
        padding:14px 16px;margin-bottom:9px}
  .card.on{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
  .top{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
  .nm{font-weight:600}
  .sub{font-family:var(--mono);font-size:.68rem;color:var(--dim)}
  .facts{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-family:var(--mono);
         font-size:.7rem;color:var(--dim)}
  .facts b{color:var(--text);font-weight:500}
  .cfg{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:11px}
  label{font-family:var(--mono);font-size:.64rem;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
  input,select{font:inherit;font-family:var(--mono);font-size:.76rem;background:var(--panel2);
    color:var(--text);border:1px solid var(--line);border-radius:4px;padding:5px 8px}
  input{width:110px}
  button{font:inherit;font-family:var(--mono);font-size:.72rem;cursor:pointer;background:var(--panel2);
    color:var(--text);border:1px solid var(--line);border-radius:4px;padding:6px 13px}
  button:hover:not(:disabled){border-color:var(--dim)}
  button.go{background:var(--accent);color:#0b0d10;border-color:var(--accent);font-weight:600}
  button:disabled{opacity:.45;cursor:not-allowed}
  .est{font-family:var(--mono);font-size:.68rem;color:var(--dim);margin-left:auto}
  .warn{color:var(--warn)}
  pre{background:#070b13;border:1px solid var(--line);border-radius:6px;padding:11px 13px;
      font-family:var(--mono);font-size:.68rem;line-height:1.5;color:var(--dim);
      max-height:190px;overflow:auto;white-space:pre-wrap;margin:0}
  code{font-family:var(--mono);font-size:.74rem;background:var(--panel);border:1px solid var(--line);
       border-radius:4px;padding:3px 7px}
  table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.72rem}
  th{text-align:right;font-weight:400;color:var(--dim);font-size:.6rem;letter-spacing:.1em;
     text-transform:uppercase;padding:5px 9px;border-bottom:1px solid var(--line)}
  th:first-child,td:first-child{text-align:left}
  td{padding:5px 9px;border-bottom:1px solid #131c2e;text-align:right;font-variant-numeric:tabular-nums}
  .note{color:var(--dim);font-size:.73rem;line-height:1.65;margin-top:9px}
  .err{color:var(--err);font-family:var(--mono);font-size:.72rem;margin-top:8px}
</style></head><body><div class="wrap">

<header>
  <h1>zero<span class="brand">-tvm</span> station</h1>
  <span class="sub" id="hdr"></span>
  <span class="pill"><span class="dot" id="dot"></span><span id="state">—</span></span>
</header>

<div id="live" style="display:none">
  <h2>Running</h2>
  <div class="grid">
    <div class="cell"><div class="v" id="pf">—</div><div class="k">prefill tok/s</div></div>
    <div class="cell"><div class="v" id="dec">—</div><div class="k">decode tok/s</div></div>
    <div class="cell"><div class="v" id="ttft">—</div><div class="k">first token</div></div>
    <div class="cell"><div class="v" id="pt">—</div><div class="k">prompt tokens</div></div>
    <div class="cell"><div class="v" id="gt">—</div><div class="k">generated</div></div>
  </div>
  <div class="cell" style="border:1px solid var(--line);border-top:0;border-radius:0 0 6px 6px">
    <div class="v" id="ctxv">—</div><div class="k">context used</div>
    <div class="bar"><i id="ctxbar" style="width:0%"></i></div>
  </div>
  <div style="margin-top:10px">
    <button onclick="clearModel()" title="unload the model and free its memory">Clear</button>
  </div>
</div>

<div id="memBox" style="display:none">
  <h2>Machine</h2>
  <div class="cell" style="border:1px solid var(--line);border-radius:6px">
    <div class="v" id="memv">—</div><div class="k">memory in use</div>
    <div class="bar"><i id="membar" style="width:0%"></i></div>
    <div class="note" id="memnote" style="margin-top:8px"></div>
  </div>
</div>

<div id="histBox" style="display:none">
  <h2>Requests <button style="margin-left:8px;padding:2px 8px" onclick="clearHistory()">clear</button></h2>
  <table><thead><tr>
    <th>time</th><th>prompt</th><th>prefill</th><th>first tok</th><th>gen</th><th>decode</th><th>ctx after</th>
  </tr></thead><tbody id="hrows"></tbody></table>
</div>

<div id="loadingBox" style="display:none">
  <h2>Loading</h2>
  <pre id="log"></pre>
</div>
<div class="err" id="failure" style="display:none"></div>

<h2>Models</h2>
<div id="models"></div>

<h2>Endpoint</h2>
<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
  <code id="base">—</code><button onclick="cp('base')">copy</button>
  <code>model: ztvm</code>
</div>
<div class="note">
  OpenAI-compatible, and <b>stable across model swaps</b> — the station holds this
  port and proxies to whichever engine is loaded, so a client never needs
  reconfiguring. In Cline: <b>OpenAI Compatible</b>, this base URL, any non-empty
  API key (no auth), model <code>ztvm</code>, and set the context window to the
  loaded model's — clients default it low and truncate silently.
  One request at a time: a single KV cache, so parallel calls queue.
</div>

<script>
const $=(i)=>document.getElementById(i)
const fmt=(n,d=0)=>n==null?'—':Number(n).toLocaleString(undefined,{maximumFractionDigits:d})
const kfmt=(n)=>n>=1024?(n/1024).toFixed(n%1024?1:0)+'k':String(n)
function cp(id){navigator.clipboard.writeText($(id).textContent)}
let MODELS=[],phase='',renderedFor=''

async function load(param){
  const m=MODELS.find((x)=>x.param===param)
  const ctx=Number($('ctx-'+m.id).value)||0
  const poolEl=$('pool-'+m.id)
  const pool=poolEl?Number(poolEl.value)||0:0
  const r=await fetch('/api/load',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({param,ctx,pool})})
  if(!r.ok){const e=await r.json();alert(e.error||'load refused')}
  tick()
}
async function clearModel(){await fetch('/api/unload',{method:'POST'});tick()}
async function clearHistory(){await fetch('/api/history/clear',{method:'POST'});tick()}

function residentGb(m){
  // What actually occupies memory. A pool build holds fewer experts, so the
  // chosen build's own label is the truth; sizeLabel is the DOWNLOAD, which
  // is a different quantity and the wrong one for "will this fit".
  const sel=$('pool-'+m.id)
  if(sel){const g=/([\\d.]+)\\s*GB/.exec(sel.options[sel.selectedIndex].text);if(g)return parseFloat(g[1])}
  return m.weightsGb
}
function estimate(m,ctx){
  const w=residentGb(m)
  if(w==null)return ''
  const kv=(ctx*m.kvBytesPerToken)/(1024**3)
  return '≈ '+(w+kv).toFixed(1)+' GB resident  ('+w+' weights + '+kv.toFixed(1)+' KV)'
}
function renderModels(loaded){
  $('models').innerHTML=MODELS.map((m)=>{
    const on=loaded&&loaded.param===m.param
    const pools=m.poolModes.length>1?'<label>pool</label><select id="pool-'+m.id+'" onchange="reEst(\\''+m.id+'\\')">'+
      m.poolModes.map((p)=>'<option value="'+p.slots+'">'+p.label+'</option>').join('')+'</select>':''
    return '<div class="card'+(on?' on':'')+'">'+
      '<div class="top"><span class="nm">'+m.name+'</span>'+
      '<span class="sub">'+m.params+'</span>'+
      (on?'<span class="sub" style="color:var(--accent)">● loaded</span>':'')+'</div>'+
      '<div class="facts">'+
        '<span>download <b>'+m.sizeLabel+'</b></span>'+
        '<span title="what the engine ships / what the checkpoint was trained for">'+
          'context <b>'+kfmt(m.defaultCtx)+'</b> default, '+kfmt(m.maxCtx)+' max</span>'+
        '<span>KV <b>'+(m.kvBytesPerToken/1024)+' KB</b>/tok</span>'+
        (m.rateLabel?'<span>measured <b>'+m.rateLabel+'</b></span>':'<span>rate <b>unmeasured</b></span>')+
        (m.ramNote?'<span class="warn">'+m.ramNote+'</span>':'')+
      '</div>'+
      '<div class="cfg"><label>context</label>'+
        '<input id="ctx-'+m.id+'" type="number" min="1024" max="'+m.maxCtx+'" step="1024" value="'+m.defaultCtx+
        '" oninput="reEst(\\''+m.id+'\\')">'+pools+
        '<button class="go" id="btn-'+m.id+'" onclick="load(\\''+m.param+'\\')">'+(on?'Reload':'Load')+'</button>'+
        '<span class="est" id="est-'+m.id+'">'+estimate(m,m.defaultCtx)+'</span>'+
      '</div></div>'
  }).join('')
}
function reEst(id){
  const m=MODELS.find((x)=>x.id===id)
  const v=Math.min(Number($('ctx-'+id).value)||0,m.maxCtx)
  $('est-'+id).textContent=estimate(m,v)
}

async function tick(){
  let s
  try{s=await (await fetch('/api/state',{cache:'no-store'})).json()}
  catch{$('state').textContent='offline';$('dot').className='dot';return}
  MODELS=s.models
  const busy=s.engine&&s.engine.busy
  $('dot').className='dot '+(busy?'busy':s.phase==='ready'?'ready':s.phase==='loading'?'loading':s.phase==='failed'?'failed':'')
  $('state').textContent=busy?'generating':s.phase
  $('base').textContent=location.origin+'/v1'
  $('hdr').textContent=s.loaded?s.loaded.param+' · '+fmt(s.engine&&s.engine.ctx)+' ctx':'no model loaded'
  $('loadingBox').style.display=s.phase==='loading'?'':'none'
  if(s.phase==='loading')$('log').textContent=s.log.join('\\n')
  $('failure').style.display=s.failure?'':'none'
  $('failure').textContent=s.failure||''
  $('live').style.display=s.phase==='ready'?'':'none'
  const key=JSON.stringify([MODELS.length,s.loaded&&s.loaded.param,s.phase])
  if(key!==renderedFor){renderedFor=key;renderModels(s.loaded)}
  // Machine memory — measured (macOS vm_stat), absent elsewhere rather than faked.
  const m=s.memory
  $('memBox').style.display=m?'':'none'
  if(m){
    const pct=m.usedGb!=null?(m.usedGb/m.totalGb)*100:0
    $('memv').innerHTML=fmt(m.usedGb,1)+'<small>/ '+fmt(m.totalGb,0)+' GB'+
      (m.compressedGb?' · '+fmt(m.compressedGb,1)+' compressed':'')+'</small>'
    $('membar').style.width=Math.min(100,pct)+'%'
    $('membar').style.background=m.swappingNow?'linear-gradient(90deg,#e2685f88,#e2685f)'
      :pct>85?'linear-gradient(90deg,#d9a44188,#d9a441)':'linear-gradient(90deg,#f0a86088,var(--accent))'
    $('memnote').textContent=m.note||(pct>85?'close to full — a second model will swap':'')
    $('memnote').className='note'+(m.swappingNow?' err':'')
  }
  // Request trend — the point is decode drifting down as the window fills.
  const H=s.history||[]
  $('histBox').style.display=H.length?'':'none'
  $('hrows').innerHTML=H.map((r)=>'<tr><td>'+new Date(r.at).toLocaleTimeString()+
    '</td><td>'+fmt(r.promptTokens)+'</td><td>'+fmt(r.prefillTokPerSec)+
    '</td><td>'+(r.ttftMs/1000).toFixed(2)+'s</td><td>'+fmt(r.genTokens)+
    '</td><td>'+fmt(r.decodeTokPerSec,1)+'</td><td>'+fmt(r.contextUsed)+'</td></tr>').join('')
  const l=s.engine&&s.engine.last
  if(l){
    $('pf').textContent=fmt(l.prefillTokPerSec)
    $('dec').textContent=fmt(l.decodeTokPerSec,1)
    $('ttft').innerHTML=(l.ttftMs/1000).toFixed(2)+'<small>s</small>'
    $('pt').textContent=fmt(l.promptTokens)
    $('gt').textContent=fmt(l.genTokens)
    const ctx=s.engine.ctx||1,pct=(l.contextUsed/ctx)*100
    $('ctxv').innerHTML=fmt(l.contextUsed)+'<small>/ '+fmt(ctx)+' · '+pct.toFixed(1)+'%</small>'
    $('ctxbar').style.width=Math.min(100,pct)+'%'
  }
}
tick();setInterval(tick,1000)
</script>
</div></body></html>`
}
