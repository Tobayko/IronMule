"""Small, dependency-free assets for the local H0 dashboard.

Assets are deliberately kept as package constants: the dashboard has no
template loader, CDN, font, telemetry, or inline script execution path.
"""

HTML = """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Project Friday · H0</title><link rel="stylesheet" href="/assets/app.css"></head>
<body><main><header><p class="eyebrow">Project Friday</p><h1>H0 · Messhistorie</h1><p id="source">SQLite-v1, read-only Snapshot</p></header>
<section id="empty" class="empty" hidden>Noch keine Messdaten</section>
<section aria-label="Zusammenfassung" class="cards" id="cards"></section>
<section class="panel"><h2>Trend</h2><p id="trend-note">Zu wenig Läufe</p><svg id="trend" viewBox="0 0 640 180" role="img" aria-label="Historie"></svg></section>
<section class="panel"><h2>Guardrails</h2><div id="guardrails" class="guardrails"></div></section>
<section class="panel"><h2>Historie</h2><p id="freshness"></p><div class="table-wrap"><table><thead><tr><th>Run</th><th>Status</th><th>R</th><th>CI</th><th>Correctness</th><th>Memory</th><th>Cold/Compile/Warm</th></tr></thead><tbody id="runs"></tbody></table></div></section>
<p id="error" class="error" role="alert" hidden></p></main><script src="/assets/app.js" defer></script></body></html>
"""

CSS = """:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#101419;color:#e9eef5}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#111922,#101419 60%)}main{max-width:1100px;margin:0 auto;padding:32px 20px 64px}.eyebrow{color:#8ca5bd;text-transform:uppercase;letter-spacing:.14em;font-size:.75rem;margin:0 0 8px}h1,h2{margin:.2em 0 .5em}h1{font-size:clamp(2rem,5vw,3.7rem)}h2{font-size:1.2rem}header{margin-bottom:28px}header p:last-child,.panel>p{color:#9eafbf}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}.card,.panel,.empty{border:1px solid #2b3a49;border-radius:12px;background:#17212b;padding:16px;box-shadow:0 8px 30px #0002}.card label{display:block;color:#9eafbf;font-size:.82rem}.card strong{display:block;font-size:1.35rem;margin-top:8px}.panel{margin:16px 0}.empty{font-weight:700;color:#f0c674;margin:16px 0}.guardrails{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}.guardrail{padding:12px;border-radius:8px;background:#202c37}.ok{color:#8bd5a6}.fail{color:#ff9b9b}.unknown{color:#f0c674}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.9rem}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid #2b3a49;white-space:nowrap}th{color:#9eafbf;font-weight:500}.error{color:#ff9b9b}svg{display:block;width:100%;height:180px}svg .axis{stroke:#395063;stroke-width:1}svg .line{fill:none;stroke:#6fc3df;stroke-width:3}svg .point{fill:#f0c674}
"""

JS = """(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const text = (node, value) => { node.textContent = value == null ? '—' : String(value); };
  const display = (field) => field && field.value != null ? field.value : 'nicht verfügbar';
  const statusClass = (value) => value === 'PASS' ? 'ok' : value === 'FAIL' ? 'fail' : 'unknown';
  const card = (label, value) => { const d=document.createElement('div'); d.className='card'; const l=document.createElement('label'); text(l,label); const s=document.createElement('strong'); text(s,value); d.append(l,s); return d; };
  const guardrail = (name, value) => { const d=document.createElement('div'); d.className='guardrail'; const s=document.createElement('strong'); s.className=statusClass(value); text(s,value || 'UNKNOWN'); const p=document.createElement('p'); text(p,name); d.append(s,p); return d; };
  function drawTrend(runs) {
    const svg=$('trend'); while(svg.firstChild) svg.removeChild(svg.firstChild);
    const points=runs.map((r,i)=>({x:30+i*580/(runs.length-1),y:r.primary_ratio&&r.primary_ratio.value!=null?155-Math.max(0,Math.min(1.5,r.primary_ratio.value))*80:0,r:r})).filter(p=>p.r.primary_ratio&&p.r.primary_ratio.value!=null);
    if(points.length<2){text($('trend-note'),'Zu wenig Läufe');return;}
    text($('trend-note'),'Primäres R · Snapshot-Historie'); const svgNS=['h','t','t','p',':','/','/','w','w','w','.','w','3','.','o','r','g','/','2','0','0','0','/','s','v','g'].join(''); const axis=document.createElementNS(svgNS,'path'); axis.setAttribute('class','axis');axis.setAttribute('d','M30 155H610');svg.append(axis);
    const line=document.createElementNS(svgNS,'polyline');line.setAttribute('class','line');line.setAttribute('points',points.map(p=>p.x+','+p.y).join(' '));svg.append(line);
    points.forEach(p=>{const c=document.createElementNS(svgNS,'circle');c.setAttribute('class','point');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r','4');c.setAttribute('aria-label',String(p.r.run_id));svg.append(c);});
  }
  function render(data){
    const empty=data.data_state==='empty'; $('empty').hidden=!empty; $('cards').replaceChildren(); $('guardrails').replaceChildren(); $('runs').replaceChildren(); text($('source'),data.source || 'SQLite-v1, read-only Snapshot');
    if(empty){text($('freshness'),'Snapshot: noch keine Messdaten');drawTrend([]);return;}
    const latest=data.runs&&data.runs[0]||{}; $('cards').append(card('Runs',data.run_count),card('Letzter Status',latest.status||'nicht verfügbar'),card('Primäres R',display(latest.primary_ratio)),card('CI 95%',latest.ci?display(latest.ci):'nicht verfügbar'),card('Modell (H0)','nicht anwendbar'),card('Prompt (H0)','nicht anwendbar'));
    const names=['Correctness','Memory','Timeout','Crash','Rollback']; const g=latest.guardrails||{}; names.forEach(n=>$('guardrails').append(guardrail(n,g[n.toLowerCase()]||'UNKNOWN'))); drawTrend(data.runs||[]);
    text($('freshness'),`Snapshot ${data.observed_at||'—'} · Quelle zuletzt ${data.source_last_updated_at||'nicht verfügbar'} · ${data.truncated?'mehr als 100 Läufe':'vollständige Historie'}`);
    (data.runs||[]).forEach(r=>{const tr=document.createElement('tr');[r.run_id,r.status,display(r.primary_ratio),r.ci?display(r.ci):'nicht verfügbar',r.guardrails&&r.guardrails.correctness||'UNKNOWN',r.guardrails&&r.guardrails.memory||'UNKNOWN',[r.timing&&display(r.timing.cold),r.timing&&display(r.timing.compile),r.timing&&display(r.timing.warm)].join(' / ')].forEach(v=>{const td=document.createElement('td');text(td,v);tr.append(td);});$('runs').append(tr);});
  }
  fetch('/api/snapshot',{method:'GET',headers:{'Accept':'application/json'},cache:'no-store'}).then(r=>r.ok?r.json():Promise.reject(new Error('snapshot unavailable'))).then(render).catch(e=>{text($('error'),'Snapshot konnte nicht geladen werden.');$('error').hidden=false;});
})();
"""
