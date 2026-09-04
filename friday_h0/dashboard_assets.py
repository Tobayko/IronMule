"""Small, dependency-free assets for the local H0 dashboard.

Assets are deliberately kept as package constants: the dashboard has no
template loader, CDN, font, telemetry, or inline script execution path.
"""

HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="theme-color" content="#050806">
  <title>Project Friday · Signal Board</title>
  <link rel="stylesheet" href="/assets/app.css">
</head>
<body>
  <div class="matrix-layer" aria-hidden="true">
    <span>01001101<br>10100110<br>00110101<br>11001010<br>01101001</span>
    <span>11010100<br>00101101<br>10110010<br>01001101<br>10010110</span>
    <span>00111010<br>11000101<br>01101010<br>10010101<br>01011001</span>
  </div>
  <main>
    <header class="masthead">
      <div class="brand-row">
        <p class="brand">// PROJECT_FRIDAY <span>:: H0</span></p>
        <div class="live-state" id="live-state" aria-live="polite"><i></i><span>SYNCING</span></div>
      </div>
      <div class="hero-grid">
        <div>
          <p class="kicker">[ HARDWARE-AWARE RUNTIME TELEMETRY ]</p>
          <h1>Signal<br><span>Board</span></h1>
        </div>
        <div class="hero-copy">
          <p>Lokale Messhistorie. Read-only. Jede Anzeige stammt aus der gebundenen SQLite-v1-Evidenz.</p>
          <dl class="source-meta">
            <div><dt>source</dt><dd id="source">SQLite-v1 Snapshot</dd></div>
            <div><dt>revision</dt><dd id="revision">waiting…</dd></div>
            <div><dt>observed</dt><dd id="observed">waiting…</dd></div>
          </dl>
        </div>
      </div>
    </header>

    <section class="console-bar" aria-label="Dashboard-Steuerung">
      <label class="search-field"><span>find_</span><input id="run-search" type="search" placeholder="run, mode, status…" autocomplete="off"></label>
      <label><span>status</span><select id="status-filter"><option value="all">alle</option></select></label>
      <label><span>mode</span><select id="mode-filter"><option value="all">alle</option></select></label>
      <label><span>window</span><select id="range-filter"><option value="10">10</option><option value="25" selected>25</option><option value="50">50</option><option value="all">alle</option></select></label>
      <button id="refresh" type="button"><span aria-hidden="true">↻</span> refresh</button>
      <button id="auto-refresh" class="active" type="button" aria-pressed="true"><span aria-hidden="true">●</span> live 5s</button>
    </section>

    <section id="empty" class="empty" hidden>
      <strong>Noch keine Messdaten</strong><span>Die Quelle ist verbunden, enthält aber noch keine Runs.</span>
    </section>
    <p id="error" class="error" role="alert" hidden></p>

    <section class="kpi-grid" id="kpis" aria-label="Zusammenfassung"></section>

    <section class="visual-grid" aria-label="Visualisierungen">
      <article class="panel panel-wide">
        <header class="panel-head"><div><p>01 / FLOW</p><h2>Status-Timeline</h2></div><span id="timeline-note">—</span></header>
        <svg id="timeline" viewBox="0 0 820 238" role="img" aria-label="Statusverlauf der gefilterten Runs"></svg>
      </article>
      <article class="panel">
        <header class="panel-head"><div><p>02 / OUTCOME</p><h2>Ergebnis-Mix</h2></div><span id="outcome-note">—</span></header>
        <svg id="outcomes" viewBox="0 0 420 238" role="img" aria-label="Verteilung der Run-Status"></svg>
      </article>
      <article class="panel">
        <header class="panel-head"><div><p>03 / MODES</p><h2>Workload-Modi</h2></div><span id="mode-note">—</span></header>
        <svg id="modes" viewBox="0 0 420 238" role="img" aria-label="Verteilung der Workload-Modi"></svg>
      </article>
      <article class="panel panel-wide">
        <header class="panel-head"><div><p>04 / EVIDENCE</p><h2>Gate-Matrix</h2></div><span>C M T X R S</span></header>
        <svg id="evidence" viewBox="0 0 820 238" role="img" aria-label="Guardrail- und Sample-Abdeckung"></svg>
        <p class="chart-key">C correctness · M memory · T timeout · X crash · R rollback · S raw samples</p>
      </article>
    </section>

    <section class="panel history-panel">
      <header class="panel-head"><div><p>05 / INDEX</p><h2>Messhistorie</h2></div><span id="freshness">—</span></header>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Run / timestamp</th><th>Mode</th><th>Status</th><th>Samples</th><th>Shape</th><th>Dtype</th><th><span class="sr-only">Aktion</span></th></tr></thead>
          <tbody id="runs"></tbody>
        </table>
      </div>
      <p id="filter-empty" class="filter-empty" hidden>Keine Runs entsprechen dem aktuellen Filter.</p>
    </section>

    <section class="panel detail-panel" id="detail" hidden aria-live="polite">
      <header class="detail-head">
        <div><p>06 / TRACE</p><h2 id="detail-title">Run-Detail</h2></div>
        <button type="button" id="detail-close" aria-label="Detailansicht schließen">× close</button>
      </header>
      <p id="detail-status" class="detail-status">loading evidence…</p>
      <div id="detail-meta" class="detail-meta"></div>
      <div class="sample-toolbar">
        <label><span>sample family</span><select id="sample-kind"><option value="all">alle</option></select></label>
        <div id="sample-stats" class="sample-stats"></div>
      </div>
      <svg id="samples" viewBox="0 0 820 260" role="img" aria-label="Rohsamples des ausgewählten Runs"></svg>
      <p id="sample-note" class="chart-key">Keine Rohsamples ausgewählt.</p>
    </section>

    <footer>
      <p><span class="prompt">friday@local:~$</span> monitor --read-only --loopback</p>
      <p id="snapshot-id">snapshot pending</p>
    </footer>
  </main>
  <script src="/assets/app.js" defer></script>
</body>
</html>
"""

CSS = r""":root {
  color-scheme: dark;
  --bg: #050806;
  --surface: #09100b;
  --surface-2: #0c160f;
  --line: #173a22;
  --line-hot: #2c6b3d;
  --green: #6dff91;
  --green-soft: #2ed45b;
  --green-dim: #6f9b78;
  --ink: #d8ffe1;
  --muted: #76917c;
  --amber: #f1cf6a;
  --red: #ff7188;
  --blue: #72d7ff;
  --mono: "SFMono-Regular", "Cascadia Code", "Liberation Mono", Menlo, monospace;
  font-family: var(--mono);
  background: var(--bg);
  color: var(--ink);
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { min-width: 320px; background: var(--bg); }
body {
  margin: 0;
  min-height: 100vh;
  background:
    linear-gradient(rgba(7, 14, 9, .82), rgba(5, 8, 6, .96)),
    repeating-linear-gradient(0deg, transparent 0 27px, rgba(75, 255, 112, .035) 27px 28px),
    repeating-linear-gradient(90deg, transparent 0 27px, rgba(75, 255, 112, .025) 27px 28px);
}
body::after {
  position: fixed;
  inset: 0;
  z-index: 30;
  pointer-events: none;
  content: "";
  background: repeating-linear-gradient(0deg, transparent 0 3px, rgba(0, 0, 0, .12) 3px 4px);
  mix-blend-mode: multiply;
}
button, input, select { font: inherit; }
button, select, input {
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 3px;
  background: #071009;
}
button { cursor: pointer; }
button:hover, button:focus-visible, select:focus-visible, input:focus-visible {
  outline: none;
  border-color: var(--green);
  box-shadow: 0 0 0 2px rgba(109, 255, 145, .12);
}
.matrix-layer { position: fixed; inset: 0; overflow: hidden; pointer-events: none; opacity: .09; }
.matrix-layer span {
  position: absolute;
  top: -11rem;
  color: var(--green);
  font-size: .65rem;
  line-height: 1.75;
  letter-spacing: .35em;
  writing-mode: vertical-rl;
  animation: matrix-drift 17s linear infinite;
}
.matrix-layer span:nth-child(1) { left: 5%; }
.matrix-layer span:nth-child(2) { left: 63%; animation-duration: 23s; animation-delay: -7s; }
.matrix-layer span:nth-child(3) { left: 91%; animation-duration: 19s; animation-delay: -12s; }
@keyframes matrix-drift { to { transform: translateY(calc(100vh + 24rem)); } }
main { position: relative; z-index: 1; width: min(1440px, calc(100% - 40px)); margin: 0 auto; padding: 26px 0 64px; }
.masthead { border-top: 1px solid var(--green-soft); border-bottom: 1px solid var(--line); padding: 14px 0 28px; }
.brand-row { display: flex; align-items: center; justify-content: space-between; gap: 20px; }
.brand, .kicker, .panel-head p, .detail-head p { margin: 0; color: var(--green); font-size: .72rem; letter-spacing: .13em; text-transform: uppercase; }
.brand span { color: var(--muted); }
.live-state { display: flex; align-items: center; gap: 8px; color: var(--green); font-size: .68rem; letter-spacing: .12em; }
.live-state i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; box-shadow: 0 0 12px currentColor; }
.live-state.paused { color: var(--amber); }
.live-state.error-state { color: var(--red); }
.hero-grid { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(320px, .7fr); gap: 8vw; align-items: end; margin-top: clamp(34px, 7vw, 92px); }
h1 { margin: 10px 0 0; color: var(--ink); font-size: clamp(3.8rem, 10vw, 9rem); font-weight: 500; line-height: .76; letter-spacing: -.08em; text-transform: uppercase; }
h1 span { color: transparent; -webkit-text-stroke: 1px var(--green-soft); text-shadow: 0 0 35px rgba(54, 255, 95, .08); }
.hero-copy > p { max-width: 56ch; margin: 0 0 28px; color: var(--green-dim); font-family: system-ui, sans-serif; line-height: 1.6; }
.source-meta { margin: 0; border-top: 1px solid var(--line); }
.source-meta div { display: grid; grid-template-columns: 90px 1fr; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--line); }
.source-meta dt { color: var(--muted); font-size: .68rem; text-transform: uppercase; }
.source-meta dd { margin: 0; overflow: hidden; color: var(--green-soft); font-size: .72rem; text-overflow: ellipsis; white-space: nowrap; }
.console-bar { position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; align-items: end; gap: 8px; margin: 0 0 18px; padding: 12px 0; border-bottom: 1px solid var(--line); background: rgba(5, 8, 6, .94); backdrop-filter: blur(14px); }
.console-bar label, .sample-toolbar label { display: grid; gap: 5px; color: var(--muted); font-size: .63rem; letter-spacing: .08em; text-transform: uppercase; }
.console-bar input, .console-bar select, .console-bar button, .sample-toolbar select { min-height: 36px; padding: 7px 10px; }
.search-field { flex: 1 1 280px; position: relative; }
.search-field > span { position: absolute; left: 10px; bottom: 11px; color: var(--green); }
.search-field input { width: 100%; padding-left: 52px; }
.console-bar button { color: var(--green-dim); }
.console-bar button.active { color: var(--green); border-color: var(--line-hot); background: rgba(46, 212, 91, .08); }
.empty, .error { margin: 18px 0; padding: 18px; border: 1px dashed var(--line-hot); background: rgba(9, 16, 11, .82); }
.empty strong, .empty span { display: block; }
.empty strong { color: var(--green); }
.empty span { margin-top: 6px; color: var(--muted); font-size: .78rem; }
.error { color: var(--red); }
.kpi-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1px; margin: 18px 0; background: var(--line); border: 1px solid var(--line); }
.kpi { min-height: 132px; padding: 16px; background: rgba(9, 16, 11, .96); }
.kpi-top { display: flex; justify-content: space-between; color: var(--muted); font-size: .64rem; letter-spacing: .08em; text-transform: uppercase; }
.kpi-glyph { color: var(--green-soft); }
.kpi strong { display: block; margin: 18px 0 9px; color: var(--ink); font-size: clamp(1.45rem, 3vw, 2.35rem); font-weight: 500; letter-spacing: -.05em; }
.kpi small { color: var(--green-dim); font-family: system-ui, sans-serif; font-size: .72rem; }
.visual-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.panel { position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 4px; background: rgba(7, 13, 9, .9); box-shadow: 0 20px 60px rgba(0, 0, 0, .16); }
.panel::before { position: absolute; top: -1px; left: 18px; width: 44px; border-top: 2px solid var(--green); content: ""; }
.panel-wide { grid-column: 1 / -1; }
.panel-head { display: flex; align-items: start; justify-content: space-between; gap: 20px; padding: 16px 18px 0; }
.panel-head h2, .detail-head h2 { margin: 5px 0 0; color: var(--ink); font-size: 1rem; font-weight: 500; text-transform: uppercase; }
.panel-head > span { color: var(--muted); font-size: .65rem; text-align: right; }
svg { display: block; width: 100%; height: 238px; overflow: visible; }
svg text { fill: var(--muted); font: 10px var(--mono); }
svg .grid { stroke: var(--line); stroke-width: 1; }
svg .axis-hot { stroke: var(--line-hot); stroke-width: 1; }
svg .timeline-line { fill: none; stroke: var(--green-dim); stroke-width: 1.5; opacity: .75; }
svg .point { cursor: pointer; stroke: var(--bg); stroke-width: 2; transition: r .15s ease; }
svg .point:hover, svg .point:focus { r: 7px; outline: none; stroke: var(--ink); }
svg .bar-bg { fill: #0d1b11; }
svg .bar { fill: var(--green-soft); }
svg .bar-secondary { fill: var(--green-dim); }
svg .tone-ok { fill: var(--green); }
svg .tone-fail { fill: var(--red); }
svg .tone-warn { fill: var(--amber); }
svg .tone-info { fill: var(--blue); }
svg .tone-unknown { fill: #203528; }
svg .sample-cell { stroke: rgba(5, 8, 6, .7); stroke-width: 1; }
svg .series-baseline { fill: none; stroke: var(--blue); stroke-width: 2; }
svg .series-candidate { fill: none; stroke: var(--green); stroke-width: 2; }
svg .series-other { fill: none; stroke: var(--amber); stroke-width: 2; }
.chart-key { margin: -8px 18px 15px; color: var(--muted); font-size: .65rem; }
.history-panel { margin-top: 12px; }
.table-wrap { overflow: auto; max-height: 620px; margin-top: 14px; border-top: 1px solid var(--line); }
table { width: 100%; border-collapse: collapse; font-size: .74rem; }
th, td { padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
th { position: sticky; top: 0; z-index: 2; color: var(--muted); background: #071009; font-size: .62rem; font-weight: 400; letter-spacing: .08em; text-transform: uppercase; }
tbody tr { transition: background .15s ease; }
tbody tr:hover { background: rgba(46, 212, 91, .05); }
.run-cell { max-width: 390px; }
.run-cell strong, .run-cell small { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.run-cell strong { color: var(--ink); font-weight: 500; }
.run-cell small { margin-top: 5px; color: var(--muted); }
.status-chip { display: inline-flex; align-items: center; gap: 7px; }
.status-chip::before { width: 6px; height: 6px; border-radius: 50%; background: currentColor; content: ""; }
.status-ok { color: var(--green); }
.status-fail { color: var(--red); }
.status-warn { color: var(--amber); }
.status-info { color: var(--blue); }
.status-unknown { color: var(--muted); }
.inspect { padding: 6px 9px; color: var(--green); border-color: var(--line-hot); background: transparent; font-size: .66rem; }
.filter-empty { padding: 28px; color: var(--muted); text-align: center; }
.detail-panel { margin-top: 12px; padding-bottom: 12px; }
.detail-head { display: flex; align-items: start; justify-content: space-between; padding: 18px; border-bottom: 1px solid var(--line); }
.detail-head button { padding: 7px 10px; color: var(--muted); }
.detail-status { margin: 0; padding: 12px 18px; color: var(--green-dim); border-bottom: 1px solid var(--line); font-size: .7rem; }
.detail-meta { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: var(--line); }
.meta-item { min-height: 78px; padding: 13px 18px; background: var(--surface); }
.meta-item span, .meta-item strong { display: block; }
.meta-item span { color: var(--muted); font-size: .6rem; letter-spacing: .08em; text-transform: uppercase; }
.meta-item strong { margin-top: 8px; overflow: hidden; color: var(--ink); font-size: .74rem; font-weight: 400; text-overflow: ellipsis; white-space: nowrap; }
.sample-toolbar { display: flex; align-items: end; justify-content: space-between; gap: 16px; padding: 16px 18px 0; }
.sample-stats { display: flex; flex-wrap: wrap; justify-content: end; gap: 8px 18px; color: var(--green-dim); font-size: .65rem; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
footer { display: flex; justify-content: space-between; gap: 20px; margin-top: 18px; padding-top: 13px; color: var(--muted); border-top: 1px solid var(--line); font-size: .64rem; }
footer p { margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.prompt { color: var(--green); }
@media (max-width: 900px) {
  main { width: min(100% - 24px, 1440px); }
  .hero-grid { grid-template-columns: 1fr; }
  .hero-copy { margin-top: 22px; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .kpi:last-child { grid-column: 1 / -1; }
  .detail-meta { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 680px) {
  .visual-grid { grid-template-columns: 1fr; }
  .panel-wide { grid-column: auto; }
  .console-bar { position: static; }
  .console-bar label:not(.search-field), .console-bar button { flex: 1 1 120px; }
  .console-bar select, .console-bar button { width: 100%; }
  .kpi-grid { grid-template-columns: 1fr 1fr; }
  .kpi { min-height: 112px; }
  .detail-meta { grid-template-columns: 1fr; }
  .sample-toolbar, footer { align-items: stretch; flex-direction: column; }
  .sample-stats { justify-content: start; }
  h1 { font-size: clamp(3.5rem, 21vw, 6rem); }
}
@media (prefers-reduced-motion: reduce) {
  .matrix-layer span { animation: none; }
  *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
}
"""

JS = r"""(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const SVG_NS = ['h','t','t','p',':','/','/','w','w','w','.','w','3','.','o','r','g','/','2','0','0','0','/','s','v','g'].join('');
  const STATUS_ORDER = ['completed', 'promoted', 'invalid', 'worker_exit', 'failed', 'timeout'];
  const state = { snapshot: null, detail: null, auto: true, inFlight: false, lastRevision: null };

  const fieldValue = (field) => field && typeof field === 'object' && Object.prototype.hasOwnProperty.call(field, 'value') ? field.value : field;
  const display = (field, fallback = '—') => {
    const value = fieldValue(field);
    return value === null || value === undefined || value === '' ? fallback : String(value);
  };
  const setText = (node, value) => { node.textContent = value === null || value === undefined ? '—' : String(value); };
  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };
  const el = (tag, className, textValue) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (textValue !== undefined) setText(node, textValue);
    return node;
  };
  const svgEl = (tag, attrs = {}, textValue) => {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, String(value)));
    if (textValue !== undefined) setText(node, textValue);
    return node;
  };
  const short = (value, size = 18) => {
    const textValue = display(value);
    return textValue.length > size ? textValue.slice(0, size - 1) + '…' : textValue;
  };
  const finite = (value) => {
    const raw = fieldValue(value);
    if (raw === null || raw === undefined || raw === '') return null;
    const number = Number(raw);
    return Number.isFinite(number) ? number : null;
  };
  const percent = (part, total) => total ? Math.round(part * 100 / total) + '%' : '—';
  const formatNumber = (value) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return '—';
    if (Math.abs(number) >= 1e9) return (number / 1e9).toFixed(2) + 'G';
    if (Math.abs(number) >= 1e6) return (number / 1e6).toFixed(2) + 'M';
    if (Math.abs(number) >= 1e3) return (number / 1e3).toFixed(2) + 'k';
    return Number.isInteger(number) ? String(number) : number.toFixed(3);
  };
  const tone = (status) => {
    const value = String(status || '').toLowerCase();
    if (['completed', 'promoted', 'pass', 'ok'].includes(value)) return 'ok';
    if (['invalid', 'failed', 'fail', 'error'].includes(value)) return 'fail';
    if (['worker_exit', 'timeout', 'unknown'].includes(value)) return 'warn';
    return 'info';
  };
  const timestamp = (value) => {
    const date = value ? new Date(value) : null;
    return date && !Number.isNaN(date.getTime()) ? date.toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'medium' }) : '—';
  };
  const shapeText = (shape) => Array.isArray(shape) && shape.length ? shape.join(' × ') : '—';

  function option(value, label) {
    const node = document.createElement('option');
    node.value = value;
    setText(node, label);
    return node;
  }

  function syncOptions(select, values, allLabel) {
    const selected = select.value || 'all';
    const nodes = [option('all', allLabel), ...values.map((value) => option(value, value))];
    select.replaceChildren(...nodes);
    select.value = values.includes(selected) || selected === 'all' ? selected : 'all';
  }

  function filteredRuns() {
    if (!state.snapshot) return [];
    const status = $('status-filter').value;
    const mode = $('mode-filter').value;
    const query = $('run-search').value.trim().toLowerCase();
    let runs = (state.snapshot.runs || []).filter((run) => {
      const id = display(run.run_id, '').toLowerCase();
      const runMode = display(run.mode, '').toLowerCase();
      const runStatus = display(run.status, '').toLowerCase();
      const haystack = [id, runMode, runStatus, shapeText(run.shape), display(run.dtype, '')].join(' ').toLowerCase();
      return (status === 'all' || runStatus === status) && (mode === 'all' || runMode === mode) && (!query || haystack.includes(query));
    });
    const range = $('range-filter').value;
    if (range !== 'all') runs = runs.slice(0, Number(range));
    return runs;
  }

  function kpi(label, value, note, glyph) {
    const node = el('article', 'kpi');
    const top = el('div', 'kpi-top');
    top.append(el('span', '', label), el('span', 'kpi-glyph', glyph));
    node.append(top, el('strong', '', value), el('small', '', note));
    return node;
  }

  function renderKpis(runs) {
    const target = $('kpis');
    clear(target);
    const completed = runs.filter((run) => tone(run.status) === 'ok').length;
    const sampled = runs.filter((run) => Number(run.raw_sample_count) > 0).length;
    const modes = new Set(runs.map((run) => display(run.mode))).size;
    const dates = runs.map((run) => new Date(run.created_at).getTime()).filter(Number.isFinite);
    const spanHours = dates.length > 1 ? (Math.max(...dates) - Math.min(...dates)) / 3600000 : 0;
    const total = state.snapshot ? state.snapshot.run_count : 0;
    target.append(
      kpi('visible runs', String(runs.length).padStart(2, '0'), total + ' total in snapshot source', '[#]'),
      kpi('completed', percent(completed, runs.length), completed + ' / ' + runs.length + ' selected', '[✓]'),
      kpi('raw evidence', percent(sampled, runs.length), sampled + ' runs carry samples', '[∿]'),
      kpi('active modes', String(modes).padStart(2, '0'), 'within current filter', '[≋]'),
      kpi('time span', spanHours < 1 ? Math.round(spanHours * 60) + 'm' : spanHours.toFixed(1) + 'h', dates.length > 1 ? 'oldest to newest run' : 'single timestamp', '[↔]')
    );
  }

  function orderedStatuses(runs) {
    const values = [...new Set(runs.map((run) => display(run.status, 'unknown').toLowerCase()))];
    return values.sort((a, b) => {
      const ai = STATUS_ORDER.indexOf(a), bi = STATUS_ORDER.indexOf(b);
      return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi) || a.localeCompare(b);
    });
  }

  function openPoint(run, node) {
    node.setAttribute('tabindex', '0');
    node.setAttribute('role', 'button');
    node.addEventListener('click', () => loadDetail(display(run.run_id, '')));
    node.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); loadDetail(display(run.run_id, '')); }
    });
  }

  function drawTimeline(runs) {
    const svg = $('timeline');
    clear(svg);
    const ordered = [...runs].reverse();
    const statuses = orderedStatuses(ordered);
    setText($('timeline-note'), runs.length ? runs.length + ' runs · click a node' : 'no signal');
    if (!ordered.length) return;
    const left = 116, right = 790, top = 34, bottom = 194;
    const stepY = statuses.length > 1 ? (bottom - top) / (statuses.length - 1) : 0;
    statuses.forEach((status, index) => {
      const y = statuses.length > 1 ? top + index * stepY : (top + bottom) / 2;
      svg.append(svgEl('line', { x1: left, y1: y, x2: right, y2: y, class: 'grid' }));
      svg.append(svgEl('text', { x: 10, y: y + 4 }, short(status, 15)));
    });
    const points = ordered.map((run, index) => {
      const status = display(run.status, 'unknown').toLowerCase();
      return {
        run,
        x: ordered.length > 1 ? left + index * (right - left) / (ordered.length - 1) : (left + right) / 2,
        y: statuses.length > 1 ? top + statuses.indexOf(status) * stepY : (top + bottom) / 2,
      };
    });
    if (points.length > 1) svg.append(svgEl('polyline', { class: 'timeline-line', points: points.map((point) => point.x + ',' + point.y).join(' ') }));
    points.forEach((point) => {
      const status = display(point.run.status, 'unknown');
      const circle = svgEl('circle', { class: 'point tone-' + tone(status), cx: point.x, cy: point.y, r: 5, 'aria-label': display(point.run.run_id) + ', ' + status });
      circle.append(svgEl('title', {}, display(point.run.run_id) + ' · ' + status + ' · ' + timestamp(point.run.created_at)));
      openPoint(point.run, circle);
      svg.append(circle);
    });
    svg.append(svgEl('text', { x: left, y: 224 }, timestamp(ordered[0].created_at)));
    const lastLabel = svgEl('text', { x: right, y: 224, 'text-anchor': 'end' }, timestamp(ordered[ordered.length - 1].created_at));
    svg.append(lastLabel);
  }

  function countBy(runs, getter) {
    const counts = new Map();
    runs.forEach((run) => {
      const key = getter(run);
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }

  function drawBars(svg, rows, total, statusColors) {
    clear(svg);
    if (!rows.length) return;
    const max = Math.max(...rows.map((row) => row[1]));
    const top = 24, rowHeight = Math.min(39, 190 / rows.length);
    rows.slice(0, 6).forEach(([label, count], index) => {
      const y = top + index * rowHeight;
      svg.append(svgEl('text', { x: 12, y: y + 11 }, short(label, 20)));
      svg.append(svgEl('rect', { x: 150, y, width: 218, height: 13, rx: 2, class: 'bar-bg' }));
      const className = statusColors ? 'tone-' + tone(label) : index ? 'bar-secondary' : 'bar';
      svg.append(svgEl('rect', { x: 150, y, width: Math.max(2, 218 * count / max), height: 13, rx: 2, class: className }));
      svg.append(svgEl('text', { x: 380, y: y + 11, 'text-anchor': 'end' }, count + ' · ' + percent(count, total)));
    });
  }

  function drawOutcomes(runs) {
    const rows = countBy(runs, (run) => display(run.status, 'unknown').toLowerCase());
    setText($('outcome-note'), rows.length + ' states');
    drawBars($('outcomes'), rows, runs.length, true);
  }

  function drawModes(runs) {
    const rows = countBy(runs, (run) => display(run.mode, 'unknown').toLowerCase());
    setText($('mode-note'), rows.length + ' modes');
    drawBars($('modes'), rows, runs.length, false);
  }

  function guardTone(value) {
    const normalized = String(value || 'UNKNOWN').toUpperCase();
    if (normalized === 'PASS') return 'ok';
    if (normalized === 'FAIL') return 'fail';
    return 'unknown';
  }

  function drawEvidence(runs) {
    const svg = $('evidence');
    clear(svg);
    const shown = runs.slice(0, 18);
    const keys = ['correctness', 'memory', 'timeout', 'crash', 'rollback', 'samples'];
    const labels = ['C', 'M', 'T', 'X', 'R', 'S'];
    const left = 250, cell = 28, top = 24, rowHeight = 11;
    labels.forEach((label, index) => svg.append(svgEl('text', { x: left + index * cell + 7, y: 13, 'text-anchor': 'middle' }, label)));
    shown.forEach((run, rowIndex) => {
      const y = top + rowIndex * rowHeight;
      svg.append(svgEl('text', { x: 10, y: y + 8 }, short(display(run.run_id), 35)));
      keys.forEach((key, colIndex) => {
        const value = key === 'samples' ? (Number(run.raw_sample_count) > 0 ? 'PASS' : 'UNKNOWN') : (run.guardrails || {})[key];
        const rect = svgEl('rect', { x: left + colIndex * cell, y, width: 20, height: 8, rx: 1, class: 'sample-cell tone-' + guardTone(value) });
        rect.append(svgEl('title', {}, key + ': ' + (value || 'UNKNOWN')));
        svg.append(rect);
      });
    });
    if (!shown.length) svg.append(svgEl('text', { x: 10, y: 40 }, 'no evidence selected'));
  }

  function statusChip(status) {
    return el('span', 'status-chip status-' + tone(status), status || 'unknown');
  }

  function renderTable(runs) {
    const body = $('runs');
    clear(body);
    $('filter-empty').hidden = runs.length > 0;
    runs.forEach((run) => {
      const tr = document.createElement('tr');
      const runCell = el('td', 'run-cell');
      const runId = display(run.run_id);
      const title = el('strong', '', runId);
      title.title = runId;
      runCell.append(title, el('small', '', timestamp(run.created_at)));
      const mode = el('td', '', display(run.mode));
      const status = document.createElement('td');
      status.append(statusChip(display(run.status, 'unknown')));
      const samples = el('td', '', String(Number(run.raw_sample_count) || 0).padStart(3, '0'));
      const shape = el('td', '', shapeText(run.shape));
      const dtype = el('td', '', display(run.dtype));
      const action = document.createElement('td');
      const button = el('button', 'inspect', 'inspect →');
      button.type = 'button';
      button.addEventListener('click', () => loadDetail(runId));
      action.append(button);
      tr.append(runCell, mode, status, samples, shape, dtype, action);
      body.append(tr);
    });
  }

  function render() {
    if (!state.snapshot) return;
    const runs = filteredRuns();
    renderKpis(runs);
    drawTimeline(runs);
    drawOutcomes(runs);
    drawModes(runs);
    drawEvidence(runs);
    renderTable(runs);
    setText($('freshness'), runs.length + ' shown · ' + (state.snapshot.truncated ? 'latest 100 source rows' : 'complete snapshot'));
  }

  function setLive(label, mode) {
    const node = $('live-state');
    node.className = 'live-state' + (mode ? ' ' + mode : '');
    setText(node.querySelector('span'), label);
  }

  async function loadSnapshot(manual = false) {
    if (state.inFlight) return;
    state.inFlight = true;
    if (manual) setLive('REFRESHING', '');
    try {
      const response = await fetch('/api/snapshot', { method: 'GET', headers: { Accept: 'application/json' }, cache: 'no-store' });
      if (!response.ok) throw new Error('snapshot unavailable');
      const data = await response.json();
      const changed = state.lastRevision !== data.source_revision;
      state.snapshot = data;
      state.lastRevision = data.source_revision;
      $('empty').hidden = data.data_state !== 'empty';
      $('error').hidden = true;
      setText($('source'), data.source || 'SQLite-v1, read-only Snapshot');
      setText($('revision'), short(data.source_revision, 28));
      $('revision').title = data.source_revision || '';
      setText($('observed'), timestamp(data.observed_at));
      setText($('snapshot-id'), 'snapshot ' + short(data.snapshot_id, 30) + ' · query_only=1');
      const runs = data.runs || [];
      syncOptions($('status-filter'), orderedStatuses(runs), 'alle');
      syncOptions($('mode-filter'), [...new Set(runs.map((run) => display(run.mode, 'unknown').toLowerCase()))].sort(), 'alle');
      render();
      setLive(state.auto ? (changed ? 'LIVE · UPDATED' : 'LIVE · 5S') : 'PAUSED', state.auto ? '' : 'paused');
    } catch (error) {
      setText($('error'), 'Snapshot konnte nicht geladen werden. Der letzte gültige Stand bleibt sichtbar.');
      $('error').hidden = false;
      setLive('SOURCE ERROR', 'error-state');
    } finally {
      state.inFlight = false;
    }
  }

  function sampleFamily(sample) {
    return display(sample.kind, 'unknown').replace(/_(baseline|candidate)$/i, '');
  }

  function metaItem(label, value) {
    const node = el('div', 'meta-item');
    const strong = el('strong', '', value);
    strong.title = String(value);
    node.append(el('span', '', label), strong);
    return node;
  }

  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function drawSamples() {
    const svg = $('samples');
    clear(svg);
    if (!state.detail) return;
    const selected = $('sample-kind').value;
    const samples = (state.detail.raw_samples || []).filter((sample) => selected === 'all' || sampleFamily(sample) === selected);
    const valid = samples.map((sample) => ({ sample, value: finite(sample.value), arm: display(sample.arm, 'other').toLowerCase() })).filter((item) => item.value !== null);
    if (!valid.length) {
      setText($('sample-note'), 'Für diese Auswahl sind keine numerischen Rohsamples vorhanden.');
      setText($('sample-stats'), '0 samples');
      return;
    }
    const values = valid.map((item) => item.value);
    let min = Math.min(...values), max = Math.max(...values);
    if (min === max) { min -= 1; max += 1; }
    const left = 64, right = 798, top = 26, bottom = 214;
    [0, .25, .5, .75, 1].forEach((ratio) => {
      const y = top + ratio * (bottom - top);
      svg.append(svgEl('line', { x1: left, y1: y, x2: right, y2: y, class: 'grid' }));
      svg.append(svgEl('text', { x: left - 8, y: y + 4, 'text-anchor': 'end' }, formatNumber(max - ratio * (max - min))));
    });
    const groups = new Map();
    valid.forEach((item) => {
      if (!groups.has(item.arm)) groups.set(item.arm, []);
      groups.get(item.arm).push(item);
    });
    groups.forEach((items, arm) => {
      const points = items.map((item, index) => {
        const x = items.length > 1 ? left + index * (right - left) / (items.length - 1) : (left + right) / 2;
        const y = bottom - (item.value - min) * (bottom - top) / (max - min);
        return x + ',' + y;
      });
      const className = arm === 'baseline' ? 'series-baseline' : arm === 'candidate' ? 'series-candidate' : 'series-other';
      svg.append(svgEl('polyline', { points: points.join(' '), class: className }));
    });
    const unit = display(valid[0].sample.unit, 'unit');
    const stats = [...groups.entries()].map(([arm, items]) => arm + ' n=' + items.length + ' med=' + formatNumber(median(items.map((item) => item.value))));
    $('sample-stats').replaceChildren(...stats.map((stat) => el('span', '', stat)));
    setText($('sample-note'), valid.length + ' samples · ' + unit + ' · y=[' + formatNumber(min) + ', ' + formatNumber(max) + ']');
  }

  function renderDetail(data) {
    state.detail = data;
    setText($('detail-title'), short(display(data.run_id), 64));
    setText($('detail-status'), display(data.status, 'unknown') + ' · ' + timestamp(data.created_at) + ' · ' + data.raw_sample_count + ' raw samples');
    const meta = $('detail-meta');
    meta.replaceChildren(
      metaItem('mode', display(data.mode)),
      metaItem('shape', shapeText(data.shape)),
      metaItem('dtype / layout', display(data.dtype) + ' / ' + display(data.layout)),
      metaItem('baseline → candidate', display(data.baseline) + ' → ' + display(data.candidate)),
      metaItem('manifest hash', display((data.hashes || {}).manifest)),
      metaItem('code hash', display((data.hashes || {}).code)),
      metaItem('spec hash', display((data.hashes || {}).spec)),
      metaItem('environment hash', display((data.hashes || {}).environment))
    );
    const families = [...new Set((data.raw_samples || []).map(sampleFamily))].sort();
    syncOptions($('sample-kind'), families, 'alle');
    const preferred = families.find((family) => family === 'pair_performance') || families[0] || 'all';
    $('sample-kind').value = preferred;
    drawSamples();
  }

  async function loadDetail(runId) {
    if (!runId) return;
    $('detail').hidden = false;
    setText($('detail-title'), short(runId, 64));
    setText($('detail-status'), 'loading evidence…');
    clear($('detail-meta'));
    clear($('samples'));
    $('detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
    try {
      const response = await fetch('/api/run?id=' + encodeURIComponent(runId), { method: 'GET', headers: { Accept: 'application/json' }, cache: 'no-store' });
      if (!response.ok) throw new Error('detail unavailable');
      renderDetail(await response.json());
    } catch (error) {
      setText($('detail-status'), 'Detail konnte nicht geladen werden.');
    }
  }

  ['status-filter', 'mode-filter', 'range-filter'].forEach((id) => $(id).addEventListener('change', render));
  $('run-search').addEventListener('input', render);
  $('sample-kind').addEventListener('change', drawSamples);
  $('refresh').addEventListener('click', () => loadSnapshot(true));
  $('auto-refresh').addEventListener('click', () => {
    state.auto = !state.auto;
    $('auto-refresh').classList.toggle('active', state.auto);
    $('auto-refresh').setAttribute('aria-pressed', String(state.auto));
    setText($('auto-refresh'), state.auto ? '● live 5s' : '○ paused');
    setLive(state.auto ? 'LIVE · 5S' : 'PAUSED', state.auto ? '' : 'paused');
    if (state.auto) loadSnapshot(true);
  });
  $('detail-close').addEventListener('click', () => { $('detail').hidden = true; state.detail = null; });

  window.setInterval(() => { if (state.auto) loadSnapshot(false); }, 5000);
  loadSnapshot(true);
})();
"""
