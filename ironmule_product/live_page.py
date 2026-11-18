"""Static, dependency-free UI for the read-only local monitor."""

LIVE_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>Live monitor · IronMule</title>
<style>
:root{--green:#08765a;--ink:#192f2a;--muted:#506960;--line:#cfdbd3;--paper:#f5f7f2;--mono:"IBM Plex Mono",ui-monospace,monospace;--sans:Inter,"Segoe UI",system-ui,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 var(--sans)}
button,input{font:inherit}
a{color:inherit}
button{cursor:pointer}
button:focus-visible,a:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #d48213;outline-offset:4px}
.shell{min-height:100vh;display:grid;grid-template-columns:210px minmax(0,1fr)}
aside{background:#102e27;color:#deeae3;display:flex;flex-direction:column;padding:30px 20px;gap:38px}
aside strong{font-size:23px;letter-spacing:-1.3px}
.brand{display:flex;align-items:center;gap:10px}
.mark{display:inline-block;width:24px;height:24px;background:linear-gradient(120deg,transparent 0 12%,#9cf1ae 13% 33%,transparent 34% 43%,#9cf1ae 44% 63%,transparent 64%);transform:skew(-6deg)}
nav{display:grid;gap:5px}
.sectionlabel{font:600 10px/1.4 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:#a8c6b8;margin:0 0 8px}
nav button,.chatlink{color:#cadfd2;background:transparent;border:0;text-align:left;text-decoration:none;border-radius:8px;padding:10px 12px;display:flex;gap:10px;align-items:center;font-size:14px;font-weight:600}
nav button[aria-current=page]{background:#244b3f;color:#effff3}
nav button:hover,.chatlink:hover{background:#204438}
.icon{font:16px var(--mono);width:16px}
.footer{margin-top:auto;border-top:1px solid #345448;padding-top:18px;color:#b7cbbc;font-size:12px}
.footer time{display:block;color:#eef5ea;font:12px var(--mono);margin-top:4px}
main{min-width:0;max-width:1450px;padding:26px clamp(20px,4vw,68px) 50px;width:100%;margin:auto}
.top{display:flex;align-items:center;justify-content:space-between;gap:14px;border-bottom:1px solid var(--line);padding-bottom:18px;color:var(--muted);font-size:13px}
.top b{color:var(--ink)}
.update{background:#e8eee6;border:1px solid #bbcbbd;border-radius:20px;padding:5px 12px;color:#254d3d;font-size:12px;font-weight:650}
h1{font-size:clamp(35px,4.7vw,68px);line-height:1.02;letter-spacing:-.065em;margin:0}
h2{font-size:22px;letter-spacing:-.045em;margin:0}
h3{font-size:15px;letter-spacing:-.025em;margin:0}
.title{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-top:34px}
.context{color:var(--muted);margin:10px 0 21px;font:13px/1.5 var(--mono);overflow-wrap:anywhere}
.state{white-space:nowrap;border:1px solid var(--line);background:white;padding:5px 10px;border-radius:18px;font:600 12px var(--mono);display:flex;align-items:center;gap:7px}
.dot{width:7px;height:7px;border-radius:50%;background:#8a9c91}
.state[data-tone=good] .dot{background:#18915b}
.state[data-tone=warn] .dot{background:#bd720d}
.state[data-tone=bad] .dot{background:#bc463c}
.hero{display:grid;grid-template-columns:minmax(0,1fr) minmax(165px,1fr);align-items:center;border-radius:15px;background:#183c32;color:#f5fcf5;padding:clamp(18px,2.6vw,38px);gap:28px}
.kicker{color:#bbdac7;font-size:13px;font-weight:650}
.big{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.big strong{font-size:clamp(48px,5.3vw,77px);line-height:1.18;letter-spacing:-.075em;font-weight:590;font-variant-numeric:tabular-nums}
.big span{color:#c2deca;font:14px var(--mono)}
.herotext{color:#c1d5c6;display:block;font-size:12px;margin-top:5px}
.chart{position:relative;height:105px}
canvas{width:100%;height:100%;display:block}
.chart small{position:absolute;bottom:0;right:1px;color:#c1d5c6;font:10px var(--mono)}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));margin:14px 0 26px;gap:12px}
.card,.panel{background:#fff;border:1px solid var(--line);border-radius:12px}
.card{padding:16px 19px}
.card span,.card small{display:block;font-size:12px;color:var(--muted)}
.card strong{display:block;font-size:clamp(23px,2.7vw,34px);letter-spacing:-.06em;font-weight:620;font-variant-numeric:tabular-nums}
.card small{font-size:11px}
.panel{padding:18px 21px;margin:12px 0}
.panelhead{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.caption{font:11px var(--mono);color:var(--muted)}
.table-shelf{overflow:auto}
.rows{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px;white-space:nowrap}
.rows :is(th,td){padding:9px 10px;border-top:1px solid #dce5dc;text-align:right;font-variant-numeric:tabular-nums}
.rows :is(th,td):first-child{text-align:left;padding-left:0}
.rows :is(th,td):last-child{padding-right:0}
.rows th{color:var(--muted);font-size:11px;font-weight:600}
.rows td{font-family:var(--mono);font-size:12px}
.rows td:first-child{font-weight:600}
.empty{color:var(--muted);margin:15px 0 4px;font-size:14px}
.details{margin-top:11px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.details dt{font-size:11px;color:var(--muted)}
.details dd{margin:2px 0;font:13px var(--mono);overflow-wrap:anywhere}
.notice{padding:10px 14px;margin:12px 0;border:1px solid #d2b277;color:#704f17;background:#fff8e9;border-radius:9px;font-size:13px}
.auth input{max-width:95%;padding:5px;border:1px solid #9daea0;border-radius:4px}
.auth button{border:1px solid #9daea0;border-radius:4px;padding:5px 9px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
[hidden]{display:none!important}
@media(max-width:800px){.shell{grid-template-columns:142px minmax(0,1fr)}
aside{padding:22px 10px}
.brand{gap:5px}
.brand strong{font-size:21px}
main{padding:20px 18px}
.hero{gap:12px}
.cards{gap:7px}
.card{padding:12px}
.panel{padding:15px}
}
@media(max-width:620px){.shell{display:block}
aside{display:block;padding:12px 14px}
.brand{float:left;margin-right:10px}
.brand strong{font-size:17px}
.mark{width:17px;height:17px}
nav{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:0}
nav button,.chatlink{font-size:12px;padding:4px 7px}
.icon,nav .sectionlabel,.footer{display:none}
main{padding:16px 13px 32px}
.title{margin-top:22px}
.hero{grid-template-columns:minmax(0,1fr) 35%;gap:6px}
.chart{height:74px}
.cards{margin-bottom:16px}
.card{padding:10px 8px}
.card span{min-height:30px}
.details{grid-template-columns:1fr 1fr}
.top{padding-bottom:12px}
}
</style></head><body><div class="shell">
<aside><div class="brand"><i class="mark" aria-hidden="true"></i><strong>IronMule</strong></div>
<nav aria-label="Workspace"><div class="sectionlabel">Workspace</div><button id="runtimeTab"><span class="icon" aria-hidden="true">⌁</span>Inference</button><button id="benchTab"><span class="icon" aria-hidden="true">▥</span>Benchmarks</button><a class="chatlink" id="chat" href="/" hidden><span class="icon" aria-hidden="true">↗</span>Chat</a></nav>
<div class="footer">Last synchronized<time id="clock">Waiting for data</time></div></aside>
<main><div class="top"><span>Workspace &nbsp; / &nbsp; <b>Live monitor</b></span><button id="refresh" class="update" aria-pressed="true">Pause updates</button></div>
<div id="message" class="notice" role="status" hidden></div><form id="auth" class="notice auth" hidden><label>Server API key <input id="key" type="password" autocomplete="off"></label> <button>Connect</button></form>
<div class="title"><h1 id="title">Inference</h1><span class="state" id="state"><i class="dot"></i><span id="stateText">Connecting</span></span></div>
<p class="context" id="context">Waiting for the first update</p>
<section class="hero" aria-labelledby="rateLabel"><div><div id="rateLabel" class="kicker">Last completion throughput</div><div class="big"><strong id="rate">—</strong><span>tok/s</span></div><small class="herotext" id="scope"></small></div><div class="chart"><canvas id="plot" aria-hidden="true"></canvas><small id="plotLabel"></small></div></section>
<div class="cards" id="cards"></div>
<section class="panel"><div class="panelhead"><h2 id="tableTitle">Request history</h2><span id="rowCount" class="caption"></span></div><div class="table-shelf"><table class="rows" id="table"></table></div><p id="empty" class="empty"></p></section>
<section class="panel"><div class="panelhead"><h2 id="detailTitle">Runtime</h2><span id="detailBadge" class="caption"></span></div><dl class="details" id="details"></dl></section>
<p class="sr" role="status" id="announce"></p></main></div>
<script>
"use strict";
const $ = (id) => document.getElementById(id),
  present = (n) => typeof n === "number" && Number.isFinite(n);
const fmt = (n, d = 1) => (present(n) ? n.toLocaleString(undefined, { maximumFractionDigits: d }) : "—");
const engineName = (e) => ({ ironmule: "IronMule", llama_cpp: "llama.cpp", vllm: "vLLM" })[e] || "—";
const at = (t) => (present(t) ? new Date(t * 1000).toLocaleTimeString() : "—");
const txt = (id, v) => {
  $(id).textContent = v;
};
function el(tag, value, cls) {
  const n = document.createElement(tag);
  if (value != null) n.textContent = value;
  if (cls) n.className = cls;
  return n;
}
function requestedView() {
  return { "#benchmarks": "benchmark", "#inference": "runtime" }[location.hash] || null;
}
let sawBenchmark = false,
  connected = false;
let data = null,
  chosen = requestedView(),
  timer = null,
  paused = false,
  busy = false,
  locked = false,
  key = "",
  lastState = "",
  points = [];
try {
  key = sessionStorage.getItem("ironmule-key") || "";
} catch {}
function notice(value) {
  txt("message", value);
  $("message").hidden = !value;
}
function table(head, rows, empty) {
  const t = $("table");
  t.replaceChildren();
  t.hidden = !rows.length;
  $("empty").hidden = !!rows.length;
  txt("empty", empty);
  if (!rows.length) return;
  const h = t.createTHead().insertRow();
  head.forEach((v) => {
    const c = el("th", v);
    c.scope = "col";
    h.append(c);
  });
  const b = t.createTBody();
  rows.forEach((vals) => {
    const r = b.insertRow();
    vals.forEach((v) => r.append(el("td", v)));
  });
}
function cards(rows) {
  const box = $("cards");
  box.replaceChildren();
  for (const [label, v, info] of rows) {
    const c = el("div", null, "card");
    c.append(el("span", label), el("strong", v), el("small", info));
    box.append(c);
  }
}
function details(rows) {
  const box = $("details");
  box.replaceChildren();
  rows.forEach(([k, v]) => {
    const d = el("div");
    d.append(el("dt", k), el("dd", v ?? "—"));
    box.append(d);
  });
}
function chart(values) {
  points = values.filter(present);
  const c = $("plot"),
    w = c.clientWidth,
    h = c.clientHeight,
    d = window.devicePixelRatio || 1;
  if (w < 1 || h < 1) return;
  c.width = w * d;
  c.height = h * d;
  const g = c.getContext("2d");
  g.scale(d, d);
  const base = h - 19;
  g.strokeStyle = "#416357";
  g.lineWidth = 1;
  [0.12, 0.45, 0.78].forEach((k) => {
    g.beginPath();
    g.moveTo(0, base * k);
    g.lineTo(w, base * k);
    g.stroke();
  });
  txt(
    "plotLabel",
    points.length > 1
      ? points.length + " observed samples"
      : points.length
        ? "1 observed sample"
        : "Awaiting samples"
  );
  if (!points.length) return;
  const max = Math.max(...points, 1),
    plot = points.map((n, i) => [
      points.length === 1 ? w / 2 : 3 + (i * (w - 6)) / (points.length - 1),
      base - ((base - 5) * n) / max
    ]);
  g.beginPath();
  plot.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.lineWidth = 2.3;
  g.strokeStyle = "#9ff2ad";
  g.stroke();
  const [x, y] = plot.at(-1);
  g.beginPath();
  g.arc(x, y, 3.5, 0, 2 * Math.PI);
  g.fillStyle = "#c6ffd2";
  g.fill();
}
const stateNames = {
  preparing: "Preparing",
  idle: "Idle",
  running: "Running",
  detached: "No server",
  not_ready: "Not ready",
  complete: "Finished",
  error: "Error",
  timeout: "Timed out",
  unavailable: "Unavailable"
};
function status(state, stale = false) {
  const name = !connected
    ? "Disconnected"
    : paused
      ? "Paused"
      : stale
        ? "Awaiting update"
        : stateNames[state] || "Unknown";
  txt("stateText", name);
  $("state").dataset.tone =
    !connected || stale
      ? "warn"
      : {
          running: "good",
          complete: "good",
          error: "bad",
          timeout: "bad",
          unavailable: "warn",
          not_ready: "warn"
        }[state] || "";
  if (name !== lastState) {
    txt("announce", name);
    lastState = name;
  }
}
function draw() {
  if (!data) return;
  const r = data.runtime,
    b = data.benchmark;
  if (b.status === "running" || b.status === "preparing") sawBenchmark = true;
  const isBench =
    (chosen || (sawBenchmark || r.status === "detached" ? "benchmark" : "runtime")) === "benchmark";
  $("runtimeTab").setAttribute("aria-current", isBench ? "false" : "page");
  $("benchTab").setAttribute("aria-current", isBench ? "page" : "false");
  $("chat").hidden = r.status === "detached";
  if (!isBench) {
    status(r.status);
    txt("title", "Inference");
    txt(
      "context",
      [r.model, r.engine, r.device].filter(Boolean).join("  /  ") ||
        (r.status === "detached" ? "No inference server is attached" : "No loaded model reported")
    );
    const rows = r.samples || [],
      last = rows.at(-1) || {};
    txt("rateLabel", "Last completion throughput");
    txt("rate", fmt(last.tokens_per_second));
    txt("scope", "Completion tokens per second, including queue and prompt time.");
    chart(rows.map((v) => v.tokens_per_second));
    cards([
      ["Time to first token", fmt(last.ttft_ms), "ms · last request"],
      ["Completed", fmt(r.completed, 0), "requests this session"],
      ["In flight", fmt(r.active, 0), fmt(r.queued, 0) + " queued"]
    ]);
    txt("tableTitle", "Request history");
    txt("rowCount", rows.length + " recent");
    table(
      ["Completed at", "Output tokens", "Throughput", "First token"],
      rows
        .slice()
        .reverse()
        .map((s) => [
          at(s.at),
          fmt(s.completion_tokens, 0),
          fmt(s.tokens_per_second) + " tok/s",
          fmt(s.ttft_ms) + " ms"
        ]),
      r.status === "detached"
        ? "Open the inference server’s /live page to watch serving requests."
        : "Completed requests will appear here with their measured timings."
    );
    txt("detailTitle", r.workers > 1 ? "Controller · first worker" : "Controller");
    txt(
      "detailBadge",
      r.gate === true ? "Quality gate passed" : r.gate === false ? "Quality gate failed" : ""
    );
    const decision = (r.decisions || []).at(-1);
    details([
      ["Active configuration", r.config],
      ["Latest width decision", decision ? [decision.action, decision.to].filter(Boolean).join(" → ") : null],
      ["Decision time", decision ? at(decision.at) : null],
      ["Engine", r.backend],
      ["Planner", r.planner],
      ["Failed requests", fmt(r.failed, 0)],
      ["Queue order", r.schedule_order ? r.schedule_order.toUpperCase() : null],
      ["Queue width limit", fmt(r.schedule_width, 0)],
      ["Observed workload windows", fmt(r.live_windows, 0)],
      ...(present(r.decision_us) ? [["Mean scheduler decision", fmt(r.decision_us) + " µs"]] : [])
    ]);
  } else {
    const rows = b.runs || [],
      last = rows.at(-1) || {},
      stamp = Date.parse(b.updated_at || ""),
      stale =
        ["running", "preparing"].includes(b.status) &&
        (!Number.isFinite(stamp) || Math.abs(data.observed_at * 1000 - stamp) > 120000);
    status(b.status, stale);
    txt("title", "Engine comparison");
    txt("context", [b.model, b.device].filter(Boolean).join("  /  ") || "No benchmark model reported");
    txt("rateLabel", "Latest measured throughput");
    txt("rate", fmt(last.tokens_per_second));
    txt(
      "scope",
      last.engine
        ? [engineName(last.engine), present(last.concurrency) ? last.concurrency + " concurrent" : null]
            .filter(Boolean)
            .join(" · ")
        : "Waiting for a completed workload."
    );
    chart(rows.map((v) => v.tokens_per_second));
    cards([
      ["Median first token", fmt(last.ttft_median_ms), "ms · latest workload"],
      ["P95 first token", fmt(last.ttft_p95_ms), "ms · latest workload"],
      [
        "Holdout quality",
        last.quality ? last.quality.passed + " / " + last.quality.total : "—",
        "passed / assessed · latest workload"
      ]
    ]);
    txt("tableTitle", "Measured workloads");
    txt("rowCount", fmt(b.run_count, 0) + " completed");
    table(
      ["Engine", "Repeat", "Concurrency", "tok/s", "TTFT p50", "Holdout"],
      rows
        .slice()
        .reverse()
        .map((v) => [
          engineName(v.engine),
          present(v.repeat) ? fmt(v.repeat + 1, 0) : "—",
          fmt(v.concurrency, 0),
          fmt(v.tokens_per_second),
          fmt(v.ttft_median_ms) + " ms",
          v.quality ? v.quality.passed + " / " + v.quality.total : "—"
        ]),
      b.status === "idle"
        ? "No benchmark has reported progress in this state directory."
        : b.status === "unavailable"
          ? "The benchmark progress report could not be read."
          : "Workloads appear as each engine finishes measuring them."
    );
    txt("detailTitle", "Run status");
    txt(
      "detailBadge",
      b.updated_at && Number.isFinite(stamp) ? "Updated " + new Date(stamp).toLocaleString() : ""
    );
    details([
      ["Stage", b.stage],
      ["Current engine", engineName(b.engine)],
      ["Concurrency", fmt(b.concurrency, 0)],
      ["Verdict", (b.verdict || "").replaceAll("_", " ") || null],
      ["Error", (b.error_code || "").replaceAll("_", " ") || null],
      ["Repeat", present(b.repeat) ? fmt(b.repeat + 1, 0) : "—"],
      ["Verdict details", (b.reasons || []).map((reason) => reason.replaceAll("_", " ")).join(" · ") || null]
    ]);
  }
}
async function poll() {
  clearTimeout(timer);
  if (paused || busy || locked) return;
  busy = true;
  const abort = new AbortController(),
    watch = setTimeout(() => abort.abort(), 5000);
  try {
    const response = await fetch("/v1/ironmule/live", {
      cache: "no-store",
      signal: abort.signal,
      headers: key ? { Authorization: "Bearer " + key } : {}
    });
    if (response.status === 401) {
      locked = true;
      connected = false;
      status("unavailable");
      $("auth").hidden = false;
      notice("Sign in to view this server’s live measurements.");
      return;
    }
    if (!response.ok) throw Error();
    const d = await response.json();
    if (d.schema !== "ironmule.live.v1") throw Error();
    data = d;
    connected = true;
    notice("");
    $("auth").hidden = true;
    txt("clock", at(d.observed_at));
    draw();
  } catch {
    connected = false;
    notice(
      data
        ? "Connection interrupted. Showing the last received measurements."
        : "Unable to reach the monitor. Reconnecting…"
    );
    txt("stateText", "Disconnected");
    $("state").dataset.tone = "warn";
  } finally {
    clearTimeout(watch);
    busy = false;
    if (!paused && !locked) timer = setTimeout(poll, 2000);
  }
}
$("runtimeTab").onclick = () => {
  location.hash = "inference";
};
$("benchTab").onclick = () => {
  location.hash = "benchmarks";
};
window.addEventListener("hashchange", () => {
  chosen = requestedView();
  draw();
});
$("refresh").onclick = () => {
  paused = !paused;
  $("refresh").setAttribute("aria-pressed", String(!paused));
  txt("refresh", paused ? "Resume updates" : "Pause updates");
  draw();
  if (paused) clearTimeout(timer);
  else poll();
};
$("auth").onsubmit = (e) => {
  e.preventDefault();
  key = $("key").value;
  $("key").value = "";
  locked = false;
  poll();
};
window.addEventListener("resize", () => chart(points));
poll();
</script></body></html>"""
