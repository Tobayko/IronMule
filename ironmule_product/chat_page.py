"""The chat page `ironmule serve` shows at `/`: one static file, no external resources.

It talks to the same server's OpenAI-compatible API, so it needs nothing the API does not
already offer. Model output is only ever set as text, never parsed as HTML.
"""

CHAT_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>IronMule</title>
<style>
:root { --bg: #07090d; --panel: #0d1118; --fg: #e6edf3; --muted: #7d8896; --line: #1c2430;
        --accent: #3ee8b5; --accent2: #38bdf8; --off: #f5a524; --mine: #0f1d2b;
        --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace; }
* { box-sizing: border-box; }
body { margin: 0; height: 100vh; display: flex; flex-direction: column; color: var(--fg);
       background: radial-gradient(1200px 500px at 80% -10%, rgba(56,189,248,.10), transparent 60%),
                   linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px) 0 0 / 100% 28px,
                   linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px) 0 0 / 28px 100%, var(--bg);
       font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 14px 20px; border-bottom: 1px solid var(--line); display: flex; gap: 14px; align-items: center;
         flex-wrap: wrap; background: rgba(7,9,13,.75); backdrop-filter: blur(6px); }
.brand { font: 700 18px/1 var(--mono); letter-spacing: .18em; text-transform: uppercase; }
.brand i { font-style: normal; color: var(--accent); text-shadow: 0 0 12px rgba(62,232,181,.6); }
.chip { font: 12px/1 var(--mono); color: var(--muted); border: 1px solid var(--line); border-radius: 6px;
        padding: 6px 8px; overflow-wrap: anywhere; }
#engine:empty { display: none; }
#engine.on { color: var(--accent); border-color: rgba(62,232,181,.5); box-shadow: 0 0 14px rgba(62,232,181,.18); }
#engine.off { color: var(--off); border-color: rgba(245,165,36,.45); }
#engine::before { content: "\\25CF  "; }
.meter { margin-left: auto; display: flex; align-items: center; gap: 12px; }
.meter canvas { width: 140px; height: 34px; }
.readout { text-align: right; font-family: var(--mono); }
#speed { display: block; font-size: 30px; font-weight: 700; line-height: 1; color: var(--accent);
         font-variant-numeric: tabular-nums; text-shadow: 0 0 18px rgba(62,232,181,.35); min-width: 5ch; }
#speedlabel { font-size: 11px; letter-spacing: .14em; color: var(--muted); text-transform: uppercase; }
#log { flex: 1; overflow-y: auto; width: 100%; max-width: 860px; margin: 0 auto; padding: 20px; }
.msg { white-space: pre-wrap; overflow-wrap: anywhere; padding: 12px 16px; border-radius: 10px; margin: 10px 0; }
.user { background: var(--mine); border: 1px solid rgba(56,189,248,.25); margin-left: 14%; }
.assistant { background: var(--panel); border: 1px solid var(--line); margin-right: 14%;
             box-shadow: inset 2px 0 0 var(--accent); }
.error { color: #ff6b6b; }
.stats { color: var(--muted); font: 12px/1.4 var(--mono); margin: -4px 14% 12px 2px; font-variant-numeric: tabular-nums; }
.stats b { color: var(--accent); font-weight: 600; }
form { display: flex; gap: 10px; width: 100%; max-width: 860px; margin: 0 auto; padding: 14px 20px;
       border-top: 1px solid var(--line); }
textarea { flex: 1; resize: none; font: inherit; padding: 12px; border-radius: 10px; border: 1px solid var(--line);
           background: var(--panel); color: var(--fg); outline: none; }
textarea:focus { border-color: rgba(62,232,181,.6); box-shadow: 0 0 0 3px rgba(62,232,181,.12); }
button { font: 600 14px var(--mono); letter-spacing: .08em; text-transform: uppercase; padding: 0 22px; border: 0;
         border-radius: 10px; color: #04120d; cursor: pointer;
         background: linear-gradient(135deg, var(--accent), var(--accent2)); }
button:disabled { opacity: .45; cursor: default; }
</style>
</head>
<body>
<header>
  <span class="brand">Iron<i>Mule</i></span>
  <span class="chip" id="model">connecting</span>
  <span class="chip" id="engine"></span>
  <div class="meter" title="completion tokens per second, from sending the message to its last token (prompt processing included)">
    <canvas id="spark" width="280" height="68"></canvas>
    <div class="readout"><output id="speed">--</output><span id="speedlabel">tok/s</span></div>
  </div>
</header>
<main id="log"></main>
<form id="form">
  <textarea id="input" rows="2" placeholder="Message. Enter sends, Shift+Enter starts a new line." autofocus></textarea>
  <button id="send" type="submit">Send</button>
</form>
<script>
const log = document.getElementById("log"), form = document.getElementById("form");
const input = document.getElementById("input"), send = document.getElementById("send");
const label = document.getElementById("model");
const engineLabel = document.getElementById("engine"), speed = document.getElementById("speed");
const speedLabel = document.getElementById("speedlabel"), spark = document.getElementById("spark");
const history = [];
let model = null;
let key = sessionStorage.getItem("ironmule-key") || "";

// What /health reports, in words: the stock path without a numeric plan is IronMule switched off.
function engineName(health) {
  if (health.backend === "current_engine") return ["IronMule engine · on", true];
  const plan = (health.execution || "").split("@")[1];
  if (plan) return ["IronMule " + plan + " · on", true];
  if (health.backend === "mlx_lm_reference") return ["stock MLX · IronMule off", false];
  return [health.backend || "", false];
}

function headers() {
  const value = {"Content-Type": "application/json"};
  if (key) value.Authorization = "Bearer " + key;
  return value;
}

function add(role, text) {
  const node = document.createElement("div");
  node.className = "msg " + role;
  node.textContent = text;
  log.appendChild(node);
  log.scrollTop = log.scrollHeight;
  return node;
}

function draw(points) {
  const ctx = spark.getContext("2d"), w = spark.width, h = spark.height;
  ctx.clearRect(0, 0, w, h);
  if (points.length < 2) return;
  const top = Math.max(...points) * 1.15 || 1;
  ctx.beginPath();
  points.forEach((value, i) => {
    const x = i / (points.length - 1) * w, y = h - value / top * (h - 6) - 3;
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.strokeStyle = "#3ee8b5"; ctx.lineWidth = 3; ctx.shadowColor = "#3ee8b5"; ctx.shadowBlur = 10; ctx.stroke();
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.shadowBlur = 0; ctx.fillStyle = "rgba(62,232,181,.12)"; ctx.fill();
}

async function api(path, options = {}) {
  let response = await fetch(path, {...options, headers: headers()});
  if (response.status === 401) {
    key = prompt("This server needs its API key:") || "";
    sessionStorage.setItem("ironmule-key", key);
    response = await fetch(path, {...options, headers: headers()});
  }
  return response;
}

async function connect() {
  try {
    const body = await (await api("/v1/models")).json();
    const models = body.data || [];
    const loaded = models.find((entry) => entry.loaded) || models[0];
    model = loaded ? loaded.id : null;
    label.textContent = model || "no model is being served";
    const [name, on] = engineName(await (await api("/health")).json());
    engineLabel.textContent = name;
    engineLabel.className = "chip " + (on ? "on" : "off");
  } catch (error) {
    label.textContent = "the server is not reachable";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || !model || send.disabled) return;
  input.value = "";
  send.disabled = true;
  add("user", text);
  history.push({role: "user", content: text});
  const out = add("assistant", "");
  const stats = document.createElement("div");
  stats.className = "stats";
  log.appendChild(stats);
  // Tokens per second = completion tokens / time since sending, the same for every engine, so
  // an answer that arrives whole (IronMule's engine) and a streamed one compare directly.
  const started = performance.now(), points = [];
  let answer = "", tokens = 0, first = null;
  const show = (final) => {
    const seconds = (performance.now() - started) / 1000, rate = tokens / seconds;
    speed.textContent = tokens ? rate.toFixed(1) : seconds.toFixed(1) + "s";
    speedLabel.textContent = final ? "tok/s · last answer" : tokens ? "tok/s · live" : "waiting";
    if (tokens) { points.push(rate); draw(points); }
    stats.textContent = "";
    const value = document.createElement("b");
    value.textContent = rate.toFixed(1) + " tok/s";
    stats.append(value, "  " + tokens + " tokens in " + seconds.toFixed(2) + " s"
      + (first === null ? "" : " · first text after " + ((first - started) / 1000).toFixed(2) + " s")
      + (final ? "" : " …"));
  };
  const ticker = setInterval(() => show(false), 100);
  try {
    const response = await api("/v1/chat/completions", {method: "POST",
      body: JSON.stringify({model, messages: history, max_tokens: 1024, stream: true})});
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error((body.error && body.error.message) || response.statusText);
    }
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split("\\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        const chunk = JSON.parse(line.slice(6));
        if (chunk.error) throw new Error(chunk.error.message || "generation failed");
        if (chunk.usage) tokens = chunk.usage.completion_tokens;
        const delta = chunk.choices && chunk.choices[0].delta.content;
        if (delta) {
          if (first === null) first = performance.now();
          if (!chunk.usage) tokens += 1;
          answer += delta;
          out.textContent = answer;
          log.scrollTop = log.scrollHeight;
        }
      }
    }
    history.push({role: "assistant", content: answer});
    clearInterval(ticker);
    show(true);
  } catch (error) {
    clearInterval(ticker);
    stats.remove();
    speed.textContent = "--";
    speedLabel.textContent = "tok/s";
    out.classList.add("error");
    out.textContent = "Error: " + error.message;
    history.pop();
  }
  send.disabled = false;
  input.focus();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

connect();
</script>
</body>
</html>
"""
