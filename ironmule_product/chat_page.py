"""The chat page `ironmule serve` shows at `/`: one static file, no external resources.

It talks to the same server's OpenAI-compatible API, so it needs nothing the API does not
already offer. Model output is only ever set as text, never parsed as HTML.
"""

CHAT_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IronMule</title>
<style>
:root { --bg: #fafafa; --fg: #1a1a1a; --muted: #6b6b6b; --mine: #e6ecfb; --line: #dcdcdc; --accent: #2f5bd3; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #141414; --fg: #ececec; --muted: #9a9a9a; --mine: #22304f; --line: #2c2c2c; --accent: #7ea2ff; }
}
* { box-sizing: border-box; }
body { margin: 0; height: 100vh; display: flex; flex-direction: column; background: var(--bg); color: var(--fg);
       font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 12px 16px; border-bottom: 1px solid var(--line); display: flex; gap: 10px; align-items: baseline; }
header b { font-size: 18px; }
header span { color: var(--muted); font-size: 14px; overflow-wrap: anywhere; }
#log { flex: 1; overflow-y: auto; width: 100%; max-width: 820px; margin: 0 auto; padding: 16px; }
.msg { white-space: pre-wrap; overflow-wrap: anywhere; padding: 10px 14px; border-radius: 12px; margin: 8px 0; }
.user { background: var(--mine); margin-left: 12%; }
.assistant { border: 1px solid var(--line); margin-right: 12%; }
.error { color: #d0453a; }
form { display: flex; gap: 8px; width: 100%; max-width: 820px; margin: 0 auto; padding: 12px 16px;
       border-top: 1px solid var(--line); }
textarea { flex: 1; resize: none; font: inherit; padding: 10px; border-radius: 10px; border: 1px solid var(--line);
           background: var(--bg); color: var(--fg); }
button { font: inherit; padding: 0 18px; border: 0; border-radius: 10px; background: var(--accent); color: #fff;
         cursor: pointer; }
button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
<header><b>IronMule</b><span id="model">connecting</span></header>
<main id="log"></main>
<form id="form">
  <textarea id="input" rows="2" placeholder="Message. Enter sends, Shift+Enter starts a new line." autofocus></textarea>
  <button id="send" type="submit">Send</button>
</form>
<script>
const log = document.getElementById("log"), form = document.getElementById("form");
const input = document.getElementById("input"), send = document.getElementById("send");
const label = document.getElementById("model");
const history = [];
let model = null;
let key = sessionStorage.getItem("ironmule-key") || "";

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
  let answer = "";
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
        const delta = chunk.choices && chunk.choices[0].delta.content;
        if (delta) {
          answer += delta;
          out.textContent = answer;
          log.scrollTop = log.scrollHeight;
        }
      }
    }
    history.push({role: "assistant", content: answer});
  } catch (error) {
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
