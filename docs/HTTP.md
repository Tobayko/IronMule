# OpenAI-compatible HTTP endpoint

*Module: `ironmule.http`. Command: `ironmule serve`.*

`ironmule serve` binds a small HTTP server to one loaded
[`Runtime`](RUNTIME.md) so an OpenAI client, Cursor, or Open WebUI can talk to a
local model. It is standard library only — no web framework is added, and the
package gains no dependency.

## Run it

```bash
ironmule serve --model mlx-community/gemma-3-4b-it-4bit --host 127.0.0.1 --port 8000
```

| Option | Default | Meaning |
| :-- | :-- | :-- |
| `--model` | the tuned default | model repo id; must already be in the local cache |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8000` | bind port |
| `--no-tuned-profile` | off | ignore this machine's tuned profile, run baseline knobs |

From Python:

```python
from ironmule import Runtime
from ironmule.http import serve

with Runtime.load() as rt:
    serve(rt, host="127.0.0.1", port=8000)
```

## Routes

| Method | Path | Behaviour |
| :-- | :-- | :-- |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/v1/models` | one entry: the loaded model's id |
| `POST` | `/v1/chat/completions` | `stream: false` → one JSON completion; `stream: true` → SSE chunks ending with `data: [DONE]` |

The request body is read as OpenAI chat completion input: `messages` (required,
non-empty), `max_tokens` or `max_completion_tokens` (default 256). `model`,
`temperature`, and other fields are accepted and ignored — decoding is greedy,
`temperature = 0`, the same as `Runtime.generate`.

## What it does not do

- **One request at a time.** The server holds a single permit. A second request
  while the model is busy gets `HTTP 429` with `Retry-After: 1`, not a queue and
  not interleaving through one engine. Streaming to one client does not benefit
  from `ThroughputMode`, so that mode is not exposed here.
- **No sampling.** `temperature`, `top_p`, `n`, `stop`, tools, and logprobs are
  not implemented. Output tokens are identical to `Runtime.generate` /
  `Runtime.stream` for the same prompt.
- **No auth, no TLS, no rate accounting.** Bind it to `127.0.0.1` or put it
  behind something that does.
- **No model download.** `--model` must resolve to an already-cached snapshot,
  the same rule as `Runtime.load`.

## Streaming detail

`stream: true` drives `Runtime.stream`, which is the sequential path decoded token
by token. Each SSE chunk carries an incremental `delta.content`; the final chunk
before `[DONE]` carries `finish_reason` (`stop` on EOS, `length` at `max_tokens`).
Incremental detokenisation decodes the visible token list each step and emits the
new suffix, so a chunk boundary can fall inside a multi-byte character on some
tokenizers; the concatenated stream is still exact.
