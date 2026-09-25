# OpenAI-compatible HTTP endpoint

*Command: `ironmule serve`. Server: `ironmule_product.http_server`.*

`ironmule serve` binds an HTTP server to one registered local model so an OpenAI
client, Cursor, or Open WebUI can talk to it. It is standard library only: no web
framework, and the package gains no dependency for it.

## Run it

```bash
ironmule setup --mode desktop
ironmule models add mlx-community/gemma-3-4b-it-4bit
ironmule serve --model mlx-community/gemma-3-4b-it-4bit --host 127.0.0.1 --port 8080
```

`models add` registers a snapshot that is already in the local Hugging Face cache.
Pass `--download` if you want IronMule to fetch it first. The server prints one line
when it is ready:

```json
{"service": "ironmule", "host": "127.0.0.1", "port": 8080, "ready": true}
```

| Option | Default | Meaning |
| :-- | :-- | :-- |
| `--model` | none | registered model to keep loaded; mutually exclusive with `--no-model` |
| `--no-model` | off | control plane only; generation returns unavailable |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8080` | bind port |
| `--api-key-env` | `IRONMULE_API_KEY` | environment variable holding the bearer token |
| `--tls-cert`, `--tls-key` | none | serve over TLS |
| `--state-dir` | `IRONMULE_HOME` or `~/.ironmule/product` | product state directory |
| `--compute-dtype` | none | opt-in numeric plan for GPUs that emulate bf16 (`float32`, `native`); changes output, see `ironmule plans` |
| `--batch-width` | off | opt-in: compute up to this many waiting `stream: false` requests as one tensor batch (2 to 8); changes output, see below |

## Routes

| Method | Path | Behaviour |
| :-- | :-- | :-- |
| `GET` | `/` | a chat page for the browser; static, served without a key, and every call it makes goes through the routes below |
| `GET` | `/health` | service, mode, backend, loaded model, queue and completion counters |
| `GET` | `/ready` | the same report; readiness is part of it |
| `GET` | `/v1/models` | the registered models, each with its exact snapshot revision |
| `POST` | `/v1/chat/completions` | `stream: false` gives one JSON completion; `stream: true` gives SSE chunks ending with `data: [DONE]` |

The request body is read as OpenAI chat completion input: `messages` (required,
non-empty) and `max_tokens` or `max_completion_tokens`. Sampling fields are accepted
and ignored. Decoding is greedy, so the same prompt gives the same tokens.

## Concurrency

Requests are **queued, not rejected while busy**. The queue depth is the mode's:
`8` in `desktop`, `64` in `server`. Generation itself is serialised through one
permit, so four simultaneous requests complete in arrival order rather than
interleaving through one engine. `--batch-width` is the one exception, below.

Two things do return `HTTP 429`:

- more than 64 accepted sockets at once, rejected in the accept loop before a worker
  thread exists, and
- a full request queue.

A load balancer in front of this should treat `429` as backpressure, not as failure.

## Batched serving (opt-in)

`--batch-width N` (2 to 8) lets one engine pass serve several requests. When the engine
becomes free, the requests with `stream: false` that are already waiting, up to N, are
computed together as one tensor batch: each weight is read once for all of them. Nothing
waits to form a batch; a lone request, and every `stream: true` request, runs exactly as
without the option.

The contract differs from the default path in one respect, and it is why the option is off
by default and never chosen automatically:

- **A batched answer can differ from the same prompt served alone.** It is still greedy, but
  computed with the batch's arithmetic, which rounds differently from a single request's;
  it can also differ between two batches that hold different neighbours. Decoding stays
  deterministic for a fixed batch, not across different traffic. `/health` reports it:
  `"backend": "tensor_batch"` and `"execution": "batched"` (`"batched@native"` with a numeric
  plan).

Everything else holds as on the default path:

- request validation, the context limit, `max_tokens`, the model's end-of-sequence and
  end-of-turn tokens, and stop strings, which are applied to each answer on its own;
- the queue, `429` and the deadline: a batch runs under the earliest deadline of its
  requests;
- isolation: every request keeps its own cache rows and receives only its own tokens; the
  requests share arithmetic, never text;
- delivery: a batched `stream: false` answer is returned when its batch finishes, as one
  completion, like any other;
- cancellation: a cancelled request leaves its batch at the next step, and the others finish
  without it.

The measured gain, the agreement with single requests and the quality gate of the batched
arithmetic are in `research/LEDGER.md`, PERF1-K.

## What it does not do

- **No sampling.** `temperature`, `top_p`, `n`, `stop`, tools, and logprobs are not
  implemented. Output is greedy and matches the engine's own generate path.
- **No model download during serving.** `--model` must name a model that
  `ironmule models add` has already registered.
- **No multi-tenant accounting.** One bearer token, checked or absent. Bind to
  `127.0.0.1` or put it behind something that does the rest.

## Verifying it yourself

```bash
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/gemma-3-4b-it-4bit",
       "messages":[{"role":"user","content":"Say hello in five words."}],
       "max_tokens":24}'
```

Add `"stream": true` to the same body and pass `curl -N` to watch the SSE chunks
arrive one token at a time.
