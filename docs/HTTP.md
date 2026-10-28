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
| `--engine` | `reference` | `reference` (earlier name `stock`): mlx-lm's own generation, streamed token by token; `tuned` (earlier name `ironmule`): IronMule's engine with this machine's tuned profile, streaming up to eight requests as one group; `rust2`: the rust2 binary on CUDA (experimental). All are greedy but compute the last prompt token differently, so answers can differ between engines |
| `--learning` | `off` | `active` or `paused`: learn group width and gather wait from this server's traffic; see [live learning](LIVE_LEARNING.md) |
| `--no-dreaming` | off | with `--learning`: the same learner without model rollouts |
| `--gpus` | none | `all` or device indices such as `0,1`: one model worker per CUDA GPU, see [Several GPUs](#several-gpus-hw1) |

## Routes

| Method | Path | Behaviour |
| :-- | :-- | :-- |
| `GET` | `/` | a chat page for the browser; static, served without a key, and every call it makes goes through the routes below. It shows which engine answers and, live, completion tokens per second from sending to the last token |
| `GET` | `/health` | service, mode, backend, the worker's actual `device` (`gpu` or `cpu`), loaded model, queue and completion counters, and a `learning` summary when learning is on |
| `GET` | `/ready` | the same report; readiness is part of it |
| `GET` | `/v1/models` | the registered models, each with its exact snapshot revision |
| `POST` | `/v1/chat/completions` | `stream: false` gives one JSON completion; `stream: true` gives SSE chunks ending with `data: [DONE]` |
| `GET` | `/v1/ironmule/learning` | live-learning status: serving, mode, candidate, the strategy that serves and why |
| `POST` | `/v1/ironmule/learning` | `{"action": "pause" \| "resume" \| "rollback" \| "compare"}`; `400` when learning is off |

The request body is read as OpenAI chat completion input: `messages` (required,
non-empty) and `max_tokens` or `max_completion_tokens`. Sampling fields are accepted
and ignored. Decoding is greedy, so the same prompt gives the same tokens.

## Concurrency

Requests are **queued, not rejected while busy**. The queue depth is the mode's:
`8` in `desktop`, `64` in `server`. With the reference engine, generation is
serialised through one permit, so four simultaneous requests complete in arrival
order rather than interleaving through one engine. `--engine tuned` streams up to
eight waiting requests as one group (SERVE4); with `--learning`, how wide a group
may be and how long it waits for company is decided by the live learner.

Two things do return `HTTP 429`:

- more than 64 accepted sockets at once, rejected in the accept loop before a worker
  thread exists, and
- a full request queue.

A load balancer in front of this should treat `429` as backpressure, not as failure.
Past the accept limit the server sends the `429` and closes at once, so a client still
sending its request may see a connection reset instead; that request was never accepted.

## Several GPUs (HW1)

`--gpus all` (or `--gpus 0,1`) starts one model worker per listed CUDA GPU, each pinned
to its card with `CUDA_VISIBLE_DEVICES` and loading its own copy of the model, so every
card must hold the whole model. Each worker keeps its own queue (the mode's depth per
card), scheduler, grouping and recovery; a lost worker is replaced on the same card. A
new request goes to the card with the fewest admitted, unfinished requests, ties to the
lower index, and stays there, so its tokens are what that one worker returns.
`/health` adds the cards under `devices`. Live learning and idle tuning model one worker,
so they need one card: `--learning` is refused with more than one, and idle tuning is
skipped. Lifecycle events go to the one stability log tagged with their card; each card
writes its own `serving-log.cuda-<card>.jsonl`. `--gpus 1` alone pins the ordinary single
worker to that card. Not measured yet: the throughput gain on Kaggle's two T4s and that
both cards return the same tokens for the same request (BACKLOG HW1).

## When the model worker fails (STAB1)

The model runs in a separate worker process. When it dies, reports a generation error
(for example CUDA out of memory), stops answering or breaks the protocol:

- every request in its group ends with an explicit error: `503 backend_unavailable`
  before the first token, or an SSE error frame and `[DONE]` after it. A partly streamed
  answer is never continued, restarted or repeated;
- the server replaces the worker, but only after the old process has been reaped, at most
  three times in ten minutes (0, 5 and 20 s apart), and only with a worker that reports the
  same model, revision, device, variant, sources and engine profile. Requests arriving
  meanwhile wait in the queue under their own deadlines;
- when recovery is refused, `/health` says why, `ready` is false and requests get `503`
  at once. Restart the server to try again.

Each request's own deadline (`request_timeout_s`, 120 s by default) ends only that
request. The worker is retired as unresponsive only when it has not finished a group a
full request timeout after the group's last deadline; with no request timeout there is no
such bound. In a group, a request's last token and completion arrive when the worker's
engine run ends, so a short request grouped with a long one can wait for it.

`/health` adds `worker` (`state`: ready, restarting, lost, refused or unavailable; `pid`,
`restarts`, `refused`, the last `exit` with reason, exit status and whether it was reaped,
and `scheduler_alive`), `in_flight` (the running group's width, open members, age and time
since the worker's last event), `groups_started` and `oldest_queued_s`. The state
directory keeps a bounded `stability-log.jsonl` (4 MiB, older half dropped): admissions,
rejections, groups, every request's end and error code, how each HTTP response ended,
worker exits with the last 4 KiB of the worker's stderr, and restarts. It holds ids, times
and codes, no prompt or answer text; the stderr tail can hold library messages, so keep the
file private like the rest of the state directory.

## CUDA memory (MEM2)

On CUDA the worker bounds MLX's cache of freed buffers to 1,024 MiB
(`IRONMULE_CUDA_CACHE_LIMIT_MIB`, in MiB; `0` keeps MLX's default, which lets the cache
grow to 0.95 of device memory). Without the bound, on a 15 GiB T4 the cache reached
11 GiB beside about 3 GiB that MLX does not count, and the worker was lost when the
device filled. The bound changes no output; whether it removes those losses and what it
costs in speed is not yet measured (its preregistered comparison was stopped before a
result). Metal is unchanged.

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
