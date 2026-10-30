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
| `--engine` | `reference` | `reference` (earlier name `stock`): mlx-lm's own generation, streamed token by token; `tuned` (earlier name `ironmule`): IronMule's engine with this machine's tuned profile, streaming up to eight requests as one group; `rust2`: the rust2 binary on CUDA (experimental); `dense`: the dense numeric plan, up to eight requests in one decode step with IronMule's own kernels (recommended on two Tesla T4s with `mlx-community/Qwen3-1.7B-bf16`: about half the cost of `tuned` on mixed traffic, answers differ, see LIMITS), unquantized Qwen 3 bf16 on CUDA below compute capability 8 only, refused elsewhere; prompt plus answer at most 2,048 tokens (larger slots wait for BACKLOG H11), and a longer request is answered `400` on its own while the rest of its group runs; a request that arrives while a group runs takes a slot a finished request freed, whatever the serving policy's `join` says (experimental, changes answers against `tuned`). All are greedy but compute the last prompt token differently, so answers can differ between engines |
| `--learning` | `off` | `active` or `paused`: learn group width and gather wait from this server's traffic; see [live learning](LIVE_LEARNING.md) |
| `--no-dreaming` | off | with `--learning`: the same learner without model rollouts |
| `--gpus` | none | `all` or device indices such as `0,1`: one model worker per CUDA GPU, see [Several GPUs](#several-gpus-hw1) |
| `--schedule` | `auto` | `auto`: a fixed schedule qualified on exactly this setup serves when one ships and no serving policy is stored (see [Qualified schedule](#several-gpus-hw1)); `reference`: the default schedule |
| `--knobs` | `auto` | `--engine tuned` only. `auto`: a knob set qualified on exactly this model, revision, GPU and MLX and mlx-lm versions serves when one ships and no tune profile is stored; it may only change knobs that leave every answer unchanged (compiled step, projection fusion, fused argmax, readback interval). None ships yet (BACKLOG KP1). `reference`: the engine's own knobs |
| `--dispatch-policy`, `--dispatch-evidence` | none | opt-in learned dispatch with the comparison that qualifies it (LD1, below) |
| `--agent` | `off` | `on`: the IronMule agent decides each group's order and width and learns online (AG1, below); one worker; not with `--learning` or `--dispatch-policy` |

## Routes

| Method | Path | Behaviour |
| :-- | :-- | :-- |
| `GET` | `/` | a chat page for the browser; static, served without a key, and every call it makes goes through the routes below. It shows which engine answers and, live, completion tokens per second from sending to the last token |
| `GET` | `/health` | service, mode, backend, the worker's actual `device` (`gpu` or `cpu`), loaded model, queue and completion counters, the `schedule` that serves, the tuned engine's `knobs` (values, their `source`, and whether they are `class_a`: only knobs that leave every answer unchanged), and a `learning` summary when learning is on |
| `GET` | `/ready` | the same report; readiness is part of it |
| `GET` | `/v1/models` | the registered models, each with its exact snapshot revision |
| `POST` | `/v1/chat/completions` | `stream: false` gives one JSON completion; `stream: true` gives SSE chunks ending with `data: [DONE]` |
| `GET` | `/v1/ironmule/learning` | live-learning status: serving, mode, candidate, the strategy that serves and why |
| `POST` | `/v1/ironmule/learning` | `{"action": "pause" \| "resume" \| "rollback" \| "compare"}`; `400` when learning is off |
| `GET` | `/v1/ironmule/dispatch` | the learned dispatch's status (LD1), or the agent's (AG1): active or not and why, policy digest, evidence, decisions, errors |
| `POST` | `/v1/ironmule/dispatch` | `{"action": "off"}` withdraws the learned dispatch without a restart; `400` when none is configured. With `--agent on`, `off` also pauses the agent's learning and `{"action": "on"}` resumes it |

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

**Learned dispatch (LD1, opt-in).** `--dispatch-policy POLICY.json --dispatch-evidence EVIDENCE.json`
replaces three scheduling choices with a policy the RL agent learned (`tools/rl4_agent.py`):
the card a new request goes to, whether a group starts its shortest requests first, and, in a
v2 policy, the group's width. It never changes what a request answers. Each request still runs
alone on one worker at its own capacity, and the hardware comparisons checked that every answer
stayed identical. The policy activates only when the evidence carries its digest, a passed
comparison, and this server's model, revision, GPU name and number of GPUs, and the engine when the evidence
names one (the shipped policies were measured on `--engine tuned`). Otherwise the
reference dispatch serves. `/health` then shows `dispatch` (`learned` or
`join_shortest_queue`) and a `learned_dispatch` block with the policy digest, the experiment,
the traffic it was validated on (`domain`), the measured gain, counts of its decisions and
errors, and the reason when it is off. After three errors while deciding, or when the operator
sends `POST /v1/ironmule/dispatch` with `{"action": "off"}`, it is withdrawn at once and the
reference serves. The withdrawal is recorded in `dispatch-withdrawals.json` in the state
directory, so a restart with the same policy and evidence keeps it off. New passed evidence for
the policy, or deleting that record, qualifies it again. It cannot be combined with `--learning`.

The first policy ships in `ironmule_product/dispatch/rl4s-t4x2-qwen3-1.7b/` (RL4-S): two T4s,
`mlx-community/Qwen3-1.7B-bf16` at revision `9cd66928`, `--engine tuned`. On spike traffic it beat the
best fixed arm by 9.0 % (95 % 6.9-11.1 %). It is not validated on traffic that mixes quiet, bursty and
spike states (BACKLOG LD1). The width-learning RL6 policy was tested on mixed traffic and is not
shipped: it beat the reference by 21.2 % but cost 6.9 % more than the best fixed schedule in bursts
(BACKLOG LD1-V).

**The IronMule agent (AG1, opt-in, experimental).** `--agent on` makes the agent the owner of two scheduling
choices: whether a group starts with the smallest output limits and how many requests it holds (1, 2, 4 or 8,
within the engine's capacity). It never changes an answer. It learns from every finished group what this machine's
work costs, keeps a metadata memory of arrivals (no text), trains a new schedule by reinforcement learning in a
separate low-priority process while the server is idle, and replays recorded arrivals with a filler prompt through
the real model to compare the new schedule with the one serving (A, B, A again). The new one serves only when its
gain's lower 95 % bound is above zero and above what two runs of the same schedule differ by, and every replayed
answer matched. Until then the server's own schedule serves. A real request cancels a replay at once. Three
errors, two windows of 24 requests with failures, or `{"action": "off"}` withdraw a serving schedule; its digest
is recorded and never serves again under this setup. What it learned lives in `agent/` in the state directory,
bound to the hardware, device, model, revision, plan and software versions; any change starts fresh. `/health`
shows an `agent` block: what serves and its measured gain, the fitted costs, the planner, the candidate and its
comparison, decision counts and the mean decision time. It refuses the dense plan, whose freed slots it does not
model yet, and runs on one worker. Its gain on hardware is not measured yet (BACKLOG AG1).

**Qualified schedule (RL6, confirmed by LD1-V).** On the setup the RL runs measured -- `--gpus all`
on two Tesla T4s, `--engine tuned`, `mlx-community/Qwen3-1.7B-bf16` at revision `9cd66928` -- the
server starts groups of at most two and, within the queue, the request with the smallest output limit
first (a request waiting longer than 10 s goes by arrival). With quiet, bursty and spike traffic
weighted equally this cost 18.4 % less than the default (95 % 14.9-22.2 %, 1,920 requests answered
identically). It changes no answer. The record and its comparison ship in
`ironmule_product/dispatch/rl6-t4x2-qwen3-1.7b/schedule.json` and serve only on an exact match of
engine, model, revision, GPU name and GPU count, and only when the state directory stores no serving
policy (`ironmule dream` writes one). `--schedule reference` keeps the default. `/health` shows the
schedule that serves under `schedule`, with its `source`. With `--dispatch-policy`, the learned
dispatch decides the order and the policy keeps its measured width.

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
