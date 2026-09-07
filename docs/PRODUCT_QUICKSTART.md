# Local model service (development preview)

IronMule now includes a portable CLI and an OpenAI-compatible text chat API.
The preview keeps one registered model in an isolated, persistent MLX worker.
It now includes bounded automatic calibration jobs and a verified local history.
The stock reference remains the default: automatic configuration promotion, an
always-on background optimizer, RL and long-running server qualification are still
open. Calibration does not silently enable an experimental backend.

## Start on an Apple Silicon Mac

Install this branch's package into your own Python environment (Python 3.10+
as declared by the package; this product slice was checked locally on 3.12):

```sh
python -m pip install .
ironmule doctor --json
ironmule setup --mode desktop
ironmule models list --family gemma --json
ironmule models add mlx-community/gemma-3-1b-it-4bit
ironmule serve --model mlx-community/gemma-3-1b-it-4bit
```

`models add` registers a complete cached snapshot; it does not load the model.
If multiple revisions exist, select one with `--revision`. To deliberately
download missing weights, append `--download`. No model download or remote
code execution happens implicitly during serving. Use `--cache-root PATH` to
select an explicit Hugging Face cache for inventory/registration.

`setup --mode server` selects a larger pending-request queue: 64 instead of
8. It does not yet enable batching, a different GPU scheduler or a measured
throughput improvement. Both profiles currently process one generation at a time.

## Send a request

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/gemma-3-1b-it-4bit","messages":[{"role":"user","content":"Explain unified memory in one sentence."}],"max_tokens":64,"temperature":0}'
```

For SSE, add `"stream":true` and use `curl -N`. A finite stream ends with
`data: [DONE]` and closes its HTTP connection. Supported request fields are
`model`, text `messages` (system/user/assistant), one of `max_tokens` or
`max_completion_tokens`, `temperature=0`, `top_p=1`, an optional uint32 `seed`,
up to four `stop` strings, `stream`, and `n=1`. Unsupported parameters fail
explicitly; tools, images, embeddings and non-greedy sampling are not supported
by this preview. Model-specific chat-template restrictions still apply.

The total tokenized prompt plus requested output must fit the current 8192-token
product limit. The HTTP body is limited to 1 MiB. A request deadline is 120 s
including queue time. Saturated request admission returns 429; unavailable
workers return 503. A worker failure does not silently restart or repeat a
request. Automatic restart is a later product stage.

## Inspect and stop

```sh
ironmule status --product
ironmule models registered
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
curl http://127.0.0.1:8080/v1/models
```

`status --product` reads saved configuration, not another running process's
liveness. `/health` is process status; `/ready` reports whether its worker is
ready. Ctrl-C or SIGTERM closes the service and its owned model worker.
`ironmule models remove MODEL_ID` removes only the registration, never weights.

The default state directory is `~/.ironmule/product`; use `--state-dir PATH`
or `IRONMULE_HOME` for a dedicated private directory. An existing shared/public
directory is rejected rather than silently changing its permissions. Product
state contains settings and model registrations, not prompts or generated text.
In-flight text stays in memory; request logging and automatic uploads are off.

## Automatic calibration jobs

```sh
ironmule optimize run --model mlx-community/gemma-3-1b-it-4bit --wait-ready 300
ironmule optimize status
ironmule optimize history --limit 100
ironmule optimize pause
ironmule optimize resume
```

The job waits for three spaced, eligible host observations before loading weights.
It then owns the frozen 90-call reference/candidate protocol, including warmups,
A/A, AB/BA, fresh workers and strict memory/time/identity checks. If no valid window
appears, it exits with code 3 (`deferred`) and records the reason; it does not
weaken thresholds, secretly retry or change the Mac's power settings.

Calibration also watches the starting worker's real RSS and system-wide swap,
then checks its actual process/MLX peaks before inference. A violation aborts
the owned worker and retains the reason in history; polling is not a hard RAM
reservation and can observe overshoot. Normal completion requires a clean
worker exit. These calibration guards do not imply a global serving-memory
scheduler. The earlier 12B swap failure is retained, but a subsequent installed
12B load and short exact-reference generation test pass under their recorded
host conditions ([12B results](PROD4P_12B_RESULTS_2026-09-07.md)). That does not
qualify long contexts or sustained load, or establish a permanent memory fix.

Model snapshots cannot opt into executable custom Python through `model_file`.
The worker checks bounded configuration before opening MLX and explicitly
disables that separate loader path; tokenizer remote code remains disabled.

Use `--readiness-only` to exercise the wait/status path without loading a model.
`--json` returns the full calibration report; the ordinary output is a concise
summary. `history --after-seq N` pages verified metadata events. Model output and
user messages are never journaled. A kernel-held lock, not an old PID/status file,
proves a running calibration job. Interrupted jobs are not called successful.

This stage is a standalone calibration job. Serving and calibration are serialized
for the same configured state directory; it cannot duplicate a live serving model
in that state. Separate state directories and other applications are not a global
GPU scheduler, so do not launch concurrent GPU workloads to create calibration
evidence. Sharing an online serving worker and automatic deferred-job rescheduling
are later integration work. Pause ends a running job at its next checkpoint;
resume clears the pause setting but does not create a hidden replacement job.

A `calibration_signal` is limited evidence for the registered workload, not a
deployment authorization. The evaluator independently checks the complete schedule,
outputs, hardware/model/software identities and recorded resource timeline.

## Network exposure

The default listener is loopback only. A non-loopback listener requires both
TLS certificate/key files and an API key supplied through `IRONMULE_API_KEY`
(or the variable selected by `--api-key-env`). Send `Authorization: Bearer KEY`
on every endpoint when a key is configured. Do not put real keys in shell
history or command-line arguments. Loopback Host/Origin checks reject unrelated
browser origins; this is not a general browser CORS gateway.

TLS handshakes are bounded and count toward the 64-connection handler limit.
At that transport limit plain HTTP receives 429; an unnegotiated TLS socket is
closed without sending plaintext. Deployments still need normal network access
controls; short integration tests are not a sustained-production readiness claim.

## Evidence and experimental paths

See [the product plan](PRODUCT_IMPLEMENTATION_2026-09-05.md) and
[the real-Gemma screen](PROD1_GEMMA_SMOKE_2026-09-07.md). Model inventory is
metadata, not a hardware qualification. The private `bounded_prefetch`
candidate is not selectable over HTTP or automatically promoted. Any measured
benefit is limited to the exact tested model, environment and workload; another
Mac or a library update must be qualified independently.
