# Local model service (development preview)

IronMule now includes a portable CLI and an OpenAI-compatible text chat API.
The preview keeps one registered model in an isolated, persistent MLX worker.
It is **not yet the autonomous optimizer**: optimization status explicitly says
`configuration_only`, and the stock reference remains the default. Long-running
server qualification, automatic configuration promotion and RL are still open.

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
