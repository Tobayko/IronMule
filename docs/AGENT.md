# The IronMule agent

The project's main goal (BACKLOG, "The main goal"): an agent that decides on its own how the model's requests
run on the hardware, keeps learning by reinforcement learning while it serves, decides in microseconds, and
serves a decision only after it proved a gain on the machine it runs on. It never changes an answer.

This page explains how the agent works and where its parts live. Protocols and every result are in
[BACKLOG.md](BACKLOG.md) under AG1 and AG2, and how to train and measure it is in
[training/README.md](../training/README.md).

## What it decides

At every group start, for the requests waiting on its card:

| Decision | Choices | Where |
| :-- | :-- | :-- |
| Order | arrival order, or shortest first | `Policy.uses_sjf` |
| What "shortest" means | the request's output limit, or the answer length it predicts (AG2) | `Policy.key` |
| Group width | 1, 2, 4 or 8, within the engine's capacity | `Policy.best_width` |

On several GPUs each card has its own agent, and the pool keeps placing requests on the card with the shortest
queue. A two-card simulation put almost all of the gain in each card's order and 1-2 points in placement
(BACKLOG AG2, "Multi-GPU").

What it optimises (AG8, 2026-11-09): by default tokens per second, with the mean first token at most 2 % later
(`objective: throughput` in its status); AG1's request cost (time in the system plus time to the first token) is
`latency`. The planner, the replay qualification and the audit all use the same objective, and the objective is part
of the agent's binding, so nothing learned under one serves under the other. In simulation at the T4's grouped
batch-1 price no width made more tokens per second than the reference's, so the throughput agent kept the reference
(BACKLOG AG8).

A decision reads a compiled table, about 7.5 µs in pure Python. A prediction is made once, when a request is
first seen, so the decision only reads a number.

## How it learns

Four steps repeat while the server runs (`ironmule_product/agent.py`):

1. **Observe.** Every finished group updates a world model of this machine by recursive least squares (prefill
   per member and per thousand prompt tokens, decode per round and per further session). Every finished live
   request goes into a traffic memory of metadata (arrival, prompt tokens, limit, tokens made, predicted
   tokens). It also teaches the length model (`ironmule_product/length_model.py`), which learns from hashed word
   features of the last message by SGD.
2. **Plan.** While idle, a low-priority planner process trains a policy by REINFORCE in a simulation of this
   machine priced by the world model, on resampled busy periods of the recorded traffic. It also prices every
   fixed schedule, including "shortest predicted first" once predictions are recorded. It proposes only what
   the model prices below the schedule that serves.
3. **Qualify.** While idle, recorded arrivals are replayed through the real model with filler prompts: the
   serving schedule, the candidate, the serving schedule again. Each replayed request makes its recorded number
   of tokens and is ordered by its recorded limit or prediction. A group-sequential judge looks at 12, 24, 48
   and 96 triplets. The candidate passes when the gain's lower bound is above zero and above the A/A floor, and
   every replayed answer matched.
4. **Serve and watch.** A qualified policy decides every group. Decision errors, failed live windows, a replay
   audit it loses, or the operator withdraw it. A withdrawn policy never serves again under that binding.

## Guarantees

- **Answers do not change.** The agent changes the order and width of grouped batch-1 work only. Every replay
  compares answer digests across its arms, and a difference vetoes the candidate.
- **Fail closed.** State is bound to the hardware, card, device, model, revision, plan and software. A foreign,
  corrupt or stale state starts fresh, and a serving policy resumes only with evidence that names it.
- **No text is kept.** The traffic memory holds metadata, and the length model holds weights over hashed buckets.
- **Opt-in.** `--agent on`. Without a qualified policy the server's own schedule serves.
- **It never stops the server or fails a request (owner, 2026-11-07).** Its setup at startup, the start of the
  engine it chose, the router's decisions, its learning and its dreams each fall back on failure: to the reference
  engine (a failed measurement), to its gated numeric plan on the reference engine and then the checkpoint's dtype
  (a chosen engine that does not start), or to a path in placement order (a router fault). Each fallback is printed
  or counted in `/health`. A plan the caller named is kept and its failure reported, never replaced.

## Its own setup: plan and engine (AG4)

The largest gains so far were settings a person found by hand: the `native` plan for a 4-bit checkpoint on an
emulating GPU, the stacked `dense` engine, and its width. With `--agent on` and no `--engine` or `--compute-dtype`,
the agent finds them itself before it serves (`ironmule/plan_choice.py`):

1. NUM1 measures each numeric plan on this device (perplexity gate, speed) and keeps the fastest that passes.
2. On that path, a stacked engine where one exists for the model. That is KB1's for Qwen 3 bf16, or KB2's for
   4-bit Gemma 3 (`ironmule/dense_gemma.py`). It must pass, on this card:
   - its load probe;
   - the quality gate on the decode path;
   - AG4's rule: more tokens per second at its best width, at least 0.9x for one request alone, and a first
     token at most 1.1x later.
   - Since AG5 (2026-11-07) the agent times every width from 1 to 8. It judges the engine at its best width and
     serves it at that width. On a T4, KB2-0's Gemma plan makes 2.0x at width 6 but 0.96x at width 8, where the
     old width-8 rule would have refused it. That plan passed its gate on 2026-11-07 (KB2-G and KB2-L).

Both gates score the same fixed text, cut into 16 chunks of 512 tokens. Since GATE2 (2026-11-07), each chunk
starts with the tokenizer's BOS, as a chat request does. Without it, Gemma 3 was unstable on every path, today's
included (KB2-D). A choice stored under the older gate is measured again on the next start.

When the stacked engine wins only under load (on a T4 with Qwen 3 1.7B it made 5x the tokens per second at
width 8, but was 18 % slower alone), and both fit the card, the agent serves both. It then routes every request,
in microseconds, to the path where it is expected to finish first (`ironmule_product/path_router.py`). It learns
both paths' costs from every finished request. In simulation, priced by that T4 measurement, the router was at
least as fast as the better path at every load, with the earliest first tokens. The hardware test (AG4-R) is
fixed in the backlog. Its run 1 on a T4 (2026-11-07) was cut by Kaggle's 12-hour limit after 22 of 36 blocks and
is invalid by its rule, but it showed where the agent falls short. On that traffic the stacked engine alone made
2.10x the single path's tokens per second, against the routed setup's 1.44x. The router learned too little to shift
towards it (1.007x over the frozen router), and it sent some requests to a path whose slot could not hold them.

## Its serving mode, learned by dreaming (AG6)

With two paths on the card, the agent now also learns how to use them (`PolicyRouter` in
`ironmule_product/path_router.py`). It has three modes: `route` (the router above), `stacked` (every request to
the stacked engine, and one its slot cannot hold to the single path) and `single`. Every 16 finished requests, once
32 are logged, it dreams: it replays the arrivals and answer lengths of its last 64 requests under each mode, in a
world model built from the prices it learned and the card's time shared between the two paths. It moves to the
mode with the most tokens per second if that mode is at least 5 % ahead and its mean first token is at most 1.1x
the current mode's. On quiet traffic, where every mode's tokens per second is just the arrival rate, it also moves to
the mode with the earliest first tokens, if that one keeps at least 98 % of the throughput and is at least 20 %
sooner (REV1). Quiet is what the router observed, not what the model estimates. Of its last 64 finished requests,
at most 1 in 16 arrived while another was being served, none is in flight, and none ended without its answer. Under
load the replay's first tokens are not reliable. The dream runs after a request has finished. A decision stays a lookup
in `stacked` or `single` (0.85 us in AG6-H) and the router's arithmetic in `route` (13.7 us on average and 62 us at
most in AG6-Q, where it served mostly in `route`). The dream runs on the thread of the request that finished, before that request's last event goes out,
so its cost (`dream_ms` under `/health`'s `paths`) also delays that one request's end; decisions are not held up by
it.
Each path still answers with its own gated plan, so no answer changes beyond the two paths' own.

On one T4 with Qwen 3 1.7B (AG6-H, 2026-11-07) the agent started in `route`, moved to `stacked` at its first dream
after 32 requests, and kept it. It made 1.51x the router's tokens per second (95 % 1.33x to 1.64x), with mean first
tokens at 0.55x. `ironmule serve --agent on` serves this way wherever the agent's setup is `route`, and `/health`
shows the mode, the dreams and the switches under `paths`.

On a fast load change it is too slow (AG6-Q, 2026-11-08, same card and model). It served a 48-request spike in
`route` and switched to `stacked` only at the spike's last request, or never. It then held `stacked` for the next 64
quiet requests, about 190 s, with first tokens 1.9x the single path's. A rule on the card's observed state alone
came within 6.3 % of the best fixed mode's tokens per second and 11.7 % of its first tokens in every phase. BACKLOG AG7 proposes making that rule the agent's starting
mode.

Measured on hardware (AG4-C run 3, one cloud CPU, `SmolLM2-135M-Instruct-8bit`): the agent chose `dequantize` by
itself and served 24.5x the default's tokens per second (95 % 12.3x to 31.3x), first tokens about 20x sooner, on
bursts and spike traffic. LIMITS has the validity domain.

The agent also keeps a list of every lever it knows (`plan_choice.LEVERS`, `/health`'s `agent_setup.levers`). For
each one it says whether it was used, measured and not chosen, not eligible on this machine, or not built yet. The
levers read from HyperQwen, Strata and vLLM (BACKLOG HQ1) are listed there as planned:
- tensor-core rows for the stacked plans;
- speculation from the request's own context, its window chosen by a policy learned online (Strata's);
- chunked prefill;
- a paged KV cache;
- an 8-bit KV cache;
- an expert cache for mixture-of-experts models.

None of them changes or shrinks the model's weights. That is the owner's boundary: only how the model runs on the
hardware changes.

Each joins the same gate and rule once it is built.

The choice is stored per hardware and model, and `/health` shows it under `agent_setup`. The owner allowed the
agent to choose plans that change answers slightly, if they pass its gate. A named `--engine` or `--compute-dtype`
is always kept. While a stacked engine serves, the online agent does not attach yet, because it does not model slot
reuse (BACKLOG AG4, step 3).

## How far ahead it checks: speculation (SP2)

Where it serves the CPU stacked engine, the agent also learns, while it serves, how many tokens to check at once
(`ironmule/speculation.py`). Each step it copies drafts from the request's own prompt and answer (the tokens that
followed the last time its latest words appeared) and the plan checks them as further rows of the request's own slot
in the same step. A request keeps the drafts up to the first one the model would not have written, plus the model's
own next token, so no token changes. How many drafts each request gets is a decision made every step from what it has
learned on this machine: the time of a step by its rows, and how often drafts of each match length were right
(Strata's draft policy). It speculates only where that is expected to make more tokens per second than a plain
step, and not at all when the engine's rows are full. SP1-0 measured why it must decide per request: at a fixed
window the same speculation made one prompt 5.7x faster and another 10 % slower. `/health` shows it under
`speculation`. Its hardware test SP2-H (2026-11-10, this container's CPU, SmolLM2-135M) did not pass its bound: one request at a time it made 1.40x the tokens per second on average (95 % from 1.21x, bound 1.3), with every answer equal and no harm under load, so it is opt-in (`--speculate context`) and the agent does not turn it on by itself. Its policy version 2 (SP3) drafts only from repeats of at least 2 tokens and prices plain and speculative steps apart; its hardware test is SP3-H in the backlog.

## llama.cpp's kernels on a CPU (GG1)

LC1 found llama.cpp 5-14x faster than IronMule's own CPU path on the same model and CPU. With `--engine ggml --gguf
PATH`, IronMule serves the model through llama.cpp's kernels in process (`ironmule/ggml_engine.py`) and adds its own
scheduling and the speculation policy version 3, which learns per row count what a step costs (median of recent steps,
each size probed a few times first) and per match length how often drafts are right. On ggml a step of several
positions takes another kernel path than one position, so answers can differ slightly; the owner allowed that behind a
quality gate (2026-11-10), which the engine passes at every start or it refuses. The agent does not choose this engine
by itself yet (GG2): a `--gguf` names it, and then its own agent (AG10, below) decides how it serves.

**How many requests at once: the engine decides (SEQ1, 2026-11-13).** The number of sequences the engine serves at
once is its largest speed lever on a CPU with AMX (ROW1: 32 sequences give 2.4x the tokens per second of 8 on Qwen3-4B),
and it costs memory: every sequence holds a KV cache as long as the longest request it promises (2,048 tokens unless
`IRONMULE_GGML_SEQ_CTX` says otherwise). At its first start on a machine the engine chooses it like the rest of its
setup (`choose_sequences` in `ironmule/ggml_engine.py`):
1. *It looks:* the memory it may use (the host's, narrowed by the container's cgroup limit, with reclaimable file cache
   counted as free), the GGUF's size and the bytes one KV cell takes (from the GGUF's metadata). Of 8, 16, 32 and 64
   it keeps the widths whose KV cache fits beside the model and a reserve of a tenth of the memory (at least 1.5 GiB).
2. *It tries:* each width that fits, on one loaded model, one context at a time: every sequence a 64-token prompt, then
   16 decode steps of all of them, median of 3.
3. *It decides:* from 8 upward it doubles while a doubling buys at least 10 % more tokens per second (`WIDTH_GAIN`);
   memory that buys less is not spent. A failed measurement keeps 8, as before.
4. *It remembers:* per GGUF, settings, request length, engine version and machine; a changed machine or model is
   measured again. `/health`'s `ggml.sequences` shows the choice, the widths that fit, the measured rates and why.
`IRONMULE_GGML_SEQS` set by hand overrides it. A finished request's KV stays in its sequence, and a new request takes
the free sequence whose kept tokens share the longest prefix with its prompt (`IRONMULE_GGML_PREFIX_REUSE=0` turns it
off), as `llama-server` does.

## The ggml engine's own agent: learning, dreaming, switching (AG10, 2026-11-13)

SEQ1 chooses the width once, at full load and at a short context. Which configuration serves best also depends on the
traffic: KV1 (docs/LAB.md) measured that on a CPU a decode step at long contexts is dominated by attention, and that
llama.cpp's flash attention is the cheaper setting at short contexts while its matmul attention (flash attention off)
is cheaper per attended token. `ironmule/ggml_agent.py` makes that choice while it serves, in the engine's worker:
1. *Configurations:* every width SEQ1 measured, each under the setting the quality gate chose and under llama.cpp's
   default (which passes the gate by definition). The measurement also times a full step at a 512-token context, so
   each configuration's start model knows its price per attended KV token. Nothing the gate did not pass is ever a
   configuration.
2. *World model, online:* a step's seconds = a + b x decode rows + c x prompt tokens + d x attended KV tokens, per
   configuration, by recursive least squares with forgetting, from every step the engine serves (the hot path only
   appends one tuple; the fit runs while idle). The startup measurement's points (one row, full width, a longer
   context, a prompt alone) are folded in at every idle call, so a fit never drifts where live steps carry no
   information. A configuration is priced by its live model after 200 live steps; before that by its start model
   times a live factor (its own after 20 steps, else that of the configuration that served most), so a configuration
   measured once is never compared with one fitted on a busier live machine.
3. *Traffic memory:* gap, prompt and answer length and the prompt tokens actually computed (after prefix reuse) of
   each finished request, no text; persisted per GGUF, machine, engine version and GPU offload in `ggml-agent/` of the
   state directory.
4. *Dreaming:* after 3 s without a request, and only when new traffic came since the last dream, windows of 32
   recorded arrivals (gaps scaled 0.5-1.5, three in four from the last 256) run through a simulation of the engine's
   continuous batching under every configuration's price, on the same windows.
5. *Switch and guard:* a configuration replaces the incumbent when the 2.5 % bootstrap bound of its dreamed
   tokens-per-second gain is above 3 % and its mean first token is at most 2 % later. The context is rebuilt on the
   loaded model in that idle spell (a request that arrives meanwhile waits for it, a few seconds at most). While the
   switch is on trial the live steps keep correcting its model; when a re-dream no longer shows a gain the agent
   switches back and does not try it again for 2,048 requests.
`/health` shows it under `ggml_agent` (what serves, the world models, the last decisions and why);
`IRONMULE_GGML_AGENT=0` turns it off (SEQ1's choice then serves alone). With `--agent on --gguf FILE` it is the agent;
the group agent of AG1 does not attach to this engine, whose continuous batching it does not model.

**FA1: a faster attention kernel under the engine.** `native/ggml_patches/fa1` replaces llama.cpp's CPU flash
attention for f16 KV caches on AVX-512 CPUs: the query heads that share a KV head are computed together, keys go in
blocks with one softmax rescale per block and vectorized exponentials, and V is accumulated in f32 (llama.cpp's
accumulates in f16). On a 4-vCPU Xeon with Qwen3-4B a decode step at 16 sequences of 1,024 tokens took 0.481 s
instead of 0.845 s; its logits were closer to the non-flash reference than llama.cpp's own flash attention, and its
perplexity ratio to llama.cpp's default was 1.00001. `python tools/build_ggml.py` builds llama.cpp with it (and with
ROW1's AMX kernels for Q8_0 weights, which serve only with `ROW1_GGML=B16` set); built from scratch and checked with the
engine's real-GGUF tests on 2026-11-13.

**A draft model for one or two requests (DM1, 2026-11-14).** One request alone sits at the memory bound (RL-MATH in
LAB.md: 92 % of what the memory can deliver), so it can only gain by getting several tokens out of one read of the
weights, and with the ROW1/FA1 kernels a verification row costs 2-3 % of a step. With `--draft-gguf SMALL.gguf` a small
model of the same family (the same tokenizer; checked at the start, else it is refused and `/health` says so) drafts
the next tokens and the served model verifies them in its step; a draft is kept only where it is the served model's own
next token. How far to draft is decided token by token by the optimal myopic rule (`ironmule/draft_model.py`): one more
draft costs a draft call and a row and adds q x p expected tokens; it goes on while that beats the step's rate so far.
p comes from the draft model's own confidence through a calibration learned online (accepted or not, by confidence
bin), the costs from this machine's measured draft calls and steps, and a direct check stops drafting at a load where
steps with drafts measured slower than plain ones (tried again every 200 steps). It drafts while at most
`IRONMULE_GGML_DRAFT_MAX_RUNNING` requests run (default 2: on a 4-vCPU Xeon it gave 1.6x for one request, 1.36x for
two and nothing from three on). `/health` shows it under `ggml.draft_model`.

**GPUs (MG2).** A llama.cpp build with a GPU backend (`tools/build_ggml.py --cuda 75`) offloads every layer
(`IRONMULE_GGML_GPU_LAYERS` sets fewer). `--gpus all` runs one engine per GPU, each seeing only its own (data parallel);
without it, several visible GPUs share one model by layers (`IRONMULE_GGML_SPLIT=layer|row|none`). The width is then
chosen within the GPU's memory. Not yet measured on a GPU: the Kaggle harness `lab/mg2/kaggle_gpu.py` is ready.

## What it knows about the hardware (AG3)

The world model learns what this machine's work costs. The agent also knows the machine's architecture
(`ironmule_product/hardware_profile.py`): the device, its published peak memory bandwidth, and the weights a
decode step reads. From these it knows the decode floor, the fastest a memory-bound step can be. Its status
reports how much of the bandwidth a measured step uses, the most a faster kernel could gain, and how much a
further session in a round adds. On a T4 that showed about 55 % bandwidth use, so up to about 1.8x is left in
kernels, and a further session costs 91 % of a step, so grouped batch-1 reads the weights once per session. Both
are levers outside the agent's own actions (BACKLOG AG3).

## Running it

```bash
ironmule serve --agent on                 # one GPU, or the CPU
```

```bash
ironmule serve --agent on --gpus all      # one agent per GPU
```

```bash
python tools/build_ggml.py                 # once: llama.cpp with IronMule's kernels; prints two exports
ironmule serve --agent on --gguf Qwen3-4B-Q8_0.gguf --model Qwen/Qwen3-1.7B  # a registered model with the GGUF's tokenizer
```

`GET /health` and `GET /v1/ironmule/dispatch` show each agent: what serves and its measured gain, the world
model, the length model's `lengths` block, the planner, the candidate and its triplets, and decision counts and
times. `POST /v1/ironmule/dispatch` with `{"action": "off"}` withdraws and pauses every agent, and `"on"` resumes
learning. Learning lives in `agent/` in the state directory, one file per binding.

## Code map

| Path | What it holds |
| :-- | :-- |
| `ironmule_product/agent.py` | the agent: world model, planner, replay qualification, judge, decisions, state |
| `ironmule/ggml_agent.py` | AG10: the ggml engine's agent: step-cost model, traffic memory, dreaming, switch and revert |
| `ironmule/draft_model.py` | DM1: the draft model's stopping rule, calibration and KV bookkeeping |
| `ironmule/ggml_engine.py` | GG1/SEQ1: llama.cpp's kernels in process, continuous batching, the gate, the width choice, GPU offload |
| `native/ggml_patches/`, `tools/build_ggml.py` | FA1 and ROW1 kernels for llama.cpp, and the build that applies them |
| `ironmule_product/length_model.py` | AG2's online answer-length model |
| `ironmule/plan_choice.py` | AG4: the agent's own choice of numeric plan and stacked engine, once per machine and model |
| `ironmule_product/path_router.py` | AG4: routes each request between two paths on one card, learning their costs online; AG6: `PolicyRouter` dreams which serving mode to use |
| `ironmule/dense_gemma.py` | KB2: up to eight 4-bit Gemma 3 requests stacked into one decode step |
| `ironmule_product/hardware_profile.py` | AG3: the device, its bandwidth, the decode floor and how much of it is used |
| `ironmule_product/learned_dispatch.py` | the policy format the agent shares with LD1 (`Policy`, `View`, order key) |
| `ironmule_product/service.py` | where the scheduler asks the agent, feeds it finished groups, and replays |
| `ironmule_product/device_pool.py` | several GPUs: placement, one agent per card in health and control |
| `ironmule_product/cli.py` | `_agent`, the binding, and attachment per card |
| `training/` | the simulator, the Kaggle harness and the RL4 agent: [training/README.md](../training/README.md) |
| `tests/agent/` | the agent's, simulator's and harness's tests |

## Where it stands

- **CPU, llama.cpp's kernels (2026-11-13).** With FA1 under the ggml engine, IronMule served Qwen3-4B Q8_0 with a
  479-token system prompt at 1.19x / 1.75x / 1.65x the tokens per second of the better stock `llama-server` setting at
  1 / 8 / 16 requests on a 4-vCPU Xeon, first tokens sooner, answers as good (LAB.md, FA1). AG10 decides that engine's
  configuration; its switching is shown in simulation (AG10 S), not yet under a changing real load.
- **No gain from order and width on the group engines (AG1).** Through AG1 V4 (T4) and V5-CPU (Kaggle CPU, cut by the
  12 h limit) those actions (order by limit, width) had no headroom over the best fixed rule above the noise; the
  gains below came from the engine (stacking, kernels), which the agent chooses.
- **AG2's simulation passed.** With real prompts and one limit for all, the agent learned answer lengths (rank
  correlation 0.63), qualified "shortest predicted first" by itself, and then served 6.9 % (T4 price) and 8.2 %
  (CPU price) below the best fixed rule in hindsight.
- **Tokens per second come first (owner, 2026-11-03).** Order cannot add them at a fixed capacity, so V-G12 was
  replaced before its data. The lever is stacking requests into one step: KB2 builds that plan for 4-bit Gemma 3
  (`ironmule/dense_gemma.py`), and the agent then decides its width.
- **KB2-0 on a T4 (2026-11-07): fast and exact, but not through the quality gate.** Six stacked rows made 2.006x
  today's tokens per second for Gemma 3 12B at 4 bit, bit for bit like each row alone, but the gate's main path
  failed (perplexity ratio up to 1.177 at 95 %), so KB2 stopped by its rule (BACKLOG KB2). KB2-D then traced the
  failure to chunks without BOS, where every path is unstable, so the gate changed for every plan (GATE2). KB2-G
  gated the same plan again on BOS-led chunks: the main path passed (0.9988, 95 % up to 1.0008), but the long path,
  4 chunks past the window, did not (95 % up to 1.0080). KB2-L repeated the long path with 16 windows and it passed
  (0.9973, 95 % up to 1.0012). The plan is now a candidate opt-in plan, and AG5-G measures it through the product.
- **AG6-H on a T4 (2026-11-07): the first hardware gain from learning while serving.** Dreaming over the traffic
  it served, the agent switched from routing to stacking and made 1.51x the tokens per second of the router it
  replaces (95 % 1.33x to 1.64x), first tokens 0.55x. It found one switch by itself; switching back where another
  mode wins is not shown yet. The run's data also exposed a worker defect, now fixed: from a batch's second run
  on, a joining request was rejected as too long for the engine.
- **AG5-G (the agent's own width on Gemma 3 12B) has no serving result yet.** Runs 1 to 3 stopped in the agent's
  startup measurement (out of memory, a timeout, out of memory again), each fixed. Run 4 measured everything and
  chose by its rule not to stack: width 6 made 2.12x, but one request alone 0.897x, below AG4's 0.9 floor.
- **KB3 (2026-11-07): the Gemma plan's attention, rewritten bit for bit.** FA2 version 3 computes the same values in
  the same order with a warp per query head and coalesced key loads; on a T4 its outputs were bit-identical, and one
  request alone made 0.928x the native path, against 0.884x before. AG5-G runs again on it (run 5).
