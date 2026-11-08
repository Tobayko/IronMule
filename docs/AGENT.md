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

`GET /health` and `GET /v1/ironmule/dispatch` show each agent: what serves and its measured gain, the world
model, the length model's `lengths` block, the planner, the candidate and its triplets, and decision counts and
times. `POST /v1/ironmule/dispatch` with `{"action": "off"}` withdraws and pauses every agent, and `"on"` resumes
learning. Learning lives in `agent/` in the state directory, one file per binding.

## Code map

| Path | What it holds |
| :-- | :-- |
| `ironmule_product/agent.py` | the agent: world model, planner, replay qualification, judge, decisions, state |
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

- **No hardware gain yet.** Through AG1 V4 (T4) and V5-CPU (Kaggle CPU, cut by the 12 h limit) the agent's
  actions (order by limit, width) had no headroom over the best fixed rule above the noise.
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
