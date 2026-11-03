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
