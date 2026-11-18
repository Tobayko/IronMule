# The IronMule agent

The project's main goal (BACKLOG, "The main goal"): an agent that decides on its own how the model's requests
run on the hardware, keeps learning by reinforcement learning while it serves, decides in microseconds, and
serves a decision only after it proved a gain on the machine it runs on. Learning updates the Pilot, the scheduler
and their hardware cost estimates. Every inference run uses the supplied language-model weights. Execution paths
that can change generated tokens must pass the applicable answer-quality checks.

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
   switch is on trial the live steps keep correcting its model. A re-dream that loses its gain triggers reversion
   and a cooldown of 512 requests for that candidate. An earlier guard compares actual worker throughput and mean
   time from admission to first token in busy windows with at least eight completions. It matches windows by
   request count, arrival spacing, prompt distribution, computed prompt tokens and output length. A trial below
   85 % of its incumbent's throughput or above 125 % of its latency in comparable windows reverts while idle.
   Recent measured successes and failures remain bound to the traffic shape, config and machine. After this
   early reversion, the restored incumbent must see 64 new completions in consecutive matching windows before a
   new trial. Arrivals in waves of fewer than eight accumulate toward a quiet-traffic trace, including single
   requests. If the mix keeps changing, the hold expires after 512 new completions; the next dream then uses the
   post-rollback traffic. An interrupted rollback resumes at the incumbent.
`/health` shows it under `ggml.agent` (what serves, the world models, the last decisions and why);
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

**GGUF on GPUs.** With the CUDA or experimental ROCm build, IronMule uses the GPU devices registered by llama.cpp.
`ironmule doctor --ggml cuda` and `ironmule doctor --ggml rocm` each execute a small graph on the requested backend.
On NVIDIA, `--gpus all` starts one worker per selected card. Each worker qualifies its own settings and batch width
with just its card visible. With several devices visible to a single worker, the default splits the model by layer.
`IRONMULE_GGML_SPLIT` selects `layer`, `row` or `none`, and `IRONMULE_GGML_GPU_LAYERS` sets the offload count.
Width selection uses the available device memory. Cached results depend on the visible devices, model, driver and
loaded native libraries. For Radeon, build with `python tools/build_ggml.py --rocm` and use `ROCR_VISIBLE_DEVICES`
to select a card. The CUDA path has published Tesla T4 results. The Radeon path still needs a real serving run.
[GPU results](../README.md#single-gpu-inference-against-vllm)

### GPU width experiments

The portable GPU comparison produces a JSON report for a single model and device. Import it with
`ironmule learn --import-gpu-report REPORT.json --state-dir STATE`. The command requires completed GPU
comparisons with confirmed layer offload. It keeps measured time ratios and quality outcomes in a compact local
record under `STATE/gpu-width-experience`. It fits a ridge regression with two outputs (relative throughput and
first-token latency) using the calibration split and reports the mean absolute log-ratio error on independent
holdout measurements next to a prediction of equal cost. Widths with mismatched outputs receive no speed credit.

`ironmule serve --gguf MODEL.gguf --agent on --state-dir STATE` recomputes the fit from distinct trials for this
GGUF and backend, preferring matching machines and device models. Of the workload shapes actually measured, it
chooses one where the model predicts higher throughput with a first token at least as early. A previously unseen
width must fit the local memory budget. The server remeasures the chosen shape on this machine, compares locally
generated tokens with the incumbent width, and gives eligible configurations' local step costs to AG10 for replay.
Imported rates do not supply those costs or change the startup width. Gated GPU stream configurations retain their
existing policy. Explicit `IRONMULE_GGML_SEQS` bypasses startup width search.

## The Pilot: a trained decision network for every step (PI1, 2026-11-14)

AG10 chooses a configuration while idle; the Pilot (`ironmule/pilot.py`) decides on every step of the ggml engine. It
is a two-layer MLP (20 inputs, 2 x 128 tanh units, four softmax heads, about 22,000 weights) whose weights ship as
`ironmule/pilot_weights.npz` for an engine on the CPU and `ironmule/pilot_weights_gpu.npz` for one whose layers run on
a GPU (PI3; open weights; the world they were trained in and the recipe are not published).

On Linux CPU, `ironmule serve --gguf MODEL.gguf --agent on` enables IronMule's local hardware model.
`IRONMULE_PILOT_LEARN=0` disables it. Without `--agent on`, `IRONMULE_PILOT_LEARN=1` opts in.
Speculation retains its usual settings. The model applies local thread decisions only when the final backend call
contains one ordinary decode token from each participating request; prompt, draft-verification, padding and
placeholder rows keep the base Pilot decision. It starts with zero weights and learns how a thread change affects measured decode time.
Inputs include available cores, foreign load, context length, concurrent decode rows and measured row and KV costs.
Weights shared by workload and action learn across machines that use the same local state store. Additional
weights specialise the prediction for this hardware and this exact binding. Training runs during idle spells.
The database `pilot/thread-model-v1.sqlite3` keeps measured time ratios, token match results, search and qualification
decisions, and the model parameters. Both gains and regressions teach it; a token mismatch earns a negative reward.
It stores timing metadata and hardware/model identifiers; user text and generated tokens stay out of its database.

For plain decode with one to four concurrent requests in the same 64-token context band (64–511 tokens), the model
ranks thread counts to test. After 64 matching observations and three idle seconds it replays generated filler
prompts on free sequences at the observed foreign load. A candidate must generate identical tokens across four
paired comparisons, improve time by over 3% in each, and beat the A/A timing spread. Timings include both thread
switches. A new request interrupts replay between backend calls. Missing slots, pending prefills, a different
load band or heterogeneous context lengths keep the stock decision outside its qualification regime.

A local per-regime thread offset can serve after qualification on this machine. Further observations trigger
requalification and a failed comparison withdraws the offset. After a restart, saved offsets are proposals again.
Inspect `thread_learning` under the Pilot status for model updates, installed offsets and the last timing ratios.
These results concern decode time. End-to-end throughput and first-token latency need a separate serving comparison.

*Scope (owner goal 2026-11-16, BACKLOG PI3).* The Pilot runs on the CPU; the model it steers is to run perfectly on
the GPU. Since PI3 (2026-11-16) an engine on a GPU takes the GPU weights: trained from scratch in dream world v7, built
from a T4's measured costs (LAB.md, PI3 probe), they keep the running cap at its maximum and a prompt budget of 128
there. On a Kaggle T4 with Qwen3-4B Q4_K_M that halves the mean first token when documents arrive among chats (5.22
against 11.76 s, 80.3 against 75.4 tokens per second) and brings it sooner for 8 and 32 requests at once (0.145 against
0.169 s, 0.43 against 0.62 s) at the same tokens per second; for short chats alone it is within the card's spread.
Their thread decisions were not steady on the CPU, so the CPU keeps its own weights. With a draft model attached
(`--draft-gguf`; on a GPU `ironmule serve` names one of the same family) one request makes 1.23x the tokens per second
on that T4 and new requests one at a time 1.19x (DM2, LAB.md), since the draft model's stop rule no longer counts
reading a new prompt against drafting. Open under PI3: why serving steps
with uneven contexts cost up to 1.5x a block's step of the same width on a GPU, and the choice of GPU when there are
several (padding 17-30 rows to 32 was tried and dropped, LAB.md, PAD2). The CPU serving path stays.

- *What it reads (20 features):* the cores this process may use (affinity and cgroup quota), how many of them other
  programs keep busy (`/proc/stat` minus this process's own CPU time, read at most every 50 ms), this machine's step
  costs measured when the engine starts (one row, each further row, 1,000 prompt tokens, 1,000 attended tokens),
  whether a GPU runs the layers and a draft model is attached, how the last step's time compares with its uncontended
  price, the running, queued and waiting prompt tokens, the mean context, arrival and token rates, its own last
  choices and the draft acceptance seen.
- *What it decides:* the step's CPU threads (an eighth to all of the usable cores), the running cap (1-64), the prompt
  tokens a step may take (32, 128, 512, 2,048 or unlimited; the shared pool honours it since PI2) and whether the draft model may draft. None of these changes an answer.
- *Why threads matter:* llama.cpp splits each matmul across its threads and waits at a barrier for the slowest.
  When a foreign process holds a core, the thread on it runs only part of the time and every other thread waits:
  on the 4-vCPU Xeon with Qwen3-4B Q8_0 a one-row step took 2.42 s with one busy foreign process at 4 threads and
  0.136 s at 3, and 5.48 s at 4 threads against 0.194 s at 2 with two (lab/pi1/threads.py). Stock `llama-server` uses every core and fell from
  7.2 to 0.24 tokens per second for one user with one foreign process (docs/LAB.md, PI1). A process busy only part of
  the time hurts too: at 25 / 50 / 75 % of every 20 ms an 8-row step at 4 threads was 1.34 / 1.75 / 2.46x slower and at
  3 threads unharmed, while the load reader saw only 0.0-0.2 / 0.23-0.32 / 0.44-0.51 cores of it (lab/pi1/half.py).
- *How it was trained:* from scratch (random start), in a dreamed world: a simulator of continuous batching whose step
  times follow the measured costs and the measured contention collapse, randomised over machines, models, traffic and
  foreign load. The shipped weights (version 2, 2026-11-16, world v6; LAB.md, PI2): from scratch, a warm start copying a
  teacher that knows the dream's true foreign load, then PPO, 47 minutes on two Kaggle T4 GPUs (seed 70 of two). World
  v6 adds what the real machine showed: with a core left free the daemons of a quiet server run on it and are seen as
  foreign load (0.12 cores at 3 of 4 threads, about 0 at 4); the world-v5 weights before them took that for a busy
  process and stayed at 3 threads on an idle machine. On the 4-vCPU Xeon they are on par with version 1 (+0.1 % tokens
  per second over 15 load cases, ahead idle, behind for one request beside a process busy 25 % of the time) and decide
  in 0.17 ms. Version 1 (2026-11-15, world v4): a warm start that copies a teacher knowing the dream's true
  foreign load, then PPO; 49 minutes on two Kaggle T4 GPUs (seed 21 of two; the CPU run before it chose differently
  in two passes near a boundary on the real machine, LAB.md). Copying the visible-load rule instead (threads = free cores)
  left two from-scratch runs at 4 threads beside a half-busy process, where 3 are 1.4x faster: that rule cannot see
  such a process. The generation before was trained on two Kaggle T4 GPUs (67 minutes) and fine-tuned on this CPU
  (29 minutes). The world and the recipe are not published.
- *Safety:* without the weights, with a feature outside the range it was trained on (|x| > 4 or not finite), or with
  `IRONMULE_PILOT=0` the engine keeps its defaults (all usable cores, no cap below the engine's, no prompt budget,
  drafting allowed). `/health` shows it under `ggml.pilot`: whether it is active and why not, the cores, the foreign
  load it saw, its current decision, decision count and mean decision time.

*Measured (2026-11-14, 4-vCPU Xeon, Qwen3-4B Q8_0, one run each; LAB.md, PI1).* Through the HTTP API, with collective
speculation off so that only the Pilot differs from stock `llama-server` (all cores): one user 9.33 / 7.07 / 5.46 tokens
per second with 0 / 1 / 2 busy foreign processes against 7.23 / 0.24 / 0.13, eight users 58.8 / 46.2 / 35.2 against
23.3 / 1.41 / 0.68; first tokens 0.3-0.7 s against 0.8-149 s. It chose 4 / 3 / 2 threads; answers were token-identical
at 4, 3, 2 and 1 threads. On an idle machine it costs nothing measurable (same engine, alternating: 9.81 against 9.81
tokens per second for one user). Those numbers are from the first shipped weights. The current ones (in the engine, `real.py`, 64 tokens, Qwen3-4B
Q8_0) choose the same threads and also give up a core when one program keeps a core half busy: 7.65 / 53.8 tokens per
second for 1 / 8 requests against 5.26 / 38.4 with the defaults (the weights before: 7.54 / 51.4); with one busy program
7.45 / 52.0 against 0.31 / 2.41 (before: 7.19 / 50.8); with two, 2 threads and 6.11 / 30.9 against 0.19 / 1.46 (16-token answers). Idle it costs nothing: in one
process, attached and detached in turn, 10.04 / 65.6 against 10.05 / 65.0 tokens per second. In the dreamed world (world v4,
240 s, 1,024 random machines) they make 12.5 % more tokens than the defaults with 45 % less waiting, and 1.8 % less
waiting than the hand rule (reward -1408 against -1454; defaults -2547; the weights before -1434).

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
ironmule serve --agent on --gguf Qwen3-4B-Q8_0.gguf  # the GGUF's own tokenizer and template; or --model to bind one
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
| `ironmule/pilot.py`, `ironmule/pilot_weights.npz`, `ironmule/pilot_weights_gpu.npz` | PI1, PI3: the Pilot, its features, decision and open weights (CPU and GPU) |
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
  configuration. The local CPU HTTP transition experiments in LAB.md measured both a switch and a rollback under
  short chats and documents; results depend on the tested workload and machine.
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
