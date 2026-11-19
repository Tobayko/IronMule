# Speed backlog

What IronMule is trying to reach, the work that is open, and what was measured and rejected. Nothing here is a
result unless it says so and names an experiment.

Cleaned up on 2026-11-16 (owner: "we clear out, so it is clear again what we want to reach here"): entries that
were answered left the file, as its own rule always said. Each keeps one line under "Closed entries" so that an
identifier in the code or the documents still resolves. The full text of every removed entry is in this file's
git history before that date. Lab notes are in [LAB.md](LAB.md), the validity domain in [LIMITS.md](LIMITS.md),
the phases to a product in [ROADMAP.md](ROADMAP.md).

## The main goal (maintainer decision, 2026-10-30)

The owner made one entry the project's main goal: a working IronMule agent. It decides on its own how the
model's requests run on the hardware, reaches a measured gain there, keeps learning by reinforcement learning
while it serves, and decides as fast as a small decision model. Every rule below still applies to it: it
keeps answer quality (since 2026-11-10 the same answer quality, not identical tokens; see below), a gain counts only
when it is measured, and it serves nothing it has not proven on the machine it runs on.

**What "gain" means (owner, 2026-11-03).** Tokens per second come first. Waiting time counts too, but less, and
the time to the first token must not get worse. AG1 and AG2 so far judge a schedule by each request's time in
the system plus its time to the first token. Ordering lowers that, but at a fixed capacity it moves no tokens per
second: the same work runs on the same engine. Tokens per second move only through what the engine can run at
once (width, true batching) and how fast its kernels are (AG3).

**The agent sets up what was found by hand (owner, 2026-11-03).** "Things we already found out, the AI should
learn and set on its own." The settings found by hand runs so far are the numeric plan (`native` about 6x on 4-bit
checkpoints of an emulating GPU), the stacked engine (H10: 48 % cheaper), the width (w8 on the dense plan, w2 on
the tuned engine), the order and the class-A knobs. The owner was asked whether the agent may also choose plans
that change answers slightly. The answer was **yes, always**: under `--agent on`, a plan that passes the agent's
own on-machine gate may serve without a further switch. The gate is the numeric plans' perplexity gate against
the reference path (bound 1.005), the plan's load probe, and a speed check. This replaces "it changes no answer"
for plan choice inside agent mode (AG4). It is never silent: the chosen plan, its gate and its reason are in
`/health` and the agent's status. The agent stays opt-in. Without `--agent on`, the caller's plan is kept as
before.

**The CPU is enough to build the agent (owner, 2026-11-10).** "The CPU is fully sufficient for this project to make
an AI that optimises itself; so we can keep training without limit." What that means under the rules above: the
agent's training (its simulator, planner, dreaming and replay) needs no accelerator, and a Kaggle CPU session or
this container spends no GPU quota, so training and proving the agent may run entirely on CPUs. A serving gain
still counts only on the machine it was measured on: a CPU result holds for that CPU and model, and says nothing
about a T4 or a Mac. CPU6 makes the CPU fast enough for that.

**Every number of devices (owner, 2026-11-10).** "Mind multi-GPU, the same as a single GPU: whether three, four or
a hundred GPUs are there, everything is to be covered." Whatever the agent can do on one device it must do on N
devices, and the product must neither refuse nor silently drop a setting because a machine has more than one.
MG1 lists where that does not hold today.

**Answer quality, not identical tokens (owner, 2026-11-10).** "We change the token-identical rule. What matters is that
the answer is correct, factually correct; it need not be identical. If the answer is still right, that is fully
enough. Change that for this project: we go for full performance optimisation with the same answer quality." What
that means from now on, for the whole project and not only for `--agent on`:
- A path, plan, kernel, batching or speculation may change tokens. Bit-identity is no longer a requirement and no
  longer a reason to reject a lever.
- It serves only after its answer quality was shown to be the same on the machine: the perplexity gate against the
  reference (bound 1.005, as NUM1 and AG4), and, once built, a correctness check on questions with checkable answers
  (QG1, below). It is never silent: `/health` names the path and its gate.
- A gain still counts only when measured as before; a quality loss beyond the gate disqualifies it, as a correctness
  failure did before.
- Tier 0 entries rejected only because tokens changed may be reopened under this rule with a quality gate (for
  example HW3's prompt-lookup speculation on CUDA and TB1's true tensor batching on the T4), under a new protocol.

**The quality bar is llama.cpp's, speed is what counts, the model stays as it is (owner, 2026-11-10, later the same
day).** "Continue with IronMule so that the hardware is used better without changing the model; only speed counts,
and for quality take the same quality as llama or somewhat better." Read as: (1) the main line is hardware
optimisation of a given model, no retraining and no changed weights; (2) the measure is speed; (3) a path's answer
quality must be at least llama.cpp's on the same model: on the perplexity gate's text its perplexity must not be
higher than that of llama.cpp's default setting on the same GGUF (ratio at most 1.000, both numbers deterministic).
This replaces the float32-on-MLX reference for the ggml engine; a GGUF llama.cpp itself serves is by definition at
llama.cpp's quality. The owner was told this reading and may correct it.

**llama.cpp's own options count as llama.cpp's quality; the three ways are taken (owner, 2026-11-11).** After GG2-H
the owner was offered three ways past parity: (1) let flash attention serve, an option of llama.cpp itself (`-fa`),
0.17 % worse in perplexity than its default on SmolLM2's GGUF, with `llama-server` given the same option for a fair
comparison; (2) a larger model on this CPU; (3) LC2 on a Kaggle T4, about 2 GPU hours. The owner's answer: "Setzt das
um" (implement it), read as all three and as the approval of LC2's GPU hours. Read for (1): a setting of the ggml
engine that is one of llama.cpp's own options (flash attention, one unified KV cache, an f32 KV cache) is at
llama.cpp's quality when its perplexity on the gate's chunks is within NUM1's bound of llama.cpp's default on the same
GGUF (the ratio's 95 % upper end at most 1.005); a comparison then gives `llama-server` the same options. GG3 and LC2
below carry it out. The owner was told this reading and may correct it.

**IronMule on the CPU, the model perfect on the GPU (owner, 2026-11-16).** First: "The LLM IronMule optimises is to
be optimised for the GPU, but also the CPU; we don't throw that away." Then, correcting the first reading: "No,
IronMule is to run on the CPU, and what is optimised is to be perfect for the GPU." Read as: (1) IronMule's decisions,
the agent and the Pilot, run on the CPU, and stay fast enough there to decide every step in milliseconds; (2) the
served LLM is what they optimise, and its main target is the GPU: the Pilot is to steer it there to the best speed the
card allows (PI3); (3) the CPU serving path, the ggml engine with the Pilot as it decides there today, stays with its
tests and numbers; (4) a gain counts only on the hardware it was measured on. The owner was told this reading and may
correct it.

## Open: the GGUF engine and the Pilot

### QG1 — A correctness check for answer quality

**Why.** The perplexity gate measures how likely the model finds a fixed text; the owner's criterion is that answers
stay factually correct. QG1 adds that check: a fixed set of questions with checkable answers (arithmetic, facts,
extraction from a given context), each answered greedily by the reference path and by the candidate; the candidate
passes when its share of correct answers is not lower than the reference's by more than the reference's own A/A
noise. It joins the perplexity gate for every path that changes tokens; its protocol is fixed before it serves a
decision.

The GGUF gate now compares 18 greedy short answers (six each for arithmetic, extraction and general knowledge),
with a repeat of the reference and a 32-token limit. QG1 v3 scores the entire final answer. It accepts an
exact value or a verified restatement, such as a complete equation with the requested operands and result,
a value assigned to the requested slot, or one of the listed affirmative factual statements. Each category
must score at least as well as the lower of
the two informative reference scores. An arm with failed generations cannot qualify. Changed GGUF settings must
pass both this check and the existing perplexity bound; the stock default remains available. QG1 covers the regular
settings and the separate shared-pool and stream checks. The T4 KG5 shape check (`1024,32,8`) uses 32 scored
requests with 960–970 prompt tokens per stream on both arms, and the flash candidate must register an actual
kernel launch during the answer check. Its small set checks for regressions in these tasks;
it does not establish general factual accuracy. The MLX numeric-plan and stacked scheduling protocols keep their
existing criteria. See `ironmule/factual_gate.py` for the prompts, grading and seeded holdouts.

### PI3 — The Pilot decides on GPUs too (owner goal 2026-11-16; accepted by the owner 2026-11-17)

**Why.** The owner's goal above: the Pilot keeps running on the CPU, and the model it steers is to run perfectly on
the GPU. Today the Pilot's thread head means nothing when the layers run on a GPU, and the
other heads (running cap, prompt budget, draft) were trained only on CPU step costs. On a T4 the engine's costs have a
different shape: small batches are dear (PAD1), a full prompt costs little next to decoding, and the draft model and
generated kernels change the price of a step (MG2, KG2).

**Mechanism.** Measure a GPU machine's step costs at startup as on the CPU (one row, each further row, prompt and
attended tokens, per GPU), give the dream world a GPU cost model built from those measurements, and add the heads a
GPU needs: the running cap and prompt budget per GPU, padding to a dear-batch boundary (PAD1), draft on or off, and
which GPU a new request joins when there are several (MG1). Train from scratch as PI1/PI2, then measure on a Kaggle
T4 x2 against the engine's defaults, `llama-server` and vLLM (MG2's harness), with the facts checked against the
reference engine.

**Kill.** If on the T4 the Pilot's decisions are not faster than the engine's defaults in tokens per second, with the
first token no later, at 1, 8 and 32 requests, the GPU heads stay off and the defaults serve there.

**Cost.** GPU training and the T4 measurement spend Kaggle GPU quota and need the owner's approval before each push.

**Result (2026-11-16, LAB.md, PI3 probe and PI3).** Approved by the owner: about 3.5 GPU hours in three steps; used
1.9 (probe 47 min, training 41 min, check 27 min). On a T4 the host threads and CPU load do not matter and the cap
must stay at its maximum; the prompt budget is the decision that pays. GPU weights (`pilot_weights_gpu.npz`, world v7,
from scratch) serve an engine on a GPU. Against the kill rule above: tokens per second equal at 1, 8 and 32 requests
(36.0 / 206.5 / 486.7 against 36.2 / 205.9 / 490.0), not faster, with the first token no later (0.048 / 0.145 / 0.43
against 0.048 / 0.169 / 0.62 s); with documents among arriving chats 80.3 against 75.4 tokens per second and the first
token 5.22 against 11.76 s. Read post hoc: the rule asked for more tokens per second, the gain is in the first token
at the same tokens per second, so the weights ship as a first-token gain and the rule's tokens-per-second part stays
open. Short chats alone: 228.5 against 238.6 tokens per second, first token 0.83 against 0.68 s, inside the T4's own
spread between rounds (207-255 for the same arm). Tried and dropped: padding 17-30 rows to 32
(PAD2, LAB.md: the probe's serving steps of 17-30 rows cost 74-88 ms against 48-61 ms for 32, but a step of 17-32 tokens
measured on its own rises evenly, 29-36 ms, and blocks of 20 / 24 / 28 requests cost 58 / 59 / 65 ms a step, so the gap
came from the steps' contexts, not their width). Also tried: compacting the KV cache and bounding the kept prefix
cache (KVD1, LAB.md): equal steps on a T4, kept off. Tokens per second (DM2, 2026-11-17, LAB.md): with the draft model,
whose stop rule no longer counts reading a new prompt against drafting, one request makes 1.23x (54.1 against 44.1
tok/s, four balanced rounds after a warm-up) and new requests one at a time 1.19x; 8 and 32 requests the same, the
facts the same. So the kill rule's tokens-per-second part holds at 1 request with a draft model attached, not at 8
and 32, where the draft model rests and the step is the kernels' alone.

**Owner decision (2026-11-17).** Asked whether to go on with a kernel project for more tokens per second at 8 and
32 requests (the generated kernels of KG1 extended to steps of 9-32 tokens; vLLM AWQ makes 419 tok/s at 8 requests on
the T4, the engine about 270), the owner chose to accept the state: the Pilot decides on the CPU in 0.11-0.19 ms, on a
GPU it brings the first token sooner (mixed arrivals 5.22 against 11.76 s) and, with a draft model attached, one
request 1.23x. More tokens per second at 8 and 32 requests on a GPU is kernel work, not a Pilot decision; it stays
open as its own entry when the owner reopens it. Open: why serving steps with uneven contexts in the shared pool
cost up to 1.5x a block's step of the same width, and the GPU a new request joins when there are several.

**Reopened by the owner the same day: KG3 (2026-11-17, LAB.md).** The cost was llama.cpp's flash-attention kernel
on the T4 reading every cell of the shared pool for every query token (no skip of masked blocks). One KV stream per
request, with three engine changes so that a step stays one pass over KV streams (a new prompt in its own call, the
same number of drafts per stream, placeholder rows for a free id between running ones) and PAD3 (every stream padded
evenly to the size measured cheaper): 1.28x the pool at 20 requests, 1.78x at 32, 1.59x for 32 short requests, 1.54x
with documents among arriving chats (first token 2.1 against 6.5 s), 1.08x for chats alone, 1.04-1.21x at 8 and
0.98x at 4; 49 of 50 facts either way. On a GPU `ironmule serve` now serves that way where its own gate passes
(`IRONMULE_GGML_STREAMS=0` keeps the pool). Through `ironmule serve` itself on the T4 (after the SEQ1 fix, LAB.md): 32
requests 518 against 422 tok/s (1.23x), 8 requests 238 against 260 (0.92x), one request the same, 49 of 50 facts;
the pool there served 8 sequences because its width measurement fails with llama.cpp's default setting on a GPU.
Still open: that measurement, the streams' 8-request step through the server, vLLM AWQ's kernels, the first token of
a block of requests (0.73 against 0.65 s at 32), and the GPU a new request joins when there are several.

**Owner decision (2026-11-17, after KG3 end to end).** Asked whether to let AG10 switch between the pool and one
stream per request by load (the pool wins at few requests when its cache is small, the streams at many), the owner
chose to leave the state as it is: streams on a GPU from 16 sequences on, the pool below. The points above stay open
until the owner reopens them.

**PB1 (2026-11-17, LAB.md).** Measured after another session suspected the GPU Pilot's prompt budget (128) of
costing 4-5x on bursts of medium prompts: refuted, the low numbers came from short answers. On the T4 with one KV
stream per request the budget cost 2-9 % on blocks of 8-32 medium prompts and gained nothing elsewhere (facts 49 of 50
in every arm), so with one KV stream per request on a GPU the engine no longer holds it (every prompt has its own call
there); the shared pool and the CPU keep it. Confirmed on the T4 after the merge: the Pilot equals the Pilot-off
engine on the blocks (32 medium prompts 102.3 against 102.4 tok/s), 49 of 50 facts.

**KG4 (2026-11-17, owner's choice of the kernel project, LAB.md).** A probe showed where a T4 step of 8-32 requests
goes (Q4_K matmuls at 97 GB/s, 8 rows dearer than 16); a Q4_K kernel on the tensor cores (`mma.sync`, generated and
tuned per GPU by `ironmule/kernel_gen.py`, KG_VERSION 3) now serves 1-32 tokens where the engine measures it faster.
End to end on the T4 against the same engine without it: 8 requests 1.12x, 32 requests 1.07x (280 against 251, 642
against 602 tok/s), 49 of 50 facts either way. A profile showed 3.7 of the isolated 5.4 ms at 32 tokens arriving in the
engine and Q6_K as the next cost; Q6_K on the tensor cores (KG_VERSION 4, its own mask, kept where it measures faster:
up to 8 tokens) brings 8 requests to 1.25x (310 against 249 tok/s), 1 request 1.37x, 32 requests 1.04x, 49 of 50 facts
either way. Open: attention (10-25 ms of a 32-request step) and Q6_K beyond 8 tokens on the same card. A new GPU
training on the KG4-era costs was checked first in the dream world (LAB.md, PI4): the shipped weights are already at
the best of the remaining decisions there, so the owner chose not to spend the quota on it.

### NEXT — What would take the GPU further (priorities measured 2026-11-17)

Measured on a Kaggle T4 with Qwen3-4B Q4_K_M after KG4 (LAB.md, KG4 and PI4); each item needs the owner's approval
before it spends GPU quota.

1. **KG5: attention for one query per KV stream.** The largest single cost left: llama.cpp's flash attention takes
   10.5 ms of a 49 ms step at 32 requests and context 300, about 25 ms at context 1,000, reading the KV cache at about
   134 GB/s of the card's 320. On Turing with f16 KV and grouped heads llama.cpp always picks its tensor-core kernel
   (`fattn.cu`: `mma_faster_for_bs1`); a decode kernel that reads each KV head once for its 4 query heads could near
   halve it. Needs a hook in `ggml-cuda/fattn.cu` (a patch beside kg1) and one rebuild of llama.cpp (about 57 minutes,
   kept as a kernel output as KG2's build is). Kill: no 10 % shorter 32-request step at context 1,000.
   The experimental implementation now builds for SM75. The [bounded KG5 experiment](../lab/kg5/README.md)
   can reuse an eligible KG2 binary via a small probe. An actual T4 result is outstanding.
2. **The library's activation quantization.** `quant_frag` costs 1.5 ms of a 32-request step against llama.cpp's 0.8
   (one warp per 32 values, 216 launches). Wider work per warp, or one quantization shared by the matmuls that read the
   same activations (Q, K, V; gate and up), with a check that the buffer is unchanged. Worth about 0.7-1.5 ms (2-3 %).
3. **Q6_K from 16 tokens on.** llama.cpp's own kernel wins there (5.25 against 6.60 ms at 16 tokens); the output
   matrix alone is 2.8-3.8 ms of it, read at about 100 GB/s. Worth up to about 2 ms at 32 requests.
4. **A tuning memory (idea from arXiv 2611.18856, "Self-Improvement Memory").** The paper keeps, beside each change,
   why it was tried, what it was predicted to fix and what it fixed, and continues better from that record than from the
   result alone (+2.4 points on average over ten paired coding runs; agent harnesses, not serving). For IronMule: the
   kernel library is tuned from nothing for every GPU and model (3-4 minutes on a T4 at the first start); a record of
   which variants won on which GPU and shape, and by how much, could start the tuning from the few likely winners.
   Shortens the first start, not the answers.
5. **The gap to vLLM.** vLLM AWQ on the same T4: 419 / 861 tok/s at 8 / 32 requests against IronMule's 310 / 627 (one
   GPU). Items 1-3 together would close part of it at 32 requests; CUDA graphs (off on Turing in llama.cpp, gr1 can
   force them for one token) are the other known difference.

Not worth the quota now: a new GPU Pilot training (PI4: the shipped weights are already at the best decisions left
on such a GPU).

### GPU kernel experience across startups

The CUDA generator binds its tuning report to the GGUF contents, visible CUDA devices, driver and generator/toolchain.
It retains numerical rejections as well as measured winners. A later startup reuses the library after checking its
digest, or skips a previously rejected search for the same binding. Resource or launch failures remain eligible
for a retry. A new file or device is measured again. Automatic generated-kernel tuning requires a worker with
one visible CUDA device, since the timing loop measures only that device. The end-to-end step comparison still
decides whether a built kernel is faster on the serving machine. Hardware performance of these cache changes has
not been measured.

### Local Pilot thread weights (opt-in, 2026-11-18)

The CPU Pilot has an optional hardware model that learns from scratch on real thread experiments and persists
gains, regressions and token mismatches. It prioritises experiments for one to four concurrent decodes at observed
CPU contention and context lengths. Local thread offsets serve after paired qualification and are withdrawn on a
failed recheck. See the Pilot section in
[AGENT.md](AGENT.md) and the Linux CPU smoke in [LAB.md](LAB.md). Open: exercise changing traffic with a
model-sized working set, record how often a candidate qualifies, and compare end-to-end throughput and first-token
latency with the shipped Pilot across alternating sessions.

### AG10 — The ggml engine's own agent: world model, dreaming, switch and revert (2026-11-13, fast mode; simulated and tested on CPU)

`ironmule/ggml_agent.py`, wired into the ggml bridge and `ironmule serve --gguf` (docs/AGENT.md, AG10). It chooses among
the configurations SEQ1 measured (width x attention setting, all within the gate) by dreaming recorded traffic through
a learned step-cost model, switches in idle time and reverts when the live steps disagree. The
[local HTTP experiment](../lab/ag10/run_pair.py) tested a short–long–short transition on SmolLM2-135M.
At 16 concurrent requests the agent switched from width 8 to 64 after request 65 and reverted after request 129.
During the returning short phase it delivered 45.85 against 84.68 tokens/s for fixed width 8 in one paired run.
See [the lab notes](LAB.md) for the scope and controls. The next question is whether an earlier live regression
check can shorten a bad trial while retaining profitable switches on fresh traffic.

### FA1 — IronMule's flash-attention kernel for llama.cpp on AVX-512 (2026-11-13, fast mode; shipped as a patch)

`native/ggml_patches/fa1`, built by `tools/build_ggml.py`. 2.69x a decode step at 16 x 1,024 tokens against
llama.cpp's flash attention, gate ratio 1.00001; end to end 1.19x-1.75x the better stock `llama-server` setting with a
long system prompt (LAB.md, KV1 and FA1). Open: an AMX-BF16 or AVX512-FP16 variant, prefill with blocks of queries,
other families' head layouts on hardware (Gemma 3, Llama 3.2, Mistral checked on this CPU, LAB.md).

### MG2 — The ggml engine on GPUs, one or many (2026-11-13; first numbers 2026-11-14)

GPU offload, split modes and one engine per GPU (`--gpus all --gguf`). `lab/mg2/kaggle_gpu.py` measures it on a Kaggle
T4 x2 against `llama-server` (one GPU, split, one server per GPU) and vLLM on Qwen3-4B, Gemma 3 4B, Mistral 7B
and Llama 3.1 8B. v4 (docs/LAB.md, 2026-11-14 MG2 v4), Qwen3-4B at 32 requests: one per GPU IronMule 1037-1059 tok/s
against 797 for llama-server (ahead on Mistral 7B and Llama 3.1 8B too); on one GPU IronMule 748, llama-server 520,
vLLM with AWQ 4-bit 893. At 8 requests on one GPU IronMule 150, llama-server 153, vLLM AWQ 431: vLLM is ahead on one
GPU, and CUDA's small-batch kernels make 5-8 rows dear (PAD1, measured in v5). Gemma 3's answers failed through the
harness (no BOS, fixed in v5); one-request numbers were inflated by SP4 on repeated prompts (off in v5). v5/v6 (2026-11-15): the shared pool with PAD1 on one GPU 1.32-1.41x the best `llama-server` arm at 8 requests on all four models, the same facts, 47-50 of 50 answers token for token unchanged; the server opens the pool on GPUs now. vLLM AWQ stays ahead on one GPU at 8 and 32 requests; one request is level with `llama-server` without a draft model and 1.15-1.32x with DM1 (1.09x `llama-server -md`; SG1 v3); CUDA graphs on a T4 (GR1) brought nothing (SG1 v2). Generated kernels (KG2, 2026-11-15): one request on a T4 1.27x the engine without them and 1.22x `llama-server` (57.0 against 46.6 tok/s), the same facts; on by default on GPUs where nvcc is present.

Open on this line beyond these entries ([ROADMAP.md](ROADMAP.md)): sampling, a CI job that runs the GGUF engine
against a real model, a fair `llama-server` arm under foreign load, and requests that a long run holds until it goes
idle. Done since the audit of 2026-11-15 (LAB.md): ST1, the engine loop's two searches no longer grow with the
context; KG2, generated GPU kernels for 4-bit weights, one request on a T4 1.27x.

## Open: the agent and the server on the MLX path

One paragraph each; mechanism, test and kill in full are in the history.

- **AG7 — the serving mode follows the card's observed state, request by request** (proposed 2026-11-08 from
  AG6-Q; not started). A rule on the card's observed state came within 6.3 % of the best fixed mode's tokens per
  second in AG6-Q, where the dreaming agent missed the spike. About 3 Kaggle session hours; fails go to Tier 0.
- **MG1 — every setting the agent has on one device, on any number of devices** (owner request 2026-11-10; open).
  Several settings are refused or dropped on more than one GPU; each needs its own test at more than two devices.
- **PL1 — which runtime under the agent, per platform** (owner question 2026-11-10; open). On a CPU llama.cpp was
  5.1x IronMule's best MLX path under load; since then the ggml engine serves GGUFs on CPUs and GPUs (MG2, above).
  Whether MLX stays the runtime anywhere but Apple Silicon is not decided.
- **STAB3 — a worker lost under a spike ends in bounded recovery or a prompt 503** (owner request 2026-11-08). The
  local part shipped; the hardware gate on a T4 is pending.
- **HW1 — serve from every GPU the machine has.** Built (`--gpus`); a two-worker pool served 6,048 requests without
  a failure. Throughput against one worker was not measured under the entry's kill (under 1.5x at four clients).
- **HW2 — split one unquantized model across both T4s** (tensor-parallel decode over NCCL; a numeric plan). Not run.
  Kill: NCCL unavailable from the wheels, under 1.2x on a 4B model, or the gate fails.
- **U2 — float16 matmuls with the checkpoint's own activations, for Gemma 3 on a T4.** Not run. Kill: not faster
  than stock, or its gate fails.
- **OV1 — build the next decode step before reading the token** (`mx.async_eval`). Kill: no resolved gain on top
  of compile + readback 4.
- **L2 — a LoRA-tuned small language model as the planner's prior.** Evidence against: Gemma 3 1B and 4B as
  planners failed the output contract every time.
- **MEM2 — CUDA memory of a long-running tuned server grows to the card's limit** (2026-10-15). Diagnosed: MLX's
  buffer cache of freed buffers grows to 11 GiB on a 15 GiB T4. Stays open.
- **HOL1 — a queued request times out behind a group that holds the worker until its own deadline** (2026-10-15).
- **HTTP1 — past the accept limit most refused clients see a reset, not the 429** (2026-10-15).
- **RSI2 — shorten the paired confirmation where the evidence is already clear** (2026-10-05).
- **CUDA1 — a deep knob combination crashes MLX's CUDA graph** (2026-10-05). Kill: not reproducible twice.
- **CPU5 — a visible answer sooner on a CPU** (2026-10-05): thinking off, or a small non-thinking model.

## Open: release and usability (from the reviews of 2026-08-27 and 2026-08-29)

- **`R2`** — request arrival, prefill and decode as one measurable lifecycle (separate phase timestamps).
- **`R3`** — a public benchmark that is balanced and end to end, with a stock `mlx_lm` arm in a fresh process.
- **`R8`** — correctness and packaging as automated release gates; nothing yet asserts that a first run works.
- **`C1`** — safe cache, chat and sampling expansion; sampling stays a separate seeded mode.
- **`Q1`** — a wider evidence matrix and sustained-load gates (1 h, 6 h, 24 h, bursts).
- **`D1`** — reproducible community bundles and the release supply chain (SBOM, signed artifacts).
- **`Q3`**, **`Q4`**, **`Q3a`** — evidence-bound choice of the optimizer's method, a hierarchical learner over
  execution and scheduling, and the path interaction of `fused_argmax` with the final Q2 incumbent.

## Open hypotheses on the MLX and Metal path

Written from 2026-08 to 2026-09 on an M1 Max; none is measured unless Tier 0 says so. One line each.

Tier 1, cheap and grounded:
- **`CHAT1`** — is the engine faster than mlx-lm's own generation in the chat? One observation (82.6 tok/s for
  `--engine stock` on Gemma 3 4B), no measurement.
- **`B57`** — qualify a kernel per shape, through the profile.
- **`B58`** — does the paired route survive a stream of arrivals?
- **`B59`** — is a native Metal path faster than `mx.fast.metal_kernel`?
- **`B64`** — now that `wired_fraction` can be confirmed, does it pay?
- **`B73`** — a second Mac, which four entries need.
- **`B41`** — pad the reduction dimension so the fast quantised kernel runs.
- **`B38`** — the exact Gemma 12B core-profile activation and canary.
- **`B1`** — is width 4 still the ceiling at 27B?
- **`B2`** — group the `lm_head`, and only the `lm_head`.
- **`B3`** — unroll k decode steps into one compiled graph.
- **`B4`** — wire the weights against page pressure.
- **`B5`** — fill the group on purpose (a bounded wait).
- **`B6`** — how the `M=4` cost ratio moves with model size.
- **`B26`** — Qwen3.8 27B, to separate model size from model family.
- **`B30`** — widen Qwen grouped batch-1 groups to five and six.

Tier 2, structural:
- **`B8`** — a decode loop that does not go through Python per operation.
- **`B9`** — record the decode step once, replay it every step.
- **`B11`** — pipeline layers across the group instead of whole steps.
- **`B12`** — jump over the `M=8` valley (width 16 was the best point of `E3`'s sweep; `E14b` found a token
  divergence, so correctness comes first).

Tier 3, algorithmic, with a quality cost to bound:
- **`B13`** — speculative decoding with a draft model (on the ggml engine this is DM1, LAB.md).
- **`B14`** — a draft head trained on the target's own hidden state (EAGLE, Medusa).
- **`B16`** — lower or mixed precision (changes the model; outside the owner's boundary of 2026-11-03).
- **`B17`** — quantise the KV cache; measure its share of a step's traffic first.
- **`B18`** — skip layers that do not change the answer.
- **`B19`** — use Gemma's 5:1 local-to-global layer pattern for the KV cache.

Tier 4, wild:
- **`B20`** — `lm_head` on the Neural Engine.
- **`B21`** — small projections on the CPU while the GPU does the big ones.
- **`B22`** — two processes on one GPU, as a control for what grouping really gains.
- **`B23`** — a weight layout built for `M=1`.

## Closed entries

Answered and removed on 2026-11-16. A failure is also in Tier 0 below; what shipped is described in
[LIMITS.md](LIMITS.md), [AGENT.md](AGENT.md) and [HTTP.md](HTTP.md). Where the outcome is not on the line, the
history has it.

| Entry | What it was | Outcome |
| :-- | :-- | :-- |
| AG1 | The IronMule agent: one online learner that decides, learns and proves its gain (`ironmule_product/agent.py`, `--agent on`) | Built. Simulation gates S and S2 passed on run 7 (2026-10-30). Through V4 it never beat the best fixed schedule |
| AG2 | The agent learns how long an answer will be, and orders by it | Simulation S-LP and its gate passed (2026-11-03); the hardware run V-G12 was replaced before its data |
| AG3 | The agent knows the machine's architecture | Not kept; report-only |
| AG4 | The agent chooses the plan and the engine itself (owner decision 2026-11-03) | Built (`plan_choice.py`, `path_router.py`). AG4-R run 1 on a T4: the stacked engine 2.10x, the router 1.44x the single path |
| AG5 | The agent learns the stacked engine's width | Rule and wiring built; the hardware test AG5-G did not run (KB2-G did not pass) |
| AG6, AG6-Q | The agent learns its serving policy by dreaming; quiet, a spike, quiet again | AG6-Q failed (2026-11-08): the agent missed a 48-request spike |
| AG8 | The agent qualifies for tokens per second, the first token no worse | Simulation: the planner found no width that makes more tokens per second at that price |
| AG9 | The online agent inside the stacked engine | Step S: no headroom |
| CPU6 | A fast BLAS under MLX and the stacked engine on a CPU | CPU6-M, A2, A2-K and Q passed; CPU6-A failed |
| SP1-0, SP2, SP3 | Speculation from the context on the CPU plan, with a learned window | Exact; killed at 1.23x; SP2-H 1.40x against a bound of 1.3; SP3-H failed. `--speculate context` stays opt-in |
| LC1 | IronMule's MLX CPU path against llama.cpp | llama.cpp 5.1x under load and 13.6x for one request |
| GG, GG2, GG3 | ggml as a CPU engine, 20 % above llama.cpp | Goal not reached under their protocols (0.81x to parity under load). The later lab work (ROW1, FA1, SP1; LAB.md) is what the README's numbers come from |
| CS1 | Contextual sparsity | Internal check: not on this model |
| STAB2, STAB4, REV1 | The agent never stops the server; slots, admission and recovery on the product's path; an external review's eight findings | Shipped |
| TP1 | Kaggle TPUs as a further device | Closed in Tier 0: the TPU's ceiling stayed under the bar the entry fixed (20.8 against 57.7 at one row, 180.8 against 290.9 at eight) |
| PROD1 | The product wrapper (approved 2026-09-05) | `ironmule_product` is its result |
| U1 | Unquantized bfloat16 checkpoints on CUDA without native bfloat16 | The float16 plan and its gate; see LIMITS.md |
| RL2, RL3 | One controller over the whole machine; length-aware placement and order on two GPUs | RL2-H2 invalid; RL3-H run 2 valid: no headroom, no learner built |
| X1 | A short request's answer depended on its group on the T4s | The KV capacity moved tokens on CUDA; fixed, each request decodes at its own capacity |
| RL4, RL4-S | A model-based RL agent that places and orders each request | The simulator gate failed; on spike traffic the frozen policy beat the best fixed arm |
| LD1, LD1-V | The learned dispatch in the product, opt-in | Product path built; on mixed traffic no evidence was written for RL6's policy, LD1 keeps its spike-bound evidence |
| RL6 | Group width and shortest-job-first on CUDA | Valid; `ServingPolicy.order` (`sjf`) shipped as the default candidate (18.4 %, 95 % 14.9-22.2 %) |
| HW3, HW4 | What a decode and a prefill step cost on the T4, and which knob sets keep every token | Valid; compile + readback 4 is class A at 0.918 of the decode time |
| TB1 | True tensor batching on CUDA | Not exact, and a small lever |
| HQ1 | Levers read from HyperQwen, Strata and vLLM | A list for later entries; the owner's boundary: the model is never changed or made smaller |
| KB1, KB2, KB3 | Batch-invariant row kernels on pre-Ampere CUDA; Gemma 3 at 4 bit stacked | KB1 exact in every variant. KB2-G did not pass (Tier 0); KB2-L passed, a candidate opt-in plan |
| GATE2 | Every quality-gate chunk starts with the tokenizer's BOS (owner decision 2026-11-07) | In force |
| KP1 | The class-A knob profile as the tuned engine's default | The knobs do not ship |
| H9, H10, H11 | The dense plan on the same traffic; with slot reuse; with 8,192-token slots | Dense plan recommended as opt-in (24.0 % cheaper), also with slot reuse; the 8,192-token slots do not ship |
| RL7, RL8 | A learner over scheduling and knobs together; width and order again | No knob learner; `shortest/w2/g0/sjf` stays best for the tuned engine |
| FA2 | A fixed-shape decode attention that reads only up to the offset | Measured in HW4 |
| MOLE1 | An MCP comparison for a package that is not in this repository | Dropped |

## Tier 0 — measured and rejected. Re-open only under the rule below.

- **The ggml engine at 1.2x `llama-server` with llama.cpp's options and on larger models (2026-11-11).** Experiments
  GG3-F and GG3-Q (this container, commit `d330998`; see GG3 above): against `llama-server` with the same options,
  SmolLM2-135M 0.87x under load (95 % 0.81x to 0.96x) and 0.98x alone, Qwen3-0.6B 1.16x (1.12x to 1.19x) and 1.52x
  (against a setting that slows `llama-server` alone), Qwen3-1.7B 1.11x (1.05x to 1.16x) and 1.04x (0.99x to 1.10x);
  against `llama-server`'s defaults Qwen3-0.6B reached 1.33x under load (1.31x to 1.35x) but 1.04x alone. No model
  reaches 1.2 in both halves against either reference. Re-open with kernels of our own below llama.cpp's, speculation
  that finds more on real traffic, or another machine, under a new protocol.

- **The ggml engine at 1.2x `llama-server`, second attempt (2026-11-11).** Experiment GG2-H (this container, commit
  `115090b`; see GG2 above): with the path around the kernels fixed, 1.00x under load (95 % 0.94x to 1.07x) and 1.14x
  one request at a time (1.09x to 1.20x), first tokens 10-20 ms later; the 1.2 goal is not reached. Both sides run
  llama.cpp's kernels at its default setting, which is the ceiling for scheduling. Re-open with a lever llama.cpp does
  not have at the same quality (kernels of our own, or a quality criterion that admits flash attention with
  `llama-server` given the same option), another model or machine, under a new protocol.

- **The ggml engine at 1.2x `llama-server` on this CPU and model (2026-11-10).** Experiment GG1-H (this container,
  commit `2793352`; see GG above): through the product, 0.81x under load (95 % 0.74x to 0.89x) and 1.05x one request
  at a time (0.94x to 1.20x), first tokens later; the gate allowed only llama.cpp's default setting. Re-open with a
  product path that streams a request's first token before the rest of its block is prefilled, a lever beyond
  llama.cpp's kernels, a quality criterion that admits flash attention (the owner's to set, QG1), or another model or
  machine, under a new protocol.

- **SP3's speculation policy (version 2) as the agent's default (2026-11-10).** Experiment SP3-H (this container,
  commit `5517512`; see SP3 above): every answer equal, but one request at a time 1.14x (95 % 1.05x to 1.25x) against
  1.3, and one load block at 0.936x against 0.98. Post hoc: the policy stopped speculating once a line fitted from 2-3
  noisy steps priced added rows at 22-30 ms each, and it never explored again. Re-open with a policy that keeps
  probing (a minimum number of speculative steps per row count before the line may veto them), under a new protocol
  on prompts not used here.

- **Speculation from the context with a learned window as the agent's default on the CPU plan (2026-11-10).**
  Experiment SP2-H (this container, commit `ee6f9d8`; see SP2 above): with SmolLM2-135M-Instruct-8bit, OpenBLAS and
  the CPU stacked engine, every answer was equal and load was unharmed (1.04x), but one request at a time it made
  1.40x the tokens per second (95 % 1.21x to 1.63x) against a bound of 1.3. `--speculate context` stays opt-in;
  `auto` leaves it off. Re-open with a changed mechanism (for example no drafts from 1-token matches, step prices
  from more samples), another model or CPU, under a new protocol on prompts not used here.

- **Speculation from the context at a fixed window on the CPU plan (2026-11-10).** Experiment SP1-0 (this container,
  commit `59f26ff`; see SP1-0 above): with SmolLM2-135M-Instruct-8bit on OpenBLAS, prompt-lookup drafts verified as
  rows of one slot were exact on all 48 Dolly prompts at `k` = 2, 4 and 7, and made 1.315x, 1.381x and 1.475x the
  sequential tokens per second, but the best lower bound was 1.23x against SP1's 1.3. Re-open with a changed mechanism
  (a window chosen per request, which (c) measured at 1.076x over the best fixed `k`, or drafts from another source),
  another model or workload, under a new protocol on prompts not used here.

- **An online-learned join order inside the stacked engine (2026-11-10).** Experiment AG9 step S (simulation at
  `a655ac7`, priced by the agent's CPU6-A2 measurement; see AG9 above): over `fifo`, no order had headroom in any of
  RL2-H2's three states by the rule fixed before the data. Longest first made up to 1.067x the tokens per second in
  `spike` with first tokens 1.46x later; shortest first brought first tokens at most 0.89x. Re-open with an engine
  whose step cost depends on the order (a different width per state, a prefill that does not stall decode), a
  different objective chosen by the owner, or hardware data contradicting the engine model, under a new protocol.

- **CPU6 step 3 as fixed: the agent's own CPU setup at 3x (2026-11-10).** Experiment CPU6-A (this container, commit
  `b71b1ce`; see CPU6 above): with SmolLM2-135M-Instruct-8bit the agent chose `dequantize` on OpenBLAS by itself and
  refused the CPU stacked engine on AG4's first-token check (1.16x later on a 10-token prompt, 14 ms). Through the
  product that served 2.85x AG4's earlier CPU choice (95 % 2.78x to 2.92x), below the bound of 3.0. Re-open with a
  changed mechanism (a stacked plan whose single-request first token is no later) or a first-token rule fixed for
  machines without `route`, under a new protocol on fresh data. **Reopened as CPU6-A2 (owner decision 2026-11-10: the mechanism, not the rule) and
  passed** at `7271b7a`: the plan's prefill head at one row, 15 first-token samples per path on a CPU; the agent chose
  the stacked engine on OpenBLAS by itself and served 12.7x (95 % 12.2x to 13.1x). This entry stays as its history.

- **AG6 dreaming as the agent's only reaction to a load change (2026-11-08).** Experiment AG6-Q (kernel
  `tobayko/ironmule-rl2-headroom-ag6q` v1, commit `cc76d3f`; see AG6-Q above): on one T4 with Qwen 3 1.7B bf16 at
  `9cd6692`, quiet traffic, a 48-request spike and quiet traffic again, 3 repetitions. The agent served the spike at
  0.582x the forced stacked engine's tokens per second, with first tokens 1.61x later. In the quiet phase after it,
  its first tokens were 1.88x the single path's, because it held the stacked engine for 64 requests. The simple
  busy rule came within 6.3 % of the best fixed mode's tokens per second and 11.7 % of its mean first token in
  every phase. A dream every 16 requests over the last 64
  does not follow such a change. Re-open with a mechanism that reacts per request (AG7), or with new evidence on
  the dreams' cadence and window.

- **KB2 Gemma 3 4-bit stacked decode plan (2026-11-07).** KB2-0 run 2 (kernel `tobayko/ironmule-cuda-probe-kb2`
  v2, commit `9d388e9`; see KB2 above): on one T4 with `gemma-3-12b-it-4bit` at `86cc6a8`, stacked rows were bit-exact
  and six of them made 2.006x today's tokens per second (95 % bound 0.4991 against 0.5), but the quality gate's main
  path read 0.984 with 95 % up to 1.177, not below 1.005 (three of 16 chunks diverge by up to 1 nat a token, in both
  directions). KB2's rule stops it at (c). Re-open with a fixed implementation found by a diagnostic of those chunks,
  re-gated under a protocol fixed before its data, or with a new model or card. **Reopened 2026-11-07 as KB2-G**
  (see KB2): KB2-D's new evidence that the gate's lead-free chunks are unstable for every path, and GATE2.
  **KB2-G (commit `4a01f49`): back in Tier 0.** Gate 2's main path passed (0.9988, 95 % up to 1.0008), but its long
  path, 4 chunks past the window, read 1.0003 with 95 % up to 1.0080. Re-open only with a new mechanism or new
  evidence, for example serving scoped to the window under its own protocol. **Reopened again as KB2-L (owner
  decision, 2026-11-07) and passed:** the long path on 16 windows read 0.9973, 95 % up to 1.0012. KB2 is no longer
  rejected; this entry stays as its history.

- **TP1 Kaggle TPU as a decode device (2026-11-07).** Probe TP0 (kernel `tobayko/ironmule-tpu-probe` v1, commit
  `08c96d4`; see TP1 above): on one TPU v5e chip, the bf16 matrix products of one `Qwen3-1.7B` decode step,
  each jitted and timed alone, allowed at most 20.8 tokens/s at 1 row and 180.8 at 8 rows, below the T4's 57.7 and
  290.9. TP1's rule closes it. Re-open with a probe that times a whole decode step compiled as one program (each
  product cost a flat 0.2-0.3 ms, a per-call floor), or with a new model or TPU generation.

- **RL2 learner for GPU placement and group schedule (2026-10-28).** Experiment `RL2-H-run1-20261028` (kernel
  `tobayko/ironmule-rl2-headroom` v1, commit `258ccbe`; see RL2-H above): on two T4s with Qwen 3 1.7B bfloat16 at
  0.2-0.95 of capacity, the oracle over 20 fixed arms beat the best single arm by -0.05 % held out (95 % bootstrap
  to +0.13 %) against 3.5 % A/A noise, and the reference was within 0.4 % of the best arm at every level. RL2's kill
  holds for these two decisions on this fingerprint: no learner for them. Re-open with a new model, checkpoint,
  card, traffic shape or decision (the MEM2 bound and threads were not measured).

- **RL3 length-aware placement and order (2026-10-29).** Experiment `RL3-H-run2-20261029` (kernel
  `tobayko/ironmule-rl2-headroom-h3` v2, commit `6eaa4b4`; see RL3 above; run 1 was invalid, X1): on two T4s with
  Qwen 3 1.7B bfloat16 under quiet, bursty and spike traffic with long prompts and 32- or 256-token answers, no
  decision had held-out headroom above the 18.7 % A/A noise. The joint decision scored -0.7 %, placement -1.2 % and
  order +0.6 %. Shortest-job-first was within 0.02 % of the reference, and segregating long from short requests
  cost 15-24 % in bursts. RL3's kill holds: no learner for placement or order by output limit, and the reference
  stays. Re-open with a new model, card, traffic shape, cost weighting or decision. A noise estimate that one A/A
  pair does not dominate also counts: here the second-largest pair was 6.6 %.

- **RL4 model-based RL agent for per-request placement and order (2026-10-29).** Experiment `RL4-SIM-20261029`
  (see RL4 above): a REINFORCE policy trained in a pool simulator fitted to RL3-H run 2 passed the fidelity gate.
  In the simulator it beat the best fixed arm by only 2.8 %, against an 18.7 % A/A noise, so the simulator gate
  failed and no learned arm went to hardware. Re-open with a testbed whose noise is several times smaller than a
  few per cent, or a decision whose simulated gain is several times larger.
  *Re-opened for `spike` by RL4-S (above):* on 30 new spike traces the frozen policy beat the best fixed arm on the
  T4s by 9.0 % (95 % 6.9-11.1 %) against 2.9 % noise. Mixed traffic is still unvalidated.

- **RL6 width-learning policy against fixed widths (2026-10-29).** Experiment `RL6-run1-20261029` (see RL6 above):
  the v2 policy that also picks each group's width gained 0.8 % (95 % -1.9 to 3.4 %) over the best fixed arm,
  `shortest/w1/g0/sjf`, against 3.0 % noise. No width learner. The fixed narrow shortest-first schedule it found
  is the candidate default (RL6).

- **A knob learner over the engine's execution knobs on the T4 (RL7, 2026-10-29).** Experiment `HW4-run1-20261029`
  (see RL7 above): seven knob sets are class A but none changes order with the group width, so knobs and
  scheduling separate and the best fixed profile plus the best schedule is the policy. Re-open when a knob's cost
  per further session changes shape against another's.

- **`head_skip_prefill` as an exact knob on CUDA (2026-10-29).** Experiment `HW4-run1-20261029`: the output head at
  one row gives other bits than the head over the whole prompt, so the prefill's logits change. Re-open with a head
  kernel that is invariant to the row count.

- **`prefill_into_fixed` on CUDA (2026-10-29).** Experiment `HW4-run1-20261029`: tokens kept, prefill 7.8 % slower.
  `E1`'s kill ("the phase it would optimise costs 1.47 ms of 537 ms") holds on the T4.

- **Prompt-lookup speculation as an exact knob on CUDA (2026-10-29).** Experiment `HW3-run1-20261029` (see HW3
  above): on a T4 every combination with `speculate_k` 3 changed tokens (first at steps 23-58 of 64), acceptance
  0.42-0.47, 0.99-1.14x. The verify forward's k + 1 rows take MLX's slower, non-invariant multi-row path. Re-open
  with a verify path whose logits are bit-identical to the sequential step (KB1).

- **True tensor batching through MLX's bf16 matmul on the T4 (2026-10-29).** Experiment `HW3-run1-20261029` (see
  TB1 above): stacked rows changed tokens and never matched bit for bit, and eight rows made a token only 1.34x
  cheaper, because MLX's bf16 product at 2-32 rows runs at 41-45 GB/s against 252 at one row. Re-open with a
  small-M kernel (KB1) or another card.

- **`fuse_projections` and `fused_argmax` on CUDA decode (2026-10-29).** Experiment `HW3-run1-20261029`: fusion
  1.010 (slower) on decode and +0.5-1.4 % on prefill; fused argmax 1.002. Neither helps on a T4 with Qwen3 1.7B
  bf16.

- **RL6's width policy as the product's learned dispatch on mixed traffic (2026-10-29).** Experiment
  `LD1-V-run1-20261029` (see LD1-V above): through the product's own pool and scheduler it beat the reference by
  21.2 % (95 % 18.3-24.3 %), but in `bursts` it cost 6.9 % more than `shortest/w1/g0/sjf` (95 % bound -12.9 %
  against 8.1 % noise), so the no-harm rule failed. No evidence for it; LD1 ships only RL4-S's spike-bound policy.
  Re-open with a policy trained against a per-state harm constraint, or traffic whose bursts it was trained on.

- **KP1 run 1: compile + readback 4 as the tuned engine's default, through the product (2026-10-29).** Experiment
  `KP1-run1-20261029` (see KP1 above): answers identical and class A at 27 capacities, but +0.2 % weighted (95 %
  -1.9 to +1.6 %) and `spike` -1.3 % (bound -5.4 % against 4.7 % noise). The worker held each token for the next,
  so the first token waited a whole chain; fixed, and re-opened as H9.

- **KP's knobs after the first-token fix (2026-10-30).** Experiment `H9-run1-20261029` (see H9): +1.7 % weighted
  (95 % 0.39-3.15 %) against a 0.81 % A/A floor, no harm in any state, `quiet` +6.1 %. Too small to resolve in 12
  traces. Re-open with more traces, or once the default's own cost changes again.

- **H11 dense slots of 8,192 tokens (2026-11-01).** Experiment `H11a-run1-20261101` (kernel
  `tobayko/ironmule-cuda-probe-h11` v1, commit `3cb1e8f`; see H11). Tested on one T4 with Qwen 3 1.7B bf16.
  Answers were bit-identical and memory fit. At 6,000-token prefixes, though, the plan's perplexity ratio was
  1.0070 (95 % to 1.0078), above the 1.005 bound. The larger slot also cost 8.1 % per step. The dense plan keeps
  2,048-token slots. Re-open with a changed mechanism for long-context accuracy, such as float32 accumulation in
  the long-key combine, and with FA2's grid cut to live chunks.

- **MOLE1 local adaptive route and indexed preparation.** Experiments
  `MOLE1-LOCAL-1-20260912-attempt1` and `MOLE1-LOCAL-2-20260912-attempt1`:
  no adaptive benefit over C; LOCAL-2 preparation missed its 5% benefit gate.
  Full results: `ironmole_mcp/docs/RESULTS.md`. Agent-level comparisons remain open.

- **R15, narrowing the competing-process gate's substring match.** Opened and closed
  2026-09-26 on inspection, no experiment: its own kill holds, because the Q3c and Q3d
  records bind `research/q3b_residual_swap_canary.py` in `runtime_code_sha256`. The gate
  stays fail-closed; q3f's real cleanup test skips with the reason (ledger R14).

### Rejected is not forbidden (project rule, 2026-09-10)

Tier 0, `NO-GO` and every closed kill entry are **historical evidence, not a permanent
ban**. They record what a specific mechanism cost on a specific fingerprint, and that is
exactly how far they reach.

Any of them may be re-opened when one of three things is new:

* **new hardware evidence** — a different chip, memory size, MLX/mlx-lm build or model
  revision than the one the entry was measured on;
* **a changed mechanism** — the reason the entry died no longer applies, stated
  explicitly against the old entry's own kill criterion;
* **a new implementation** — a different code path, kernel or execution route, not a
  re-run of the same one.

Re-opening costs one thing: the new entry must name the old entry, quote the old kill
criterion, and say which of the three conditions above is met. Repeating an experiment
with no such statement is still forbidden, because that is what wastes GPU time. Refusing
a method *only* because an older attempt failed is equally forbidden, because that is what
freezes a runtime.

- **`B15` exact-but-pruned `lm_head` (2026-09-09).** Measured offline on
  `gemma-3-4b-it-4bit` over 72 real greedy decode steps: with a correct
  Cauchy-Schwarz cluster bound, `95.5%` of the 262,208 rows survive at `k=16384`
  and `98.5%` at `k=4096`, against a kill threshold of `25%`. The token was exact
  on 72 of 72 steps, so the bound is right and the geometry is the problem. The
  bound needs a cluster radius of `0.106`; k-means reaches `0.387`. It is not a
  clustering failure: sampling 256 rows, only `1.25` rows on average lie within
  the required radius of any row, and the mean nearest-neighbour distance is
  `0.235`. Achieving the radius would need roughly one cluster per row, and the
  centroids alone would then cost more than the rows they replace. Row norms are
  nearly uniform (`0.006` to `1.085`, median `1.000`), so the pure norm bound
  prunes `0.02%`. Experiment `B15_4B_offline_bound_20260909_attempt1`; do not
  re-run with a different clustering.

- **`B48` sharing the aligned projections too (2026-09-09).** Kernel GO for `K=4096` and
  `K=15360`, bit-identical, but the product gain against `PairedThroughputMode()` was
  `0.9903` and `0.9920` against a `5%` goal. `share_aligned` stays `False`. Full entry in
  `research/LEDGER.md`.

- **`B49` one weight sweep for four requests (2026-09-09).** Bit-identical, but four shared
  over two shared pairs is `1.1084`, CI `1.1014 – 1.1205`, slower in 20 of 20 blocks. Wider
  sharing is not the lever at `K=3840`. Full entry in `research/LEDGER.md`.

- **`B56` a per-request objective (2026-09-10).** Implemented, default unchanged, `NO_GO`
  in `B56` and `B56b` (`research/LEDGER.md`).

- **`B68` MLX command buffer limits (2026-09-10).** `MLX_MAX_OPS_PER_BUFFER` and
  `MLX_MAX_MB_PER_BUFFER` swept in 48 fresh processes on 4B and 12B: no interval below
  `1.0`, three outright losses, `DEFAULT_WINS` on both models, MLX `50/50` stays.
  Experiment `B68_command_buffer_sweep_20260910`. Proposed again on 2026-09-23 as a first
  step; not re-run, because nothing about the mechanism or the hardware changed.

- **`B10` fewer kernels per step (2026-09-09).** Bounded by `B24S`: norms, adds, rope,
  attention and copies together are `0.2` to `0.7%` of decode GPU time, so fusing all of
  them away returns under `0.7%`. Experiment `B24_metal_trace_series_20260909_attempt1`.

- **Fusing `lm_head` with argmax (proposed 2026-09-23; derived, not measured).** The
  saving is the logits write and re-read, about `0.5` to `1 MB` per token against
  `2.561 GB` of weights read per 4B token (`B24R`), and argmax sits inside the `0.40%`
  non-matmul share of 4B GPU time (`B24S`). That bound is below every equivalence margin in
  use. The `fused_argmax` knob already moves argmax into the compiled body; `E11` rejected
  it and `Q3a` asks whether that changes on the final incumbent. Re-open only with a
  profile that shows the head's epilogue as a cost.

- **Judging a module by searching its source text (`B82`).** Four studies wrote checks that
  matched the module's own prose rather than its code. Structural claims are checked on the
  syntax tree. See `research/LEDGER.md` B82.

- **`B76` a contextual state model for the `(4, 8)` stack ratio (2026-09-11).** Ridge over
  `load_1min`, `memory_free_percent` and `swap_used_gb`: prediction error `0.0087` against a
  constant model's `0.0058`, larger than the whole between-session spread. Those features
  carry no signal; a new feature set needs a mechanism first.

- **`B77` plain mean against a Bayesian posterior (2026-09-11).** Identical action sequence
  and regret over fourteen sessions; the data cannot tell them apart until the spread moves.

- **PROD8 12B 1077-token integration screen (2026-09-07).** First real stock-reference request hit the frozen 6 s host deadline (6.002551 s observed); owned worker reaped, no product/HTTP run. Experiment `PROD8_12B_long_context_20260907_attempt1`; no shorter-context or relaxed-limit retry. Phase-aware follow-up hypothesis: root `BACKLOG.md` PROD9.

- **PROD2/PROD6 12B short-prompt calibration (2026-09-07).** Complete installed 90-call protocol, exact outputs/resources and three normal worker exits; no qualified net gain at any cap under the frozen noise rule. Experiment `PROD6_12B_installed_20260907_attempt1`; no adoption or same-design retry. Results: `docs/PROD6_12B_RESULTS_2026-09-07.md`.

Listed so the next person does not spend a week rediscovering them.

- **PROD2/PROD6 1B short-prompt calibration (2026-09-07).** Complete installed 90-call protocol and clean worker exits, but no qualified signal above the frozen noise floor; no adoption. Experiment `PROD6_1B_installed_20260907_attempt1`; do not repeat for a luckier result.

- **PROD2/PROD3 4B short-prompt calibration (2026-09-07).** Complete installed 90-call protocol, exact outputs and resources valid, but all three caps remain below the frozen noise qualification; no adoption. See `PROD3_4B_installed_20260907_attempt1` and `docs/PROD3_RESULTS_2026-09-07.md`; do not rerun the same design for a luckier result.

- **`Q3c` direct replication attempts (2026-08-31).** Run 1 was refused before
  a phase because load `8.294921875 > 8` (raw SHA-256
  `5270c0f38e50984cd26223aa2a9817982fc5a1861ddbe2caa3cff98393c9e8d5`); run 2
  aborted after `105` samples / `27.395 s`, swap `2,353,654,661 B` to
  `2,625,172,930 B`, delta `271,518,269 B` (`258.94 MiB > 128 MiB`), with
  cleanup unverified. No timings, identity, performance result or promotion
  exists. Raw run 2 SHA-256 is
  `d94db80402254c87c0e4a0128cf802e1eaa59d42c4459c2f208077f48c38b8df`; retain
  both records and do not rerun Q3c.

- **`Q3d` model-free recovery gate (2026-08-31).** The gate itself passed:
  `61` samples over `60.020192667 s`, maximum adjacent gap
  `1.013944625 s`, swap `2,651,722,874 B` throughout and delta `0 B`.
  Raw SHA-256 is
  `4699a49b174db31580a9701ef2075f8b1964d309b0f857dd7779fb230cfccb83`
  (`34,144` bytes); the summary SHA-256 is
  `3b43e267000ba15b9d9079d9f118e59c1cd51dbcdfecc067c20995b01a0a1c3e`
  (`970` bytes). The one permitted Q3c invocation was then refused before
  `Popen` because macOS `/bin/ps` rejected `sid` (`rc=1`, `ps: sid: keyword not
  found`). No model, inference process, timing, identity or performance data
  exists. Final state is `Q3C_FAILED` with `BASE/current incumbent` fallback;
  do not repeat Q3d or pool its gate with Q3c. Q3e is the separately frozen
  portability-repair path.

- **`Q3e` portable process probe and one Q3c invocation (2026-08-31).** The
  repair removed unsupported macOS `ps sid` and the exact Phase-R run completed
  with token/resource identity intact. Its descriptive incumbent/BASE total
  ratio was `0.857466859207542`, CI
  `[0.8551668079699586, 0.8611021999710893]` (`14.2533140792%` faster), but
  cleanup conservatively rejected four stable unrelated same-UID launchd
  services that appeared after the worker baseline. The raw result is
  `research/raw/Q3e_q3c_final_20260831.json`, SHA-256
  `1df6c81dc824911016e687883c535f1ec314f3e03b51303b04c38ae71bb6f4ea`, size
  `2,205,857` bytes; the terminal note is
  `research/raw/Q3e_terminal_result_20260831.md`. Status is `FAILED`, Phase N
  did not run, and fallback remains `BASE/current incumbent`. Do not pool or
  rerun Q3e; Q3f is the separately frozen attribution path.

- **`Q3f` terminal attribution path (2026-09-01).** The single permitted Q3f
  execution reached the exact local Gemma 4B worker only far enough to fail with
  `ABRunError: child 0 start callback failed`; no child marker, ledger, timing,
  token, Phase-N or accepted performance evidence exists. All 14 preflight
  checks passed, AC/thermal/low-power gates were green, and swap stayed at
  `2,609,643,520 B` across 25 samples with zero delta. Cleanup reaped the
  worker and rejected the run closed because guard/ledger evidence was
  unavailable and same-UID Spotify Helper PID `52017` could not be proven
  unrelated. Raw SHA-256 is
  `e82accdbd52857e6201fa2b34984765e61658ecfd3956d5c903a49c1e6de70a9`
  (`2,487,533` bytes); terminal note is
  `research/raw/Q3f_terminal_result_20260901.md`. Status is `FAILED`, with
  `BASE/current incumbent` fallback. Do not retry Q3f, pool its record or say
  that historical Q2 speed was reproduced; any child-visibility race fix needs
  new authorization and preregistration.

- **`R11/R12/E15` fork-per-block memory-integrity path (b700377).** Closed by the
  complete four-block E15 after-file (SHA-256
  `d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c`): four fresh
  PIDs, flat per-block peaks, no swap growth, no correctness divergence. This is an
  engineering memory result only, not a speed claim; do not repeat the exact attempt.

- **`B25` KV cache reallocation during decode.** Nothing reallocates. The fixed-shape
  cache is allocated once per `serve()`: `mx.zeros` appears only in
  `_empty_fixed_state`, and `_caches_from_state` wraps the existing arrays rather than
  copying them. Measured, not just read: writing 56 tokens through `FixedKVCache`
  moved active memory from `65,644 B` to `32,876 B` — it *fell* by exactly one full
  keys+values copy (`32,768 B`), which is the warmup's double buffer being released
  once MLX takes the `slice_update` donation. Shape constant throughout.
  `tests/test_cache_allocation.py` holds both properties. The predecessor's `4.4263%`
  from cache growth copies (candidate 21) is gone with the growing cache it was
  measured on, so there is nothing here to claim.

  **One caveat worth carrying, because it shapes how the tuned gain reads.**
  `FixedKVCache.update_and_fetch` builds `mx.array(0)` and `mx.stack(...)` on every
  call, and `make_mask` an `mx.arange(capacity)` — per layer, per decode step. Under
  `compiled_fixed_cache=True` these are bound once when `mx.compile` traces the body,
  so they cost nothing. Under the untuned `BASELINE` they are rebuilt every step. Part
  of that knob's `0.9679` is therefore host work that stops happening, not GPU work
  that got faster — which is the documented purpose of `mx.compile`, and `E5` already
  put `3.3 ms` of host work on each step. How the ratio splits between constant rebuild
  and kernel fusion is not measured. The tuned gain against `BASELINE` is real, because
  `BASELINE` is what an untuned install actually runs; it is not a gain against a
  well-optimised floor.

- **`R11` measurement is gated on swap, not on a byte count.** The literal
  `12 * 1024**3` was wrong in both directions: it refused `gemma-3-12b-it-4bit` at a
  `17.51 GB` block that never swapped, and it passed a run that had to be discarded
  because swap climbed `2816 MB`. `ironmule/bench.py` now carries `MemoryGate`, which
  aborts on a *delta* against the run's own swap baseline — an absolute value would
  refuse every run on a machine already carrying backlog — with a coarse peak backstop
  derived from `hw.memsize` rather than typed in. Wired into `e14b_arms.py`,
  `e15_service.py` and `e12_window_falsification.py`; `e16_replication.py` keeps its
  per-child ceiling, which was always coherent because it forks. The self-check replays
  both reference runs from `B7`'s recorded numbers, so the gate is tested against what
  actually happened rather than against what sounds reasonable. A gate that can read
  neither swap nor installed memory reports itself `inert` instead of silently passing.

- **`R9` `ironmule.tune` resolves to the function, not the module.** `__init__.py:35`
  re-exports `tune` from `.tune`, rebinding the name in the package namespace, so
  `import ironmule.tune as m` yields the function and the module is reachable only via
  `importlib.import_module("ironmule.tune")`. Renaming would break the public API, so
  this is documented rather than changed — the decision its kill criterion allowed for.
  Checked across the package: exactly one of fifteen submodules is shadowed this way.
  `tests/engine/test_ironmule.py` asserts that, so a second one cannot appear unnoticed; the
  quirk stays a footnote instead of becoming a pattern.

- **B27e mirrored cross-commit control.** Four fresh 4B processes, source-surface
  digest `ec242c…`, all correctness/resource gates green. OLD/D1 block ratios were
  within 2%; mirrored D1/OLD made D1 appear 5.8–7.9% faster. Final
  `ORDER_OR_TEMPORAL_DRIFT`; a consistent D1 slowdown was not reproduced, but B27d
  remains formally inconclusive. Do not rerun the same two-block unconditioned design.

**Two lists, one of them closed.** The entries below marked `E*` come from this
project's ledger. The ones marked `cycle *` come from the predecessor project, whose
candidate catalogue is `EXPERIMENT_BACKLOG.md` in the source repository — 24 candidates
over 17 sealed cycles, most of them closed. That file is history and takes no new
entries; everything current goes here.

- **Prompt-lookup speculation.** `E0c`: `2.9x` slower, acceptance `0.17` per drafted
  token. The workload does not repeat itself enough. This does **not** kill `B13`,
  which uses a different draft source.
- **Projection fusion as a decode win.** `E5`: paired across six processes, decode
  ratio `0.9990` with a CI containing 1.0. Kept for its `-1.10%` prefill effect only.
  `E0b`'s `+2.4%` was noise.
- **A dedicated `M=1` fast path.** `E2`: the apparent `M=1` cost was a `~0.49 ms`
  `eval`+`synchronize` round trip, not kernel behaviour.
- **`prefill_into_fixed`.** `E1`: the phase it would optimise costs `1.47 ms` of
  `537 ms`. Killed before it was benchmarked.
- **Padding `M=322` to 384.** `E2`: `~4.74 ms` predicted against `4.43 ms` measured.
- **An adaptive width controller.** Realised width already adapts; nothing measured
  beats a fixed 4. Revisit only if `B1` finds the optimum is model-dependent.
- **A draft model drafting for a larger target** (predecessor candidate 11, 1B for 4B).
  Measured and rejected, recorded as `0.560x`. **The sign convention on that row is
  ambiguous** — elsewhere in that file a ratio below 1.0 marks a win — so re-derive the
  direction from the cycle's raw data before trusting it either way. See `B13`, which is
  the same idea at a target/draft ratio the predecessor never tried.
- **`mx.compile` over decode subgraphs on a growing cache** (predecessor candidate 12).
  `-23.8%` dispatch and **wrong tokens from position 2**. Rejected, and the older
  device-model compile numbers were invalidated with it. The same idea on a *fixed-shape*
  cache is what this runtime ships: cycle 16 measured `0.9296` with identical tokens
  across 18 arm executions, and `E0a` reproduced it. The lesson is about the cache, not
  about compiling. See `B3`.
- **A custom Metal kernel chosen without profiler evidence** (predecessor candidate 14).
  Locked because cycle 9 localised no single kernel hotspot. Still the right rule:
  `B24` before `B10` or `B23`, or it is guesswork with a compiler attached.
- **Bundled host readback**, reading the stop token only every `N` steps (predecessor
  candidate 18, cycle 17). Readback 8 was faster in every pair, but `0.9581` missed a
  preregistered 5% threshold, so the recorded decision is
  `no_clear_speedup_baseline_retained` and the `4.19%` is calculated rather than claimed.
  A valid negative result. It also points the wrong way for the scaling problem: a fixed
  per-step host cost is a *smaller* share of a longer step, so 27B would show less, not
  more. Do not re-run it hoping for a better draw.
- **B28 / `qwen_native_true_batch_v1` rejected at the correctness gate.** Widths 2, 3
  and 4 produced exact visible tokens and stop reasons with no fallbacks, but the final
  hybrid `kv_hash` differed from the sequential reference. Swap delta was `0 B`; no
  token-rate or other performance measurement is valid. Do not route this path.
- **B29c / `qwen_native_b1_v1` below target.** Widths 2, 3 and 4 passed exact
  correctness, final state and 16-token continuation with no fallback and zero swap
  delta. Candidate `16.0722` versus Interactive `15.6740` (`1.02541x`) and versus
  Throughput `16.0687` (`1.000219x`) remain below the `1.10` gate. No route.

- **B40 width sweep (experiment B40, `INCONCLUSIVE`).** W2/W4 and W3/W4 were
  directionally slower on all six blocks, but material epoch drift prevented a
  valid selection; all 18 children and safety gates were clean. No retry is
  authorized, and no W2/W3/W4 timing may be cherry-picked or treated as a
  selected profile. W4 remains the unchanged operational baseline.

## Rules

Fast mode (owner, 2026-11-11; AGENTS.md) suspends the protocol rules for experimental work until the owner says
otherwise. What stays in force for this file:

- **Correctness gates come first.** A path that gets answers wrong is out, whatever its speed.
- **A finished entry leaves this file**: one line under "Closed entries", and one in Tier 0 if it was rejected.
- **Rejected is not forbidden** (above): a reopening names the old entry and what is new.
- **Measure optimisations in combination**, not as the product of single gains.
- **Add what you learn the same day**, with its mechanism and what would close it.
