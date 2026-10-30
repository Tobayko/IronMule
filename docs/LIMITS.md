# Known limits and validity domain

Stated as measured, not as hoped. Anything not listed here was not tested.

## Validity domain of every performance number

| | |
| :-- | :-- |
| Model | `mlx-community/gemma-3-4b-it-4bit`, revision `93724907`, for every preregistered result. `gemma-3-12b-it-4bit` was measured once outside the preregistered series, see `X1` in the ledger |
| Quantisation | 4 bit, group size 64 |
| Framework | MLX `0.32.0`, mlx_lm `0.31.3` |
| Machine | Apple M1 Max, 32 GB unified memory, 32 GPU cores |
| Power | AC. Battery and low-power mode were never measured |
| Decoding | greedy, `temperature = 0` |
| Batch | 1 per execution; grouping never changes tensor shapes |
| Context | 276 to 2048 tokens |
| Concurrency | up to 8 requests |

No claim holds outside this box. Another model, another quantisation, another MLX
build or another machine requires re-measurement, which is what the fingerprint
exists to force.

**B27 D1 is not a runtime optimization or selected profile.** It adds an immutable,
stdlib-only evidence contract that current runtime modules do not import. The sealed
B27d post-change engineering screen preserved exact outputs/resources and passed its
12B regression gates, but both 4B arms were common-mode 5.7–6.4% slower and crossed the
preregistered 5% intervals. Its result is therefore
`INCONCLUSIVE_POTENTIAL_REGRESSION`, not proof that D1 is performance-neutral or that
the slowdown is causal. The separate B27e mirrored cross-commit control then found D1
within 2% in OLD/D1 order and 5.8–7.9% apparently faster in D1/OLD order. Its frozen
class is `ORDER_OR_TEMPORAL_DRIFT`: a consistent D1 slowdown was not reproduced, but
neutrality is still unproved. D1 remains non-imported and unactivated.

**D2 exact model identity is a validity guard, not a speed optimization.** Runtime
fingerprint v2 and tuned-profile conditions v2 now bind the exact local revision,
complete manifest, architecture, canonical quantisation and tokenizer digest. Legacy
or incomplete profiles are rejected, and model identity is hashed only during local
resolution/load rather than `serve()`. This does not qualify or activate any profile,
route a request, persist D1 evidence or extend a performance claim. The same-day D2a
4B/12B pre-change baseline and sealed D2b post-change screen completed with exact
identity/correctness/resources. D2b's frozen result is `NO_REGRESSION_OBSERVED` under
its 5% engineering gates, not a speed, quality, qualification or activation claim.
See [`D2_IMPLEMENTATION.md`](D2_IMPLEMENTATION.md).
Since 2026-10-13 resolution also refuses a snapshot whose weights the loader cannot read: no
`model*.safetensors`, a broken link or empty file among them, a gap in a numbered shard set, or an index
whose shards are partly present (a stale upstream index beside a consolidated file stays loadable). It
checks names, links and sizes, not tensor contents: a truncated non-empty file still fails only at load.
Identities of complete snapshots are unchanged; `model_identity.py`'s source digest, which online-runtime
identities include, changed, so evidence bound to the old digest needs requalification. The product
bridge builds its identity from an already loaded model and is not gated again.

**Q3 is currently an offline contract/replay track only.** `ironmule.adaptive` and
the replay builder define and validate state/action/outcome/failure/uncertainty
records without importing the runtime or MLX. They do not select methods, persist
profiles, route requests or activate an optimizer; `BASE` and runtime coordinate
descent remain unchanged. The current replay is data-insufficient for BO, surrogate,
coordinate-descent or contextual-bandit comparison and has no sequential horizon,
so offline RL is not applicable. No BO/RL, performance, hardware or generalisation
claim follows from the replay inventory.

## The OPT1 coordinator (`native/optimizer`, opt-in)

Demonstrated: one cycle per application on Kaggle hardware (Linux x86_64 CPU session; Linux with a Tesla T4),
with two adapters (IronMule knobs on Qwen3 4B: `head_skip_prefill` qualified at 0.928 by `tune.confirm`; an own
LZ77 job: 4 threads qualified at R 0.399), activation only after the evaluator passed, and real rollback. Built
and tested on macOS arm64 (unit and integration tests only, no hardware claim); Windows: compile check only, its
probe leaves memory and GPU unknown. It acts only on its own workers and explicitly selected applications; power,
kernel and global scheduler settings are unsupported. Sequential RL and the Dream-RSI meta loop were measured once
on one 4-CPU host and lost to a fixed setting (ledger OPT1); no learned concurrency policy is active anywhere.
Energy and temperature are not measured. PROBE1 (ledger): `imopt-cpujob`'s foreground probe is no sound basis
for foreground-aware optimisation on the M1 Max as defined: its `e2e` p95 alone is 1.69 x its value beside one
background thread (processing 3.3-4.3 x slower alone; cause not identified), it counts generating its own data, and
under load a near-constant 5 ms of its own timer lateness dominates its tail; on the Kaggle host the dependence
runs the other way (0.84). Foreground terms of OPT3 and OPT5, whose targets came from the alone state, were
therefore lax; their verdicts stand as recorded, and OPT5's "no learning need" rests on build time at idle only.
OPT5 (ledger): on the M1 Max the coordinator kept cargo's default `jobs`
for this repository's Rust test build (fewer jobs only lengthened it; no better foreground at idle; busy unresolved
with three builds per cell); no build setting is active. OPT3 (ledger): the learning campaign (`imopt campaign`) resumes from its
journal after a kill, shown on an M1 Max with the CPU job; there the thread decision has no headroom (one fixed
setting is best at every load level), so no RL claim exists for macOS. On the 4-CPU Kaggle CPU host (Xeon
2.20 GHz, 2 cores x 2 threads) the calibration passed its gate narrowly (OPT2-CAL: headroom 0.065, 95 % interval
0.027-0.162), an upper bound that releases OPT2 there and is no gain; a policy that sees only the previous slice's
level is estimated post hoc at about 1 %, and OPT2 has not run. On
that Mac the CPU adapter's qualification refused a large gain under desktop noise (reference kept); the IronMule
adapter ran one cycle there on `gemma-3-1b-it-4bit` (resumed after a kill; `compiled_fixed_cache` qualified by
`tune.confirm` at 0.906, activated and rolled back), which covers that model, revision and tune workload only. No Kaggle qualification applies to the Mac. Replay results never qualify anything.
Qualification protocol 2 (OPT4b, opt-in by `"protocol": 2` in a goal) passed its contract on the M1 Max and a
Kaggle CPU session for the CPU job only: no null or 1 % cycle accepted, large gains accepted under desktop and
injected noise where protocol 1 vetoed them. It is not the default (`goals/cpujob-p2.json` selects it explicitly); load that slows
only the reference runs is invisible to both protocols. Since OPT3-D (c) every adapter call runs supervised
(macOS and Linux): after the coordinator dies, SIGKILL included, its adapter process groups end within 10 s and
a resumed coordinator waits for them; descendants that leave the adapter's process group, a supervisor itself killed with SIGKILL, and the
episode workers of `compare`/`campaign` (they end on stdin EOF, outside the lease) are not covered, and on
other platforms (Windows) adapters run unsupervised. Tested on macOS arm64 and Linux x86_64 (commit a57f986),
graceful SIGTERM termination included. The CPU adapter's activation and rollback on macOS are still shown by
the integration tests only (OPT3-D (a): protocol 2 vetoed a stalled run, reference kept).

## The IronMule agent (`--agent on`, AG1, opt-in)

No hardware result yet: nothing here claims that the agent makes a server faster. What has run is the agent's
code against a simulated server and on this repository's tests (BACKLOG AG1, S). It decides two things only:
order and group width. Placement across GPUs, engine, plan, knobs and output are outside it. Its world model
assumes one group at a time with members prefilled one after another, so it refuses the dense plan's slot reuse
and runs on one worker. Its replays use arrival times, prompt lengths and output limits with a filler prompt
(capped at 128 tokens and 2 s gaps), as RSI3-LIVE's do. Prompt-content effects and answers longer than 128 tokens
are therefore not represented. A schedule it qualifies is proven only for the binding it was measured under, and on
the traffic that machine saw. Where idle spells are shorter than a replay window, nothing qualifies and the
server's own schedule keeps serving.

## Live learning (`--learning`, RSI3-LIVE, opt-in)

Shown end to end on one Apple M1 Max with `gemma-3-1b-it-4bit` (rev `2d44e83d`) under `--engine tuned`
and desktop load (ledger RSI3-LIVE-P1): real streamed requests become stored transitions, model and Q
updates, bounded dream rollouts, a versioned candidate and replay qualification on the GPU; pause,
restart with identity check, client cancellation and the five-arm comparison ran. No learned policy
qualified and none serves anywhere; the comparison was descriptive (four traces) and favoured no learned
arm. No speedup is claimed. Not run: CUDA, the CPU device, other models, the reference engine (capacity 1:
learning keeps the reference there), sustained bursts above capacity, a preregistered comparison.
Decisions cover group width and gather wait only; `join` must be off. Replay qualification uses arrival
times, prompt lengths and output limits from real traffic with a fixed filler prompt, so prompt-content
effects (prefix or cache reuse) are not represented. The summed first-content model is inaccurate on that
Mac (relative error 12-17), which stops most rollouts.

RSI3-LIVE-P2 (ledger, same Mac and model, desktop load): real requests displaced replay windows with p95 62 ms
extra wait over 15 requests in all four window phases; a prefill already in progress is not interrupted and bounds the
wait, so longer prompts or larger models can exceed the 100 ms limit, which then suspends replay qualification for that
identity (not measured beyond this model). Cancelled grouped streams now stop at their next token boundary in the
engine (end of a chain with chained readback); requests without a cancel predicate are unchanged. The decomposed
first-content model met its criterion there (relative error 0.35, latest segment; groups with failed, cancelled or
rejected requests are not in it); groups of 5-8 were few (12) with two-sigma coverage 0.67. Most dream rollouts still
stop on model uncertainty (3,784 steps in the hour). No candidate reached a verdict and no comparison completed, so
the performance limits above are unchanged.

RSI3-LIVE-P3 (ledger, one Kaggle T4, `gemma-3-1b-it-4bit`, `--engine tuned` in server mode, synthetic bursty
traffic): the contextual bandit, RL and RL with dreaming did not beat the best simple strategy; the only valid run was at
light load (rejected, no qualified gain), the run at the registered load was invalid by its own calibration check and
showed the learned arms 3-4 % worse. No learned policy is activated anywhere. Not measured: other GPUs, larger models,
memory pressure, real user traffic, the CPU device.

RL2-H (backlog RL2; one Kaggle "GPU T4 x2" session, `mlx-community/Qwen3-1.7B-bf16`, `--engine tuned`, one HW1
worker per card, Poisson traffic at 0.2-0.95 of measured capacity, 64- or 128-token answers): no learner can gain
from deciding placement (shortest queue or fill) or group width and gather wait per load level. The per-level best of
20 fixed arms beat the best single arm by -0.05 % held out against 3.5 % A/A noise, and the default (shortest queue,
width up to 8, no wait) was within 0.4 % of the best arm at every level, so it stays and no whole-machine learner is
built for these decisions. Not measured: other models and checkpoints, longer prompts or answers, bursty or real
traffic, one card, the MEM2 cache bound, detokenizing threads.

RL3-H (backlog RL3, run 2; the same session type, model and pool under quiet, bursty and spike traffic with
1,000-token prompts and 32- or 256-token answers): placing or ordering requests by output limit gains nothing
measurable. No decision's held-out headroom cleared the 18.7 % A/A noise, and separating long from short requests
cost 15-24 % in bursts. A REINFORCE agent trained in a simulator fitted to that run (RL4) gained 2.8 % in
simulation, too little to test on this hardware. Tested where that testbed is quiet, in spike traffic only (RL4-S, 30
new traces, all answers identical across arms), the same frozen policy beat the best fixed arm by 9.0 % (95 %
6.9-11.1 %, A/A noise 2.9 %). The product can now run it as an opt-in learned dispatch (LD1), off by default and
bound to that measured setup: two T4s, that model and revision. It is unvalidated on traffic that mixes quiet,
bursty and spike states, and on every other model, GPU or card count.

RL6 (same setup, `bursts` and `spike` traffic, 2,304 requests identical across arms): starting the shortest output
limit first with groups of at most two cost 18.0 % less than the default, width 8 in arrival order (95 % 15.0-20.5 %;
width 1 18.6 %, width 4 11.9 %). LD1-V (same setup, `quiet`, `bursts` and `spike` weighted equally, 1,920 requests
identical across arms) confirmed it with quiet traffic included: 18.4 % (95 % 14.9-22.2 %), width 1 19.5 %. The
server now runs it by default on exactly that setup (two T4s, `--engine tuned`, that model and revision, no stored
serving policy); everywhere else the default is unchanged and unmeasured. A width-learning RL policy matched that fixed schedule (+0.8 %, 95 %
-1.9 to 3.4 %) but did not beat it. Run through the product's learned dispatch on mixed traffic, it beat the default
by 21.2 % but cost 6.9 % more than width 1 in bursts, so it is not qualified (LD1-V).

Decode on one T4 (HW3, HW4 and TB1, Qwen3 1.7B bf16, MLX 0.32.4, 6 prompts x 64 tokens): compiling the decode
step plus reading the host every 4 tokens made decode 8.2 % faster (0.918, 95 % 0.914-0.921), and HW4 found its
logits and state bit-identical to the untuned engine's. Skipping the output head over the prompt
(`head_skip_prefill`) changes the prefill's logits on this card. Prompt-lookup speculation changed tokens there,
and so did stacking requests into one forward (true tensor batching), which also made a token only 1.34x cheaper
at eight rows, because MLX's bf16 product at 2-32 rows is 5-7x slower than at one row. A decode step reaches 70 %
of the weight-bandwidth floor (252 GB/s measured). Prefill of a 1,026-token prompt takes 1.7 s. A research plan
that stacks up to eight requests into one decode step with IronMule's own kernels (KB1-1) answered every request
bit for bit the same at 1, 3 or 8 rows, decoded eight requests 3.8x faster than today's grouping, and passed the
numeric plans' quality gate on the decode path; it changes answers against today's engine (as a numeric plan does).
With one host read every 4 steps it serves a request alone at 0.93x (short prompt) and 0.99x (long) of today's tuned
decode, and each further row costs about 3 ms (KB1-2). It is in the product as the opt-in `--engine dense`.

Through the product on two T4s, with quiet, bursty and spike traffic weighted equally, it cost 48.3 % less than
the default (95 % 45.0-51.4 %, H10), and no state was worse: quiet 13.6 %, bursts 53.0 %, spike 48.0 %. That run
measured it as it ships: version 2 prefill, freed slots refilled by newcomers, each request done with its last
token, at its default width of eight (width four cost 8.3 % more). Without slot reuse it had cost 24.0 % less
(H9). It is recommended on exactly that setup (two Tesla T4s, `mlx-community/Qwen3-1.7B-bf16` at `9cd66928`), as
an opt-in numeric plan: about half of its answers differ from `--engine tuned` (258 of 576 compared were the
same). It passed the numeric plans' quality gate.

It needs unquantized Qwen 3 bf16 on CUDA below compute capability 8 and takes about two minutes to start. The
recommendation covers prompt plus answer up to 2,048 tokens, the slot size it was measured with and still
serves; a longer request is answered `400` on its own. Slots of up to 8,192 tokens (the context limit) are built
and wait for H11's part (a) on the T4, then part (b) decides whether the recommendation extends to them; neither
has run yet. Not measured either: other GPUs, other models, real user traffic.

Stability (STAB1, ledger; one Kaggle T4, `gemma-3-1b-it-4bit`, `--engine tuned` in server mode, run 1 of P3's synthetic
traffic): the worker's CUDA memory grows to the 15 GiB card's limit within each run, and under sustained overload a
prefill allocation or a CUDA graph then fails within about half an hour (four losses in six overload replays). Before
STAB1 that stopped serving for good; now the server replaces the worker (at most three times in ten minutes, same
identity only, old process reaped first): through three overload replays it served every window, the group in flight
got explicit errors (5 requests), each replacement took about 13 s and no accepted request was lost. A SIGKILLed
worker was replaced in 17.6 s with identical answers afterwards. Not changed: the memory growth (MEM2); a queued
request can time out behind a group that holds the worker until its own deadline, and a grouped request's completion
waits for its engine run (HOL1); past the accept limit a refused client often sees a connection reset instead of the
429 (HTTP1). Not measured: other GPUs and models, the Metal path, `--engine rust2`, real user traffic.

MEM2 (CUDA, one T4): MLX's cache of freed buffers, not held requests or CUDA graphs, filled the device under
sustained serving. The worker now bounds that cache to 1 GiB on CUDA (`IRONMULE_CUDA_CACHE_LIMIT_MIB`, `0` = MLX's
default); its effect on worker losses and on speed is not measured yet (the comparison was stopped before a result).

## `--engine rust2` (RUST2-SRV, opt-in, CUDA)

Measured on one Kaggle Tesla T4 with Qwen3 4B and 8B (mlx-community 4-bit, DeepSeek DSpark drafts), one user,
eight chat prompts of 256 tokens: 6-8x less wall time per answer and 10-12x sooner first tokens than the product's reference
stream on MLX's CUDA backend (ledger RUST2-SRV step 2). It is a different engine: its tokens are exactly greedy for its
own kernels, not identical to MLX's (126-128 of 128 teacher-forced tokens agree, RUST2-P), so answers differ from
the reference. Only Qwen3 dense 4-bit checkpoints, one NVIDIA card, prompt plus answer at most 1024 positions, one
request at a time (no grouping), no chat-turn reuse. Not measured: other GPUs, other models, concurrency, Metal
(rust2 is CUDA only; on Apple Silicon IronMule runs MLX as before).

## The local learning lifecycle, as of `B82`

The runtime can measure its own machine, learn a preference from that measurement, act
on it, watch it, drop it and re-earn it. What that sentence is and is not evidence for:

**Demonstrated, on one Apple M1 Max with 32 GB of unified memory:**

| | study |
| :-- | :-- |
| cold-start learning from no prior knowledge, in under five minutes | `B75` |
| the effect's sign stable over fourteen independent sessions | `B76` |
| machine-state features make the forecast worse, not better | `B77` |
| a persistent controller that never qualifies without enough evidence | `B78` |
| evidence-gated learned dispatch on real requests, opt-in | `B79` |
| passive drift monitoring that stays quiet through ordinary use | `B80` |
| a fail-closed fallback, and a kill switch that survives a restart | `B79`, `B80` |
| explicit requalification, and every way it must refuse | `B81` |
| one live requalification that passed and handed the action back | `B82` |

**Not demonstrated, and therefore not claimed:**

- **Cross-hardware generalisation.** Every number here is one machine. `B73` is the
  entry that needs a second real Mac and it has not run.
- **Any other Apple Silicon generation.** M2, M3 and M4 are untested. The fingerprint
  exists to refuse them rather than to guess.
- **Reinforcement learning or a contextual bandit.** Neither is implemented. `B77`
  measured a contextual model over load, free memory and swap as worse than a constant
  one on prediction error, coverage and regret, so those features are excluded until new
  evidence reopens the question.
- **A universal performance gain.** The qualified action covers one model, one
  quantisation, one shape and one workload class.
- **Automatic activation.** `enable_local_learned_dispatch` defaults to `False` and
  nothing in the system can change that on its own.

**The qualified action's validity box.**

| | |
| :-- | :-- |
| Action | the `(4, 8)` threadgroup geometry for the `K = 3840` quantised matvec |
| Model | `mlx-community/gemma-3-12b-it-4bit`, bound by exact identity and revision |
| Workload class | `single_short`: one request, no session plan, at most 32 new tokens |
| Route | the sequential path only; grouped and paired routes are excluded at source |
| Typical ratio | `0.965`, range `0.954` to `0.977` over fourteen sessions (`B76`) |
| Reproduced | `0.9607`, `0.9584`, `0.9673` in one requalification (`B82`) |
| Not typical | `B69`'s `0.8469`. `B77` found that run's A/A control an order of magnitude wider than `B76`'s, with the candidate interval overlapping it. The sign stands, the magnitude does not |

**What a monitor may and may not conclude.** Observations from ordinary dispatches carry
no counterfactual: the alternative did not run. They may move a qualified action to
`REQUALIFICATION_REQUIRED` and can never create or strengthen a qualification. Only a
paired comparison under the original gates moves the preference.

**A drift alarm is not free and the threshold says so.** A shift counts only when it is
both statistically clear and larger than the smallest gain the action's own qualified
interval supports. Below that the action still wins and there is nothing to recheck.

## Apple-Silicon inference claims

Apple's M4/M5 comparison is evidence for that specific MacBook Pro setup and
workload: it reports 120 -> 153 GB/s memory bandwidth and 19--27% subsequent-token
generation improvement, while TTFT benefits from the M5 GPU Neural Accelerators.
It must not be turned into a universal M5 or MLX multiplier:
[Apple ML Research, MLX and the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).

The M5 GPU Neural Accelerators are not the separate Apple Neural Engine (ANE).
The former are reached through the GPU/Metal ML path used by that MLX report; ANE
deployment is a separate Core ML decision. IronMule has no ANE kernel path and
does not infer one from MLX:
[Apple Core ML documentation](https://developer.apple.com/documentation/coreml).

External BaseRT, vllm-mlx and FusionML publications are design context, not
IronMule measurements or guarantees:
[BaseRT](https://arxiv.org/abs/2607.00501),
[vllm-mlx](https://arxiv.org/abs/2601.19139), and
[FusionML](https://arxiv.org/abs/2607.22785). Their reported speedups cannot be
copied into a runtime target without the same model, quantisation, chip,
framework, workload and correctness protocol.

The local E4/E5 results remain the relevant evidence for this repository: the
achieved GEMV bandwidth depends on matrix size, and projection fusion did not
produce a robust decode win. B35's 12B observation was order-sensitive and
inconclusive; that narrow statement is superseded by B36 only for the exact
12B revision, 322/32 workload, M1 Max host, default wired/cache policy and
full-hash/prefault protocol. B36 qualifies the core profile under those
conditions but does not activate it or generalize it. These are recorded in
[`research/LEDGER.md`](https://github.com/Tobayko/IronMule-Research/blob/main/research/LEDGER.md) and
`research/raw/B35_review.md` and
[`research/raw/B36_review.md`](https://github.com/Tobayko/IronMule-Research/blob/main/research/raw/B36_review.md).

No global 60--70% bandwidth-efficiency constant is validated here. The phase
diagnostic therefore requires per-run measured inputs and never clamps an
efficiency above one; such a value is only an input-consistency warning. No
zero-allocation decode-loop claim, M5 speed claim, Metal-kernel claim, or ANE
claim follows from the diagnostic or from B35.

**Two models outside the box have now been measured, and the gain is not
size-independent.** Under the strict plan, with three runs each and a realised width of
`4.00` throughout, the gain falls monotonically: `+19.24%` at 4B, `+15.42%` at 12B,
`+11.81%` at 27B. Peak memory grew `2.78 -> 7.80 -> 16.78 GB`; 27B fits this machine
with room to spare.

Those runs were not preregistered and are recorded as `X1` in the ledger, so they are
exploratory observations rather than validated results. They are stated here anyway
because the alternative — describing the domain as one model while three have been run,
and leaving the impression that the headline number travels unchanged to larger models —
would be less honest, not more careful.

All three are Gemma 3 at 4 bit on the same machine, so nothing here separates model size
from model family, and nothing here extends to another Mac.

**Qwen compatibility is narrowly qualified by X2.** The validated Gemma all-KV
legacy path is retained. Qwen compatibility covers only revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, the strict/greedy path, and workloads
up to 6 concurrent requests with the tested 2/3-request × 8-token and 6-request ×
48-token gates. `ArraysCache` with
non-`None` `lengths` or `left_padding` is rejected, as is hybrid speculation;
these are fail-closed boundaries. Qwen performance is unqualified. The compiled
tiny gate was exact, but its `30.76 GB` peak is a memory warning, not a speed claim.

## Self-learning, as of RSI1b and CPU4

**What the learning planner showed, and where.** On one Kaggle Tesla T4, on five 4-bit models it
had never seen (Gemma 3 270M, 1B, 4B, Qwen3 0.6B, Phi-3.5 mini), the planner that picks the next
hardware test used 3-6 tests instead of 11, needed 3061 s of tuning in total against 4007 s for the
fixed order, and found on average as much confirmed speed-up (20.22 against 19.82 percentage
points). The fixed order was better on Gemma 3 1B. Random order "won" Gemma 3 4B on the time-penalised
score only because it found nothing and so skipped the paired confirmation. One machine class, one
run per method; it is not measured on Apple Silicon or on a CPU. Summary:
[RSI1b](../evidence/kaggle/rsi1b-experiment-planner-heldout/summary.json); the earlier, inconclusive
attempt is [RSI1](../evidence/kaggle/rsi1-experiment-planner-t4/summary.json).

**What replay can and cannot say.** Replay only re-orders tests that were really run; it cannot
judge a test nobody ran, and it never stands in for a hardware test. If replay finds the fixed order
better on a machine, the planner switches itself off there.

**The learning parts need Rust.** The planner (`native/experiment_planner`) and the online
controller (`native/online_controller`) are built from source with `cargo`. The one-line installer
does not build them: without the planner, tune uses the fixed order; without the controller,
`ironmule autopilot` stops and says how to build it.

**CPU-only machines are slow.** On a 4-core Kaggle Xeon, `ironmule start` with Qwen3 0.6B answered
48 tokens in 55-64 s with the weights unpacked to float32 and in 446-456 s without (one run,
unpaired). Qwen3 spends short answers in its `<think>` block first. Linux x86_64 only. Summary:
[CPU4](../evidence/kaggle/cpu4-ironmule-cpu4/summary.json).

## Limits of the runtime itself

**Group width is capped at 4.** Not a tuning knob: width 8 regressed in E14b and E2
and E3 independently, at the `M=8` kernel regime. Requesting more raises.

**One capacity per request.** The KV cache is fixed-shape. Each request is
allocated what it would get alone, from its own prompt length and output limit (X1:
on CUDA the cache length moves decode logits, so a shared group capacity made an
answer depend on the group). Capacity above 8192 is refused rather than allocated.

**Grouping helps only when groups fill.** With 2–3 token answers the realised width
fell to 1.83 and the gain fell below threshold in both E15 and E16. A service whose
queue empties faster than a group fills gains nothing.

**Median latency always worsens under grouping**, by 26% to 54% depending on
workload. A latency-sensitive single-request path should stay in `InteractiveMode`.

**Fallback discards work.** A failed group restarts its requests from prefill. That
is correct and wasteful; it is not an optimisation path.

**Ragged prompt lengths within a group are untested on hardware.** Every measured group used a
shared cache offset. On two Kaggle T4s (RL3-H, Qwen3 1.7B bf16) one short request was answered
differently when grouped with about 1,000-token prompts, because the group's KV capacity then
followed the long prompts and on CUDA the cache length moves decode logits (X1: up to 0.75, a
token flipped in 2 of 16 prompts within 32 tokens; the CPU build was bit-exact). Each request now
decodes at its own capacity, which the runtime contract tests check; on the T4s the re-run then
answered all 1,680 requests identically across six arms. Opt-in reuse that re-pads a stored state to another capacity (`ChatTurnsPlan`'s
turn cache) is not exact on CUDA for the same reason.

**Sustained load is untested.** All measurements used eight requests. Queue effects
that only appear under continuous arrival are unknown.

**Token identity is to IronMule's own engine, not always to mlx-lm.** Every tuned knob is checked against
IronMule's untuned engine. That engine matched mlx-lm `stream_generate` on Gemma 3 1B (8/8 chat answers of 128
tokens) but not on Gemma 3 4B on a Tesla T4 (0/8, first difference after 10-113 tokens; EXACT1). On that
combination IronMule's answers are deterministic and knob-invariant, but not mlx-lm's; mlx-lm's own
`stream_generate` and a plain greedy loop over the same model disagree there on 8/8 as well (EXACT3), so bf16
greedy output on that GPU depends on the order of the computation, and no single order is the reference.

## Limits of the evidence

**E13's quality bound is `1.14` accuracy points on a contaminated set.** SQuAD v1.1
is public and probably in the model's training data. The paired design rules out a
different task distribution between plans, but contamination raises model confidence
and therefore lowers the experiment's sensitivity to plan divergence. The bound may
be optimistic for unseen material.

**Only extractive question answering was tested for quality.** Summarisation,
reasoning, code and multi-turn were not.

**E16's frozen verdict is `CONFOUNDED_BY_PROCESS_STATE`**, assigned by two criteria
that are demonstrably misspecified: one anchored RSS growth to warmup instead of the
first repeat, the other was not scoped to the required workloads. Every instrument
that actually measures accumulation reads zero. The frozen class is reported
unchanged; the substantive reading is replication.

**Kernel counts are retired.** MLX exposes no machine-readable dispatch counter, so
no absolute dispatch time can be derived and none is claimed. `completion_wait` in
the telemetry is a wait, not GPU time: device work may already have begun during
submission.

**`W = 1` is not `SequentialService`.** They differ in interleaving as well as
synchronisation. E15 measured `G(W1) ≈ 0`, which is what makes the rest of the
comparison interpretable.

## Not implemented on purpose

- No adaptive controller. Realised width already adapts on its own, and nothing
  measured shows a controller beating a fixed width of 4.
- No true tensor batching. It diverged at batch 8 in E14b, one token, reproducibly.
- No speculative decoding. Acceptance was 0.17 per drafted token and decode was
  2.9x slower.
- No automatic plan selection. Plans change output; that is a caller decision.
