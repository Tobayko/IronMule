# Completion audit — external review, 2026-08-27

This is a skeptical, item-by-item audit of
user-supplied review attachment (local path omitted)
against the current IronMule checkout. It is an audit, not a release note and not a
performance claim. A backlog mapping is never treated as completion evidence.

## Method and status vocabulary

Every explicit item in the review is listed below. Status values are deliberately
closed: `complete`, `partial`, `open_approval`, `open_resource`, `open`, `external`,
`deprioritized`, or `rejected`. Evidence names current files, tests, raw evidence, or
states that evidence is missing. “Next gate / kill” is the smallest condition that
would close, reject, or safely defer the item.

Local evidence supplied for this audit is: `108 passed / 11 deselected` in `7.62s`
real time (`8.46s` by `/usr/bin/time`), maximum RSS `346,996,736 B`, `0` swaps;
`10/10` Gemma integration tests
in `26.58s`, maximum RSS `3,348,430,848 B`, `0` swaps; green `ironmule doctor`; green
wheel build/metadata/zipimport CLI smoke; and an R3 smoke with identical tokens but
no performance claim. The R3 JSON SHA-256 is
`36ba45933b3de344116812e34bb451d19124b0ab35db3d3ee659b768dacc6209`. The local
wheel evidence is not a clean installed-wheel run. The public benchmark still shares
one loaded process/model and has no stock `mlx_lm` arm.

## P0.1–P0.17: correctness and measurement

| ID | Status | Evidence | Next gate / kill |
|---|---|---|---|
| P0.1 first prefill token | complete | `tests/test_ironmule_runtime.py::test_prefill_token_eos_stops_and_is_counted`, `::test_max_tokens_one_stops_after_prefill_token`; local suite evidence | Keep both executor paths token/count/stop-identical; regress if either diverges |
| P0.2 TTFT definitions | partial | `engine_start_ns` is now before prefill and nonnegative; `tests/test_ironmule_runtime.py`; separate prefill/admission phase timestamps are absent | Architecture-approved lifecycle timestamps; kill if delayed-arrival and phase ordering remain unmeasurable |
| P0.3 delayed arrivals | open_approval | `Request.arrival_ms` and staggered tests exist, but prefills still occur before admission/arrival | Move admission/prefill ownership to an approved scheduler; kill any claim based on pre-arrival work |
| P0.4 end-to-end wall | complete | `ironmule/benchmark.py`, `tests/test_benchmark.py`: `outer_wall_ms` is primary and executor wall is diagnostic | Retain outer-wall denominator; reject reports using executor-only wall |
| P0.5 order/cache bias | partial | Separate arm plans, even AB/BA repeats and raw samples in `ironmule/benchmark.py`; one loaded process/model is shared | Fresh-process blocks or explicitly keep R3 open; kill cross-arm cache/process claims |
| P0.6 fused argmax | complete | `tests/test_ironmule.py::test_mlx_backend_step_honours_fused_argmax_contract` | Keep both fused/logits contract tests; kill unsupported profile reuse |
| P0.7 correctness status | complete | `Telemetry.correctness_check_performed` and checked-request fields; `tests/test_ironmule_runtime.py::test_telemetry_does_not_present_zero_as_a_correctness_check` | Require the status field wherever correctness is displayed; no shadow-check performance claim |
| P0.8 stock MLX baseline | open_approval | No stock `mlx_lm` arm; public benchmark intentionally has only IronMule modes | Define exact prompt/stop/revision contract and obtain architecture approval; otherwise do not add arm |
| P0.9 mismatch exit | complete | `tests/test_benchmark.py::test_main_mismatch_exits_nonzero_and_persists_structured_diff`; JSON diff and exit 2 | Keep first-token/stop/count context; kill any zero exit on mismatch |
| P0.10 profile validation | partial | System conditions fail closed; `tests/test_r6_r7.py` profile rejection coverage | Complete model-bound profile validation only with P0.11 identity contract; kill stale profile reuse |
| P0.11 revision/quantisation | open_approval | Current fingerprint/profile evidence does not prove qualified model revision and quantisation identity | Approved model hash/revision/bits/group-size contract plus tests; kill any broadened validity claim |
| P0.12 prompt drift | complete | `tests/test_r6_r7.py::test_revalidate_uses_tokenized_prompt_length_from_canary` | Preserve current-tokenization check; kill stored-length-only revalidation |
| P0.13 import environment | complete | `tests/test_r6_r7.py` and isolated CLI probes; no import-time global policy claim remains | Keep caller environment unchanged; kill subprocess probe that mutates parent state |
| P0.14 dependency bounds | complete | `pyproject.toml` qualified bounds and metadata tests | Re-run against every supported MLX pair before widening bounds |
| P0.15 unsupported candidates | complete | `tests/test_r6_r7.py` typed unsupported-candidate continuation | Require typed disposition in every tuning result; kill abort-on-one-candidate behavior |
| P0.16 hardware discovery cache | complete | `tests/test_r6_r7.py::test_static_facts_queries_system_profiler_once_per_process` | Keep process cache; persistent cache remains a separate scoped decision |
| P0.17 version source | complete | `tests/test_r6_r7.py` metadata/version checks and current package metadata | Keep one authoritative version source; kill release if package/tag diverge |

## Product improvements 1–24

| # | Status | Evidence | Next gate / kill |
|---:|---|---|---|
| 1 persistent `ironmule serve` | open_approval | No daemon/service command; `S1` remains design work | Approve loopback service contract and bounded queue; kill unbounded service scope |
| 2 OpenAI-compatible API | open_approval | No `/v1` server routes | Define exact request/stream/error contract after service approval; kill unsupported compatibility promises |
| 3 token streaming | open | `Result` is returned after completion; no stream API | GPU-readback streaming test; kill simulated/post-hoc streaming |
| 4 durable queue | open_approval | `Runtime.serve()` accepts a fixed sequence, not arrivals over time | Scheduler design with queue/backpressure tests; kill pre-admission prefill |
| 5 scheduler-owned prefill | open_approval | Current prefill happens before executor admission | Resolve R2/P0.3 architecture gate; kill service-TTFT claims before that |
| 6 cancellation | open | No cancellation/disconnect contract or tests | Add cancellation state machine and device-safe tests; kill if partial groups leak state |
| 7 time limits/queue cap | open | No persistent service queue limit | Specify timeout semantics and bounded queue tests; kill silent unbounded waiting |
| 8 backpressure | open | No service admission/backpressure implementation | Resource-gated overload test; kill if memory grows without bound |
| 9 separate interactive/batch lanes | open | Only caller-selected modes exist | Service lane policy and tail-latency evidence; kill automatic mode switching without contract |
| 10 async Python API | open | No async API | API plus cancellation/ordering tests; kill wrapper that hides blocking work |
| 11 capacity buckets | open_approval | Current `serve()` sizes one capacity from longest prompt; limitation documented | Architecture approval and mixed-length memory/correctness matrix; kill any cache-state divergence |
| 12 cache LRU | open_approval | `PrefixCache` is a bounded single-plan snapshot, not LRU | Budget/eviction/tenant tests; kill unbounded cache growth |
| 13 multiple capacities/session | partial | Capacity matching exists, but no multi-capacity session store | Add explicit capacity classes and hit/miss evidence; kill implicit cross-capacity reuse |
| 14 thread/process safety | open | No concurrent-call/thread safety contract | Race/isolation tests and service ownership decision; kill shared mutable state |
| 15 multi-turn chat | open | Single rendered user prompt helper only | Model-family template tests; kill exact-mode ambiguity across turns |
| 16 system/user/assistant roles | open | No public multi-role API | Role/template contract and tests; kill undocumented role flattening |
| 17 custom stop sequences | open | Only backend EOS IDs are supported | Stop matcher contract and token-identity tests; kill accidental EOS-only wording |
| 18 temperature/top-p/seed | open | Runtime is exact greedy | Separate seeded sampling mode and quality/distribution tests; kill mixing sampling into exact mode |
| 19 compatibility register | open | No model-family compatibility registry | Revision/family matrix with explicit unsupported states; kill optimistic fallback |
| 20 automatic context/memory limits | partial | Capacity calculation and memory telemetry exist; no service policy | Define limits/OOM behavior and tests; kill any claim of automatic protection |
| 21 health/readiness endpoints | open | `ironmule doctor` is a CLI prerequisite check, not a service endpoint | Service health contract; kill readiness that does not test model/device state |
| 22 JSON logging/metrics export | partial | Benchmark `--json` and telemetry snapshots exist; no service exporter | Stable schema/export tests; kill conflating diagnostics with correctness checks |
| 23 warm model | partial | `Runtime.load()` loads a model for a process; no persistent daemon | Warm-service lifecycle and memory evidence; kill startup claims from one-shot load |
| 24 Homebrew/`uv tool` install | open | Package metadata/pip path only | Packaging approval and clean install matrix; kill install claim without clean-wheel evidence |

## Product detail sections 4.1–4.8

| Section | Status | Evidence | Next gate / kill |
|---|---|---|---|
| 4.1 `ironmule serve` | open_approval | No command or endpoint implementation; `S1` backlog is not evidence | Approved loopback MVP with health, queue and failure contract |
| 4.2 token streaming | open | No stream method or GPU-readback event path | End-to-end streamed-token test; kill buffered imitation |
| 4.3 dynamic grouping | partial | `ThroughputMode` groups requests supplied together; no arrival-time collector | Measure bounded collection windows in a service; kill added tail latency beyond budget |
| 4.4 capacity bucketing | open_approval | One `serve()` capacity follows longest prompt | Approved bucket policy plus memory/token regression matrix |
| 4.5 prefix-cache management | partial | `PrefixCache` hit/miss/capacity behavior tested; no LRU/budget/expiry/tenant controls | Add explicit ownership and eviction tests; kill cross-prompt reuse |
| 4.6 prefix mismatch rejection | open | Prefix matching helper exists, but public reusable plan does not require match | Default `ValueError` contract and test; kill silent chunked fallback |
| 4.7 multiple chat messages | open | No `generate_messages` API | Per-family template tests and API review |
| 4.8 separate sampling mode | open | Exact greedy only | Seeded sampling design and quality gate; kill approximate behavior in exact mode |

## Test, CI and security items 1–21

| # | Status | Evidence | Next gate / kill |
|---:|---|---|---|
| 1 GitHub Actions unit tests | partial | `.github/workflows/ci.yml` exists; remote job has not run | Successful remote macOS run; kill green-by-file-presence assumption |
| 2 build/fresh install | partial | Local wheel build/metadata/zipimport smoke only; no clean installed-wheel run | Execute clean install job remotely; kill installation claim before it runs |
| 3 CLI smoke | complete | Local `doctor`, info/help and zipimport smoke evidence | Preserve dependency-light smoke; kill model download in smoke |
| 4 Apple-Silicon test system | partial | Local Gemma integration `10/10`; no CI matrix | Reproducible runner/resource approval; kill general chip claim |
| 5 nightly model test | open_resource | No nightly workflow/model run | Approved model cache/runner and retention policy; kill unbounded spend/download |
| 6 first EOS token | complete | `tests/test_ironmule_runtime.py` EOS regression | Keep sequential/grouped parity |
| 7 `max_tokens=1` | complete | `tests/test_ironmule_runtime.py` cap regression | Keep prefill token as total-cap member |
| 8 every tuning profile | partial | `tests/test_r6_r7.py` profile identity and unsupported cases | Enumerate all knob combinations/model families; kill incomplete profile coverage |
| 9 very different prompt lengths | partial | Heterogeneous synthetic prompts and caps are tested; no full service matrix | Add preregistered real-length matrix; kill extrapolated performance claim |
| 10 property-based scheduler tests | open | No property-based suite | Add bounded generated scheduler corpus; kill before claiming exhaustive scheduling |
| 11 1/6/24-hour load | open_resource | No duration runs | Resource-approved soak with RSS/MLX/swap/TTFT gates |
| 12 OOM/device simulation | partial | Injected device failure/fallback tests exist; OOM is not covered | Controlled OOM/rollback test; kill unsafe worker execution |
| 13 abort/disconnect | open | No cancellation or disconnect tests | Service architecture plus deterministic disconnect tests |
| 14 Python/MLX matrix | open_resource | One local qualified environment only | Runner/version matrix with exact raw evidence |
| 15 coverage report | open | No checked-in coverage artifact or threshold | Add reproducible coverage job; kill percentage without scope |
| 16 Ruff/Mypy/Pyright | open | No configured static-analysis gate | Choose tool and baseline; kill lint-only confidence |
| 17 Dependabot/dependency scan | open | No configured scan evidence | Security workflow and triage policy |
| 18 CodeQL | open | No CodeQL workflow/result | Add approved CodeQL workflow; kill unreviewed alert closure |
| 19 SBOM | open | No release SBOM | Generate and verify per artifact |
| 20 signed artifacts | open | No signing/provenance workflow | Key/provenance approval and verification test |
| 21 official tag/release | open_approval | No new release created in this audit | Explicit release approval after residual gates close |

### Section 5.1: integration skip policy

| Item | Status | Evidence | Next gate / kill |
|---|---|---|---|
| 5.1 skip only model/access/Metal unavailability | complete | `tests/test_ironmule_runtime_integration.py` classifier tests; unexpected errors propagate | Keep allowlist narrow; kill broad `except Exception: skip` |

### Section 5.2: all 18 boundary tests

| # | Boundary test | Status | Evidence | Next gate / kill |
|---:|---|---|---|---|
| 1 | first token EOS | complete | Runtime executor tests | Preserve EOS count/stop contract |
| 2 | `max_tokens=1` | complete | Runtime executor tests | Preserve total physical cap |
| 3 | empty text after EOS | complete | `tests/test_ironmule_runtime.py::test_runtime_serve_prefill_eos_result_contract` | Preserve empty visible text and EOS stop contract |
| 4 | one request ends, others continue | complete | Ragged scripted schedules and early finisher tests | Keep per-session state isolation |
| 5 | all requests end first step | complete | `tests/test_ironmule_runtime.py::test_grouped_all_prefill_eos_has_no_decode_round_or_width` | Preserve no-decode/no-width behavior for all-EOS groups |
| 6 | very different prompt lengths | complete | Heterogeneous prompt test corpus | Add real-model matrix before general claim |
| 7 | very different output lengths | complete | Ragged caps/early finish tests | Preserve stop/count parity |
| 8 | matching-capacity cache hit | complete | Prefix cache copy/hit tests | Add service-level reusable hit evidence |
| 9 | different-capacity cache miss | complete | Prefix capacity mismatch test | Preserve exact capacity keying |
| 10 | prefix mismatch | partial | `PrefixCache.matches` negative test; no default plan rejection | Add public `ValueError` contract |
| 11 | `fused_argmax=True` profile | complete | Backend fused/logits unit tests | Add real profile gate when model exists |
| 12 | old MLX profile | partial | Framework identity rejection coverage exists | Complete with explicit qualified-version fixtures |
| 13 | changed model revision | open_approval | No revision-bound local model fixture | P0.11 identity contract and fixture |
| 14 | client abort during group | open | No client/service cancellation path | Service design and deterministic abort test |
| 15 | device error after partial group | complete | Injected grouped device failure/fallback tests | Add real-device gate only with resource approval |
| 16 | memory limit exceeded | open_resource | No controlled OOM test | Isolated worker/resource limit and rollback evidence |
| 17 | two concurrent `serve()` calls | open | No concurrency contract/test | Define ownership and race test |
| 18 | multiple threads on one runtime | open | No thread-safety contract/test | Synchronization design and stress test |

### Section 5.3: duration/load profiles and metrics

| Profile/metric | Status | Evidence | Next gate / kill |
|---|---|---|---|
| 10-minute smoke | open_resource | No duration run | Resource-approved run with bounded memory |
| 1-hour stability | open_resource | No duration run | RSS/swap/cache-growth gate |
| 6-hour soak | open_resource | No duration run | Thermal/cache/recovery gate |
| 24-hour long-term | open_resource | No duration run | Explicit production-like resource approval |
| 1,000-request burst | open_resource | No burst runner | Queue/backpressure evidence |
| continuous arrival rate | open_resource | No persistent service | Admission/queue architecture first |
| RSS | partial | Local maximum RSS values recorded | Repeat per load profile; kill single-snapshot inference |
| MLX peak memory | partial | Runtime telemetry exposes peak memory | Repeat per matrix cell |
| swap | partial | Local evidence reports zero swaps | Record baseline/delta during every load run |
| thermal state | partial | Environment/benchmark captures it | Require stable thermal protocol |
| throughput | partial | Existing research ledger and benchmark code | New protocol/raw samples before claim |
| TTFT p50/p95/p99 | partial | p50/p95 telemetry exists; p99/load evidence absent | Add p99 and duration matrix |
| fallbacks | complete | Telemetry and fallback tests | Preserve reason records |
| aborts | open | No abort implementation | Add service cancellation first |
| queue length | open | No persistent queue | Instrument approved scheduler |
| cache hit rate | partial | Prefix cache hit/miss counters/tests exist | Export per-cell hit-rate evidence |

## Performance B1–B26 and B28

These are hypotheses or research rows, not completion claims. Existing ledger/raw files
are cited only when they contain a directly relevant measurement; the backlog itself is
not evidence.

| ID | Status | Evidence | Next gate / kill |
|---|---|---|---|
| B1 width at 27B | open_resource | 4B width evidence exists; no qualified 27B sweep | Run preregistered widths with token gate; kill if width 4 wins/noise |
| B2 group output head | open | Backlog mechanism only; no new exact-logit study | Bitwise logits corpus; kill any nonzero logit difference |
| B3 unrolled decode | open | Prior fixed-cache compile evidence is historical | Fresh paired fixed-shape test; kill token divergence/compile loss |
| B4 wired weights/page pressure | open_resource | No controlled wired-limit run | Loaded-machine swap experiment; kill no reproducible effect |
| B5 bounded group wait | open | No wait-bound sweep | Measure throughput and tail latency; kill if latency trade loses |
| B6 M=4 vs M=1 cost | open | Historical control rationale only | Paired cost measurement; kill if no explanatory value |
| B7 time-share instrumentation | open | Current telemetry has service/engine diagnostics but no full kernel breakdown | Instrument before kernel work; kill any unsupported attribution |
| B8 native decode loop | open_approval | No native loop implementation | Architecture approval and microbenchmark; kill absent absolute gain/correctness |
| B9 replay recorded GPU flow | open_approval | No implementation | Controlled graph/replay design; kill if data-dependent output cannot be exact |
| B10 fewer kernels | open_approval | No profiler-backed candidate | B7/profiler evidence first; kill guess-driven kernel work |
| B11 layer pipelining | open_approval | No implementation | Microbenchmark plus exactness gate |
| B12 width 16 | deprioritized | Historical width-8 token divergence blocks priority, but does not kill the untested width-16 hypothesis | Revisit only with a new exactness-first design; kill on any token/cache divergence or failed fixed gate |
| B13 draft model | partial | Prior predecessor draft route rejected/slower; target-family route unmeasured | Measure acceptance/cost for target; kill below preregistered gate |
| B14 MTP/draft head | open_approval | No implementation or trained head | Model-specific design and quality gate |
| B15 reduced output head | open_approval | No exact bound proof | Prove token-safe pruning before timing |
| B16 mixed precision | open_approval | No approved quality matrix | Separate approximate mode and quality gate |
| B17 KV quantisation | open_approval | No qualified exact/quality result | Separate mode and long-context evidence |
| B18 adaptive layer skip | deprioritized | Not valid for the exact mode because of quality risk; a separate approximate mode remains possible | Keep out of exact mode; revisit only with explicit approximate-mode quality gate |
| B19 Gemma local/global layers | open | No new model-specific measurement | Model-specific profile and exactness gate |
| B20 Neural Engine head | open_approval | No public programmable-kernel evidence | Public API feasibility proof; kill unsupported ANE claim |
| B21 CPU projections | open_approval | No controlled overlap measurement | Profiler-backed experiment; kill if memory/latency worsens |
| B22 two GPU processes | open_resource | No current control run in this audit | Controlled resource experiment; kill if no explanatory value |
| B23 M=1 weight layout | open_approval | No implementation | Profiler/format evidence first |
| B24 real GPU counters | open_approval | MLX counter evidence absent | API/reproducible counter proof; kill invented dispatch numbers |
| B25 KV reallocation | open | No new measurement | Controlled allocation experiment; kill no stable benefit |
| B26 Qwen/Gemma scaling | open_resource | Qwen compatibility work exists; no qualified comparative performance | Exact same protocol/revision/resources; kill family claim without raw data |
| B28 Qwen hybrid cache | partial | `tests/test_qwen_hybrid_integration.py`, X2 compatibility raw evidence and B28 harness; no qualifying performance result | Complete correctness/resource gates and preregistered measurement; kill token/cache divergence or no gate |

## Performance detail sections 6.1–6.4

| Section | Status | Evidence | Next gate / kill |
|---|---|---|---|
| 6.1 B7 instrumentation first | open | Existing telemetry is not a kernel/submit/readback decomposition | Add measured phase breakdown before native/kernel work; kill unsupported bottleneck story |
| 6.2 B1 large-model width | open_resource | No qualified 27B width sweep in supplied evidence | Run widths with correctness-before-speed gate |
| 6.3 B8 native decode | open_approval | No native decode path | Architecture approval, absolute tok/s and exact token gate |
| 6.4 B13 draft acceptance | partial | Historical rejection is not target-family acceptance evidence | Measure acceptance/cost on target; kill below decision threshold |

## Hardware, models, quantisation and workload matrix

| Matrix dimension | Status | Evidence | Next gate / kill |
|---|---|---|---|
| M1 | open_resource | No base-M1 result | Community/runner cell with raw evidence |
| M1 Max | complete | Local validated machine is M1 Max; local Gemma integration | Do not generalize beyond this cell |
| M1 Pro | open_resource | No result | Community/runner cell with raw evidence |
| M1 Ultra | open_resource | No result | Same |
| M2 Pro | open_resource | No result | Same |
| M2 Max | open_resource | No result | Same |
| M2 Ultra | open_resource | No result | Same |
| M3 Pro | open_resource | No result | Same |
| M3 Max | open_resource | No result | Same |
| M4 | open_resource | No result | Same |
| M4 Pro | open_resource | No result | Same |
| M4 Max | open_resource | No result | Same |
| different memory sizes | open_resource | No comparative cells | Record RAM, swap and peak memory per cell |
| Gemma | partial | Primary Gemma evidence and 10/10 local integration | More revisions/loads before family-wide claim |
| Qwen | partial | X2 compatibility and local-only integration gate | Performance remains unqualified; resource gate |
| Llama | open_resource | No result | Model/revision-compatible cell |
| Mistral | open_resource | No result | Same |
| Phi | open_resource | No result | Same |
| under 4B | open_resource | No matrix | Exact model/revision raw run |
| 7B–14B | open_resource | No matrix | Same |
| 20B–32B | partial | Exploratory 12B/27B evidence only | Repeated qualified cells; no extrapolation |
| 2-bit | open_resource | No quality/performance cell | Separate quantisation gate |
| 3-bit | open_resource | No quality/performance cell | Same |
| 4-bit | partial | Qualified Gemma 4-bit group-size 64 cell | Keep revision/group-size bound |
| 6-bit | open_resource | No cell | Same |
| 8-bit | open_resource | No cell | Same |
| BF16 | open_resource | No cell | Memory-approved reference |
| context 128 | open_resource | No dedicated cell | Matrix run |
| context 512 | open_resource | No dedicated cell | Matrix run |
| context 1,024 | partial | Existing contexts include 1024 boundary evidence | Exact cell and raw output |
| context 2,048 | partial | Existing ledger context range reaches 2048 | Exact cell and raw output |
| context 4,096 | open_resource | No cell | Matrix run |
| context 8,192 | open_resource | No cell | Matrix run |
| context beyond 8,192 | open_resource | No cell | Model-specific approval |
| output 2–3 tokens | partial | Short-answer evidence exists | Repeat under current protocol |
| output 16 tokens | open_resource | No dedicated cell | Matrix run |
| output 48 tokens | partial | Current benchmark/B28 workload uses 48 | Keep raw exactness evidence |
| output 128 tokens | open_resource | No cell | Matrix run |
| output 512 tokens | open_resource | No cell | Memory/resource gate |
| output until EOS | partial | EOS boundary tests; no matrix | Add unconstrained-EOS workload |
| concurrency 1 | complete | Interactive/runtime unit and local smoke | Do not infer grouped gain |
| concurrency 2 | partial | Grouped fake/runtime coverage | Real-model cell |
| concurrency 4 | partial | Group width cap and local Gemma integration | Real-model repeated cell |
| concurrency 8 | partial | Historical validity domain; no current completion audit run | Repeated raw cell |
| concurrency 16 | open_resource | No cell | Resource-approved service run |
| concurrency 32 | open_resource | No cell | Same |
| continuous input | open_resource | No persistent service | Service architecture first |
| all-at-once arrival | partial | Fixed `serve()` batches | Arrival-owned service test |
| evenly distributed arrival | partial | Synthetic staggered arrivals | Pre-arrival prefill defect remains |
| random arrival | open_resource | No test | Scheduler/resource gate |
| bursts | open_resource | No burst run | Backpressure gate |
| sustained high load | open_resource | No duration run | Soak gate |
| alternating long/short | partial | Heterogeneous synthetic cases | Real mixed-load matrix |
| normal chat | partial | Single user prompt helper | Multi-turn API and template tests |
| summarization | open_resource | No dedicated quality cell | Task corpus |
| document questions | partial | Apollo/document question tests | Broader model/task matrix |
| RAG | open_resource | No retrieval integration | Fixture and leakage policy |
| programming | open_resource | No quality cell | Private code corpus |
| JSON output | open_resource | No structured-output contract | Schema/validity tests |
| tool calling | open_resource | No tool-call API | Service/API design |
| long reasoning | open_resource | No quality cell | Cost/quality gate |
| translation | open_resource | No quality cell | Multilingual corpus |
| extraction | partial | Extractive Apollo/Qwen questions | Uncontaminated corpus |
| short classification | open_resource | No quality cell | Task corpus |
| code quality | open_resource | No code benchmark | Private held-out tasks |
| summary quality | open_resource | No summary benchmark | Held-out human/metric protocol |
| instruction following | open_resource | No dedicated test | Task corpus |
| long documents | partial | Long document/prefix tests exist | Broader context matrix |
| structured JSON quality | open_resource | No schema evaluation | Validity/semantic checks |
| multi-step conversation | open_resource | No multi-turn API | Conversation corpus |
| private unknown data | open_resource | No private corpus in repo | Approved non-public evaluation |
| German tasks | open_resource | No language matrix | Multilingual evaluation |
| multiple model families | open_resource | Gemma/Qwen only, Qwen performance open | Three-family evidence before claim |

## Community benchmark bundle and submission

| Item | Status | Evidence | Next gate / kill |
|---|---|---|---|
| structured collection schema | complete | `COMMUNITY_BENCHMARKS.md` required fields and empty reviewed table | Keep one fingerprint per row |
| benchmark bundle command | open | No `--bundle` implementation | Implement deterministic artifact bundle; kill private-prompt leakage |
| complete console/JSON/hardware/model evidence | partial | Current benchmark JSON/raw schema covers local protocol; no bundle command | Bundle schema test and checksum |
| optional submission command | open_approval | No `--submit`; issue template exists | Preview plus explicit consent and no auto-upload |
| external results reviewed | open | `COMMUNITY_BENCHMARKS.md` says none reviewed | Review first external submission with raw evidence |

## Documentation and positioning 9.1–9.5

| Section | Status | Evidence | Next gate / kill |
|---|---|---|---|
| 9.1 README simplification/order | partial | README remains evidence-rich; P0 benchmark wording corrected | User-review pass without dropping caveats |
| 9.2 cautious M1–M4 wording | complete | README explicitly says designed for Apple Silicon, validated primarily on M1 Max, and labels results preregistered/exploratory; other chips are community evidence only | Preserve evidence-linked labels; no M1–M4 generalization |
| 9.3 visible demo | open | No terminal video/demo artifact | Record reproducible demo; kill fabricated comparison |
| 9.4 decision overview | complete | README includes the compact single-chat/latency, concurrent-throughput, shared-document and short-answer mode table with caveats and runtime links | Keep recommendations subordinate to workload measurements |
| 9.5 complete CLI | partial | `doctor`, `info`, `benchmark`, lazy `tune`/`revalidate`/`status`, and read-only local-cache `models` route now exist; no serve/cache-management commands by design | Add further commands only with explicit contracts; service/cache scope remains open |

## Release strategy 0.1.1–1.0

| Version | Status | Evidence | Next gate / kill |
|---|---|---|---|
| 0.1.1 | partial | Changelog/action plan record P0 fixes; R2/R3/R6/R8 residuals remain | Close or explicitly approve residuals; no release before wheel/CI/model identity gates |
| 0.2.0 | open_approval | Service/streaming/queue work absent | Approve S1 and loopback MVP |
| 0.3.0 | open | Compatibility/cache/load/community work absent | Complete C1/Q1/D1 evidence |
| 0.4.0 | open | B7/width/memory/native microbenchmarks absent | Complete research gates with raw data |
| 1.0 | open_resource | Multi-chip/family/quantisation/24-hour/service requirements unfulfilled | Resource-approved matrix and reliability gates; kill premature stability claim |

## Licence points 1–12 and business offers

| # | Licence point | Status | Evidence | Next gate / kill |
|---:|---|---|---|---|
| 1 | independent legal review | open_approval | No independent legal review artifact | Obtain counsel review; kill legal-compliance claim before completion |
| 2 | developer one-page summary | complete | `COMMERCIAL.md`, `LICENSE.md` plain-language sections | Keep summary subordinate to binding terms |
| 3 | company one-page summary | complete | `COMMERCIAL.md` production/evaluation bands | Legal consistency review |
| 4 | permitted/paid examples | complete | `COMMERCIAL.md` concrete use boundaries | Keep examples synchronized with licence |
| 5 | consulting-company rule | open_approval | No explicit dedicated consulting rule | Add reviewed boundary example |
| 6 | SaaS rule | complete | `COMMERCIAL.md` hosted/managed service boundary | Keep binding-term cross-check |
| 7 | internal enterprise production | complete | `COMMERCIAL.md` and LICENSE sections 10/12 | Legal review of edge cases |
| 8 | fork rule | complete | `COMMERCIAL.md` states permitted forks and production boundary | Preserve notice obligations |
| 9 | contact/licence process | partial | GitHub issue/profile path exists; placeholder email remains | Replace placeholder and confirm process owner |
| 10 | commercial licence template | open_approval | No model contract | Counsel-approved template |
| 11 | no licence phone-home telemetry | complete | No licence-control telemetry path in source/docs | Keep no-registration/no-phone-home promise |
| 12 | future dual licensing | open_approval | Only discussed as future option | Legal/business decision; no implied commitment |
| business offer: production licence | partial | Pricing/bands described in `COMMERCIAL.md`; no signed process | Approved contract and enquiry workflow |
| business offer: enterprise support | open | No support SLA/offer | Define separately; kill implied warranty |
| business offer: new model families | open | No service offering | Scope and support terms |
| business offer: custom profiles | open | No offering workflow | Scope, reproducibility and liability terms |
| business offer: Mac fleet management | open | No fleet product | Product/security design |
| business offer: benchmark/audit reports | open | No service artifact | Method and independence policy |
| business offer: integration support | open | No offering workflow | Contract scope |
| business offer: prioritized fixes | open | No support process | SLA/triage decision |
| business offer: long-term versions | open | No maintenance policy | Version/support policy |

## Explicit non-prioritisation points

| Item | Status | Evidence | Next gate / kill |
|---|---|---|---|
| large graphical UI | deprioritized | No GUI; terminal/API are current scope | Revisit only after service need is demonstrated |
| Docker-first Apple GPU execution | deprioritized | Native macOS/Metal is current path | Revisit only for auxiliary services |
| universal “up to 5x” claim | deprioritized | Docs now narrow TTFT wording and validity domain | Kill any absolute universal speed claim |
| true tensor batching before divergence resolved | deprioritized | Historical width-8 divergence; B28 remains gated | Exactness/resource approval first |
| approximate optimisations in exact mode | deprioritized | Exact greedy contract and separate-mode policy | Only separate approved mode with quality gate |

## Recommended sequence, item by item

| Step | Status | Evidence | Next gate / kill |
|---:|---|---|---|
| 1 first token/EOS/cap | complete | P0.1 tests and local suite | Preserve contract |
| 2 TTFT timestamps | partial | Pre-prefill engine start fixed; lifecycle remains open | R2 architecture gate |
| 3 end-to-end wall | complete | Benchmark outer-wall tests | Keep denominator |
| 4 benchmark order/cache bias | partial | Plans/AB-BA/repeats fixed; shared process remains | Fresh-process decision |
| 5 fused argmax | complete | Backend unit tests | Keep both contracts |
| 6 stored profile validation | partial | System fail-closed tests; identity open | P0.11 |
| 7 import environment | complete | Isolated import/probe tests | Preserve parent state |
| 8 dependency bounds | complete | Metadata/tests | Requalify before widening |
| 9 CI and boundary tests | partial | Workflow and broad local tests; remote clean install absent | Remote CI/resource gate |
| 10 official 0.1.1 release | open_approval | Residual R2/R3/R6/R8 | Explicit release approval |
| 11 `ironmule serve` | open_approval | S1 absent | Service architecture |
| 12 streaming | open | No stream API | GPU-readback stream gate |
| 13 durable queue | open_approval | No persistent queue | Admission design |
| 14 dynamic grouping | partial | Fixed-batch grouping only | Arrival scheduler |
| 15 cancellation/backpressure | open | No service contract | Failure/load tests |
| 16 OpenAI API | open_approval | No API routes | Contract approval |
| 17 capacity buckets | open_approval | Single capacity per serve | Memory/correctness matrix |
| 18 cache management | partial | Single prefix snapshot | LRU/budget design |
| 19 B7 instrumentation | open | Full phase breakdown absent | Measure before optimize |
| 20 width at 12B/27B | open_resource | No qualified sweep | Resource-approved repeats |
| 21 Qwen/Gemma comparison | open_resource | Qwen compatibility only | Same revision/protocol/resources |
| 22 memory pressure/KV allocation | open_resource | No controlled run | Swap/RSS gate |
| 23 longer contexts | open_resource | Narrow current range | Matrix evidence |
| 24 native decode microbenchmarks | open_approval | No native path | Architecture/perf gate |
| 25 draft models/custom kernels | deprioritized | Downstream of B7/exactness and explicitly gated | Do not start before prerequisites |

## Bottom line

The review's completed local correctness fixes are real and evidenced. The remaining
completion blockers are not hidden by this audit: R2 admission/lifecycle architecture,
R3 fresh-process/stock-arm scope, R6 model identity, R8 remote CI/clean installation,
resource-heavy matrix/load work, and product/service architecture remain open. No
performance conclusion is upgraded by this document.
