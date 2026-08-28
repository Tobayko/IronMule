# B39 — Combined core profile and service mode

Experiment ID: B39
Registered: 2026-08-28, before any B39 process
Status: exploratory qualification only; no activation

The exact local Gemma 3 12B snapshot and the X1-strict workload are frozen:
the first six benchmark questions, six fresh q0-q5 requests per repeat,
StrictOneShotPlan per request, max_tokens 48 and greedy decoding.

Four explicit arms are tested:

- A: Knobs() with InteractiveMode
- B: Knobs() with ThroughputMode(max_width=4)
- C: compiled_fixed_cache=True and head_skip_prefill=True with InteractiveMode
- D: the same core knobs with ThroughputMode(max_width=4)

Eight balanced blocks use orders ABDC, BCAD, CDBA, DACB, DACB, CDBA, BCAD,
ABDC. Each arm gets one fresh serial process, one model load, two warmups and
five measured repeats. A pilot consists of exactly one complete four-arm block
with two warmups and one measured repeat and is never promotional.

The primary comparisons are D/A and D/B wall-time ratios, calculated from
block medians. B/A, C/A and D/C are descriptive. Physical and visible rate
ratios use rate_X/rate_Y. All ratios use deterministic 10,000-resample block
bootstraps; normal 95% CIs are descriptive and Bonferroni-conservative 97.5%
CIs are required for D/A and D/B family decisions. Interaction
I = D*A/(B*C) is diagnostic only. Material position/order/epoch drift is
INCONCLUSIVE, not REJECTED.

Every child records complete per-request physical/logical/visible token arrays,
stop reasons, counts, service/engine TTFT, latency p50/p95, queue wait, realized
width, fallbacks, peak memory, RSS, swap, crash snapshots and fingerprints.
Any fallback, correctness error, missing request record, identity mismatch,
timeout, crash, relevant crash report, missing instrumentation, peak over 12 GiB,
candidate/core peak ratio over 1.10 or swap increase over 256 MiB is a hard
failure. No retry is allowed. The parent writes an atomic partial sidecar after
each child and activation_allowed is always false.

The result is QUALIFIED only as an exploratory, exact-scope observation when
all hard gates pass, D/A and D/B wall medians are below 0.995 with upper CIs
below 1, physical and visible rate lower CIs exceed 1, and no material drift
is present. X1's 0.8458 is a historical stretch flag only and never a gate.
