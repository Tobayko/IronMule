# PROD2 bounded-prefetch pilot — preregistration

This document freezes the small real-hardware pilot for the optional
`bounded_prefetch` worker variant. It is not a production activation decision
and it makes no superiority claim. The reference variant remains the default.

The report also records start time, the actual MLX device-info dictionary and
the installed stock `mlx_lm/generate.py` SHA-256. The source manifest includes
the complete `ironmule/*.py` import surface, not just its statistics helper.
Request wall time is a conservative accounting bound for inference work;
model-load wall time is reported separately and is not labelled GPU time.

## Amendment before 1B audit attempt 2

Attempt 1 is retained as failed. The worker loaded, but the first cell's
readiness checkpoint raised `NameError` because `os.cpu_count()` lacked its
module import. No generation call or forward-count result was obtained. Add
the missing import and exercise the real checkpoint before restarting with a
new report. No threshold, schedule, budget or interpretation rule changes.

## Amendment before 1B audit attempt 3

Attempt 2 is retained as a readiness failure, with no generation calls: the
observed one-minute CPU load rose above the unchanged 0.8-per-logical-core gate.
Readiness is now also checked before loading a worker, and every accepted or
rejected numeric checkpoint is persisted before its gate is applied. Do not
lower the threshold or terminate unrelated user/system jobs to obtain a run.
The separate CI-discovered worker-start repair adds Python isolated mode to
prevent the package's `types.py` from shadowing the standard library; the
generation algorithm remains unchanged. All attempted runs stay separate.

## Amendment before 1B full pilot attempt 2

The 1B audit attempt 3 passed all six traced calls, including exact outputs and
forward-count deltas. Full pilot attempt 1 then stopped after four measured
calls: `summarise` was imported in `run`, not in the function using it. The
artifact retains those calls; it contains no completed AB/BA pair and cannot
support an effect estimate. Move that import to its actual scope and add a
pinned static undefined-name check (Ruff 0.16.6) to CI. No schedule, gate,
budget or claim changes; the full pilot restarts as a separate new attempt.

## Scope and exact baseline

The candidate is derived from the installed `mlx-lm==0.31.3` greedy
`generate_step`. The stock loop computes `next_y` before yielding, including
before the final visible token at a finite limit. The candidate omits only that
final unused forward when its fresh private cache is discarded. Prompt-prefill
segmentation and every earlier visible-token operation remain unchanged.

The pilot accepts one exact cached Gemma model ID and binds its snapshot
revision and direct `model*.safetensors` byte total. The worker independently
recomputes that byte total, rejects a changed registration, and refuses weights
at or above the device's reported `max_recommended_working_set_size`.

## Invocation

The harness is inert without an explicit new output path and `--execute`:

```bash
.venv/bin/python tools/product_bounded_bench.py \
  --execute \
  --model mlx-community/gemma-3-4b-it-4bit \
  --output experiments/prod2_bounded_prefetch_2026-09-07.json
```

`--audit-only` uses one fresh worker and runs traced reference/candidate warmups
at limits `1`, `8`, and `32`. It always records `performance_claim: false` and
`activation_allowed: false`; it is a forward-count audit, not a timing result.

## Registered schedule

Full mode uses three fresh workers. Their limit orders are respectively:

```text
worker 0: 1, 8, 32
worker 1: 8, 32, 1
worker 2: 32, 1, 8
```

Each limit has exactly eight serialized HTTP calls:

1. traced reference warmup;
2. traced candidate warmup;
3. untraced reference A/A request one;
4. untraced reference A/A request two;
5. untraced reference then candidate pair;
6. untraced candidate then reference pair.

That is `3 workers × 3 limits × 8 calls = 72 real calls per model`. Every
worker has one loaded `MLXWorkerClient`. Two fixed-variant forwarding leases
connect it to two actual loopback `ProductService` + HTTP servers. The servers
are called sequentially; no fabricated tokens or fake model output is used.

The lease records the underlying token IDs, text, finish reason, prompt-token
count, completion-token count, done metrics, and (when requested) model-forward
count before forwarding the result to HTTP. Every call is compared with the
reference output for that limit. A traced forward-count delta is valid only as
`reference − candidate = 1` when `completion_tokens == limit`, otherwise `0`.
Timed calls never enable tracing.

## Gates and accounting

The shared `_bench.harness_preconditions` and `friday_evidence.BudgetGuard`
are mandatory. The registered limits are 120 seconds total GPU work, 6 seconds
continuous work, 4 seconds required break after every call, 60 seconds between
workers, and a 20-minute wall bound. The harness requires AC power and reads
Low Power mode from Apple's public `ProcessInfo.isLowPowerModeEnabled` flag; it
does not infer Low Power from a missing `pmset` field. Each cell also requires
ProcessInfo thermal state `0` or `1` (states `2`/`3` reject the cell), and
normalized host load no greater than `0.8` logical-core equivalents.

Before each HTTP call, the harness reserves a conservative 6-second work block
against the existing 25%-of-60-second duty window. If the rolling window cannot
fit that reserve, it takes additional registered 4-second breaks until it can;
this pacing is outside the timed HTTP wall interval.

`ironmule.bench.MemoryGate` is created once before the first worker load and
checked after each call, after each worker load, and after worker cleanup. A
missing swap reading at any check rejects the study. The check uses the actual
`done.metrics.peak_memory` value in GB, converted to
bytes, together with one study-wide swap baseline and installed-memory limits.
Missing swap telemetry or an inert memory gate aborts the pilot rather than
supporting an unguarded claim. Each cell records cheap AC/power, thermal, and
load checkpoints; the report does not claim any unavailable foreign GPU sensor.

Reports are written incrementally before model resolution and hardware
preflight, after every call, after every worker, and in every failure path.
Failed attempts remain in the new output file. A source manifest
covers `ironmule_product`, `friday_evidence`, `ironmule/bench.py`, the shared
measurement harness, the pilot tool, this preregistration, `pyproject.toml`,
and `BACKLOG.md`; the manifest must be unchanged at completion.

## Statistics and interpretation

HTTP completion wall time is the primary timed metric. For each limit, the two
AB/BA pairs produce a within-worker median paired ratio using the shared
`paired_ratio` bootstrap. The three worker medians then form the per-limit
cluster input for the shared bootstrap; this is explicitly not a claim of six
independent workers. A/A reference timings are retained as per-limit noise
ratios.

The pilot cannot activate the candidate. A/A is summarized separately for each
limit as a paired ratio. The pilot-only noise threshold is
`max(2%, 3 × maximum absolute A/A relative deviation)`; a cluster confidence
interval crossing `1.0`, or a cluster movement inside that threshold, is
classified as `inconclusive_noise_overlap`. If A/A noise overlaps the observed
gain, no superiority statement is made. Early
EOS before the configured limit must show a zero forward delta and does not
support the saved-forward mechanism. Token, text, finish, count, memory,
resource, source-integrity, or budget failures invalidate the corresponding
attempt and remain visible in the report.
