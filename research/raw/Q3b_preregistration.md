# Q3b preregistration — residual-swap safety canary

Written 2026-08-31, before any Q3b hardware execution. This document and its
companion SHA-256 file are immutable once a canary starts.

## Question and scope

Can the exact locally cached `mlx-community/gemma-3-4b-it-4bit` snapshot at
revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2` run one fresh baseline arm
and one fresh candidate arm, with known residual swap, without creating a new
swap or memory-pressure hazard and with identical outputs? This is a safety
canary only. It makes no performance, optimisation, reinforcement-learning,
promotion, or activation claim.

## Exact arms and workload

The only model is the pinned local 4B snapshot above. The baseline is the final
Q2 incumbent:

```text
fuse_projections=False, compiled_fixed_cache=True, fused_argmax=False,
head_skip_prefill=True, prefill_into_fixed=False, readback_every=2,
speculate_k=0, speculate_ngram=3, capacity_slack=0, wired_fraction=0.0
```

The candidate is identical except `fused_argmax=True`. The two arms are run in
two separate fresh single-arm stages, baseline first and candidate second. Each
stage uses one fresh model child, one warmup, three measured repeats, greedy
generation, and `max_tokens=32`. Raw logical and physical tokens, token counts,
stop reasons, capacities, timings, and MLX peak are retained. No ratio,
performance classification, or promotion decision is computed.

## Start gates and bounds

Dry-run is the default; `--execute --output PATH` is required for execution and
the output is created exclusively with mode 0600. Before the first model child,
the harness must prove AC power, low-power mode off, nominal thermal state,
exact pinned model identity, matching preregistration SHA, known installed
memory, a clean Git tree apart from `research/data/squad-dev-v1.1.json`, and a
runtime-code hash binding parent and worker. The initial memory-pressure free
percentage must be at least 35%. Initial swap is recorded and must be known and
at most 4 GiB. Three load samples must be known, have max at most 8.0 and spread
at most 2.0. The process gate takes two bounded inventories,
`pid=,rss=,%cpu=,args=` and `pid=,comm=`. The args snapshot is authoritative
for relevance: every Claude-related or known inference/model record must have
an exact same-PID `comm` record. Extra `comm` records are ignored, and an
irrelevant args record may lack a `comm` row. If a relevant Claude args record
is missing from `comm`, a read-only `kill(pid, 0)` probe may classify it as
already gone; alive, permission-denied, and unknown probe results fail closed.
Known inference/model tokens block directly even before comm matching.
The only Claude exception is a process whose `comm` path is lexically inside
the exact `/Applications/Claude.app/Contents/` boundary, after the complete
bundle has passed absolute `/usr/bin/codesign --verify --deep --strict` and
bounded `/usr/bin/codesign -dv --verbose=4` metadata checks for identifier
`com.anthropic.claudefordesktop`, team `Q6L2SF6YDW`, and first authority
`Developer ID Application: Anthropic PBC (Q6L2SF6YDW)`. Generic Claude
CLI/server/backend processes, outside-bundle paths, untrusted bundles, and
malformed inventories are hard blockers. The allowed desktop process still
contributes to loadavg; its CPU value is not discarded from the load gate.
The two codesign calls use a dedicated 5.0-second timeout because the observed
deep verification took 1.487 s and metadata inspection 0.030 s on this host;
all other OS commands retain the 1.0-second bound. This timeout exception is
limited to the two codesign calls and does not relax any trust, output-size, or
nonzero/exception fail-closed check.
The two snapshots are not atomic: a process created after the args snapshot
may appear only in `comm` and is ignored there, while repeated pre-child and
post-stage gates are the detection boundary for later relevant processes. This
inventory limitation supports safety gating only and creates no performance
claim.

The stage worker remains stdlib-only until the capability payload, runtime-code
hash, stage name, and finite worker deadline have all been validated. It then
activates only `Path(__file__).resolve().parents[1]`, which must contain
`ironmule/__init__.py`, at `sys.path[0]`. `importlib.util.find_spec("ironmule")`
must resolve exactly to that package origin and search path before any IronMule
import; a shadow path or a preloaded foreign `ironmule` module refuses the stage.
The parent dry-run and a direct `--stage-worker` invocation without a valid
capability must not activate this path or import IronMule/MLX.

The parent owns a monotone 180-second deadline. Each model child has a
35-second timeout and each fresh stage worker has a 120-second cap. A bounded
0.25-second swap sampler runs for the whole stage and its maximum, not merely
the endpoint, is checked against the initial reading. The worker records a
synchronous swap sample before starting the periodic sampler and another
synchronous sample after the stage; every sample has a monotonic timestamp and
a monotonic offset from worker start. Sampling evidence is bounded to 512
records, and successful results require `sampler_errors=[]`, at least two
samples, equal-length sample/time/offset arrays, and a final sample. The maximum
allowed timestamp gap is 1.75 seconds (0.25-second interval + 1.0-second
command timeout + 0.5-second scheduling margin); a larger gap fails closed.
The synchronous worker-start sample is compared with the parent preflight
reading and a delta above 128 MiB refuses the stage before importing or
starting any model child. `before_child` rechecks sampler errors and the
current sampled high-water delta immediately before every child. While a child
is active, any periodic sampler command/read/parse/timestamp error or observed
high-water delta above 128 MiB atomically captures bounded evidence and emits a
flushed `@SAFETY` JSON marker containing only `reason`, `samples`, `times`,
`offsets`, and `errors`; it then immediately sends `SIGTERM` to the worker's
process group (`os.killpg(os.getpgrp(), SIGTERM)`), which also contains the
active A/B child. Marker and kill failures are fail-loud. Any command, read,
parse, timestamp, or sampler-thread error is recorded and hard-fails the stage.
A stage worker owns one process group and all timeout/crash/OSError paths
terminate and reap it before returning. The
180-second total budget reserves cleanup/snapshot time; the next stage is never
started after a failed gate.

## Per-stage resource gates

After every stage, compare the sampled maximum swap with the initial preflight
reading: the increase must be at most 128 MiB (a decrease is allowed), the fresh
post-stage swap endpoint must also be known, free memory must remain at least
20%, and both the stage's MLX peak and child RSS must be at most 60% of
installed memory. The post-stage snapshot must prove the prior
child is reaped and contain fresh swap, memory-pressure, RSS, load, and process
evidence before Stage 2 may start. The post-stage AC, low-power, thermal, load,
and inference process gates must also remain valid. A stage must have complete raw
repeats, deterministic logical and physical token sequences, exact token counts,
stop reasons, capacities, decode-step count, prompt-token count, and
deterministic flags. Baseline and candidate must match on every one of those
cross-stage identity fields, with no fallback, timeout, crash, malformed output,
or residual model process.

The final synchronous swap sample is taken only after the model child has been
reaped. A final read/parse/sampler error or a final sampled high-water delta
above 128 MiB is still terminal safety evidence: it emits the same bounded
`@SAFETY` event and prevents a normal result marker. If the worker-group TERM
fails, it immediately attempts KILL on the same process group; only failure of
both signals is a kill failure.

## Descriptive timing context on a complete PASS

Only after both stages have independently passed validation and all cross-stage
identity checks, a final `SAFETY_CANARY_PASS` may include a `descriptive_timing`
object. It is prominently labelled `descriptive_only=true`,
`performance_valid=false`, `order_confounded=true`, and
`statistical_confidence=none`. For each stage it reports the exact-raw-repeat
median total, prefill, and decode time in milliseconds, logical and physical
output-token counts, total output tokens per second, and decode steps per
second. It may report candidate-over-baseline ratios and direction-labelled
percent-faster values: `100 * (1 - ratio)` for lower-is-better total, prefill,
and decode time, and `100 * (ratio - 1)` for higher-is-better total-throughput
and decode-steps-throughput. A negative value means slower in either direction.
It contains
no confidence intervals, winner, or promotion decision. This object is never a
gate, never changes status, and is absent for failed or incomplete stages.

## Decision and kill criteria

Only a complete two-stage run satisfying every gate can be labelled
`SAFETY_CANARY_PASS`. It always has `promotion_allowed=false` and
`performance_valid=false`; timing fields are safety evidence only and must not
be interpreted as a ratio. Any unknown or violated start/per-stage gate stops
before the next stage, terminates/reaps active processes, writes `FAILED` with
`BASE` fallback, and retains bounded partial raw evidence, any `@SAFETY` event,
cleanup/reap status, and the failure reason. A safety marker is terminal even if
the worker has no final `@@` marker; a nonzero exit or missing final marker also
requires bounded process-group cleanup before the parent returns.
Any token, physical-token, count, stop-reason, determinism, identity, memory,
swap, timeout, cleanup, or process failure kills the canary. Q3a's
preregistration, hash, code path, and evidence remain unchanged.
