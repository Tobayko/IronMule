# Q3c preregistration — Q2 replication and fused-argmax preservation

Written 2026-08-31, before any Q3c implementation or hardware execution. This
document and `Q3c_preregistration.sha256` are frozen once implementation or
measurement starts. Q3c has no automatic profile promotion, routing, or
activation path.

## Question and scope

Can the exact locally cached Gemma 3 4B snapshot reproduce the historical Q2
incumbent under a fresh-process paired protocol, and does adding exactly
`fused_argmax=True` preserve that incumbent-level result? This is one local
replication/performance study, not a new hardware/model validity claim. Q3b's
ordered single-arm timing is safety-only evidence and is not a cross-run input
to Q3c.

## Exact model, workload, and arms

Use only the local Hugging Face snapshot
`mlx-community/gemma-3-4b-it-4bit`, revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`. The prompt must produce exactly
`322` prompt tokens under the existing tokenizer/chat-template path. Use greedy
generation and `max_tokens=32`.

The untuned `BASE` is the exact `Knobs()` default:

fuse_projections=False, compiled_fixed_cache=False, fused_argmax=False,
head_skip_prefill=False, prefill_into_fixed=False, readback_every=1,
speculate_k=0, speculate_ngram=3, capacity_slack=0, wired_fraction=0.0

The exact Q2 incumbent is:

fuse_projections=False, compiled_fixed_cache=True, fused_argmax=False,
head_skip_prefill=True, prefill_into_fixed=False, readback_every=2,
speculate_k=0, speculate_ngram=3, capacity_slack=0, wired_fraction=0.0

The candidate is the Q2 incumbent with exactly `fused_argmax=True`; every other
knob remains identical.

There are two independent phases. Each phase calls the existing
`ironmule.ab.run` contract exactly once with six fresh OS processes, two warmup
repetitions and seven measured repetitions per arm. Process order is fixed and
alternates `AB`, `BA`, `AB`, `BA`, `AB`,
`BA`:

1. Phase R (replication): `BASE` versus the exact Q2 incumbent.
2. Phase N (new candidate): `BASE` versus the Q2 incumbent plus
   `fused_argmax=True`.

Each phase retains its own six unique PIDs, complete arm plans, all raw
per-repeat arrays, process order, identity binding, resource/sampler history,
and cleanup/reap evidence. The two phases are not pooled. Q3b values are not
multiplied into either phase or into a cross-run gain.

## Reused Q3b safety and resource policy

The Q3b residual-swap/live policy is inherited without relaxation:

- Before start, AC power, low-power mode off, nominal thermal state, known
  installed memory, exact local model/revision/manifest, clean Git binding,
  complete runtime-code identity, known initial swap `<=4 GiB`, and start free
  memory `>=35%` are required. Three load samples must be known, with
  `max<=8.0` and `spread<=2.0`.
- A bounded periodic sampler runs throughout every phase. The sampled
  high-water swap increase is measured against the phase's initial reading.
  Any increase above `128 MiB` is terminal safety evidence and immediately
  aborts the current worker process group.
- After each phase, free memory must be `>=20%`; MLX peak and child RSS must
  each be `<=60%` of installed memory. Fresh swap endpoint, memory-pressure,
  RSS, load, process inventory, and cleanup/reap evidence are mandatory.
- Any sampler command/read/parse/timestamp/thread error, unknown power/thermal/
  memory/swap/load/process state, timeout, crash, malformed output, residual
  model process, or cleanup failure fails closed.
- The bounded process inventory blocks all known model/inference activity.
  Only the exact signed Claude Desktop process may be ignored: its `comm` path
  must be lexically inside `/Applications/Claude.app/Contents/`, and the
  complete bundle must pass absolute codesign checks for identifier
  `com.anthropic.claudefordesktop`, team `Q6L2SF6YDW`, and first authority
  `Developer ID Application: Anthropic PBC (Q6L2SF6YDW)`. Claude CLI/server/
  backend, outside-bundle paths, malformed/untrusted records, and unknown
  states remain blockers. Known model tokens block before the Claude exception.
  The PID/ppid ancestry and snapshot-race rules are inherited exactly.
- The worker must emit bounded argv-free `@SAFETY` evidence on a live abort,
  send `SIGTERM` to the worker process group, escalate to `SIGKILL` if needed,
  and prove group cleanup. Partial child records and all cleanup errors remain
  in the raw result; a safety marker or missing final marker is terminal.

## Exact time bounds

The parent owns one monotonic study deadline of `600 s` (10 minutes, below the
user's 30-minute maximum). Each phase has a hard `270 s` bound consisting of:

- one existing `ab.run` worker capped at `240 s`;
- each fresh child capped at `35 s`;
- `30 s` reserved for post-phase snapshot and process-group cleanup.

The study reserves a final `60 s` for inter-phase identity checks and terminal
cleanup. Thus `270 s + 270 s + 60 s = 600 s` exactly. Phase N starts only after
Phase R has passed all safety, identity, raw-completeness and cleanup gates. A
failed Phase R therefore stops the study before Phase N.

## Identity rule — “same values”

Every arm and every measured repeat in both phases must preserve exactly the
same logical token IDs, physical token IDs, logical/physical counts, stop
reasons, capacities, deterministic flags, prompt-token count, and decode-step
count. These are evaluator-owned fields: the result must retain the complete
arrays and reconstruct the booleans from raw data. A missing, inferred,
self-asserted, or mismatching field is a hard identity failure, not a speed
result. The same model revision, manifest, runtime-code hash, and phase plan
must also be bound in the raw evidence.

## Metrics and statistical procedure

For each arm, first take the median of the seven raw repeats within each fresh
process. Report total, prefill, and decode median milliseconds, candidate/base
paired ratios and deterministic 95% bootstrap CIs, and direction-labelled
percent faster as `100 * (1 - ratio)` for lower-is-better time.

Also report physical output tokens/s and decode steps/s per process, their
candidate/base paired ratios and 95% bootstrap CIs, using
`100 * (ratio - 1)` for higher-is-better rates. Rates use the physical output
count and corresponding total or decode duration from the same raw repeat; the
formula and denominator are retained in the derived record.

Use the existing deterministic `ironmule.ab.run`/`paired_ratio` convention:
10,000 paired resamples, seed `20260825`, preserving process pairs. No raw
repeat, process, phase, or Q3b observation may be pooled or multiplied.

## Historical Q2 target and preregistered decisions

The historical Q2 paired target is ratio `0.8568`, with 95% CI
`[0.8549; 0.9402]`. Q2 also reported a stored screening gain of `14.57%`;
that screening percentage is retained as historical context and is not
recomputed or treated as the paired Q3c estimate.

Phase R reproduces the Q2 incumbent only when all of the following hold:

1. the new incumbent/BASE median ratio is within `±0.03` absolute of `0.8568`;
2. the new 95% CI contains `0.8568`; and
3. the new 95% CI high is `<1.0`.

Phase N preserves the candidate only when both of these hold:

1. the candidate/BASE 95% CI high is `<1.0`; and
2. the candidate/BASE median ratio is no more than `0.005` above the
   replicated Phase-R incumbent/BASE median ratio.

The candidate's gain is reported descriptively with its direction and CI. These
criteria do not authorize profile persistence, activation, routing, or any
claim outside this exact local workload.

## Failure, fallback, and evidence record

If either phase is failed, incomplete, inconclusive, outside its time/resource
bound, not exactly identity-preserving, or misses its preregistered performance
criterion, retain untuned `BASE` and the current Q2 incumbent. Record the
failure reason, partial raw evidence, identity, resource history, and cleanup
status. Never declare a winner from a partial phase and never auto-promote the
candidate.

The complete raw result is the auditable record. No Q3b timing is combined with
the Q3c result, and no presentation layer is part of the execution or decision
contract.
