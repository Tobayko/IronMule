# Cycle 19 — fused greedy selection in the fixed compiled decode (v2)

**Study ID:** `fused-greedy-compile-20260825-02`
**Run ID:** `fused-greedy-compile-validation-20260825-02`
**Candidate:** `fixed_compiled_fused_greedy`
**Baseline:** `fixed_compiled_external_greedy`
**Claim:** `formal_claim=false`

## Scope and authorization

This is one new, local, pre-registered A/B study. It uses the already local
`mlx-community/gemma-3-4b-it-4bit` snapshot at revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`. Model files, weights, quantization,
prompt, tokenizer, cache capacity and Matmul mathematics are unchanged. The
study changes only where the already used greedy `mx.argmax` is placed relative
to the existing fixed-state compiled forward.

Cycle 18 (`fused-greedy-compile-20260825-01`) is a separate, terminal
zero-pair harness failure. Its worker and parent used different canonical
environment fingerprints, so authorization failed before model loading and
the parent reported a terminal provenance mismatch. Cycle 19 has a new study
and run ID, a new marker/result location and a new sealed specification; it is
not a rerun or reinterpretation of Cycle 18.

The baseline is the validated Cycle-17 N=1 architecture: the compiled fixed
forward returns `(logits, state)`, the parent-side Python wrapper applies the
same `mx.argmax(logits[:, -1, :], axis=-1)`, and one host boundary materializes
the scalar token. The candidate applies that exact `mx.argmax` expression inside
the compiled body and returns `(token, state)`. Both arms are N=1 and use the
same single host boundary, with exactly one host API conversion per physical
token, after the compiled call. There is no readback batching, no
second synchronization, no Matmul-off arm and no alternate sampler.

## Hypotheses

H1: the candidate reduces the complete Decode-Kritischer-Pfad time, including
greedy selection and the one host boundary. It wins only when the paired median
`candidate / baseline <= 0.99` and the paired bootstrap 95% upper bound is
`< 1.0`.

H2: physical, logical and visible greedy token sequences and visible text are
identical in every pair and every process. Each arm must also be deterministic
across all six processes. Any mismatch is terminal
`correctness_failed`; no quality score or cleanup can replace this gate.

H0/negative result: there is no clear gain, or the extra compile boundary/state
shape is neutral or slower. The result then retains the baseline or is
`inconclusive`, with no activation.

## Fixed workload

Both arms receive byte-identical Cycle-14/16/17 planner input and the same
rendered chat template. Greedy sampling is `temperature=0`; the maximum is 32
physical tokens and at most 31 decode forwards after the prefill-selected first
token. The first EOS stops generation earlier.
Eight warmup forwards run on a disposable fresh fixed cache before each measured
decode. The first EOS belongs to the logical sequence; visible text excludes
EOS and any physical tail. The tail remains fully timed and hashed. The cache
is discarded after each arm. No Multi-Turn or general quality claim is made.
Each arm performs exactly two prefills: one disposable prefill for the warmup
state and one fresh prefill for the measured state. Prefill, fixed-cache
conversion and compile-cold materialization are recorded separately. The primary
timer starts immediately before greedy selection from the measurement-prefill
logits, includes every N=1 host boundary and the EOS decision, and stops only
after explicit state discard and before `BudgetGuard.record_gpu`.

Six fresh Python processes run one model load each and execute both arms on
fresh cache state. The fixed order is:

1. baseline → candidate
2. candidate → baseline
3. baseline → candidate
4. candidate → baseline
5. baseline → candidate
6. candidate → baseline

There is no retry, no outlier removal and no parallel model execution.

The paired bootstrap uses exactly 10,000 resamples with frozen seed `20260825`.

The parent and worker bind exactly the same canonical environment fingerprint:
offline variables, the fixed tuple of removed unsafe variables
`(PYTHONHOME, PYTHONPATH, PYTHONINSPECT, PYTHONSTARTUP)`, resolved project
Python and machine. A mismatch is an authorization/provenance failure and is
preserved as a bounded terminal error without starting model work.

## Resource and execution gates

The child alone owns `BudgetGuard` GPU accounting and pauses. AC power,
`Device(gpu, 0)`, Apple M1 Max/32 GB, offline environment, exact local
snapshot/revision/weight hashes, clean sealed Git state, RSS <= 5 GiB, swap
delta 0, continuous work <= 6 s, total GPU work <= 120 s and wall time <= 1200
s are mandatory. A 60-second/15%-duty window is respected with the registered
break schedule. A timeout, memory/resource failure, budget rejection, wrong
snapshot/device or malformed worker event stops safely and preserves partial
evidence; it is never retried.

The parent creates a private 0700 marker directory and exclusive 0600 marker
before the first child. Existing marker or result blocks execution. Worker
stdout is exactly one strict JSON object (no duplicate keys, NaN, multiline or
oversize output). Model imports occur only after authorization gates.

## Evidence

Each arm records model/revision/snapshot/weight hashes, Git and dirty-state
fingerprints, code/spec/prompt/environment hashes, PID/load count, arm order,
prompt/token/rendered hashes, physical/logical/visible token counts and hashes,
EOS position and tail count, visible text hash, compile timings, TTFT, complete
Decode-Kritischer-Pfad time, token rate, RSS/MLX peak, swap, BudgetGuard
observations/charges and terminal reason. The primary metric is measured; the
bootstrap ratio and decision are calculated from the six pairs.

## Decision table (frozen before measurement)

| Gates and result | Decision |
|---|---|
| Resource or budget gate fails | `resource_or_budget_failed` |
| Any token/text identity gate fails | `correctness_failed` |
| Candidate compile/shape/API cannot run, without resource failure | `candidate_not_runnable` |
| All gates pass; median <= 0.99 and bootstrap upper < 1.0 | `fused_greedy_compile_wins_exact_scope` |
| All gates pass; candidate median > 1.0 and bootstrap lower > 1.0 | `fused_greedy_compile_regression_baseline_retained` |
| All gates pass but neither speed condition is clear | `fused_greedy_compile_inconclusive` |
| Fewer than six complete pairs for any other reason | `incomplete_evidence` |

Every outcome remains `formal_claim=false` and permits no candidate execution,
automatic activation, persistent service or product claim.
