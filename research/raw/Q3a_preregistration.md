# Q3a preregistration — path interaction at the final Q2 incumbent

Written 2026-08-31, before any Q3a hardware execution. This document and its
companion SHA-256 file are immutable once a pilot starts.

## Question and hypothesis

Q2 tested `fused_argmax` early, then retained the incumbent
`compiled_fixed_cache=True`, `head_skip_prefill=True`, and `readback_every=2`.
Q3a asks whether adding `fused_argmax=True` after that path has been assembled
changes the measured outcome. The hypothesis is only path interaction; no profile
promotion or general performance claim is made.

## Exact arms and workload

Both arms use the local cache only and the exact model ID
`mlx-community/gemma-3-4b-it-4bit`, pinned to revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`, greedy generation, `max_tokens=32`,
and the existing `ironmule.ab.run` harness. The final Q2 incumbent is:

```text
fuse_projections=False, compiled_fixed_cache=True, fused_argmax=False,
head_skip_prefill=True, prefill_into_fixed=False, readback_every=2,
speculate_k=0, speculate_ngram=3, capacity_slack=0, wired_fraction=0.0
```

Arm A is that incumbent. Arm B is identical except `fused_argmax=True`.
Run six fresh child processes, alternating AB/BA order, with two warmups and
seven measured repeats per arm. The result must preserve raw samples, token
identity, determinism, per-arm MLX peaks, conservative child RSS, and model,
runtime, code, and environment identities.

## Start gates and time bound

The dry-run is the default. `--execute` is required to start a worker and an
exclusive 0600 output. Before the first model process, the harness must prove:

- exactly one allowlisted local model snapshot at the pinned revision is readable;
- AC power, low-power mode off, and nominal thermal state (both separate `pmset`
  lines must explicitly report no thermal and no performance warning);
- no competing loaded model process;
- start swap is at most 256 MiB and installed memory is known;
- this file matches `Q3a_preregistration.sha256`, the Git tree is clean (apart from
  the known SQuAD artifact), and the runtime-code hash is bound;
- the monotone wall deadline is 300 s, with a 35-second timeout per child and a
  worker phase capped at 240 s and at least 10 s reserved for postflight (the
  pilot's conservative estimate is 270 s). The worker owns the process group;
  direct A/B children inherit it and are individually terminated/reaped on timeout.

If any fact is unknown, the pilot refuses before importing MLX. The parent passes a
one-shot nonce capability and the expected identity over an inherited pipe; a
direct worker invocation has no capability and exits before importing IronMule or
MLX. The 35-second child allowance is a pilot bound, not a performance estimate.

## Hard gates and recording

After the worker, the end environment must still satisfy AC, low-power off, nominal
thermal state, the three-sample load gate, and the no-competing-model process gate.
Swap change must remain at most 256 MiB (a decrease is allowed). Every child and
arm must have exactly seven raw timing repeats and two warmups, positive MLX peak,
and complete raw structure, including per-repeat logical/physical token arrays,
token counts, stop reasons and capacities. Token, token-count and stop-reason
identity plus determinism must hold. Conservative MLX peak and child RSS
must remain below 60% of installed memory. Missing model revision or manifest,
timeout, crash, fallback, incomplete data, or any gate breach is a failure: stop,
record `FAILED` with `BASE` fallback, and retain the reason and completed-child
markers.

## Decision and kill criteria

The only passing status is `INFORMATION_GAIN` with interpretation
`PATH_INTERACTION_ONLY`; `promotion_allowed` is always false. Before measurement,
fix the total candidate/incumbent ratio classification: `GAIN` iff `ci_high <
0.995`, `LOSS` iff `ci_low > 1.005`, `PRACTICALLY_NEUTRAL` iff the complete CI is
inside `[0.995, 1.005]`, otherwise `INCONCLUSIVE`. A clean run is information only;
it must not be described as a direct statistical comparison with early Q2.

Q3a is killed and no activation is allowed on any preflight refusal, timeout,
crash, resource/swap breach, raw-data omission, identity drift, token mismatch,
or nondeterminism. The current deterministic tuner and stored profile remain
unchanged in every case.
