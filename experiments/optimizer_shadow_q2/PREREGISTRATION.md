# Optimizer Shadow Q2 — Sealed Preregistration

**Experiment ID:** `optimizer-shadow-q2-4b-20260830-01`  
**Status:** `SEALED`  
**Scope:** one future, manually started, five-minute real shadow session  
**Measurement status:** no measurement performed; this document does not itself authorize execution

This document freezes the question and safety contract for one later real run.
It is not a result and does not authorize a model load. The machine-readable
sealed contract is [`PREREGISTRATION.json`](PREREGISTRATION.json).

## 1. Question and boundary

Can the already existing IronMule Q2 procedure produce one safe, reproducible
**shadow-only** recommendation for the exact Gemma 3 4B identity below, while
preserving the baseline answer and all resource gates? The run may recommend a
candidate; it may not activate, promote, or alter a product/runtime profile.

The result is valid only for the exact bound hardware, software, model,
workload, IronMule source, and power state. It does not support a claim for
another Mac, Gemma size (including 1B/12B/27B), revision, tokenizer, prompt, or
runtime version. No synthetic data can produce a terminal performance claim.

## 2. Exact identity

### Model and Q2 evidence

| Field | Frozen value |
| --- | --- |
| model id | `mlx-community/gemma-3-4b-it-4bit` |
| model revision | `93724907d4ed1745d2fe50baadf3b0b01a65abf2` |
| architecture | `gemma3` |
| quantisation | `bits=4`, `group_size=64` |
| quantisation SHA-256 | `4952fcd6b27eda69be81c1a695ca32544e62b6b6edc197d191ea132c0afc314b` |
| complete model manifest SHA-256 | `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae` |
| tokenizer SHA-256 | `afbd505ba5b2603a0a7e0c09e6d51672c448953e8fc1f7141e98fd0c264d7129` |
| model identity SHA-256 | `2730e8b13b892b576452493dfb1983c0948c175d02c50099475385f8bac97bd2` |
| canonical model identity file | [`MODEL_IDENTITY.json`](MODEL_IDENTITY.json), SHA-256 `b6f37cb3c6bfc1b92844d97fe95749c438ba8f1c96d709a11089985c5f35a1de` |
| Q2 profile contract | `friday.q2_profiles.confirmation_ratio.v1`, version `1` |
| Q2 profile contract SHA-256 | `a8af067a512ed98013ae124811725a7f258d91036f7d641bc3c7c7a27035b001` |
| Q2 profile source SHA-256 | `0a1104b248b4aaf532ee8ef7d9c9c0c06196dde0c5111450ee9386358d15509b` |

The canonical identity file is an exact, self-authenticating copy of the
`model_identity` object in archived Q2 profile source SHA-256
`0a1104b248b4aaf532ee8ef7d9c9c0c06196dde0c5111450ee9386358d15509b`. The
existing Friday collector validates its schema, quantisation digest, manifest
digest, tokenizer digest and `identity_sha256` before use. The Q2 profile is evidence for the identity and workload binding only. Its
reported speedup is not imported as a result for this experiment.

Version 0.1 accepts only an exact, locally cached Hub identifier in the form
`org/name`. Local directories, `local:` sources, absolute or relative paths,
Backslashes, Colons, and `.`/`..` path tricks are intentionally unsupported and
must fail closed with `unsupported_model_source` before any model resolver or
download fallback is reached. The frozen model id above is the permitted form.

### Workload

The prompt body is deliberately not stored here. The family is the exact source
symbol `ironmule.tune.DEFAULT_PROMPT`; the source hash and token count bind its
identity without reproducing prompt content.

| Field | Frozen value |
| --- | --- |
| prompt family | `ironmule.tune.DEFAULT_PROMPT` |
| prompt token count | `322` |
| output cap | `32` tokens |
| tokenizer | exact SHA above |
| generator | `ironmule.runtime.Engine.generate` |
| execution plan | `single_shot` |
| context bucket | `prompt_tokens_322` |
| batch / concurrency | `1 / 1` |
| sampling | greedy, temperature `0`, no prompt logprobs |
| workload mode | `interactive` |
| power mode | `ac` (Q2 profile spelling: `AC`) |

The sealed contract is the adjacent
[`WORKLOAD_CONTRACT.json`](WORKLOAD_CONTRACT.json). Its canonical SHA-256 is
`f191a7168a73b34aaf1984bdf09c23d7fa27c1fdbb04f70af2fb54289b5d23fa`. A
self-referential hash field is not placed in
the JSON because the existing strict `friday.workload_contract.v1` parser allows
only the schema and the exact workload fields.

### Host and software

The sealed contract is restricted to the already evidenced Apple M1 Max / 32 GB scope:

- Apple M1 Max, Apple GPU, 32 GB unified memory;
- macOS `26.5.2`, MLX `0.32.0`, mlx-lm `0.31.3`, Python `3.12.13`;
- the collector must obtain fresh public identity facts and fail closed on any
  mismatch or unknown value;
- a separate, clean IronMule checkout exists at commit
  `03e884cb28a05d090d20844460fc3afc8e738a91` (`.worktrees/friday-optimizer-ironmule`)
  and is the only permitted execution checkout;
- the current Claude checkout
  `/Users/tobiasburandt/Project_Friday/.worktrees/ironmule-b7` is forbidden as
  an execution target and remains untouched.

## 3. Code and source binding

The IronMule fixed execution registry is the reviewed adapter registry with
hash `b4a988f2c2746278fbb8a8d31e0ee5270ffabbbbe14b5049a030e96a46b164c1` and
aggregate source digest `a8fd88611235be92aceb38db79702dacca4828f1c8c29ae5f68568574fa7d23f`.
The checkout must be clean: no tracked or untracked files, no symlinked
ancestors, exact HEAD, exact interpreter identity, and unchanged fixed-file
manifest. A separately reviewed adapter-owned staging directory is mandatory.

Observed source references at C0 seal time:

| Reference | SHA-256 | Role |
| --- | --- | --- |
| IronMule `ironmule/tune.py` at `03e884` | `e2121dd351db715f6b95b105127f3765d7d223d69e70d917a1712965fc225d99` | exact prompt, tune, confirmation |
| IronMule Q2 preregistration | `ab8a0740ae42600f80fa1a4f2f01aa751fd8b62ff7d9430fa8b5f36f9d64aef0` | prior contract/evidence |
| IronMule Q2 run log | `a3c0839fa15f605718275f99e11ba2f4e8dab2a0a23568c5d276fd7f2f26eac9` | prior outcome only |
| IronMule E15 preregistration | `939a3c40683433e6fc2e24c4409304a4a762fbae52c7b528b6a4de1216b70a92` | memory/process dead-end context |
| IronMule handover | `6e7a2e56bc41fa0924ca221e4a0af7c9b779a6c5c99178aec95910f76219d05b` | branch/evidence context |
| Friday `collector.py` (C0 seal state) | `b63cbb068752434167d0b72427b4a47e269aace4e7d3639fad93046b1027d77c` | identity collection |
| Friday `ironmule_stage_worker.py` (sealed C0 state) | `bcfd23530abc10d31e888df34e6ab6486c1f9adaab5be839fb2366f19a667c01` | bounded stage protocol |
| Friday `ironmule_adapter.py` (sealed C0 state) | `69d84b337ceb1ca2f716980cc5c6b1862077301f99963bf98b887f9ddfd8cc78` | staging/authorization |

Final sealed optimizer code:

- `final_optimizer_code_commit: 8b63b7b406bad7b380918ff5c2970fab4b36d5af`
- `final_optimizer_code_sha256: e257925030300536c75d766d7cad515cee8341999c504707d39bdab96b498ce0`
- `stage_worker_sha256_at_seal: bcfd23530abc10d31e888df34e6ab6486c1f9adaab5be839fb2366f19a667c01`
- `adapter_sha256_at_seal: 69d84b337ceb1ca2f716980cc5c6b1862077301f99963bf98b887f9ddfd8cc78`

The canonical sealed JSON has SHA-256
`91f84603e867edc2b6d06c4cd3af3045eb65acffd2ca423d58512d3c14983709`.
The canonical fingerprint report has report SHA-256
`3fc28eb3853d356a0e61d670240d9a4f96382e18b7c503e4ee11417ec949ff6e` and
fingerprint SHA-256
`11242a3a1343fc2b56653a89f30a0f7204b3f5fa5b61b1d1ee171c37a065abe5`.

Any mismatch at seal or run time invalidates the session; no hash is guessed.

The machine-bound Tune-Search-Contract SHA-256 is
`f9f343f0e0609608432fe8cdcb28d5730a899ecc98a795afdeafcd52a4e26a25`. It binds
the exact `SEARCH` order and values from `03e884`, the `runtime.Knobs` defaults,
`KEEP_IF_RATIO_BELOW=0.995`, `CONFIRM_PROCESSES=6`, `CONFIRM_REPEATS=7` and
confirmation `warmup=2`. A returned profile or trial outside those defaults and
allowlisted values is inconclusive and can never qualify.

The frozen search sequence is:
`compiled_fixed_cache=[true]`, `fused_argmax=[true]`,
`head_skip_prefill=[true]`, `prefill_into_fixed=[true]`,
`readback_every=[2,4,8]`, `speculate_k=[4]`, `capacity_slack=[128]`,
`wired_fraction=[0.6]`, `fuse_projections=[true]`. The bound `Knobs` defaults are
`fuse_projections=false`, `compiled_fixed_cache=false`, `fused_argmax=false`,
`head_skip_prefill=false`, `prefill_into_fixed=false`, `readback_every=1`,
`speculate_k=0`, `speculate_ngram=3`, `capacity_slack=0`, and
`wired_fraction=0.0`.

The interpreter is bound as a regular resolved executable through its exact
target hash and inode identity; its local path is an internal gate value and is
not persisted. When the adapter runs inside a virtual environment, it derives
the active `purelib` internally, verifies the regular `pyvenv.cfg`/directory
identity, and binds exactly that value through an internal `PYTHONPATH` hash.
No user-supplied `PYTHONPATH`, path override, download or installation is
accepted.

## 4. Allowlist and stages

The only candidate is `combined_core_profile`. It is tested only through the
existing controlled IronMule adapter and worker:

- calibration stage: one real `ironmule.ab.run` A/A call with two identical
  `BASELINE` arms (`aa_left`/`aa_right`), balanced AB/BA order and fixed
  preregistered process/repeat/warmup bounds; it does not call the tuner
  screening loop;
- test stage: one `tune(confirm_winner=True)` call. The worker temporarily
  wraps `tune.confirm` to capture the exact single A/B confirmation already
  executed by `tune`; no second confirmation run is permitted;
- required Q2 prerequisites: `fixed_compiled_cache` and
  `head_skip_prefill`;
- the Q2 profile's `readback_every=2` is evidence, not a free or independently
  selectable candidate in this sealed contract;
- no free flags, source/code/kernel generation, model changes, quantisation
  changes, downloads, installations, activation, canary, promotion, or rollback
  mutation are permitted.

There is exactly one session and no retry. Calibration and test are part of the
same five-minute budget. No interim result may alter the contract, thresholds,
baseline, or action space.

## 5. Start, readiness, and deadline gates

The user must explicitly provide the one-time start authorization for this exact
experiment. The session then has a hard `300` second wall-clock deadline,
including readiness waiting, calibration, test, cleanup, and result validation.
The session waits rather than guessing when any required fact is unavailable.

Before every stage and at each control point, all of these must hold:

1. AC power is positively observed; battery or unknown power blocks.
2. Low-power mode is positively off; unknown blocks.
3. Stable idle is demonstrated by multiple bounded samples of CPU/system load,
   memory and swap; one instantaneous sample is insufficient.
4. No Claude, MLX, Python, Node, model server, or other heavy foreign process is
   present. A newly appearing foreign load is checked at the next post-stage
   readiness checkpoint; that stage/result is then discarded safely. This sealed contract
   does not claim guaranteed mid-stage termination.
5. A private atomic readiness lease is held and revalidated.
6. Swap and resource counters are readable; missing counters are inconclusive,
   never safe by assumption.

The worker runs offline with a private temporary home, bounded stdout/stderr,
fresh process isolation, timeout, and the exact staged source manifest. Source,
worker, stage-spec, interpreter, commit, model and workload must remain bound.
The one-shot stage authorization is HMAC-protected; the result is bound by its
strict schema and commit/source/fingerprint/registry hashes, not by a result HMAC.
Any TOCTOU, HMAC, lease, process, or manifest failure is a safe
inconclusive result with baseline retained.

Before either Calibration or Test imports MLX or performs model work, the worker
re-resolves the local model source and checks model id, revision, complete
manifest, architecture, quantisation bits/group size and tokenizer against the
Stage-Spec identity. The same exact identity and the profile conditions must be
present in any returned Test profile. An unknown or contradictory power value is
rejected strictly (`AC is not exactly True` or `low_power is not exactly False`).

## 6. Correctness and resource decision rules

Correctness is checked before performance is considered:

- exact token IDs and count against the same baseline (IDs are transient gate
  inputs and are never persisted);
- exact decoded-text equivalence hash derived from the bound tokenizer and token
  sequence; decoded text itself is never stored;
- source-derived stop equivalence from identical token/count data plus the bound
  max-token, capacity and EOS contract; an observed stop reason is not claimed;
- prompt/tokenizer/model/source identity unchanged;
- no NaN/Inf, cache, shape, or process-isolation violation.

The result is `inconclusive` if any confirmation pair, engine TTFT, decode-only
tokens/s, peak MLX memory, peak RSS, swap delta, timeout status, process identity,
or required resource field is missing. A malformed, partial, crashed, timed-out,
or foreign-load stage is never treated as a short successful run.

The primary measurements, if present, are engine TTFT in the bound
`engine_prefill_to_first_token` scope and decode-only tokens/s. The
candidate cannot be recommended unless correctness passes, both required stages
are complete, and the conservative paired evidence is available. No standalone
screening value is a terminal claim. In this sealed contract, activation and promotion are
always disabled regardless of measured values.

Resource gates:

- `max_seconds=300` total session deadline;
- `max_peak_memory_bytes=12884901888` (12 GiB MLX peak ceiling, carried from the
  reviewed E15 safety rule);
- `max_rss_bytes=12884901888`;
- `max_swap_delta_bytes=0`;
- bounded result/output sizes and controlled worker timeout;
- any unknown or contradictory resource reading => `inconclusive`.

## 7. Terminal classifications

The following order is frozen at seal:

1. `SAFETY_ABORT` — power, foreign load, lease, timeout, swap, memory, source,
   HMAC, process, or integrity violation.
2. `CORRECTNESS_FAILURE` — any reproducible token/text/count/stop/hash/state
   mismatch.
3. `INCONCLUSIVE` — missing confirmation pairs, TTFT/decode/resource data,
   incomplete stage, unstable idle, unknown identity, or insufficient evidence.
4. `SHADOW_RECOMMENDATION` — only when every required gate passes; this remains
   a recommendation and does not activate anything.

Synthetic fixtures may test parser and error handling only. They cannot close a
terminal real-run classification or establish speed, quality, hardware, or
generalisation claims.

## 8. Private result and history schema

No prompt content, token IDs, decoded text, PIDs, local paths, stdout or stderr
is stored. The private result envelope is `friday.optimizer.session-result.v1`
and contains only redacted numeric AA/AB raw series, bounded identities and
equivalence hashes:

```text
experiment_id, schema_version, status, classification,
session_id_hash, user_start_authorization_hash,
hardware_fingerprint, model_identity_sha256, workload_contract_sha256,
ironmule_commit, ironmule_source_digest, optimizer_code_commit,
optimizer_code_sha256, worker_sha256, adapter_sha256,
candidate_id, stage_statuses, confirmation_pair_count,
correctness{token_equal,text_equivalence_hash,count_equal,count_hash,
stop_equivalence,source_derived_stop_equivalence},
metrics{ratios,confidence_intervals},
measurement_evidence{calibration,test},
pairs{pair_id,order,arms{total_ns,prefill_ns,decode_ns,decode_steps,
deterministic,mlx_peak_bytes,token_hash,count_hash,text_equivalence_hash}},
resources{mlx_peak_bytes,rss_peak_bytes,swap_delta_bytes,wall_seconds},
gate_reasons, created_at
```

The redacted raw measurement contract has been verified to preserve
`decode_steps` in every raw arm alongside the numeric timing series and their
equivalence hashes. It is retained as measurement evidence, not inferred from
token counts.

History is written as an ordinary redacted Optimization Memory v2 event through
`OptimizationMemoryV2`/`HistoryWriter`; this sealed contract does not invent a separate
history schema or event format. Its bounded payload retains the result/evidence
hashes and approved numeric summaries only. It has no prompt text, model output,
token IDs, decoded text, PIDs, local paths, stdout/stderr, secret/HMAC material,
or arbitrary command output. The read-only UI may expose redacted status/metrics
only; no write endpoint is part of this sealed contract.

## 9. Seal checklist

Before any user-started run, a maintainer must:

- verify the sealed code, worker, adapter, workload and model hashes in
  `PREREGISTRATION.json`;
- canonicalize and hash `WORKLOAD_CONTRACT.json`;
- verify the exact model manifest/tokenizer identity from the local snapshot;
- verify the existing separate clean IronMule checkout at `03e884...`;
- re-read the current collector, stage-worker, adapter, Q2 contract and relevant
  dead ends; any changed source reopens this seal;
- record the user authorization, then start exactly one session;
- retain the private result/history only after strict schema and HMAC validation.

All seal checklist items are closed. This seal does not itself start a
measurement; execution still requires the separate explicit user start.

## 10. Known constraints and blockers

- The separate clean IronMule checkout at the exact commit exists and is
  verified; the Claude worktree remains explicitly forbidden and untouched.
- Final optimizer, adapter, and stage-worker hashes are sealed in the adjacent
  machine-readable contract. Any future source change reopens this seal.
- The static exact runtime fingerprint is present in `FINGERPRINT_REPORT.json`.
  Only live readiness, the one-time user start authorization and the runtime
  result remain outstanding. No model or benchmark work has started.
- The historical Q2 profile records the exact prompt token count but does not
  publish a separate semantic prompt-family label. This sealed contract therefore binds
  the source-qualified symbol `ironmule.tune.DEFAULT_PROMPT` and its source hash;
  replacing it with a different label at seal requires a new audit, not a guess.
