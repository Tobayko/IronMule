# B27a — Current-main engineering baseline

**Status before measurement:** sealed design; no B27a model timing has been observed.

## Purpose

Capture a reproducible engineering baseline for the current IronMule main snapshot
before any evidence-layer architecture changes. This is not a new optimisation,
qualification, promotion, routing decision, or stock-MLX comparison.

The source branch starts at Git commit
`d422fdb00fced3238dfaa6b5e9e993294adb72cd`. The only pre-baseline product-tree
change is the B27 backlog entry; the runner and corpus auditor are research-only.

## Frozen harness and inventory

- `research/b27_main_baseline.py` SHA-256
  `1e0dc2f287eb5b2fe40e681fe97f2091e28dfb5924f80aa486235b77baf04ad3`
- `tests/test_b27_main_baseline.py` SHA-256
  `704e65fc61845603d9feb0eecf04adc819796979715014af5d8b97c84cb0d99c`
- `research/b27_evidence_inventory.py` SHA-256
  `7ef99d3f0eaad0be363b238bf0388395b85b1572bc5e4258678ac9f48859fbc1`
- `tests/test_b27_evidence_inventory.py` SHA-256
  `0626e494cbde919cb7c58c949a99f1b01ceece6514fa1c58a253f6f1e2f9ac0b`
- read-only inventory dataset SHA-256
  `ee414c9ee51c6e583ada094444ce66d5e22dca6c15c197dda1d7cd004e30bf32`

The two new test files passed together with `5 passed` before sealing. The runner
forces `BASELINE = Knobs()`, resolves only an already cached local snapshot, creates
one fresh process per model, and does not read or write a tuned profile.

## Existing model cells, in fixed order

1. `mlx-community/gemma-3-4b-it-4bit`, exact cached revision
   `93724907d4ed1745d2fe50baadf3b0b01a65abf2`.
2. Only after cell 1 finishes cleanly:
   `mlx-community/gemma-3-12b-it-4bit`, exact cached revision
   `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`.

No model above 12B is run. No model or software is downloaded or installed. Each
runner hashes the complete resolved model manifest before loading and records the
revision, architecture and quantisation metadata.

## Protocol per model

- strict one-shot execution plan;
- six concurrent requests and 48 maximum physical output tokens per request;
- baseline arm `InteractiveMode`, candidate/reference-preservation arm
  `ThroughputMode(max_width=4)`;
- both arms use the unchanged baseline knobs;
- two warmups and six measured repeats per arm;
- balanced alternating AB/BA order;
- primary endpoint: complete `Runtime.serve` outer wall;
- physical token rate is the paired rate endpoint;
- raw warmup/repeat snapshots, token IDs, stop reasons, counts, fingerprints,
  environment, peak MLX memory, system state and swap delta are retained;
- one loaded model is shared between arms inside a cell, so this is an engineering
  baseline and not fresh-process qualification evidence.

## Hard preflight and correctness/resource gates

- run outside the sandbox and strictly serially; no xdist;
- `xcodebuild -checkFirstLaunchStatus` must return 0;
- AC power;
- pre-spawn system swap at or below 256 MiB;
- no other local model process;
- token IDs, stop reason and physical/visible counts must match between arms;
- zero fallbacks and zero recorded correctness errors;
- swap growth from process preflight to completion at or below 256 MiB;
- all requested warmups/repeats present and the runner exits normally.

Any failed or unavailable gate gives `INCONCLUSIVE`. There is no retry, pooling,
threshold repair, repeat-until-good or selective result suppression. Cell 2 is not
started if cell 1 reveals a system-safety issue that could contaminate it.

## Interpretation fixed in advance

`BASELINE_CAPTURED` means only that the current main behavior was recorded under this
protocol with its raw variation and gates intact. The observed throughput ratio is
reported but is not promoted to `QUALIFIED`; no minimum speed threshold is imposed and
a slower result remains valid baseline evidence. Existing B39d/B40 qualification is
not pooled with B27a.

The run deliberately has no stock `mlx_lm` arm and no fresh process per arm. Those are
explicit architecture/method gaps for B27 Phase C, not facts to infer from this run.
