# B27d — D1 post-change regression screen

**Status before measurement:** sealed; no post-D1 model timing has been observed.

## Question

Did adding the approved, non-imported stdlib-only D1 evidence contract change the
existing strict baseline or grouped batch-1 execution behavior under the same 4B/12B
engineering cells?

This is a regression screen, not a qualification, promotion, activation or stock-MLX
comparison. The D1 module itself has no execution path.

## Frozen implementation and review

Parent phase-A–C commit: `467d5b8`. The measurement must run from the clean commit
that contains this preregistration and the following exact source hashes:

- `ironmule/evidence.py`:
  `d605eecdf43e460e7a355aa63333380fb6b633ac098cb2848a0474338de74b74`
- `tests/test_evidence.py`:
  `bc3e602db36db564e0535b8ac7a499045de82a87384ff55be3538f636c20a376`
- `research/b27_main_baseline.py`:
  `288c5ec77c9f82b0e2c79b4c6da34e104209dcf8be0a62277ebea636de46737e`
- `tests/test_b27_main_baseline.py`:
  `ddc0df033aa63324849bd5d19e57506ea4be03497c92bd4e1a86019cba6fdf3a`
- `research/b27_compare_post_change.py`:
  `231013f3b5e6336643baecc27bc5d9d10ca8f890d357dfe94edd5bd134f6889e`
- `tests/test_b27_compare_post_change.py`:
  `bb77e14d058fa5c27455be730c2cc4521a3f05d2b961c03e120aca1d5633b60d`
- D1 review SHA-256:
  `9d146b69d5644a02fb40a127e8927085a47a7a70d2a92e1f6daaa991e6d4a91f`

The implementation commit hash and resulting runtime-tree digest are recorded by the
children and must agree across both post cells. Runtime/service/plan/executor/tuner
imports must remain unchanged and D1 must remain absent from their import graph.

## Frozen pre-change references

- Gemma 3 4B raw SHA-256:
  `e1e9b7ce3248b83fced553334b452404bf47931d02e2352f2aed8d96f55607a0`
- Gemma 3 12B raw SHA-256:
  `7276ee6505a58ca176561f8e66f2087616d9682aa44273d1f7ddad51a6311d98`
- pre-change runtime-tree SHA-256:
  `ec242cc4872014d7994c6e11cf0b32bbf145ecca4eac32088c697059e2e48385`

These B27a2 records are not pooled with B39d/B40 or any other experiment.

## Exact model cells and order

1. `mlx-community/gemma-3-4b-it-4bit`, revision
   `93724907d4ed1745d2fe50baadf3b0b01a65abf2`.
2. Only after cell 1 exits cleanly and the system recovers:
   `mlx-community/gemma-3-12b-it-4bit`, revision
   `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`.

No model above 12B, download, installation or network fallback is allowed.

## Protocol per cell

- fresh serial process per model;
- exact local cache revision and complete manifest hash;
- `--experiment-id B27d`;
- strict one-shot plan;
- `BASELINE=Knobs()`, no stored profile;
- Interactive baseline and Throughput W4 candidate/reference-preservation arm;
- six requests, 48 physical output tokens;
- two warmups and six measured repeats per arm;
- alternating AB/BA order inside the cell;
- same complete `Runtime.serve` outer wall and physical-rate definitions as B27a2;
- one loaded model shared between arms; no fresh-process-per-arm claim.

## Hard preflight and correctness/resource gates

- run outside the sandbox, serially, with xdist disabled;
- Xcode/Metal/MLX prerequisites green;
- AC, low-power false, no competing model process;
- system swap `<=256 MiB` and memory-free percentage `>=80%` immediately before each
  cell; otherwise that cell does not start;
- exact model/framework/hardware/protocol binding against B27a2;
- token IDs, stop reason and physical/visible counts identical between arms;
- zero fallback and correctness errors;
- swap delta `<=256 MiB`, no crash/timeout/residual process;
- all warmups/repeats and raw samples present.

Any domain change is `REVALIDATION_REQUIRED/EVIDENCE_DRIFT` before timing is
interpreted.

## Frozen post/pre regression rules

For every model and both arms, independently bootstrap the post/pre median ratio from
the six raw post and six raw pre samples with 10,000 resamples and fixed seed
`20260828`.

- wall passes when the 95% CI high is `<=1.05`;
- physical token rate passes when the 95% CI low is `>=0.95`.

Ordered classifications:

1. incomplete model/evidence set -> `INCONCLUSIVE_INCOMPLETE`;
2. any hardware/software/model/protocol/system-domain difference ->
   `REVALIDATION_REQUIRED`, regression kind `EVIDENCE_DRIFT`;
3. same-domain token/correctness/resource failure -> `CODE_REGRESSION`, regression kind
   `HARD_CORRECTNESS_OR_RESOURCE_REGRESSION`;
4. same-domain performance interval miss -> `INCONCLUSIVE_POTENTIAL_REGRESSION`,
   regression kind `POTENTIAL_CODE_REGRESSION`;
5. all cells/arms pass -> `NO_REGRESSION_OBSERVED`, regression kind `NONE`.

No class activates D1 or changes a profile. A failure or drift is retained as the
result; there is no retry, pooling, threshold change, repeat-until-good or selective
suppression.

## Required outputs

- ignored full B27d 4B/12B raw JSON;
- path-free tracked comparison summary with both input/output hashes and intervals;
- updated ledger, backlog status and local history dashboard;
- final serial non-integration and real 4B integration verification;
- clean ProjectAtlas/runtime/MCP and Git state.
