# B27e — Mirrored 4B cross-commit common-mode control

**Date:** 2026-08-29
**Status before measurement:** sealed; no B27e model timing observed.

## Purpose

Discriminate B27d's common-mode 4B slowdown between an association with D1 commit
presence and unmodelled temporal/order drift. B27e is a new control with only new
samples. It does not retry, pool or replace B27d.

## Exact targets and source invariant

- OLD commit: `467d5b8`;
- D1 commit: `0b14eb6`;
- D1 `ironmule/evidence.py` SHA-256:
  `d605eecdf43e460e7a355aa63333380fb6b633ac098cb2848a0474338de74b74`.

Before any child, create separate detached worktrees at those commits and initialize
their private ignored ProjectAtlas state. The parent must prove that the declared 16
runtime/benchmark source files have one byte-identical aggregate digest, D1 is absent
from OLD and exact-hash present in D1. The ProjectAtlas state is excluded by the fixed
source manifest and remains ignored.

## Frozen harness and review

- `research/b27e_cross_commit.py`:
  `bde2181490389e3838c73be1ed2d6c2e58a4bdfa094ab8ee3497528133a1283d`
- `tests/test_b27e_cross_commit.py`:
  `dd17487301362783cd02aceca222877bf0ef19a37e79478b5536e534b5f6eefe`
- shared baseline harness:
  `e6d981583384d4b526af32eb508579a79815bebabea0c64c8a2f4d99ebfe74d4`
- baseline harness tests:
  `c023bb53f3c83c52ca1c2d1a924c4269fb4e6fa8656967da17f6677a8880967b`
- review SHA-256:
  `7803639a8ebaf4ec8fa900253522aae7c5c14741059bf3e2f531f054ef2774bf`.

The orchestration commit containing this preregistration is recorded before launch and
must be clean. The same orchestration/measurement harness is used for both targets;
only the imported target root differs.

## Model and exact protocol

Model: `mlx-community/gemma-3-4b-it-4bit`, exact cached revision
`93724907d4ed1745d2fe50baadf3b0b01a65abf2`. No download/network fallback.

Fixed children and order:

1. block 0 position 0: OLD;
2. block 0 position 1: D1;
3. block 1 position 0: D1;
4. block 1 position 1: OLD.

Every child is a fresh serial process using:

- `experiment_id=B27e`;
- strict one-shot plan and `BASELINE=Knobs()`;
- Interactive and Throughput W4 arms;
- six requests × 48 physical output tokens;
- two warmups and six measured repeats per arm;
- alternating AB/BA order inside the child;
- complete `Runtime.serve` outer wall and physical token rate;
- one exact target model load, no profile reuse;
- atomic raw output and exact target package/commit/runtime-tree binding.

No B27a2/B27d timing enters a B27e ratio.

## Preflight and hard gates before every child

- AC and low-power false;
- system swap `<=256 MiB`;
- memory free `>=80%`;
- no competing/residual model process;
- Xcode/Metal/MLX prerequisites green before the study;
- child timeout 600 seconds;
- token IDs, stop reason and physical/visible count identical between arms;
- zero fallback/correctness error, swap growth `<=256 MiB`, no crash/timeout;
- exact hardware/framework/model/protocol binding inside each mirrored block.

Any failed gate stops later children. No retry or threshold change.

## Frozen comparison and classification

For each mirrored block and each Interactive/Throughput arm, calculate D1/OLD from the
six-repeat medians:

- wall ratio (lower is better);
- physical-rate ratio (higher is better).

Threshold `5%`, applied to all four block/arm comparisons:

1. incomplete evidence -> `INCONCLUSIVE_INCOMPLETE`;
2. domain drift -> `REVALIDATION_REQUIRED`;
3. correctness/resource failure -> `CODE_REGRESSION`;
4. every wall and rate ratio within `[0.95,1.05]` ->
   `COMMITS_INDISTINGUISHABLE`, supporting `COMMON_MODE_TEMPORAL_DRIFT` for B27d;
5. every wall `>1.05` and rate `<0.95` -> `D1_SLOWER_REPRODUCED`;
6. every wall `<0.95` and rate `>1.05` -> `D1_FASTER_REPRODUCED`;
7. any mixed/order-dependent direction -> `ORDER_OR_TEMPORAL_DRIFT`, leaving B27d
   inconclusive.

No classification qualifies, routes or activates D1. Even a commit association is not
a general performance claim and requires a mechanism before architecture advances.

## Required evidence

- ignored full parent and four child JSON records;
- tracked path-free public summary and deterministic verification;
- Ledger/Backlog/Limits/dashboard update;
- final serial non-integration and real 4B integration tests;
- clean ProjectAtlas/MCP/Git status and a separate result commit.
