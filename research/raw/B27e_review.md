# B27e — Mirrored cross-commit control review

**Date:** 2026-08-29
**Status:** `READY_FOR_MEASUREMENT`; no B27e model timing observed.

## Question and independence

B27e is a new mechanism-level control, not a B27d retry. It uses only four new 4B
child records in two mirrored blocks (`OLD,D1` then `D1,OLD`). B27a2 and B27d values
are explanatory context only and are never pooled into B27e.

Targets are exact detached commits:

- OLD: `467d5b8` (Phase-A–C baseline, no D1 module);
- D1: `0b14eb6` (approved D1 implementation).

Git's exact diff over the 16 declared execution-surface files is empty. D1 adds only
`ironmule/evidence.py`, SHA-256
`d605eecdf43e460e7a355aa63333380fb6b633ac098cb2848a0474338de74b74`,
which the runtime import-boundary tests prove is not imported.

## Frozen harness review

- `research/b27e_cross_commit.py`:
  `bde2181490389e3838c73be1ed2d6c2e58a4bdfa094ab8ee3497528133a1283d`
- `tests/test_b27e_cross_commit.py`:
  `dd17487301362783cd02aceca222877bf0ef19a37e79478b5536e534b5f6eefe`
- shared current baseline harness:
  `e6d981583384d4b526af32eb508579a79815bebabea0c64c8a2f4d99ebfe74d4`
- baseline harness tests:
  `c023bb53f3c83c52ca1c2d1a924c4269fb4e6fa8656967da17f6677a8880967b`

The same current baseline harness is loaded for OLD and D1. Before model import, each
fresh child prepends exactly one target root to `sys.path`; it verifies the imported
`ironmule.__file__`, hashes that target's runtime tree and records the exact target
commit. Timed work remains the existing complete `Runtime.serve` protocol.

The parent:

- proves both worktrees are on their sealed commits;
- proves the declared execution surfaces have one identical digest;
- proves D1 is absent from OLD and exact-hash present in D1;
- runs all children serially with a 600-second timeout;
- applies AC, swap, memory-free and model-process gates before every child;
- writes parent evidence atomically after every step;
- stops on any failed preflight, timeout or child failure without retry.

## Classification review

Domain or correctness/resource failures precede timing. For every mirrored block and
both Interactive/Throughput arms, D1/OLD wall and rate medians are calculated from six
new repeats.

- all ratios inside ±5% -> `COMMITS_INDISTINGUISHABLE`;
- D1 wall >1.05 and rate <0.95 in all four block/arm comparisons ->
  `D1_SLOWER_REPRODUCED`;
- inverse in all comparisons -> `D1_FASTER_REPRODUCED`;
- mixed/order-dependent direction -> `ORDER_OR_TEMPORAL_DRIFT`;
- incomplete/domain/hard-gate failures retain their own inconclusive/regression class.

No class qualifies or activates D1.

## Verification

- B27e + baseline focused tests: `10 passed`;
- full serial non-integration suite: `151 passed, 11 deselected` in `5.15 s`;
- source-surface Git diff: empty;
- no model, download, install or network action during review.

## Limits

Two mirrored blocks are a mechanism discriminator, not qualification evidence or a
general noise model. A mixed result leaves B27d inconclusive. B27e never changes the
runtime, D1 profile state or activation status.

**Decision:** ready to seal and commit the B27e protocol before creating target
worktrees or starting the first 4B child.
