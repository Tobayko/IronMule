# D2a — Same-day pre-D2 engineering baseline

**Date:** 2026-08-29
**Status before measurement:** sealed; no D2a timing observed.

## Purpose

Capture a new pre-D2 4B/12B baseline immediately before exact-model-identity source
changes. D2b will compare only against D2a. B27a2/B27d/B27e timings are not pooled.

## Frozen inputs

- parent pre-D2 code commit before documentation: `857c023d1243d55801537fdeeeba9275fa767a30`;
- measurement must run from the clean commit containing this preregistration;
- runtime-tree SHA-256 must remain
  `d7577af8e83778b9753ad4bf721656a16d923a9f848040e406178b7dcffc8a21`;
- baseline harness SHA-256
  `e6d981583384d4b526af32eb508579a79815bebabea0c64c8a2f4d99ebfe74d4`;
- D2 contract SHA-256
  `ebfb372f6f1dc32a1a73122a004fe5a7a393243f06d313f3a2f8951a641bf38d`;
- review SHA-256
  `8327a778358314bc38a7171130dbb85a19a04f48fda6571d33e165dca5316bf5`.

No D2 source implementation may be present in D2a.

## Exact cells and order

1. `mlx-community/gemma-3-4b-it-4bit`, revision
   `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, expected full manifest
   `a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae`.
2. After clean completion and system recovery:
   `mlx-community/gemma-3-12b-it-4bit`, revision
   `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`, expected full manifest
   `aef12412aba9e6351a457381fb723672d253f8473b837757beff716d3c8e2b31`.

No model above 12B, download, install or network fallback.

## Protocol per cell

- `experiment_id=D2a`;
- exact local cached snapshot;
- strict one-shot plan and `BASELINE=Knobs()`;
- no stored tuned profile;
- Interactive and Throughput W4 arms;
- six requests × 48 physical output tokens;
- two warmups and six measured repeats per arm;
- alternating AB/BA order;
- complete `Runtime.serve` outer wall and physical token rate;
- fresh serial process per model, one model shared between arms;
- atomic raw evidence with environment, model manifest, runtime tree, outputs and
  resources.

## Hard gates

- clean source/commit and frozen hashes;
- Xcode/Metal/MLX prerequisites green;
- AC, low-power false, memory free >=80%, swap <=256 MiB, no model process before
  each cell;
- exact revision/manifest/architecture/quantisation;
- token IDs, stop reason and physical/visible count identical between arms;
- zero fallback/correctness errors, swap growth <=256 MiB, no crash/timeout/residual;
- all warmups/repeats present.

`BASELINE_CAPTURED` is an engineering reference, not qualification. Any failure makes
that cell `INCONCLUSIVE`; later cells stop on safety/resource failure. No retry,
pooling, threshold change or selective suppression.

## Required outputs

- ignored `D2a_gemma4b_pre_20260829.json` and `D2a_gemma12b_pre_20260829.json`;
- tracked path-free D2a summary with input hashes and no absolute path;
- Ledger/dashboard update;
- only then may D2 implementation begin.
