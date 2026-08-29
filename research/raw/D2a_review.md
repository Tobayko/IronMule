# D2a — Exact-identity pre-change baseline review

**Date:** 2026-08-29
**Status:** `READY_FOR_BASELINE`; no D2a timing observed.

The user approved D2, but no D2 source change exists yet. D2a freezes a new same-day
pre-change reference so D2b does not depend on B27d's cross-day 4B ambiguity or pool
B27e's control samples.

## Bound state

- clean branch commit before D2 source:
  `857c023d1243d55801537fdeeeba9275fa767a30`;
- runtime-tree SHA-256 expected from the unchanged D1 tree:
  `d7577af8e83778b9753ad4bf721656a16d923a9f848040e406178b7dcffc8a21`;
- baseline harness SHA-256:
  `e6d981583384d4b526af32eb508579a79815bebabea0c64c8a2f4d99ebfe74d4`.

The harness forces baseline knobs, strict plan, local cached snapshots and complete
service timing. Stored profiles are not used. D2a writes identity available in the
pre-D2 harness (exact revision/full manifest/architecture/quantisation) and retains
complete raw samples.

## Cells and gates

Gemma 3 4B revision `93724907d4ed1745d2fe50baadf3b0b01a65abf2`, then Gemma
3 12B revision `86cc6a8dedbc456dd0e4af01a9d09f396f77e558`. Each cell is a
fresh serial process with six requests × 48 tokens, two warmups and six measured
Interactive/Throughput repeats. AC, memory-free >=80%, swap <=256 MiB, no model
process, exact arm output and zero fallback/resource errors are hard gates.

No B27a2/B27d/B27e timing is pooled. A failure is retained and later D2 comparison
does not start from an incomplete reference. No retry or threshold editing.

**Decision:** ready to seal D2a and commit the contract/review before model execution.
