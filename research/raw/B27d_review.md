# B27d — D1 pre-measurement review

**Review status:** `READY_FOR_POST_CHANGE_MEASUREMENT`; no performance result reviewed.

## Scope checked

The user explicitly approved D1 on 2026-08-28. Review is limited to the approved
stdlib-only evidence contract and its post-change measurement harness. Persistence,
runtime integration, automatic selection, profile mutation, activation, stock
`mlx_lm`, new dependencies and new models remain out of scope.

Frozen source hashes at review time:

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
- approved contract document:
  `9fedda6276f9a0a491390a0be02c06dc43d05cd82a1e33f79bb305c182516bfb`

## Findings and resolutions

1. **Profile-deserialization bypass found and closed.** A direct serialized trusted
   profile could initially have supplied evidence IDs without their corresponding
   qualified records. Construction now requires a private factory token; deserialization
   requires the supplied `EvidenceRecord` objects and compares the rebuilt canonical
   profile byte-for-byte with the serialized form.
2. **Role identity strengthened.** Terminal evidence now requires distinct non-empty
   Researcher, Reviewer and Evaluator identities. A Researcher may create only
   `HYPOTHESIS`; a Reviewer cannot qualify; self-evaluation is rejected.
3. **Qualified evidence gates strengthened.** Primary metrics require measured raw
   samples, median, p95 and interval. MLX active/peak, RSS and swap evidence, crash,
   timeout, fallback and repeated uncertainty gates are all required.
4. **Summary import no longer invents resource facts.** The B27 public-summary adapter
   marks state identity, absolute swap, crash and complete resource gates unverified;
   the imported record is `INCONCLUSIVE/SUMMARY_ONLY` and cannot construct a trusted
   profile.
5. **Validity comparison expanded.** Post-change comparison binds chip, machine, RAM,
   GPU, hardware fingerprint, runtime/framework/model identity, quantisation,
   protocol, power/low-power/thermal, swap class and memory-free class before timing
   interpretation.
6. **Runtime boundary confirmed.** `ironmule/evidence.py` imports only stdlib modules.
   Package root, Runtime, service, plans, executors, tuner, benchmark, telemetry and
   fingerprint do not import it. The new types expose no `run()` or `select()` method.

## Verification before hardware measurement

- D1 focused contract suite: `15 passed`;
- final D1/baseline/comparison focused set: `26 passed`;
- full serial non-integration suite: `146 passed, 11 deselected` in `5.21 s`;
- existing real Gemma-4B integration suite: `10/10` in `21.24 s`;
- pre-integration system swap: `0 B`; no competing model process;
- no package, model, network or installation action.

## Remaining limits

- D1 is representation and validation only; it has no persistence or migration.
- No current runtime module consumes a D1 record or profile.
- The post-change comparison is an engineering regression screen across separate
  sessions, not a qualification or stock-MLX experiment.
- The reviewer and evaluator remain logical records in D1; no autonomous role runner
  is implemented.

**Decision:** the approved D1 implementation is ready to be committed and subjected
to the separately sealed B27d 4B/12B post-change protocol. Any observed value remains
unseen at this review point.
