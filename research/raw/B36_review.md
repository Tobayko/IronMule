# B36 independent raw and arithmetic audit

Scope: read-only audit of B36_gemma12b_results_20260828.json, with all 16
pair medians and 10,000 bootstrap draws recomputed independently. No model,
Python/MLX runtime or UI action was taken by the audit.

## Protocol and gates

The raw result contains 16 serial pairs and 32 fresh child processes: eight
baseline -> candidate and eight candidate -> baseline. Every child has one
model load, two warmups and five measured repeats. All 32 children report
returncode 0, no crash, identity and canonical-correctness gates true, and
complete post-evidence.

The complete warmup/repeat physical/logical/visible token, stop-reason and
decode-step digest is
0c04b2910e0b8e5adc2c66108a79f4cbf233bf7fc8465f0a4d30418b6533019e.

## Independent ratios

Ratios are candidate / baseline. Each pair is the ratio of the two five-repeat
medians; the aggregate is bootstrapped over 16 pair ratios.

| Metric | Median | 95% CI | Bootstrap |
| --- | ---: | ---: | --- |
| total | 0.927147428180255 | [0.9197363534291831; 0.9303748490885659] | 10,000, seed 20260828 |
| prefill | 0.9183106745417602 | [0.9081866453423364; 0.9218801379522791] | 10,000, seed 20260829 |
| decode | 0.9540158794083631 | [0.9419388082376179; 0.9577180135679649] | 10,000, seed 20260830 |

Order strata:

| Stratum | Median | 95% CI |
| --- | ---: | ---: |
| AB | 0.9250279042521969 | [0.8883771936776205; 0.9283929297931295] |
| BA | 0.9294847335372622 | [0.925744092622901; 0.9404202103921636] |

Absolute order interaction is 0.0044568292850653, below the preregistered
0.02 limit. One pair has a decode ratio above one; this is retained as a
diagnostic observation and does not change the aggregate.

## Resource and identity evidence

- Maximum MLX peak: baseline 7,946,637,412 B; candidate 7,830,608,598 B.
- Maximum RSS: baseline 3,692,576,768 B; candidate 4,526,096,384 B.
- Candidate/baseline peak ratio: 0.985399004889189.
- Maximum process-start-to-end swap delta: 0 B.
- H1, delayed H1 and final H2 Foundation probes: low power 0, thermal raw
  value 0, returncode 0; memory free percentages 74, 75, 66.
- B36 preregistration SHA-256:
  7bf3997b19dc55d3b75be977c0da8d42d6ab554232ce2bf40617429c478897a4.
- B36a clarification SHA-256:
  ee5b3e9b250d75eb69ed6e38f9661f656da743098bef318966dc055099c9e492.
- Model manifest digest:
  3de99933cacc693c88d807c4f5e4dade6d1fe719cacc570841e222940f0a9eb2.
- Code digest:
  5566ee87f1656d9dcaceb05edf6a155ee2a35dd784c81a46fbb6dab30e499ddc;
  the current 61-file code fingerprint matches exactly, as does commit
  f3478e07d58e3bf054b3ae0503925dbb15f7edf1.

The candidate uses only compiled_fixed_cache=True and head_skip_prefill=True;
wired-limit and cache-limit mutations were not applied. Full manifest hashing
before and after load intentionally prefaulted the same artifact for both arms.

## Decision and permitted claim

The raw summary is QUALIFIED under the frozen B36 rules, while
activation_allowed remains false. The result is scoped to the exact local
Gemma 3 12B revision, 322-token B35 prompt, 32-token greedy generation,
Apple M1 Max 32 GB, MLX 0.32.0, mlx-lm 0.31.3, AC power, Foundation nominal
state, default wired/cache policy and the full-hash/prefault protocol.

This is not a general model-, chip-, framework- or runtime-performance claim.
The higher candidate RSS despite lower MLX peak is recorded without a hidden
memory claim. No profile activation or routing change follows from B36/B36a.
