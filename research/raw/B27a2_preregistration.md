# B27a2 — Module-invocation correction

**Status before amended measurement:** sealed invocation correction; no benchmark
arm has run and no timing has been observed in B27a or B27a1.

B27a1 resolved and fully hashed the exact Gemma 4B snapshot, then stopped at stage
`model_binding` with `ModuleNotFoundError: No module named 'ironmule'`. Directly
executing a file under `research/` makes that directory, rather than the repository
root, Python's import root. The failure record is retained without overwrite as
`B27a1_gemma4b_main_baseline_20260828.json`, SHA-256
`dd09d2e2cc4a2ad9ac95272c4d464ff029730e82dcc3475d374683e0ad1e2260`.
System swap remained `0 B`; no model was loaded and no benchmark arm ran.

## Sole amendment

Invoke the unchanged frozen harness as the repository module
`python -m research.b27_main_baseline` from the repository root. Its `--help` path
was validated successfully before sealing. The harness and test hashes remain the
B27a1 values:

- harness `d683e99f47cbdecc10fc005815e15fe16cd02d0e4d38844d38f34b2739ad5b46`;
- tests `c5cc474cdc19aeec1f27e303aa264e0bf94c09dffc90b8976ada7297287814e8`.

No model, revision, workload, arm, knob, order, warmup, repeat, metric, gate,
threshold or interpretation changes. Corrected outputs use the B27a2 prefix and are
never pooled with either preserved pre-measurement failure.
