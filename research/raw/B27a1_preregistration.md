# B27a1 — Cached-snapshot resolver correction

**Status before amended measurement:** sealed correction; no model was loaded and no
timing was observed in B27a.

The first B27a invocation stopped during `model_binding` after `0.224 s`. The local
Hugging Face resolver rejected the cached Gemma 4B snapshot because two optional
repository files (`.gitattributes` and `README.md`) were absent. System swap stayed
`0 B`, memory-free percentage stayed `87%`, and no benchmark arm ran. The complete
failure record is retained without overwrite as
`B27a_gemma4b_main_baseline_20260828.json`, SHA-256
`e5e7ab91218a4e7a7dcd2544efc3b44fbfdbed6fefce70cadd4f5c1c366e306a`.

## Sole amendment

Replace `snapshot_download(local_files_only=True)` in the research runner with
`scan_cache_dir()` plus an explicit exact commit revision. The selected cached
revision must occur exactly once and its snapshot directory must exist. The resolved
local directory is still passed to IronMule's offline loader; there is no network
fallback and no download or installation.

The corrected runner now requires `--revision`. Frozen hashes:

- `research/b27_main_baseline.py` SHA-256
  `d683e99f47cbdecc10fc005815e15fe16cd02d0e4d38844d38f34b2739ad5b46`
- `tests/test_b27_main_baseline.py` SHA-256
  `c5cc474cdc19aeec1f27e303aa264e0bf94c09dffc90b8976ada7297287814e8`

The resolver regression and all baseline-runner CPU tests pass (`4 passed`).

No B27a model, workload, arm, knob, order, warmup, repeat, metric, correctness gate,
resource gate, threshold or interpretation changes. Corrected outputs use the B27a1
prefix and are never pooled with or substituted for the preserved pre-measurement
failure record.
