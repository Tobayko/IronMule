# Documentation index

## User documentation

- [BENCHMARKS.md](BENCHMARKS.md) — measured throughput by model, device and mode.
- [HTTP.md](HTTP.md) — the OpenAI-compatible server: routes, auth, limits.
- [RUNTIME.md](RUNTIME.md) — the Python `Runtime`/`Engine` API and execution plans.
- [LIMITS.md](LIMITS.md) — what is and is not qualified, and why.
- [RELEASING.md](RELEASING.md) — the release checklist.
- [BACKLOG.md](BACKLOG.md) — open performance hypotheses and Tier 0's rejected ones.
- [PRODUCT_QUICKSTART.md](PRODUCT_QUICKSTART.md) — HTTP service preview: serving, calibration jobs, network exposure.
- [PAIRED_OPT_IN.md](PAIRED_OPT_IN.md) — the opt-in paired-decode service mode.
- [DATA1_PORTABLE_COLLECTION.md](DATA1_PORTABLE_COLLECTION.md) — portable, quota-bounded collection in `friday_evidence`.
- [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) — how to submit a benchmark from your own machine.
- [D2_IMPLEMENTATION.md](D2_IMPLEMENTATION.md) — the exact model-identity implementation, as built.
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — status numbers pinned by `tests/claims/test_documented_claims.py`.
- [assets/](assets/) — figures and video the docs above embed.
- [releases/](releases/) — dated release notes.

## Qualification specifications

Preregistrations and qualification protocols that `friday_evidence/provenance.py`
or a `tools/product_*.py` harness hashes by path. Their bytes stay identical to what
a sealed run measured against; do not edit, rename or move them. Seven are historical
German-language drafts, marked below.

- [PHASE1_MATMUL_SPEC.md](PHASE1_MATMUL_SPEC.md) — H0 measurement-system preflight (German).
- [H1_VORREGISTRIERUNG_ENTWURF.md](H1_VORREGISTRIERUNG_ENTWURF.md) — historical H1 preregistration draft (German).
- [H1H2_EVIDENZ_ARCHITEKTUR.md](H1H2_EVIDENZ_ARCHITEKTUR.md) — H1/H2 evidence architecture draft (German).
- [PROD2_BOUNDED_PREFETCH_2026-09-07.md](PROD2_BOUNDED_PREFETCH_2026-09-07.md) — bounded-prefetch pilot preregistration.
- [PROD4_LOAD_MEMORY_SPEC.md](PROD4_LOAD_MEMORY_SPEC.md) — observed load memory limits (German).
- [PROD4_PROCESS_MEMORY_SPEC.md](PROD4_PROCESS_MEMORY_SPEC.md) — 12B load process-memory footprint (German).
- [PROD6_MODEL_POLICY_SPEC.md](PROD6_MODEL_POLICY_SPEC.md) — no executable model files from snapshots (German).
- [PROD8_LONG_CONTEXT_SPEC.md](PROD8_LONG_CONTEXT_SPEC.md) — exact 12B long-context integration screen.
- [PROD10_GPU_CAPTURE_SPEC.md](PROD10_GPU_CAPTURE_SPEC.md) — private Metal GPU capture, diagnostic only.
- [PROD10_OPEN_VALIDATION_SPEC.md](PROD10_OPEN_VALIDATION_SPEC.md) — native validation after removing artificial hardware gates.
- [PROD10_SERVER_SOAK_SPEC.md](PROD10_SERVER_SOAK_SPEC.md) — one-hour 12B mixed-concurrency endurance protocol.
- [PROD11_NATIVE_CORRECTNESS_SPEC.md](PROD11_NATIVE_CORRECTNESS_SPEC.md) — native correctness qualification.
- [PROD12_SERVER_MEMORY_SPEC.md](PROD12_SERVER_MEMORY_SPEC.md) — separate server/worker memory observation (German).
- [PROD14_NATIVE_SCREEN_SPEC.md](PROD14_NATIVE_SCREEN_SPEC.md) — native variant correctness screen.
