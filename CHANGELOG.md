# Changelog

All notable public changes to IronMule are documented here. Measurements and research conclusions are preserved as recorded in [`research/LEDGER.md`](research/LEDGER.md); this changelog does not reinterpret them.

## [Unreleased]

Review follow-ups completed locally; this section is not a release or a performance claim.

- **R1 completed:** prefill-produced tokens now participate in the shared token/count/stop contract, including EOS and `max_tokens=1` coverage.
- **R4 completed:** the service backend honours both fused-token and logits output contracts.
- **R5 completed:** telemetry distinguishes an unperformed correctness check from a checked request/error count; benchmark mismatches are structured and fail closed.
- **R7 completed:** environment probing, version/dependency checks, unsupported tuning candidates, hardware discovery caching, and version ownership are covered by fail-closed tests.
- **R3 partial:** the public benchmark uses complete service/outer-wall timing, independent arm plans, balanced warmups/repeats, raw evidence, and structured mismatch exits; fresh-process isolation and a stock `mlx_lm` arm remain open by design.
- **R6 partial:** system/profile conditions fail closed and current workload drift is checked; model revision and quantisation identity remain open under approval-dependent P0.11 work.
- **R8 partial:** macOS build/wheel/CLI/unit-test workflow is checked in; local wheel build/metadata/zipimport CLI smoke is green, while the clean installed-wheel job and remote CI execution remain open.
- **B37 completed:** adds a pure, fail-closed phase/roofline diagnostic with per-run
  effective-bandwidth inputs; it does not alter runtime gates, profile selection, or
  performance claims.
- **B37a completed:** hardens diagnostic numeric bounds and requires structured
  measured-effective versus nominal-peak bandwidth provenance.
- **B36 completed:** records the arm-isolated Gemma 3 12B core-profile result as
  a narrowly scoped qualification; activation and generalization remain disabled.
- **DOC1 completed:** rewrites the public README in plain language, moves the
  problem and quick start ahead of research detail, links measured claims to their
  evidence, corrects the unpublished PyPI install path, and sharpens discovery
  metadata without extending the measured M1 Max validity domain.
- **B39d completed:** publishes the exact-scope Gemma 3 12B result: `22.03%`
  higher token rate and `18.05%` lower complete service wall time versus baseline
  interactive, with a separate `6.58%` / `6.17%` incremental comparison under the
  same throughput mode. No profile is activated automatically.
- **B3-U2 pilot recorded:** 8/8 isolated processes and 240 measured requests passed
  token/state correctness and safety gates. This is not a speed result; confirmation
  remains blocked by missing persisted per-child host-state evidence.
- **B27 D1 implemented, not activated:** adds stdlib-only immutable execution-strategy,
  validity-domain, evaluator-owned evidence and trusted-profile contracts. Runtime,
  tuner, plans, modes and executors do not import D1; persistence, selection and
  activation remain outside the approved scope. The sealed post-change screen passed
  12B but found a common-mode 4B potential regression, so its frozen result is
  inconclusive and no performance-safety claim follows. A separate mirrored OLD/D1
  4B control did not reproduce a D1 slowdown, but was order/temporal-drift sensitive;
  it also makes no neutrality or activation claim.

## [0.1.0] — 2026-08-26

Initial public release, prepared for publication.

- **Runtime:** MLX inference runtime for local LLMs on Apple Silicon with explicit execution plans, prefix KV caching, grouped batch-1 execution, correctness checks, telemetry, and validity fingerprints.
- **Evidence:** Includes the preregistered experiment ledger, raw result summaries, negative findings, and the narrow validity domain in [`docs/LIMITS.md`](docs/LIMITS.md).
- **Documentation:** Adds a short install/quickstart path, measured benchmark summary, API/runtime guide, fair-code licensing explanation, and community benchmark submission workflow.
- **Branding:** Adds the source logo asset and a validated 1280×640 GitHub social-preview image.
- **Community:** Adds issue templates for bugs, feature requests, and benchmark submissions, plus a pull request checklist.

### Measurement note

The values in the release notes and README are copied from the existing ledger. They apply only to the stated model, MLX versions, M1 Max machine, power mode, decoding, contexts, batch, and concurrency conditions. No universal speedup claim is made.

[0.1.0]: docs/releases/v0.1.0.md
