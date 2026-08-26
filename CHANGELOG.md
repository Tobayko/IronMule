# Changelog

All notable public changes to IronMule are documented here. Measurements and research conclusions are preserved as recorded in [`research/LEDGER.md`](research/LEDGER.md); this changelog does not reinterpret them.

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
