# Changelog

All notable public changes to IronMule are documented here. Measurements and research conclusions are preserved as recorded in [`research/LEDGER.md`](research/LEDGER.md); this changelog does not reinterpret them.

## [Unreleased]

Review follow-ups completed locally; this section is not a release or a performance claim.

- **New: an OpenAI-compatible HTTP endpoint (`ironmule serve`, `ironmule.http`).**
  Standard library only, no new dependency. `POST /v1/chat/completions` (streaming
  and non-streaming), `GET /v1/models`, `GET /health`. It serves one loaded model
  one request at a time on the interactive path and answers `HTTP 429` while busy;
  it adds no sampling and no batching, and token output is identical to
  `Runtime.generate`. `Runtime.stream` was added to yield decoded deltas token by
  token (sequential path, same tokens as `generate` under `InteractiveMode`).
  See [`docs/HTTP.md`](docs/HTTP.md). `README.md` no longer says IronMule provides
  no streaming — it now provides a local one.

- **MLX peak memory is now measured per arm, not per process.** `mx.get_peak_memory()`
  is a process-wide high-water mark and `mx.reset_peak_memory()` was called nowhere in
  the repository. `ironmule/ab.py` loads a fresh model for every arm inside one child
  process and read the peak once after the arm loop, so the number filed under an arm
  was the maximum over every arm that process had run. It now resets before each arm
  and records `mlx_peak_bytes` per arm; the top-level value keeps its old meaning as
  the maximum across arms. `E15` limitation `M2` already recorded this for the research
  harnesses `E14`, `E14b` and `E15`; that it also affected the shipped library did not.
  The same reset is added to those three harnesses, where the inflated number also feeds
  a 12 GiB abort guard, so from the second block onward a run could be cut short by
  arithmetic rather than by memory. That mechanism is real, but it was not what aborted
  the 12B run that prompted this: its `17.51 GB` was measured on block 1, with nothing
  accumulated, so 12B exceeds that guard honestly. See `R10`.
- **`B25` closed: nothing reallocates the KV cache during decode.** Writing 56 tokens
  through `FixedKVCache` moves active memory from `65,644 B` to `32,876 B` — it falls
  by exactly one keys+values copy as the warmup double buffer is released, and the
  shape never changes. `tests/test_cache_allocation.py` holds it. The predecessor's
  `4.4263%` from cache growth copies belonged to a growing cache this runtime no longer
  has.
- **Two documentation corrections.** `research/LEDGER.md` said `E14b`'s token
  divergence reproduced "in all four processes" twenty lines above limitation `M2`,
  which states those were blocks in one interpreter; `docs/BACKLOG.md` repeated it in
  the `B12` entry. Both now say blocks and point at `M2`.

- **The stored gain now comes from the confirmation, not the screening.** `tune()`
  computed `gain` from `best_result`, the single-process screening measurement, while
  the six-process paired confirmation it had just run only decided accept/reject and
  never reached the profile. `ironmule.status()` therefore reported the weaker of two
  numbers the tuner already held, next to a `tokens identical` claim that did come from
  the confirmation. The first real end-to-end tuning run (`Q2`, 2026-08-29) stored
  `0.1457` from screening where the paired measurement was `0.8568`, i.e. `14.32%`.
  `status()` also states the paired 95% interval now: for that run `[5.98%; 14.51%]`,
  whose lower bound is nowhere near the headline number it used to print alone.

- **First-run experience fixed.** On a machine with no Hugging Face cache — every new
  clone, and every CI runner — `ironmule models` and `ironmule benchmark` raised a raw
  `huggingface_hub` traceback. A single `scan_local_cache()` helper now reads a missing
  cache directory as an empty one, so `models` prints an empty list and any command
  asking for an uncached model fails with the command that fixes it (`hf download …`)
  instead of a stack trace. The CLI reports `ModelIdentityError` as a message and exit
  `1`. The three ways a model can fail to resolve now say different things: nothing
  cached, cached but not at the requested revision, or several revisions cached and
  none pinned. The middle one used to claim the model was not cached at all, which
  sent the user to doubt their cache instead of their pin, and the last one used to
  advise a `--revision` flag the CLI does not have. The library still raises where the
  CLI reports. Subprocess regression tests cover both, plus the case
  where the MLX import itself is broken and `doctor` must still start. One behaviour
  change: `ironmule models` now reaches the cache through the shared helper in the
  `ironmule` package, so it needs the runtime import that `huggingface_hub` alone used
  to satisfy. A broken MLX install therefore makes `models` print the `ironmule doctor`
  hint and exit `1` instead of listing; `doctor` itself still imports neither.
- **Documentation links are now a test.** `tests/test_docs_links.py` walks every tracked
  Markdown file and fails on a relative link that does not resolve. It found 18 dead
  links: `research/LEDGER.md`, `docs/LIMITS.md` and `research/raw/B39_review.md` linked
  to raw benchmark JSON that `.gitignore` deliberately keeps local, so those links were
  dead for everyone who cloned. Raw files are now named as inline code, and the ledger
  header says plainly that raw evidence is local and the redacted
  `*_public_summary_*.json` is what ships.

- **R1 completed:** prefill-produced tokens now participate in the shared token/count/stop contract, including EOS and `max_tokens=1` coverage.
- **R4 completed:** the service backend honours both fused-token and logits output contracts.
- **R5 completed:** telemetry distinguishes an unperformed correctness check from a checked request/error count; benchmark mismatches are structured and fail closed.
- **R7 completed:** environment probing, version/dependency checks, unsupported tuning candidates, hardware discovery caching, and version ownership are covered by fail-closed tests.
- **R3 partial:** the public benchmark uses complete service/outer-wall timing, independent arm plans, balanced warmups/repeats, raw evidence, and structured mismatch exits; fresh-process isolation and a stock `mlx_lm` arm remain open by design.
- **R6/D2 implemented locally:** Runtime fingerprint v2 and tuned-profile conditions
  v2 bind the exact cached revision, complete manifest, architecture, canonical
  quantisation and tokenizer digest. Ambiguous cache resolution, identity conflicts,
  source changes during load, and legacy/incomplete profiles fail closed. D2 adds no
  routing, EvidenceRecord persistence or profile activation. Its sealed same-day
  4B/12B post-change screen preserved exact identities/outputs/resources and passed
  every 5% gate: `NO_REGRESSION_OBSERVED`, with no qualification or activation.
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
