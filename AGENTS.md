# IronMule Agent Guide

## Mission and authority

The agent maintains IronMule, an evidence-gated local LLM runtime for Apple Silicon
using MLX. Correct output and an honest reference path come before performance.

This is the shared engineering guide for Codex, Claude and Gemini. Providers may
use different tools; correctness, evidence, testing, safety and completion standards
remain the same. Keep shared behavior here, not in separate provider rulebooks.

This guide routes to existing sources of truth; it does not replace them. Read only
relevant context. When documentation and code disagree, inspect tests and history,
report the discrepancy and preserve the contract until deliberately resolved. An old
handoff, proposed design or result is not authorization to change current behavior.

## Repository map and context routing

Start with affected files and nearby tests. For a first substantive contribution,
read [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md). A typo correction
needs only the affected text and applicable documentation conventions.

| When working on | Load as needed |
| --- | --- |
| Product purpose, usage or public claims | [README.md](README.md); for claims, [docs/LIMITS.md](docs/LIMITS.md) and the supporting [ledger](https://github.com/Tobayko/IronMule-Research/blob/main/research/LEDGER.md) entry in the research repository |
| Runtime, executor, plans, cache or scheduling in [ironmule/](ironmule/) | [docs/RUNTIME.md](docs/RUNTIME.md), affected implementation and [runtime contract tests](tests/engine/test_ironmule_runtime.py) |
| Service, workers, HTTP, calibration or state in [ironmule_product/](ironmule_product/) | [docs/HTTP.md](docs/HTTP.md), affected modules and matching `test_product_*` / `test_automatic_*` files in [tests/engine/](tests/engine/) |
| Learned dispatch, qualification or recovery | LIMITS, relevant modules and tests: [learner](tests/test_b78_local_learner.py), [activation](tests/test_b79_activation.py), [monitoring](tests/test_b80_monitoring.py), [requalification](tests/test_b81_requalification.py), [readiness](tests/test_b82_readiness.py) |
| Performance, benchmarks or optimisation experiments | [docs/BACKLOG.md](docs/BACKLOG.md), including Tier 0 and its reopening rule |
| Shared evidence in [friday_evidence/](friday_evidence/) | Affected module, callers and tests; see Friday Evidence below |
| Packaging, dependencies, CLI or environment | [pyproject.toml](pyproject.toml), CONTRIBUTING and [CI](.github/workflows/ci.yml) |
| Validation selection | [pytest.ini](pytest.ini), [tests/conftest.py](tests/conftest.py), affected tests and CI |
| Releases | [docs/RELEASING.md](docs/RELEASING.md), including its full pre-publication gates |
| Figures or aggregate evidence views | [tools/make_figures.py](tools/make_figures.py) for [docs/assets/](docs/assets/) |

[tools/](tools/) now holds only what the product, the kept tests, CI or the README
figures need -- measurement/verification harnesses for one specific kept test, plus the
figure renderer. Inspect the selected tool and its contract before running it.

The research workflow that used to route through this file -- the experiment ledger,
preregistration and sealed-study conventions, the Kaggle harnesses, the SSOT corpus and
the rest of Project Friday -- lives in a separate repository,
[IronMule-Research](https://github.com/Tobayko/IronMule-Research). Consult it for that
work; this guide covers the product repository only.

ProjectAtlas is optional package-development tooling under CONTRIBUTING; it is not a
dependency of IronMule and nothing in the package, the tests or CI imports it.

## Working method and scope

1. Check the working tree, requested outcome, affected subsystem and existing
   contracts. Preserve other people's changes.
2. Use filenames, headings and focused searches before large documents. Follow the
   relevant routing row; do not invent APIs, paths or repository rules.
3. Make the smallest complete change within the existing architecture. Avoid unrelated
   refactoring, speculative abstractions, unnecessary dependencies and gratuitous
   public API changes.
4. Reproduce the bug or establish the appropriate baseline, implement, then validate.
   For performance: make it work, measure, identify the bottleneck, optimise, validate
   correctness, and measure again.
5. Fix failures introduced by the change, update owning documentation and review the
   final diff against the original request.

Normal local inspection, history searches, edits, focused refactoring, test additions,
tests, existing static checks, bug reproduction and appropriate local benchmarks do
not need repeated confirmation. Follow actual study gates and tool permissions;
normal engineering autonomy does not authorize bypassing either.

Minimal-change modes (for example the ponytail plugin at `ultra`) shorten the solution,
never the evidence path. Their YAGNI never covers backlog or evidence discipline: read
[docs/BACKLOG.md](docs/BACKLOG.md) and the research ledger before proposing performance
work, and record results where they belong (see Performance and evidence below).

## Runtime correctness

- Preserve the caller's execution plan. Strict and reusable-session plans can produce
  different output; cache reuse must be exact against the same plan without reuse.
  Do not silently substitute a plan to obtain a faster result.
- Preserve token IDs, physical/visible counts, EOS and length stop reasons, request
  ordering, per-request state isolation and cache correctness. Grouped batch-1
  preserves tensor shapes; it is not true tensor batching.
- Preserve fallback output: discard failed grouped work, restart from the defined
  state on the sequential path and record reasons. Keep service and engine TTFT
  distinct; no recorded errors is not proof that correctness was checked.
- Executor, plan or cache changes must keep
  [tests/engine/test_ironmule_runtime.py](tests/engine/test_ironmule_runtime.py) green.
  Use its [real-model counterpart](tests/engine/test_ironmule_runtime_integration.py)
  and affected cache/model tests for hardware behavior. Extend coverage for changed
  grouping, arrivals, token generation or recovery.
- Preserve exact identity and fail-closed qualification. Missing, corrupt, foreign,
  stale or mismatched evidence cannot authorize an optimisation. Consult
  [identity](tests/engine/test_model_identity.py),
  [evidence](tests/engine/test_evidence.py) and
  [router](tests/engine/test_router.py) tests when changing these boundaries.
- Learned dispatch stays opt-in and off by default. Comparative evidence may qualify
  an action; ordinary observations may withdraw it but cannot qualify or strengthen
  it. Readiness cannot restore qualification. Preserve explicit requalification,
  persistent kill state and reference fallback; use the lifecycle tests above.

## Performance and evidence

Before proposing an optimisation, inspect [docs/BACKLOG.md](docs/BACKLOG.md)'s Tier 0
and the cited ledger entries. Add unlisted optimisation work with its mechanism and
kill criterion before starting. Reopening a rejected route requires naming the old
entry, quoting its kill criterion and documenting new hardware evidence, a changed
mechanism or a new implementation. Rejection is scoped historical evidence, not a
permanent ban.

A performance claim needs a measurement on the target hardware: identify the measured
machine, software, exact model identity, quantisation, plan, workload and control.
Compare against the appropriate reference and the measured complete stack; distinguish
absolute performance from baseline-relative gain, latency from throughput, and kernel
timing from end-to-end benefit. Correctness failure disqualifies a performance result.
Changed identity requires applicable revalidation or requalification.

Close answered backlog entries as CONTRIBUTING specifies: shipped results go to the
research repository's ledger; rejected results to Tier 0 with the entry number and
experiment ID. Record new findings and update LIMITS when the validity domain changes.

The actual measurement work -- preregistration, sealed-study and separate-process
requirements, warmups, paired measurements, A/A controls, the Kaggle harnesses and the
SSOT corpus that indexes it all -- happens in
[IronMule-Research](https://github.com/Tobayko/IronMule-Research), which this repository
no longer contains. A change proposed here still needs that evidence before it ships;
it is produced and recorded there, not fabricated locally.

## Friday Evidence and historical records

Reuse `friday_evidence` for new shared statistics, storage, provenance and observation
logic; do not copy infrastructure out of frozen study packages.

- **Storage/schema/canonicalisation:** inspect [storage.py](friday_evidence/storage.py),
  [migrations/](friday_evidence/migrations/), [canonical.py](friday_evidence/canonical.py),
  [provenance.py](friday_evidence/provenance.py) and
  [contract tests](tests/test_friday_evidence.py). Preserve append-only records,
  deterministic hashes, idempotency, conflict rejection, private storage and read-only
  verification. The store verifies the initial migration's digest and rejects unknown
  schemas; it is not an automatic migration framework. Document and validate schema
  compatibility explicitly; do not rewrite old SQL to make existing evidence appear
  valid.
- **Identity/resources/budgets:** inspect affected helpers and
  [identity](tests/engine/test_product_identity.py),
  [process-memory](tests/engine/test_process_memory.py) or
  [observation](tests/engine/test_open_observation.py) tests. Missing telemetry remains
  unavailable, not a fabricated zero. Keep observation separate from admission.
  Historical BudgetGuard rules belong to their sealed studies in the research
  repository; PROD10's rules (its results live there too) govern new product
  validation. Do not restore superseded blanket limits or remove gates from other
  qualification protocols by analogy.
- **Portability:** follow [docs/DATA1_PORTABLE_COLLECTION.md](docs/DATA1_PORTABLE_COLLECTION.md),
  [portable/](friday_evidence/portable/) and matching `tests/test_portable_*` tests.
  Preserve backend/identity binding, provenance, holdout separation, quota checks and
  verified cleanup. Imported diagnostics are not automatically training evidence or
  permission to activate a local runtime path.

Sealed preregistrations, result records and frozen study packages remain byte-identical;
they live in the research repository now, and that rule travels with them. A handful of
dated documents still sit in this repository -- `docs/PHASE1_MATMUL_SPEC.md`,
`docs/H1_VORREGISTRIERUNG_ENTWURF.md`, `docs/H1H2_EVIDENZ_ARCHITEKTUR.md`,
`docs/PROJECT_STATUS.md` and `requirements-apple-silicon.txt` -- because
`friday_evidence/provenance.py`'s frozen `SPEC_FILES`/`SOURCE_DIRS` tuple hashes them for
its own provenance self-consistency check
(`tests/test_friday_evidence.py::RootProvenanceContractTest`); do not delete, rename or
edit them without checking that test first.

Keep private raw data and local databases under the existing ignore policy. Publish
only deliberately redacted artifacts; never expose prompts, secrets or local paths.
Regenerate figures with their tool and run its `--check`; do not edit rendered curves
or source measurements to improve a headline.

## Testing and documentation

Run the smallest relevant tests first, then broaden for the affected surface. Common
engine checks from CONTRIBUTING are:

```sh
python -m pytest tests/engine -m "not integration"
python -m pytest tests/engine -m integration
```

The second command needs a local model snapshot. Respect pytest's parallelism and
file grouping; use `-n 0` for sequential diagnosis. `tests/conftest.py` leaves a test
file uncollected only when it needs `mlx` and `mlx` is not installed; do not disable
that gate or present skipped/uncollected tests as passes. Scripted contract tests do
not replace execution on the target hardware for hardware claims.

Use existing CI checks for affected packaging, static-check and figure surfaces.
Releases require the broader RELEASING checklist, including clean installed-package
validation. A text-only correction normally needs a diff/link check, not GPU tests
or the complete suite. [tests/engine/test_docs_links.py](tests/engine/test_docs_links.py)
checks tracked Markdown; check new untracked documents explicitly as well.

CONTRIBUTING owns English-language, no-emoji, privacy, commit, branch and attribution
conventions; [LICENSE.md](LICENSE.md) owns licensing. Update the relevant source
when behavior or claims change instead of growing this guide. Use the
[PR template](.github/pull_request_template.md) for contributions.

**Existing attribution conflict:** CONTRIBUTING prescribes a single provider-specific
co-author trailer. This guide does not amend that policy or the licence. Flag its
mismatch with actual tool authorship for maintainer resolution before an affected
commit; do not invent authorship or silently replace the policy.

## Safety boundaries

Obtain explicit authorization before publishing releases/packages, destructive history
changes, deleting important remote resources, changing production infrastructure,
unexpected paid external compute, changing credentials/secrets or sending external
communications. Local preparation and validation need no approval at every step;
existing explicit authorization need not be repeated.

Preserve local-only product behavior, explicit model-download commands, privacy and
authentication. Do not disable OS/thermal protection, change system security or power
settings, kill unrelated processes, or bypass sandbox permissions. Run generated
kernel code through the controlled worker with its correctness, timeout, resource and
rollback contract. Respect cancellation and verify cleanup of owned workers/jobs.

## Definition of done

A task is complete when the requested outcome is implemented, the original request
has been rechecked, relevant contracts hold, proportionate validation has run and
failures introduced by the change are fixed. Performance work also requires applicable
evidence, backlog/ledger updates and bounded claims. Update relevant documentation
and review the diff for unrelated changes and private data.

Report what changed, exact checks and outcomes, skips or unavailable validation, and
remaining limitations or known regressions. Code written, a silent fallback or an
unexecuted test is not proof of completion.
