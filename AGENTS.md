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
| Product purpose, usage or public claims | [README.md](README.md); for claims, [docs/LIMITS.md](docs/LIMITS.md) and the supporting [ledger](research/LEDGER.md) entry |
| Runtime, executor, plans, cache or scheduling in [ironmule/](ironmule/) | [docs/RUNTIME.md](docs/RUNTIME.md), affected implementation and [runtime contract tests](tests/engine/test_ironmule_runtime.py) |
| Service, workers, HTTP, calibration or state in [ironmule_product/](ironmule_product/) | [docs/HTTP.md](docs/HTTP.md), affected modules and matching `test_product_*` / `test_automatic_*` files in [tests/engine/](tests/engine/) |
| Learned dispatch, qualification or recovery | LIMITS, relevant modules and tests: [learner](tests/test_b78_local_learner.py), [activation](tests/test_b79_activation.py), [monitoring](tests/test_b80_monitoring.py), [requalification](tests/test_b81_requalification.py), [readiness](tests/test_b82_readiness.py) |
| Performance, benchmarks or optimisation experiments | [docs/BACKLOG.md](docs/BACKLOG.md), including Tier 0 and its reopening rule; relevant ledger entries, limits and the specific preregistration/harness in [research/](research/) or [experiments/](experiments/) |
| Shared evidence in [friday_evidence/](friday_evidence/) | Affected module, callers and tests; see Friday Evidence below |
| Project Friday packages under [research/](research/) | [Research overview](docs/README_PROJECT_FRIDAY.md), relevant [research backlog](docs/PROJECT_FRIDAY_BACKLOG.md) entry and study specification |
| Packaging, dependencies, CLI or environment | [pyproject.toml](pyproject.toml), CONTRIBUTING and [CI](.github/workflows/ci.yml); [scripts/](scripts/) contains research setup and environment checks |
| Validation selection | [pytest.ini](pytest.ini), [tests/conftest.py](tests/conftest.py), affected tests and CI |
| Releases | [docs/RELEASING.md](docs/RELEASING.md), including its full pre-publication gates |
| Figures or aggregate evidence views | [tools/make_figures.py](tools/make_figures.py) for [docs/assets/](docs/assets/); [docs/SSOT.md](docs/SSOT.md) for the derived evidence index |
| Clean-up, or whether a file is code or evidence | [docs/REPOSITORY_MAP.md](docs/REPOSITORY_MAP.md) |

[tools/](tools/) contains measurement, analysis and verification harnesses. Inspect
the selected tool and its contract before running it; unrelated work does not need
all experiments or the entire ledger.

ProjectAtlas is optional package-development tooling under CONTRIBUTING. For the
Project Friday research workflow, follow the local, untracked `docs/PROJECTATLAS_INTEGRATION.md`:
use the installed, version-matched skill and MCP context first, with its CLI fallback
and incremental freshness policy. Initialize only absent local state. A documented
tool failure permits direct navigation; index output is never benchmark evidence.

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
never the evidence path. Their YAGNI never covers the [SSOT](docs/SSOT.md) chain: read
the backlog and ledger before work, record results where they belong, and after adding
evidence rebuild and verify the corpus (`python tools/ssot.py build`, then `verify
--check-sources`).

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

Before proposing an optimisation or starting a substantial experiment, inspect
relevant backlog evidence and Tier 0, then follow the cited ledger entries. Add
unlisted optimisation work with its mechanism and kill criterion before starting.
Reopening a rejected route requires naming the old entry, quoting its kill criterion
and documenting new hardware evidence, a changed mechanism or a new implementation.
Rejection is scoped historical evidence, not a permanent ban.

Use the applicable preregistration and existing harness. Fix the question, workload,
baseline, candidate, gates and kill criterion before measuring; follow the repository's
sealed-preregistration and separate-process requirements for work that could ship.
Use warmups, repeated paired measurements, balanced order and appropriate A/A controls;
report raw samples, median, spread and uncertainty. Do not borrow a noise floor from
another regime or infer a gain from one run.

Compare against the appropriate reference and measured complete stack. Distinguish
absolute performance from baseline-relative gain, latency from throughput, and kernel
timing from end-to-end benefit. Measure combinations; do not add or multiply isolated
gains. Correctness failure disqualifies a performance result.

Claims must identify measured hardware, software, exact model identity, quantisation,
plan, workload and control. A foreign machine, simulation or model-free test cannot
establish local MLX/Metal/model performance or correctness. Changed identity requires
applicable revalidation or requalification. When installing into an evidence-bound
environment, record its identity before/after and which qualifications need renewal.

Keep code, configuration and preregistration inputs stable throughout an active run.
Never change thresholds, methodology, evidence or exclusions after seeing results
merely to improve the number. Preserve failed, negative and inconclusive outcomes.
A corrected method or implementation needs a distinct, prospectively identified run;
it does not retroactively repair an old verdict.

Close answered backlog entries as CONTRIBUTING specifies: shipped results go to the
ledger; rejected results to Tier 0 with the entry number and experiment ID. Record
new findings and update LIMITS when the validity domain changes.

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
  valid. Check [sealed evidence](tests/test_sealed_evidence.py) when affected.
- **Identity/resources/budgets:** inspect affected helpers and
  [identity](tests/engine/test_product_identity.py),
  [process-memory](tests/engine/test_process_memory.py) or
  [observation](tests/engine/test_open_observation.py) tests. Missing telemetry remains
  unavailable, not a fabricated zero. Keep observation separate from admission.
  Historical BudgetGuard rules belong to their sealed studies; the research backlog's
  PROD10 rules govern new product validation. Do not restore superseded blanket limits
  or remove gates from other qualification protocols by analogy.
- **Portability:** follow [docs/DATA1_PORTABLE_COLLECTION.md](docs/DATA1_PORTABLE_COLLECTION.md),
  [portable/](friday_evidence/portable/) and matching `tests/test_portable_*` tests.
  Preserve backend/identity binding, provenance, holdout separation, quota checks and
  verified cleanup. Imported diagnostics are not automatically training evidence or
  permission to activate a local runtime path.

Sealed preregistrations, result records and frozen study packages remain byte-identical;
do not rehash or refactor them to fit the current layout. Documentary corrections must
be explicit and leave raw data and frozen criteria intact. The backlog documents
studies that still verify but can no longer re-seal after the layout change.

Keep private raw data and local databases under the existing ignore policy. Publish
only deliberately redacted artifacts; never expose prompts, secrets or local paths.
The SSOT index is regenerable derived data, not a replacement for source evidence.
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
file grouping; use `-n 0` for sequential diagnosis. The research suite has target-
environment collection gates in `tests/conftest.py`; do not disable them or present
skipped/uncollected tests as passes. Scripted contract tests do not replace execution
on the target hardware for hardware claims.

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

[docs/HANDOFF_PROMPT.md](docs/HANDOFF_PROMPT.md) and
[docs/GEMINI_SELF_LEARNING_SYSTEM.md](docs/GEMINI_SELF_LEARNING_SYSTEM.md) contain
historical task instructions and experimental memory. Consult their evidence only
when relevant; use this guide and current owning sources for shared behavior,
including backlog reopening and current product-validation rules.

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
