# Repository map

What each part of the tree is for, whether it may change, and what checks it. The
[agent guide](../AGENTS.md) routes work to the right sources; this page records the status
of each area, so a clean-up can tell code from evidence.

| Area | Content | Status | Checked by |
| :-- | :-- | :-- | :-- |
| `ironmule/`, `ironmule_product/`, `friday_evidence/`, `ironmule_cli.py`, `ironmule_inventory.py` | The shipped package (`pyproject.toml`) | Live code | `tests/engine/`, CI |
| `tests/engine/` | The package's own suite; runs anywhere | Live | CI, Kaggle (TEST1) |
| `tests/test_*.py` | The research suite, bound to the target Mac (`tests/conftest.py`); `test_numeric_plans.py` and `test_documented_claims.py` read only committed files and run everywhere | Live, target device only (those two: everywhere) | The Mac; those two also CI and Kaggle |
| `research/friday_*` | Research packages; their manifests hash paths relative to `research/` | Frozen where a study sealed them | The research suite |
| `research/LEDGER.md` | Experimental conclusions | New entries appended; recorded results not rewritten | `tests/engine/test_docs_links.py` |
| `research/raw/`, `research/product_history_*`, `experiments/*/` data | Raw evidence and result records | Byte-identical once recorded | SSOT `verify --check-sources`, `tests/test_sealed_evidence.py` |
| `experiments/kaggle_compat/*.py` | Kaggle harnesses and their shared scripts (`cross.py`, `perf1.py`, `quality.py`) | Live; a run's harness is archived with it | Kaggle runs |
| `experiments/kaggle_compat/results/<run>/` | One directory per Kaggle run: result JSON, logs, `submitted-notebook.py` and `submitted-kernel-metadata.json` as they ran | Byte-identical once recorded | SSOT |
| `tools/` | Measurement, analysis and verification harnesses | Live; many are the harness behind one ledger entry | Their ledger entries |
| `docs/` | Contracts (`RUNTIME.md`, `HTTP.md`, `LIMITS.md`), backlogs, preregistrations, specifications and historical reports | Preregistrations and dated reports are frozen | `tests/engine/test_docs_links.py` |
| `docs/assets/` | Figures rendered from committed evidence | Generated | `tools/make_figures.py --check`, CI |
| `examples/`, `profiles/`, `scripts/` | Usage examples, measured tuning profiles, environment setup | Live | Manual |
| `ironmole_mcp/` | An independent TypeScript MCP runtime; only its benchmark report reuses `friday_evidence.statistics` | Separate project | Its own `npm test` |

## Code identity

`friday_evidence/identity.py` hashes every `.py` file under `ironmule/`,
`ironmule_product/` and `friday_evidence/` into the product's code identity. Any change there,
a deleted helper included, changes that identity, and stored calibrations and qualifications
on the target device must be renewed. Batch clean-ups in these packages into one change.

## Kaggle harnesses

A harness is written as a template in `experiments/kaggle_compat/`, submitted with its
placeholders filled, and archived byte for byte in its run's `results/` directory. Once the
run is archived, the archived copy is the record and the template is removed; git history
keeps the reviewed template. Shared scripts that later runs reuse stay.

## Rebased history

Pull request 9 was rebase-merged, so its 16 commits reached `main` with new hashes and
identical trees (`f9e1d30` is `7af3733` on `main`, `9394838` is `c4cb89a`). The ledger and the
archived notebooks of TEST1, TEST2, TEST2-G and the clean-up check name the original hashes;
GitHub keeps them under `refs/pull/9/head`.

## Clean-up status (2026-09-24)

- Removed: four private helpers nothing called (`ironmule/q4_corpus.py`,
  `ironmule/q4_optimizer.py`) and thirty-five Kaggle templates whose runs are archived (`perf1_run1` to `perf1_run18`, `test1`,
  `test1_qwen`, `test2`, `test2g`, `oss1`, `backlog1` to `backlog9`, `shot1`, `perf1k`, `tests1`).
- Merged 2026-09-25: `research/port2-model-families` (PERF1 runs 14-18, the MoE row kernel, BOS
  on every `perf1.py nll` chunk), with run 18's output archived from its Kaggle download.
- Byte-identical files in the tree are per-run logs (the same download or environment
  output in several runs). They are evidence of each run and stay.
- 55 scripts in `tools/` are named nowhere else in the repository (the B15, B24 and B42 to
  B56 harnesses, and the `bench_*`, `profile_*`, `sweep_*`, `demo_*` and `test_live_*`
  scripts added on 2026-09-03). Their outputs may live only in the target device's
  `.friday-data/`, so they stay until that is checked there.
- `ironmole_mcp/` depends on IronMule only through `friday_evidence.statistics` in its
  benchmark report and could move to its own repository.
- The Kaggle templates `port1_run5` to `port1_run8` and `port2_run1` to `port2_run9` have no
  archived submitted copy and stay.
- Checked on Kaggle (CPU notebook, `results/cleanup-check-84944ce6/`): engine suite 1260
  passed, 28 skipped, 0 failed; ruff, figures, CLI smoke and SSOT `verify --check-sources` pass.
