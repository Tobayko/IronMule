# Evidence

The numbers under this directory are redacted summaries of measurements made on private
hardware. The raw runs — every request's full timing series, generated text, token IDs and
the local paths of the machine that measured them — stay on that machine; they were never
committed, and the ones that used to be are gitignored and were purged from the published
history on 2026-09-28.

What is here instead is exactly what `tools/make_figures.py`, `ironmule/numeric_plans.py`
and the two claim tests (`tests/claims/test_numeric_plans.py`, `tests/claims/test_documented_claims.py`)
read: paired ratios, medians, confidence intervals, per-chunk perplexities, identical-request
and failed-process counts, and the model identity (`model_id`, `revision`) and framework
versions (`mlx`, `mlx-lm`) each number was measured against. Nothing else survives the
export: no prompt or generated text, no token-id lists, no local or container paths, no
hostnames.

`evidence/kaggle/<run-dir>/<file>.json` mirrors `experiments/kaggle_compat/results/<run-dir>/<file>`;
`evidence/kaggle/<run-dir>/freeze.json` is the two package versions this project pins
(`mlx`, `mlx-lm`) parsed out of that run's `pip freeze`, which otherwise pins everything
else installed too. `evidence/apple/<file>.json` mirrors `research/raw/<file>`, and
`evidence/apple/head_skip_formal.json` mirrors `experiments/head_skip_formal/results.json`.

## Reproducing this directory

`tools/export_evidence.py` produces every file here from the raw runs. It only runs on a
machine that still has the private data:

```sh
python tools/export_evidence.py               # raw data under this repository's root
python tools/export_evidence.py /path/to/raw   # raw data under another root
python tools/export_evidence.py --check        # re-export and fail on any difference
```

A handful of measurements this project cites are not under this directory: they stay behind
their own explicit skip (in `tests/conftest.py` or the test itself) because their raw file is
not one of the three sources this export covers, or because the study they belong to lives in
a sealed database rather than a plain JSON file. Those checks run only on the machine that
measured them.

## Protocol

Each run's preregistration, method and full decision record — what PORT2, PERF1, NEXT1-C and
the others mean, and why a given cell is what it is — is the `research/LEDGER.md` entry named
in the code that cites it, in the separate research repository:
[IronMule-Research](https://github.com/Tobayko/IronMule-Research/blob/main/research/LEDGER.md).
