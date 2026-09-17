# PORT2 run 9, first attempt: ran without the PORT2 fixes

This run was submitted with an empty patch. The submit script built the patch from the
working tree's `git diff`, and by then the work was committed, so the guest checked out
the pre-PORT2 commit and applied nothing. `clone` exits 128 because `git apply` refuses an
empty file, and every stage that needs the fixes fails a second later: `families.py`
imports `_new_cache`, which does not exist at that commit, and `float16` is not in
`COMPUTE_DTYPES`, so the float16 quality arm exits 1.

Three stages do not depend on the fixes and are kept, because they answer real questions:

* `gemma4-unified-load.json` — `ValueError: Model type gemma4_unified not supported`, which
  is why the Gemma 4 12B checkpoints cannot run on MLX at all.
* `gemma4-26b-load.json` — Gemma 4 26B-A4B loads and decodes correctly at 14.20 GB
  resident, peak 14.30 GB. The one-card ceiling is higher than Mistral 3's 13.26 GB showed.
* `quality-gemma4-e2b-{bf16,float32}.json` — usable as data, but **not** as a quality gate:
  the bfloat16 reference itself has a perplexity of 22 212 on WikiText-2, so the ratio is
  between two numbers that mean nothing. Run 9b screens E2B against a stock decode to find
  out whether the model or the harness is responsible.

Kept rather than deleted so the mistake and what survived it stay attached to each other.
The successful re-run is `port2-run9b-*`.
