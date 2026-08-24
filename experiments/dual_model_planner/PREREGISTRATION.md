# Vorregistrierung — Zwei lokale Gemma-Modelle als evidenzgebundene Planner

**Studie:** `dual-model-evidence-planner-20260824-01`
**Zyklus:** `15`
**Status:** vor jedem weiteren Studienartefakt geschrieben; nach dem Hashen unveränderlich
**Claim:** `formal_claim=false`

## Zweck und enge Abgrenzung

Diese Studie vergleicht ausschließlich die beiden bereits vorhandenen lokalen
MLX-Modelle in einem einzigen, geschlossenen Planungsfall. Sie prüft, ob sie die
vorhandene Evidenz als strikt begrenzte Auswahlantwort wiedergeben können und
welches Modell unter den gemessenen Bedingungen weniger Ressourcen benötigt.

Die Studie misst keine allgemeine Modellqualität, keine allgemeine
Planungsfähigkeit, kein Lernen, kein Training und keine Gewichtsänderung. Das
Modell darf weder Code erzeugen noch ausführen, Dateien ändern, einen Kandidaten
starten oder eine Produktaktivierung auslösen. Auch ein Bestehen erlaubt nur die
Dokumentation dieses einen Falls, keine automatische Nutzung.

## Fest gebundene Modelle und Hardware

Beide Modelle werden ausschließlich nacheinander aus dem bestehenden lokalen
Snapshot-Resolver `tools/_bench.py:resolve_local_model_snapshot` geladen. Ein
fehlender Snapshot oder eine fehlende Abhängigkeit beendet die Studie ohne
Download oder Installation.

| Schlüssel | Modell | Revision |
|---|---|---|
| `1b` | `mlx-community/gemma-3-1b-it-4bit` | `2d44e83dc9e80843d22fb941d3d699a0b1351aa6` |
| `4b` | `mlx-community/gemma-3-4b-it-4bit` | `93724907d4ed1745d2fe50baadf3b0b01a65abf2` |

Zielgerät ist der netzbetriebene Apple M1 Max mit 32 GiB und MLX GPU-Gerät.
Gebunden sind die projektlokale `.venv`, MLX `0.32.0` und mlx-lm `0.31.3`.

## Bytegleicher Planungsfall

Beide Modelle erhalten bytegleich denselben folgenden Prompt. Die Eingabe wird
als fester String eingebettet, mit `temperature=0`, greedy Sampling und
höchstens `32` Ausgabetoken. Der erwartete Prompt-Fingerprint ist:

`c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b`

```text
You choose exactly one next Project Friday experiment.

Hardware: Apple M1 Max, 32 GB unified memory. Use only the evidence below.

Measured evidence:
- persistent_service_qualification: keeping Gemma 4B loaded reduced paired time to first output by 65.3032%; all greedy outputs matched exactly. Multi-turn and parallel-request qualification are still missing.
- batched_readback: isolated decode readback accounts for 12.98% per output token, but batching the checks can emit extra tokens and therefore needs a later correctness study.
- host_readback_upper_bound: 15.3% is only an upper bound, not a directly usable implementation.
- kv_cache_preallocation_ab: 4.4263% of decode time is correlated with reallocations, but the first step is confounded and the cache change still requires separate architecture permission.

Fixed selection policy:
1. Prefer the largest already confirmed end-to-end lever that also closes a required missing workload.
2. Do not choose a diagnostic upper bound.
3. Do not choose a permission-blocked cache change.
4. Choose exactly one ID from this list: persistent_service_qualification, batched_readback, host_readback_upper_bound, kv_cache_preallocation_ab.

Return only a JSON object with exactly one key named candidate_id and no prose, markdown, or explanation.
```

The only accepted complete answer is the exact JSON object
`{"candidate_id":"persistent_service_qualification"}`. No Markdown,
codeblock, preface, extra key, duplicate key, trailing text or automatic cleanup
is accepted. A contract error is recorded as a negative result; it does not
stop the remaining non-critical model executions.

## Fixed balanced execution schedule

There are exactly six paired observations. Every observation starts a fresh
Python process, loads exactly one model once, has no shared model/token/KV state,
and exits before the next process starts. The following schedule is immutable:

| Paar | Ausführungsreihenfolge |
|---:|---|
| 1 | `1b → 4b` |
| 2 | `1b → 4b` |
| 3 | `1b → 4b` |
| 4 | `4b → 1b` |
| 5 | `4b → 1b` |
| 6 | `4b → 1b` |

Thus each model runs exactly six times and the pair order is fixed before any
measurement. Failed model work is never repeated. Contract and priority errors
are non-critical and do not shorten the six-pair schedule. A timeout,
malformed worker event, memory limit, swap growth, wrong hardware, wrong
snapshot, unavailable required package, loss of mains power or budget violation
aborts safely at once; all completed raw events remain in the result.

## Pre-registered gates and limits

### Identity and determinism

For each model, all six completed events must carry the registered model ID and
revision, one load, a distinct fresh PID, the fixed raw-prompt hash, the
rendered chat-template prompt bytes and their SHA-256, and a valid MLX GPU
execution. The rendered prompt bytes and hash must be identical across all
twelve runs; the raw prompt hash alone is not sufficient. Token IDs, decoded
text and finish reason must be identical within each model across all six runs.
Different models are allowed to have different output token IDs, but not
different input prompt bytes. Missing or mismatched token, text, finish-reason,
or rendered-prompt identity is `correctness_failed`, terminal, and is never
replaced by a quality score.

Before the first worker, the parent resolves each registered snapshot and
hashes the exact execution manifest once: the resolved `config.json`,
`tokenizer_config.json`, any present tokenizer file, and every registered
weight file. Safe Hugging Face cache symlinks are resolved and must remain
inside the local repository. The absolute resolved snapshot path, revision,
content manifest SHA-256 and weight SHA-256 mapping are bound into each child
environment. The child checks that binding and records a resolved-file stat
manifest (device, inode, size and `mtime_ns`) immediately before and after
the single MLX load; a stat change or identity mismatch aborts the run. After
the twelve runs, the parent hashes the two execution manifests once again and
requires equality with the preflight hashes before any successful decision.
These content hashes are provenance checks outside the measured child process
wall interval; the per-run stat checks are recorded but are not charged as
GPU work.

For each complete pair, the UTF-8 decoded output bytes from 1B and 4B are also
compared as a concrete cross-model fact and reported as `exact_text_equal`.
The total is reported as `x/6`; every raw response variant and output hash is
retained. Cross-model text equality and token equality are descriptive only:
different models are not required to produce the same text or token sequence,
and these comparisons never weaken or replace the strict contract, priority or
within-model identity gates.

### Planner contract and priority

The structural parser records separately whether an answer is strict JSON with
exactly one `candidate_id` key and one of the four fixed IDs. The contract gate
is stricter: `contract_ok` is true only for the exact UTF-8 bytes
`{"candidate_id":"persistent_service_qualification"}`. Any whitespace,
Markdown, preface, trailing text, duplicate key or other candidate is a
contract failure; no whitespace is normalized or accepted as a repair. The
priority count still records the six semantic candidate values independently.

### Hardware and resources

- Apple M1 Max, `arm64`, `32 GiB`, MLX default GPU device.
- Mains power is required for the complete study.
- Process peak RSS and MLX peak memory are each at most `5 GiB` per run.
- Swap usage is readable and must not increase during any run or across the
  study. Unknown swap usage fails the resource gate.
- Required package versions are MLX `0.32.0` and mlx-lm `0.31.3`.

### Budget and pacing

One shared `BudgetGuard` covers all twelve serial model processes with these
fixed values:

| Limit | Wert |
|---|---:|
| GPU work total | `120 s` |
| zusammenhängende GPU-Arbeit | `6 s` |
| Pflichtpause | `4 s` |
| Duty-Faktor / Fenster | `0.15 / 60 s` |
| Gesamtzeit | `1200 s` |
| Kandidaten-Abkühlung | `60 s` |
| interne Zielauslastung für Pausen | `0.10` |

The parent checks mains power immediately before every worker, stops the
worker's measured model duration before calling `BudgetGuard.record_gpu()` or
taking a required pause, and never counts Guard rest as model time. A worker
has a fixed `90 s` timeout, a child watchdog that aborts generation after
`6 s` from generation start through final `mx.synchronize()`, and emits at
most `1,000,000` bytes of stdout. The parent enforces the stdout ceiling while
the child is running and kills the process group on overflow; only one strict
JSON event is accepted. The watchdog measures continuous generation work, not
model loading or the later Guard pauses.

The child environment removes `PYTHONHOME`, `PYTHONINSPECT`, `PYTHONPATH` and
`PYTHONSTARTUP`, and sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1` and `PYTHONNOUSERSITE=1`.

## Recorded evidence

The raw result stores, for the study and every execution: model ID and exact
revision; resolver identity; the preflight and post-study snapshot execution
manifest and weight SHA-256 hashes; the bound absolute snapshot path and
before/after stat manifests; rendered
chat-template prompt bytes and SHA-256; Git
revision and dirty state; code, specification, prompt and environment
fingerprints; target hardware, software versions and power source; pair and
schedule position; PID and load count; prompt and output token IDs; token hash;
decoded response; finish reason; strict parser result and candidate ID; model
load time, TTFT, model-generation time, process wall time and token rate; peak
RSS and MLX memory; swap before/after/delta; BudgetGuard summary; and any
terminal error or partial-run information. Raw text and token IDs remain in the
evidence file; the read-only UI never inserts them into HTML.

## Fixed statistics

For each model, contract and expected-candidate counts are reported as `x/6`.
For every execution, `model_load_ns` is the monotonic interval from just before
the single MLX load to its return. `model_work_ns` is the monotonic interval
from generation start through the final `mx.synchronize()`. `ttft_ns` is from
generation start to the first yielded token, `process_wall_ns` is the parent
monotonic interval from process start to process exit, and `token_rate` is
`output_tokens / (model_work_ns / 1e9)`. The parent content-manifest hashing is
performed once before the schedule and once after it, outside those
per-process timing intervals; it is never included in `process_wall_ns` or
`model_work_ns`.
Store the integer nanosecond values and derive seconds to at least nine decimal
places. For TTFT, model-generation time and process wall time, report seconds
as the median and median absolute deviation (MAD), with no outlier removal.
Peak RSS, MLX peak and swap deltas are reported per model and as maxima.

For each metric, pair the six 1B observations with the six 4B observations by
pair ID and calculate the six ratios `1B / 4B`. The reported paired ratio is the
median of those six ratios. Its deterministic paired bootstrap 95% interval is
pre-registered as follows: pair IDs are explicitly stored in ascending order;
use Python `random.Random` seed `20260824`, draw `10,000` samples of six pair
IDs with replacement, calculate the median ratio for each sample, and use the
2.5th and 97.5th percentile with linear interpolation. Store the seed, resample
count, statistic, percentile method, pair IDs, raw six ratios, median and
interval. No observation is discarded or winsorized. If fewer than six complete
pairs exist, the ratio and interval are `null` and the resource or hardware
failure is reported instead.

## Immutable decision table

Functional gates are evaluated per model after the full schedule. The decision
precedence is fixed: (1) any resource, hardware, power or BudgetGuard abort is
`resource_or_budget_failed`; (2) any within-model token, text, finish-reason or
rendered-prompt mismatch is `correctness_failed`; (3) the remaining table is
applied to contract and priority gates. A structural or contract failure is
non-critical and the remaining scheduled pairs still run.

| 1B gate | 4B gate | Fixed decision |
|---|---|---|
| pass | fail | `planner_1b_qualified_exact_case` |
| fail | pass | `planner_4b_qualified_exact_case` |
| pass | pass, and 1B paired runtime median is at most 5% slower **and** 1B peak memory is at least 25% lower | `both_qualified_1b_preferred` |
| pass | pass, but the two preference thresholds are not both satisfied | `both_qualified_no_automatic_preference` |
| fail | fail | `no_planner_qualified` |
| any | any, with resource/hardware/BudgetGuard abort | `resource_or_budget_failed` |
| any | any, with a within-model token/text/finish/prompt identity mismatch and no resource abort | `correctness_failed` |

Here “1B at most 5% slower” means `median(1B/4B) <= 1.05` for process wall
time. “Mindestens 25% geringer” means `1 - max_peak_rss_1B /
max_peak_rss_4B >= 0.25`; both thresholds are required. The decision never
executes a candidate or activates a product and always remains
`formal_claim=false`.

## One-shot protection and reproducibility

Before the first worker, the harness creates an exclusive private start marker
under `.friday-data/dual-model-planner/` with directory mode `0700` and file
mode `0600`. An existing marker or result file blocks execution. The result is
written atomically and is immutable evidence. The exact study/run IDs, schedule,
prompt, parser, revisions, thresholds and table are frozen by this file's
SHA-256 before any hardware execution.
