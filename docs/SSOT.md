# Evidence corpus

The local evidence SSOT is `.friday-data/ssot.sqlite3`. It contains original source
bytes, canonical records and every source occurrence. It is the shared entry point
for historical data and analysis; it does not qualify execution paths or supersede
the original experiment contracts.

Public product claims belong in [README.md](../README.md), experimental conclusions
in [research/LEDGER.md](../research/LEDGER.md), validity limits in [LIMITS.md](LIMITS.md)
and open work in the [backlogs](PROJECT_FRIDAY_BACKLOG.md). Agent behavior belongs in
[AGENTS.md](../AGENTS.md). Keep those responsibilities separate from the data store.

## Build, inspect and verify

```sh
python tools/ssot.py build
python tools/ssot.py status
python tools/ssot.py verify --check-sources
python tools/friday.py status --ssot --plain
python tools/friday.py status --ssot --json
```

The existing terminal status and its JSON output use the same corpus snapshot. There
is no additional web server. The status reports coverage and recent canonical records;
it does not label an unverified or unavailable corpus as healthy.

A build scans local evidence and reachable Git history, writes a private temporary
SQLite file, checks its source inventory and integrity, then atomically replaces the
old corpus. An unreadable source, unsupported database, changing input, active SQLite
journal or missing historical archive member prevents publication. Sources are never
modified or checkpointed by the builder. Stop the owning writer before rebuilding a
database with an active journal; do not discard its journal to make a build pass.

`verify` independently decompresses and hashes every stored source object, checks
payload identities and record coverage, and validates SQLite references. With
`--check-sources`, it also compares the current filesystem, archive manifest and Git
history against the captured inventory. A changed source means that a new snapshot is
needed; the previous snapshot still records the original bytes.

A verified earlier corpus can supply already compressed objects during a rebuild:

```sh
python tools/ssot.py --database .local/rebuilt-corpus.sqlite3 build \
  --reuse .friday-data/ssot.sqlite3
python tools/ssot.py --database .local/rebuilt-corpus.sqlite3 verify --check-sources
```

Reused objects are decompressed and hash-checked before copying. This avoids compressing
large trace buffers again without trusting a damaged previous index. Inspect and
verify a candidate before replacing the default corpus.

## Coverage and original bytes

The source inventory covers:

- Local `.friday-data/` artifacts, including historical archive specifications,
  partial results, binary tensors and GPU trace buffers. Models, installed tools,
  caches, transient locks and the derived corpus itself are excluded.
- Structured data in `research/`, `experiments/` and `profiles/`, plus
  `docs/EXPERIMENT_MATRIX.json`.
- Reachable historical Git data blobs in those data paths, including versions whose
  files have since been changed or deleted.

The content-addressed historical archive's manifest is reconciled entry by entry,
including its duplicate references. Partial files remain partial artifacts when they
cannot be parsed; their bytes are retained without inventing successful records.
Filesystem aliases inside the evidence tree retain their link target and share one
stored object. Aliases outside the allowed roots are rejected.

Original bytes are stored once per SHA-256 in bounded, compressed chunks. They remain
recoverable even after the original file is unavailable:

```sh
python tools/ssot.py export-source <source-sha256> <new-output-file>
```

Export refuses an existing destination and removes its own incomplete output on
failure. A corpus snapshot is an archive of data, not a backup of the entire repository
or an automatic restorer of filesystem ownership, permissions and directory layout.
Preserve the original sealed studies and Git history.

## Canonical records and source occurrences

| Relation | Responsibility |
| --- | --- |
| `source_objects`, `source_chunks` | One lossless representation of each original file content |
| `sources` | Every original path or Git blob, its content hash, format and alias metadata |
| `payloads` | One canonical JSON payload per content hash |
| `canonical_records` / `runs` | Canonical records; `runs` is the query view |
| `record_occurrences` / `origins` | Every location at which a canonical record appeared, including provenance and attribution |
| `metrics` / `measurement` | Numeric projections of canonical records, counted once per record |

A hash-verified journal event has one canonical identity across its database and
exports. A journal export's envelope remains in the source object; it is not indexed
again as a measurement containing all the same events. An unchanged Git blob is an
additional origin of the corresponding current source, not another experiment.

Other identified records bind their original identifier together with payload,
provenance, time and measurement context. Equal values alone do not prove equivalence:
independent repetitions and records without adequate identity remain distinct. Their
payload bytes are still shared. Do not collapse those occurrences to improve counts
or statistical confidence.

The existing historical archive manifest also establishes artifact copies. An archived
file and its identical current counterpart share record identity when their original
filename and full content hash match that manifest. Within-file repetitions retain
their separate positions; unrelated equal-valued reports are not merged.

```sh
python tools/ssot.py query "SELECT run_id, study, native_id, status FROM runs LIMIT 20"
python tools/ssot.py query "SELECT * FROM origins WHERE run_id = 1"
python tools/ssot.py metrics tokens_per_second
```

A record may be a journal event, report, configuration or historical document. The
word `runs` is retained for query compatibility; counting its rows is not counting
independent experimental trials. Check source kind, identity, study protocol, units
and validity before comparing metrics or applying a regression threshold.

## Projections and attribution

Large numeric arrays are summarized in the metric projection, and per-record metric
limits are reported through `metrics_truncated`. This does not discard the underlying
source bytes or complete canonical payload. `payload_json` in the convenience view
omits large payloads; `payloads` and `source_content` retain access to the complete data.

Some historical files intentionally contain non-finite headroom values or numerical
test vectors. Their original values remain unchanged and their payload format is
explicitly `legacy_nonfinite_json`. Finite metric projections remain available;
non-finite values are not turned into zero, null or invented performance evidence.

Attribution keeps its evidence label. Explicit payload fields and provider-naming
commit trailers are evidence; commit scope, file history or nearby timestamps are
weaker inferences. A commit without a tool marker remains `unattributed`; absence of
a marker is not evidence that a particular provider authored it. Multiple marked tools
remain explicit. These rules describe the derived data, not the contribution policy
in [CONTRIBUTING.md](../CONTRIBUTING.md).

Historical records, negative results and frozen verdicts are not rewritten. An
archived or superseded result is preserved as history, not promoted into a current
performance claim. Schema changes affect this regenerable corpus, not the migration
contracts or bytes of the source evidence databases.

## Test coverage

[tests/engine/test_ssot.py](../tests/engine/test_ssot.py) exercises lossless recovery,
source mutation, canonical identity, historical Git versions, archive copies, legacy
numeric encodings and failure recovery. It runs in the existing engine CI suite.
Real local history is additionally checked with `verify --check-sources` and
[tests/test_sealed_evidence.py](../tests/test_sealed_evidence.py).

## Privacy

The corpus contains private raw material and remains local under the existing ignore
policy. Do not commit the SQLite file, trace buffers, prompts or unredacted source
exports. Public reports contain deliberately reviewed aggregates and provenance;
[research/LEDGER.md](../research/LEDGER.md) owns the corresponding scientific claims.
