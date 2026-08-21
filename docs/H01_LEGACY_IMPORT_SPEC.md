# H0 → H0.1 Generation Inventory and Versioned Adapter Contract

Status: **inventory and generation adapters frozen; explicit batch-execute
contract registered before its implementation or first execution**. This
revision supersedes the attempted generic Legacy-Warmup extraction contract.
The prior production dry-run was a valid NO-GO: one fully stored historical
completed result could not be replayed through the current H0 projection
normalizer because its closed warmup shape belongs to another code generation.
That failure is evidence of a version boundary, not permission to loosen the
current H0 contract or special-case one run.

## Adapter and materialization scope

This phase may construct detached, canonical H0.1 legacy-observation bundles
from exact archived test fixtures or an explicitly authorized read-only dry-run.
After a successful full-source replay it may also persist all importable bundles
to one explicitly supplied H0.1 SQLite-v1 target. It never classifies
stationarity, promotes a candidate, changes H0, or reinterprets H0. The
historical action remains `no_h0_conclusion`.

The public legacy compatibility API has two exact modes. `execute=False`
requires `target=None` and has no target side effect. `execute=True` requires an
explicit filesystem target and never uses a default. Exact non-boolean execute
values, a dry-run target, a missing execute target, source/target aliasing,
symlinks, or URI-like paths fail closed. Unsupported provenance is reported as
excluded; provenance claimed by a descriptor with any shape/tag mismatch is a
hard `claimed_known_malformed` failure before parsing or target open.

All source candidates are verified, matched, parsed and converted to canonical
bundles before an execute opens the target. Source file identity and SHA-256 are
rechecked immediately before target open and again after persistence. These
checks detect bounded pre/post changes but are not claimed as a general
filesystem TOCTOU proof.

Execute passes the complete ordered importable set to H0.1 storage as one batch.
Storage rebuilds every bundle before `BEGIN IMMEDIATE`, then verifies exact
SQLite-v1 schema, connection/file binding and every existing row inside the
transaction. Every requested entity is either byte-identical and `idempotent`,
or newly inserted and replayed. Any conflicting entity, schema/file drift,
trigger rejection or replay failure rolls back all insertions from that call.
After commit the importer reopens the target read-only and requires each entity
to replay to the exact planned canonical bundle. Persistence outcomes preserve
source order and are returned separately from the canonical compatibility
report.

## Source trust and deterministic candidate set

The source H0 database is immutable. It is accepted only as a regular,
non-symlink filesystem path and is hashed with file-descriptor identity checks.
The public H0 storage layer opens it using SQLite URI `mode=ro`, requires
`query_only=1`, exact H0 SQLite-v1 identity/schema, and `integrity_check = ok`
inside one read transaction. SQLite's resolved main path and file identity must
match before and after `BEGIN`; parent/file identity and the complete SHA-256
must remain unchanged after close. These are tested invariants for the named
file, not a general filesystem TOCTOU claim.

The candidate set is every stored H0 `common_result` whose run mirror has
`mode = eager_baseline`, ordered exactly by:

1. `runs.created_at_unix_ns` ascending;
2. `runs.run_id` ascending.

There is no value, outcome, stability, or usefulness filter. Before deriving
the first structural fingerprint, the inventory reconstructs every selected
closed manifest and complete stored evidence bundle and calls H0
`Storage.verify_common_result_bundle`. Therefore a schema drift, wrapper/hash
tamper, child-row mismatch, duplicate/missing Common Result, or malformed
candidate aborts the entire inventory before any generation selector is made.

## Value-independent structural fingerprint v1

The algorithm identifier is `sha256_recursive_json_structure_v1`. It hashes a
closed token stream containing only:

- exact recursively sorted object keys and object cardinalities;
- array kind, length, item order, and each item's recursive structure;
- scalar type classes `null`, `bool`, `int`, `float`, and `string`;
- explicit container terminators and fixed-width length prefixes.

Boolean is distinct from integer. Key, scalar-type, container-kind, item-order,
or list-length changes alter the digest. Scalar values—including all raw warmup
durations—never enter this fingerprint. Exact signed-int64, finite-float,
string, depth, node, and container bounds still apply before a digest is
accepted.

Declared keys named exactly `schema_version` are inventoried separately with
their escaped JSON-pointer paths, exact type classes, and scalar values (or a
structural hash for a container-valued malformed declaration). Their canonical
list hash is part of the selector. Thus ordinary numeric values remain excluded
while declared schema generations remain distinguishable.

For each candidate the inventory outputs the stored code, spec, environment,
manifest, result, complete evidence, evidence-bundle, and Common-Result wrapper
SHA-256 digests; result and optional failure-diagnostic structural fingerprints;
declared schema tags; source status/classification/action/error code; registry
match; selection index and creation time. It never outputs a raw warmup sequence.

## Static versioned adapter registry

Registry schema and descriptor schema are both version 1. A descriptor is an
immutable tuple of:

- bounded `adapter_id`, exact outcome, and `parser_id`;
- exact parent code/spec/environment SHA-256 provenance;
- exact source status;
- result structure and declared-schema-tag hashes;
- an exact optional diagnostic structure/tag-hash pair.

The descriptor also binds registry version 1, exact outcome and parser ID. The
descriptor SHA-256 covers the complete canonical descriptor. The registry
SHA-256 covers its ordered complete descriptor set. Duplicate adapter IDs or
duplicate selectors are rejected.

Matching is provenance-first and has exactly three states:

- `matched`: one descriptor matches the complete selector;
- `unsupported_generation`: no descriptor claims the three parent provenance
  hashes; inventory records an explicit exclusion;
- `claimed_known_malformed`: provenance is registered but status, structural
  fingerprint, schema tags, or diagnostic contract differs. A materializing
  importer must abort, never downgrade this to unsupported.

The static registry contains exactly the four frozen inventory generations:

- runtime-unavailable provenance `246eb77f…` is the recognized exclusion
  `no_warmup_runtime_unavailable_v1` and never produces a bundle;
- completed provenance `5f62c419…`, result structure `39782447…`, is parsed by
  `completed_eager_warmup_v1` (exact 11-value historical warmup shape);
- warmup-unstable provenance `aae3245e…`, result structure `0747fd2d…` and
  diagnostic structure `6d76f955…`, is parsed by
  `warmup_unstable_diagnostic_v1` (exact schema-1 16-value diagnostic);
- completed W1v3 provenance `101cdadf…`, result structure `ef39c352…`, is parsed
  by `completed_eager_warmup_w1v3` (exact eight-block/eight-value shape).

The source contains the complete unabridged hashes; ellipses above are prose
only. Matching always precedes value access. The current H0 projection
normalizer is neither imported nor called. No run ID, timestamp, apparent
stability, warmup value, or measurement statistic is a registry selector.

## Entity binding

The entity binding accepts only hashes—not raw warmup values—and binds each
detached entity ID to:

- adapter ID, complete adapter-descriptor SHA-256, and selector SHA-256;
- source run ID and creation time;
- parent manifest/result/evidence/bundle SHA-256 values;
- the canonical raw-warmup-sequence SHA-256.

Changing registry, descriptor, parser, selector, source lineage, or raw sequence
hash changes the entity ID deterministically. Manifest, observation and lineage
repeat the registry/descriptor/selector/raw hashes; H0.1 storage replays each
binding and the exact adapter-specific warmup length before accepting bytes.

## Archived verified fixtures

Four canonical source-bundle fixtures are retained under
`tests/fixtures/h01/`. Each was exported only after the frozen inventory from a
read-only, `query_only=1`, integrity-checked H0 handle after public
`verify_common_result_bundle` replay. They contain the manifest, result,
children and bundle hashes needed for the same verifier, but no source path or
secret. File SHA-256 values are pinned in tests: A `7f273423…`, B `22876ade…`,
C `f5058482…`, D `2f60fb80…`. A changed fixture byte, wrapper, or child must fail
before adapter parsing.

## Canonical inventory and compatibility report

The inventory schema is `friday_h01.h0_generation_inventory.v1`. It contains
the exact source path/identity/hash, selection rule, frozen registry identity,
closed counts, ordered candidate records, and `inventory_sha256` over the body.
Counts distinguish `matched`, `unsupported_generation`, and
`claimed_known_malformed`.

The compatibility report is `friday_h01.legacy_h0_import_report.v2` with exact
mode `adapter_dry_run` or `adapter_execute`. It records every selected candidate in source order,
all parent provenance/bundle hashes, adapter outcome/parser/registry/descriptor/
selector identities, raw-warmup hash, count, exact descriptive statistics,
entity ID or explicit exclusion, and report hash. The report describes the
verified plan; transaction outcomes are a separate returned tuple so an
idempotent replay does not change the report. Archived fixtures remain test
evidence only. Production execution additionally requires the explicit project
authorization and immutable-source pre/postflight recorded in the work journal.
