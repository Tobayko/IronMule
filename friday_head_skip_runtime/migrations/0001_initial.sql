BEGIN IMMEDIATE;

PRAGMA application_id = 0x48535231;
PRAGMA user_version = 1;

CREATE TABLE metadata (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    runtime_id TEXT NOT NULL CHECK(runtime_id = 'head-skip-runtime-20260824-01'),
    migration_sha256 TEXT NOT NULL
        CHECK(length(migration_sha256) = 64
              AND migration_sha256 NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE records (
    record_id TEXT PRIMARY KEY
        CHECK(length(record_id) = 64 AND record_id NOT GLOB '*[^0-9a-f]*'),
    previous_record_id TEXT
        CHECK(previous_record_id IS NULL OR
              (length(previous_record_id) = 64
               AND previous_record_id NOT GLOB '*[^0-9a-f]*')),
    runtime_id TEXT NOT NULL CHECK(runtime_id = 'head-skip-runtime-20260824-01'),
    entity_key TEXT NOT NULL UNIQUE CHECK(length(entity_key) BETWEEN 1 AND 160),
    record_kind TEXT NOT NULL CHECK(record_kind IN (
        'policy_overhead', 'runtime_validation_attempt',
        'runtime_validation', 'runtime_failure'
    )),
    status TEXT NOT NULL CHECK(length(status) BETWEEN 1 AND 96),
    created_at_unix_ns INTEGER NOT NULL
        CHECK(typeof(created_at_unix_ns) = 'integer'
              AND created_at_unix_ns BETWEEN 0 AND 9223372036854775807),
    report_json TEXT NOT NULL,
    report_sha256 TEXT NOT NULL
        CHECK(length(report_sha256) = 64
              AND report_sha256 NOT GLOB '*[^0-9a-f]*'),
    provenance_json TEXT NOT NULL,
    provenance_sha256 TEXT NOT NULL
        CHECK(length(provenance_sha256) = 64
              AND provenance_sha256 NOT GLOB '*[^0-9a-f]*')
);

CREATE INDEX idx_runtime_records_kind_created
    ON records(record_kind, created_at_unix_ns, record_id);

CREATE INDEX idx_runtime_records_status_created
    ON records(status, created_at_unix_ns, record_id);

CREATE TRIGGER records_no_update
BEFORE UPDATE ON records
BEGIN
    SELECT RAISE(ABORT, 'runtime history is append-only');
END;

CREATE TRIGGER records_no_delete
BEFORE DELETE ON records
BEGIN
    SELECT RAISE(ABORT, 'runtime history is append-only');
END;

CREATE TRIGGER metadata_no_update
BEFORE UPDATE ON metadata
BEGIN
    SELECT RAISE(ABORT, 'runtime metadata is immutable');
END;

CREATE TRIGGER metadata_no_delete
BEFORE DELETE ON metadata
BEGIN
    SELECT RAISE(ABORT, 'runtime metadata is immutable');
END;

COMMIT;
