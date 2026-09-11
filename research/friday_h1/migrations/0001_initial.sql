BEGIN IMMEDIATE;

PRAGMA application_id = 0x48315632;

CREATE TABLE metadata (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    schema_version INTEGER NOT NULL CHECK(schema_version = 2),
    study_id TEXT NOT NULL CHECK(study_id = 'h1v2-dispatch-n8-20260821-01'),
    migration_sha256 TEXT NOT NULL
        CHECK(length(migration_sha256) = 64
              AND migration_sha256 NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE records (
    record_id TEXT PRIMARY KEY
        CHECK(length(record_id) = 64 AND record_id NOT GLOB '*[^0-9a-f]*'),
    study_id TEXT NOT NULL CHECK(study_id = 'h1v2-dispatch-n8-20260821-01'),
    entity_key TEXT NOT NULL UNIQUE CHECK(length(entity_key) BETWEEN 1 AND 96),
    record_kind TEXT NOT NULL CHECK(record_kind IN (
        'preregistration',
        'calibration_session',
        'calibration_summary',
        'confirmation_seal',
        'confirmation_session',
        'study_decision',
        'session_failure'
    )),
    stage TEXT NOT NULL CHECK(stage IN ('preregistration', 'calibration', 'confirmation')),
    session_id TEXT NOT NULL CHECK(length(session_id) <= 8),
    status TEXT NOT NULL CHECK(length(status) BETWEEN 1 AND 96),
    formal_claim INTEGER NOT NULL CHECK(formal_claim IN (0, 1)),
    created_at_unix_ns INTEGER NOT NULL
        CHECK(typeof(created_at_unix_ns) = 'integer'
              AND created_at_unix_ns BETWEEN 0 AND 9223372036854775807),
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL
        CHECK(length(payload_sha256) = 64
              AND payload_sha256 NOT GLOB '*[^0-9a-f]*'),
    provenance_json TEXT NOT NULL,
    provenance_sha256 TEXT NOT NULL
        CHECK(length(provenance_sha256) = 64
              AND provenance_sha256 NOT GLOB '*[^0-9a-f]*')
);

CREATE INDEX idx_records_kind_created
    ON records(record_kind, created_at_unix_ns, record_id);

CREATE INDEX idx_records_status_created
    ON records(status, created_at_unix_ns, record_id);

CREATE TRIGGER records_no_reinsert
BEFORE INSERT ON records
WHEN EXISTS (
    SELECT 1 FROM records
    WHERE entity_key = NEW.entity_key OR record_id = NEW.record_id
)
BEGIN
    SELECT RAISE(ABORT, 'formal evidence cannot be replaced');
END;

CREATE TRIGGER records_no_update
BEFORE UPDATE ON records
BEGIN
    SELECT RAISE(ABORT, 'formal evidence is append-only');
END;

CREATE TRIGGER records_no_delete
BEFORE DELETE ON records
BEGIN
    SELECT RAISE(ABORT, 'formal evidence is append-only');
END;

CREATE TRIGGER metadata_no_update
BEFORE UPDATE ON metadata
BEGIN
    SELECT RAISE(ABORT, 'formal metadata is immutable');
END;

CREATE TRIGGER metadata_no_delete
BEFORE DELETE ON metadata
BEGIN
    SELECT RAISE(ABORT, 'formal metadata is immutable');
END;

PRAGMA user_version = 2;

COMMIT;
