CREATE TABLE evidence_metadata (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    migration_sha256 TEXT NOT NULL
        CHECK(length(migration_sha256) = 64 AND migration_sha256 NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE evidence_records (
    record_id TEXT PRIMARY KEY
        CHECK(length(record_id) = 64 AND record_id NOT GLOB '*[^0-9a-f]*'),
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('native', 'legacy_summary')),
    source_key TEXT NOT NULL CHECK(length(source_key) BETWEEN 1 AND 200),
    tool TEXT NOT NULL CHECK(tool IN (
        'dispatch', 'cooldown', 'loop', 'model-loop', 'codegen', 'roofline', 'fusion'
    )),
    workload_key TEXT NOT NULL CHECK(length(workload_key) BETWEEN 1 AND 200),
    result_status TEXT NOT NULL CHECK(length(result_status) BETWEEN 1 AND 128),
    raw_measurements_available INTEGER NOT NULL
        CHECK(raw_measurements_available IN (0, 1)),
    observed_at_unix_ns INTEGER
        CHECK(observed_at_unix_ns IS NULL OR observed_at_unix_ns >= 0),
    recorded_at_unix_ns INTEGER NOT NULL CHECK(recorded_at_unix_ns >= 0),
    report_json TEXT NOT NULL,
    report_sha256 TEXT NOT NULL
        CHECK(length(report_sha256) = 64 AND report_sha256 NOT GLOB '*[^0-9a-f]*'),
    provenance_json TEXT NOT NULL,
    provenance_sha256 TEXT NOT NULL
        CHECK(length(provenance_sha256) = 64 AND provenance_sha256 NOT GLOB '*[^0-9a-f]*'),
    git_revision TEXT NOT NULL CHECK(length(git_revision) BETWEEN 40 AND 64),
    git_dirty INTEGER NOT NULL CHECK(git_dirty IN (0, 1)),
    code_sha256 TEXT NOT NULL
        CHECK(length(code_sha256) = 64 AND code_sha256 NOT GLOB '*[^0-9a-f]*'),
    spec_sha256 TEXT NOT NULL
        CHECK(length(spec_sha256) = 64 AND spec_sha256 NOT GLOB '*[^0-9a-f]*'),
    environment_sha256 TEXT NOT NULL
        CHECK(length(environment_sha256) = 64 AND environment_sha256 NOT GLOB '*[^0-9a-f]*'),
    hardware_key TEXT NOT NULL
        CHECK(length(hardware_key) = 64 AND hardware_key NOT GLOB '*[^0-9a-f]*'),
    UNIQUE(evidence_kind, tool, source_key)
);

CREATE INDEX idx_evidence_recorded
    ON evidence_records(recorded_at_unix_ns DESC, record_id);
CREATE INDEX idx_evidence_tool_recorded
    ON evidence_records(tool, recorded_at_unix_ns DESC, record_id);
CREATE INDEX idx_evidence_status_recorded
    ON evidence_records(result_status, recorded_at_unix_ns DESC, record_id);

CREATE TRIGGER evidence_records_no_update
BEFORE UPDATE ON evidence_records
BEGIN
    SELECT RAISE(ABORT, 'evidence records are append-only');
END;

CREATE TRIGGER evidence_records_no_delete
BEFORE DELETE ON evidence_records
BEGIN
    SELECT RAISE(ABORT, 'evidence records are append-only');
END;

CREATE TRIGGER evidence_metadata_no_update
BEFORE UPDATE ON evidence_metadata
BEGIN
    SELECT RAISE(ABORT, 'evidence metadata is immutable');
END;

CREATE TRIGGER evidence_metadata_no_delete
BEFORE DELETE ON evidence_metadata
BEGIN
    SELECT RAISE(ABORT, 'evidence metadata is immutable');
END;
