BEGIN IMMEDIATE;
PRAGMA foreign_keys = ON;
PRAGMA application_id = 0x46524830;
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS db_identity (
    key TEXT PRIMARY KEY CHECK (key = 'identity'),
    value TEXT NOT NULL CHECK (length(value) BETWEEN 1 AND 128)
);

INSERT OR IGNORE INTO db_identity(key, value)
VALUES ('identity', 'friday_h0.sqlite.v1');

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY CHECK (length(run_id) BETWEEN 1 AND 128),
    phase TEXT NOT NULL CHECK (phase = 'H0'),
    mode TEXT NOT NULL CHECK (mode IN (
        'eager_baseline', 'compile_comparison', 'aa_gpu',
        'analysis_slow', 'analysis_known_win', 'analysis_wrong_fixture',
        'analysis_missing_data', 'control_timeout', 'control_exit_70'
    )),
    manifest_json TEXT NOT NULL,
    manifest_hash TEXT NOT NULL CHECK (length(manifest_hash) = 64),
    code_sha256 TEXT NOT NULL CHECK (length(code_sha256) = 64),
    spec_sha256 TEXT NOT NULL CHECK (length(spec_sha256) = 64),
    environment_sha256 TEXT NOT NULL CHECK (length(environment_sha256) = 64),
    revision TEXT,
    revision_missing_reason TEXT,
    created_at_unix_ns INTEGER NOT NULL CHECK (created_at_unix_ns >= 0),
    CHECK ((revision IS NULL) <> (revision_missing_reason IS NULL))
);

CREATE TABLE IF NOT EXISTS status_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    event_kind TEXT NOT NULL CHECK (length(event_kind) BETWEEN 1 AND 128),
    status TEXT NOT NULL CHECK (length(status) BETWEEN 1 AND 128),
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL CHECK (length(payload_hash) = 64),
    recorded_at_ns INTEGER CHECK (recorded_at_ns IS NULL OR recorded_at_ns >= 0)
);

CREATE TABLE IF NOT EXISTS raw_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    session_id TEXT NOT NULL CHECK (length(session_id) BETWEEN 1 AND 128),
    sample_kind TEXT NOT NULL CHECK (length(sample_kind) BETWEEN 1 AND 128),
    sample_index INTEGER NOT NULL CHECK (sample_index >= 0),
    block_index INTEGER NOT NULL CHECK (block_index >= 0),
    arm TEXT NOT NULL CHECK (length(arm) BETWEEN 1 AND 128),
    value REAL NOT NULL CHECK (value >= 0),
    unit TEXT NOT NULL CHECK (length(unit) BETWEEN 1 AND 128),
    observed_at_ns INTEGER CHECK (observed_at_ns IS NULL OR observed_at_ns >= 0),
    UNIQUE (run_id, session_id, sample_kind, block_index, arm, sample_index)
);

CREATE TABLE IF NOT EXISTS scalar_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    metric_name TEXT NOT NULL CHECK (length(metric_name) BETWEEN 1 AND 128),
    scope TEXT NOT NULL CHECK (length(scope) BETWEEN 1 AND 128),
    value REAL,
    missing_reason TEXT CHECK (missing_reason IS NULL OR length(missing_reason) BETWEEN 1 AND 256),
    unit TEXT NOT NULL CHECK (length(unit) BETWEEN 1 AND 128),
    recorded_at_ns INTEGER CHECK (recorded_at_ns IS NULL OR recorded_at_ns >= 0),
    CHECK ((value IS NOT NULL) <> (missing_reason IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS correctness_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    case_name TEXT NOT NULL CHECK (length(case_name) BETWEEN 1 AND 128),
    metric_name TEXT NOT NULL CHECK (length(metric_name) BETWEEN 1 AND 128),
    value REAL NOT NULL,
    unit TEXT NOT NULL CHECK (length(unit) BETWEEN 1 AND 128),
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    detail_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    artifact_name TEXT NOT NULL CHECK (length(artifact_name) BETWEEN 1 AND 128),
    artifact_kind TEXT NOT NULL CHECK (length(artifact_kind) BETWEEN 1 AND 128),
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    metadata_json TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_migrations(version, name) VALUES (1, '0001_initial');

CREATE TRIGGER IF NOT EXISTS schema_migrations_append_only_update
BEFORE UPDATE ON schema_migrations BEGIN SELECT RAISE(ABORT, 'schema_migrations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS schema_migrations_append_only_delete
BEFORE DELETE ON schema_migrations BEGIN SELECT RAISE(ABORT, 'schema_migrations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS db_identity_append_only_update
BEFORE UPDATE ON db_identity BEGIN SELECT RAISE(ABORT, 'db_identity is append-only'); END;
CREATE TRIGGER IF NOT EXISTS db_identity_append_only_delete
BEFORE DELETE ON db_identity BEGIN SELECT RAISE(ABORT, 'db_identity is append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_append_only_update
BEFORE UPDATE ON runs BEGIN SELECT RAISE(ABORT, 'runs are append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_append_only_delete
BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT, 'runs are append-only'); END;
CREATE TRIGGER IF NOT EXISTS status_events_append_only_update
BEFORE UPDATE ON status_events BEGIN SELECT RAISE(ABORT, 'status_events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS status_events_append_only_delete
BEFORE DELETE ON status_events BEGIN SELECT RAISE(ABORT, 'status_events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS raw_samples_append_only_update
BEFORE UPDATE ON raw_samples BEGIN SELECT RAISE(ABORT, 'raw_samples are append-only'); END;
CREATE TRIGGER IF NOT EXISTS raw_samples_append_only_delete
BEFORE DELETE ON raw_samples BEGIN SELECT RAISE(ABORT, 'raw_samples are append-only'); END;
CREATE TRIGGER IF NOT EXISTS scalar_metrics_append_only_update
BEFORE UPDATE ON scalar_metrics BEGIN SELECT RAISE(ABORT, 'scalar_metrics are append-only'); END;
CREATE TRIGGER IF NOT EXISTS scalar_metrics_append_only_delete
BEFORE DELETE ON scalar_metrics BEGIN SELECT RAISE(ABORT, 'scalar_metrics are append-only'); END;
CREATE TRIGGER IF NOT EXISTS correctness_metrics_append_only_update
BEFORE UPDATE ON correctness_metrics BEGIN SELECT RAISE(ABORT, 'correctness_metrics are append-only'); END;
CREATE TRIGGER IF NOT EXISTS correctness_metrics_append_only_delete
BEFORE DELETE ON correctness_metrics BEGIN SELECT RAISE(ABORT, 'correctness_metrics are append-only'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_append_only_update
BEFORE UPDATE ON artifacts BEGIN SELECT RAISE(ABORT, 'artifacts are append-only'); END;
CREATE TRIGGER IF NOT EXISTS artifacts_append_only_delete
BEFORE DELETE ON artifacts BEGIN SELECT RAISE(ABORT, 'artifacts are append-only'); END;

COMMIT;
