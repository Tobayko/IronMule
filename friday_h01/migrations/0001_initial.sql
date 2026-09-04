BEGIN IMMEDIATE;

PRAGMA application_id = 0x48303131;

CREATE TABLE bundles (
    entity_id TEXT PRIMARY KEY
        CHECK(length(entity_id) BETWEEN 1 AND 160),
    entity_kind TEXT NOT NULL
        CHECK(entity_kind IN (
            'paced_session',
            'paced_study',
            'legacy_h0_warmup_observation'
        )),
    status TEXT NOT NULL,
    action TEXT NOT NULL
        CHECK(action = 'no_h0_conclusion'),
    created_at_unix_ns INTEGER NOT NULL
        CHECK(typeof(created_at_unix_ns) = 'integer'
              AND created_at_unix_ns BETWEEN 0 AND 9223372036854775807),
    manifest_sha256 TEXT NOT NULL
        CHECK(length(manifest_sha256) = 64
              AND manifest_sha256 NOT GLOB '*[^0-9a-f]*'),
    trace_sha256 TEXT NOT NULL
        CHECK(length(trace_sha256) = 64
              AND trace_sha256 NOT GLOB '*[^0-9a-f]*'),
    result_sha256 TEXT NOT NULL
        CHECK(length(result_sha256) = 64
              AND result_sha256 NOT GLOB '*[^0-9a-f]*'),
    lineage_sha256 TEXT NOT NULL
        CHECK(length(lineage_sha256) = 64
              AND lineage_sha256 NOT GLOB '*[^0-9a-f]*'),
    bundle_sha256 TEXT NOT NULL UNIQUE
        CHECK(length(bundle_sha256) = 64
              AND bundle_sha256 NOT GLOB '*[^0-9a-f]*'),
    manifest_json TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    lineage_json TEXT NOT NULL,
    bundle_json TEXT NOT NULL,
    CHECK(
        (entity_kind = 'paced_session'
         AND status IN ('h01_session_complete', 'h01_invalid'))
        OR
        (entity_kind = 'paced_study'
         AND status IN (
             'h01_stationarity_supported',
             'h01_complete_unresolved',
             'h01_invalid'
         ))
        OR
        (entity_kind = 'legacy_h0_warmup_observation'
         AND status = 'legacy_observation')
    )
);

CREATE INDEX idx_bundles_kind_created
    ON bundles(entity_kind, created_at_unix_ns DESC, entity_id);

CREATE INDEX idx_bundles_status_created
    ON bundles(status, created_at_unix_ns DESC, entity_id);

CREATE TRIGGER bundles_no_reinsert
BEFORE INSERT ON bundles
WHEN EXISTS (
    SELECT 1
    FROM bundles
    WHERE entity_id = NEW.entity_id
       OR bundle_sha256 = NEW.bundle_sha256
)
BEGIN
    SELECT RAISE(ABORT, 'bundles cannot be replaced');
END;

CREATE TRIGGER bundles_no_update
BEFORE UPDATE ON bundles
BEGIN
    SELECT RAISE(ABORT, 'bundles are append-only');
END;

CREATE TRIGGER bundles_no_delete
BEFORE DELETE ON bundles
BEGIN
    SELECT RAISE(ABORT, 'bundles are append-only');
END;

PRAGMA user_version = 1;

COMMIT;
