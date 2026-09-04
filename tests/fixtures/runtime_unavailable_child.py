"""Deterministic supervisor child; it never imports benchmark or MLX."""

from pathlib import Path

from friday_h0.protocol import (
    PRODUCTION_MANIFEST_BYTES,
    PRODUCTION_RESULT_BYTES,
    close_manifest,
    fallback_result,
    read_capped_json,
    write_json_atomic,
)


def main() -> int:
    value, _ = read_capped_json(
        Path.cwd() / "manifest.json",
        limit=PRODUCTION_MANIFEST_BYTES,
    )
    manifest = close_manifest(value)
    result = fallback_result(
        manifest=manifest,
        status="invalid",
        classification="runtime_unavailable",
        code="runtime_unavailable",
        message="test fixture forces runtime unavailable",
        evidence={"rss_peak_bytes": None, "rss_missing_reason": "test_fixture"},
    )
    write_json_atomic(Path.cwd() / "result.json", result, limit=PRODUCTION_RESULT_BYTES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
