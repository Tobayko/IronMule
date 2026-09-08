"""Pure controller/protocol tests for PROD14; no fake native success."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import product_variant_qualification as screen


def test_fixed_variants_case_and_output_accounting():
    assert screen.VARIANTS == ("prefix_reuse", "current_engine")
    assert screen.long_case()["name"] == "long_8"
    assert screen.long_case()["max_tokens"] == 8
    assert screen.long_case(128)["max_tokens"] == 128
    assert screen.POLICY["startup_timeout_s"] is None
    assert screen.POLICY["rss_stop_bytes"] is None
    assert set(screen.ORACLES) == set(screen.MODELS)


def test_exact_output_binds_all_completion_fields():
    base = screen.output_metadata([7, 8], "ok", {
        "prompt_tokens": 1077, "completion_tokens": 2, "finish_reason": "length"})
    assert screen.exact_output(base, dict(base))
    for key in ("output_sha256", "token_sha256", "text_sha256", "prompt_tokens",
                "completion_tokens", "finish_reason"):
        changed = dict(base); changed[key] = "different"
        assert not screen.exact_output(changed, base)


def test_prefix_sequence_is_closed_over_real_metadata_shape():
    metadata = {"prefix_cache": {"status": "accepted", "commit_status": "no_pending_commit",
        "trace": {"status": "completed", "cache_hit": True, "cache_stored": False, "reused_tokens": 1076},
        "stats": {"entries": 1, "bytes": 4096}}}
    screen.prefix_expectation(metadata, hit=True)
    with pytest.raises(screen.ValidationFailure, match="prefix_hit_sequence_invalid"):
        screen.prefix_expectation(metadata, hit=False)
    metadata["prefix_cache"]["commit_status"] = "completed"
    with pytest.raises(screen.ValidationFailure, match="prefix_hit_sequence_invalid"):
        screen.prefix_expectation(metadata, hit=True)


def test_engine_metadata_requires_current_identity_profile_and_knobs():
    engine = {"current_identity": {"model": "fixture"},
              "profile_source": "baseline_no_compatible_profile", "selected_knobs": {},
              "fallback_used": False, "fallback_count": 0}
    assert screen.engine_metadata({"engine": engine}) is engine
    with pytest.raises(screen.ValidationFailure, match="engine_metadata_incomplete"):
        screen.engine_metadata({"engine": {"profile_source": "fixture"}})


def test_engine_metadata_rejects_any_completion_fallback():
    engine = {"current_identity": {}, "profile_source": "fixture", "selected_knobs": {},
              "fallback_used": True, "fallback_count": 1}
    with pytest.raises(screen.ValidationFailure, match="engine_fallback_observed"):
        screen.engine_metadata({"engine": engine})


def test_oracle_failure_is_closed_and_never_requests_implicit_stock(monkeypatch, tmp_path):
    model = next(iter(screen.MODELS))
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"status": "failed", "samples": []}))
    monkeypatch.setitem(screen.ORACLES, model, oracle)
    with pytest.raises(screen.FreshReferenceRequired, match="fresh_reference_required") as caught:
        screen.load_correctness_oracle(model, screen.MODELS[model], {}, {})
    assert caught.value.reason == "oracle_status_identity_provider_or_rows_invalid"


def test_cli_is_closed_without_explicit_execute(tmp_path):
    tool = Path(screen.__file__)
    result = subprocess.run([sys.executable, "-I", str(tool), "--model",
        next(iter(screen.MODELS)), "--variant", "prefix_reuse",
        "--state-dir", str(tmp_path), "--output", str(tmp_path / "out.json")],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "--execute" in result.stderr
    assert not (tmp_path / "out.json").exists()


def test_help_bootstrap_imports_no_mlx_or_model(tmp_path):
    tool = Path(screen.__file__)
    code = ("import json,runpy,sys; sys.argv=['tool','--help']; "
            "\ntry: runpy.run_path(" + repr(str(tool)) + ",run_name='__main__')"
            "\nexcept SystemExit: pass"
            "\nprint(json.dumps(sorted(k for k in sys.modules if k=='mlx' or k.startswith('mlx.') or k=='mlx_lm')))" )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout.splitlines()[-1]) == []
