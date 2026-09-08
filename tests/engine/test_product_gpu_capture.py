"""Pure control-contract tests for PROD10 GPU capture; never import MLX."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import product_gpu_capture as capture


def test_isolated_cli_bootstraps_exact_sibling_helper():
    harness = Path(capture.__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-I", str(harness), "--help"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--execute" in result.stdout


def test_capture_process_reuses_proven_reference_protocol_and_cleanup():
    assert capture.CaptureProcess.read is capture.ReferenceProcess.read
    assert capture.CaptureProcess.close is capture.ReferenceProcess.close


def test_private_trace_is_exact_fresh_descendant(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    target = tmp_path / ".friday-data" / "profiles" / "run-1" / "capture.gputrace"
    assert capture._private_trace(target) == target.resolve()
    assert target.parent.is_dir()
    assert target.parent.stat().st_mode & 0o077 == 0
    with pytest.raises(capture.CaptureFailure, match="trace_exists"):
        capture._private_trace(target)
    with pytest.raises(capture.CaptureFailure, match="trace_path_not_private"):
        capture._private_trace(tmp_path / ".friday-data" / "profiles" / "capture.gputrace")


def test_reference_requires_four_stable_bound_rows(monkeypatch, tmp_path):
    keys = {
        "output_sha256": "a" * 64, "token_sha256": "b" * 64,
        "text_sha256": "c" * 64, "finish_reason": "length",
        "prompt_tokens": 1077, "completion_tokens": 8,
    }
    identity = {name: char * 64 for name, char in zip(
        ("model_sha256", "environment_sha256", "hardware_sha256", "code_sha256"), "def0")}
    raw = {"status": "passed", "model_id": capture.MODEL_ID, "revision": capture.REVISION,
           "identity_before": identity, "identity_after": identity,
           "samples": [{"backend": "stock", "case": "long_8", "repeat": index,
                        "warmup": index == 0, **keys} for index in range(4)]}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(capture, "RAW_REFERENCE", path)
    assert capture._expected_stock() == (keys, identity)
    raw["samples"][3]["token_sha256"] = "9" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(capture.CaptureFailure, match="stock_reference_not_stable"):
        capture._expected_stock()


def test_stream_close_failure_still_stops_capture():
    class Stream:
        def __iter__(self): return iter(())
        def close(self): raise RuntimeError("close failed")
    class Metal:
        stopped = False
        def start_capture(self, _path): pass
        def stop_capture(self): self.stopped = True
    class MX:
        metal = Metal()
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs): return list(range(1077))
    with pytest.raises(RuntimeError, match="close failed"):
        capture._sample(object(), Tokenizer(), MX(), lambda *_args, **_kwargs: Stream(), {},
                        capture=Path("capture.gputrace"))
    assert MX.metal.stopped is True


def test_trace_metadata_does_not_count_symlink_aliases_and_deduplicates_hardlinks(tmp_path):
    trace = tmp_path / "capture.gputrace"
    trace.mkdir()
    payload = trace / "MTLBuffer-0"
    payload.write_bytes(b"abc")
    os.link(payload, trace / "MTLBuffer-0-hardlink")
    os.symlink(payload.name, trace / "MTLBuffer-0-alias")

    metadata = capture._trace_metadata(trace)

    assert metadata["exists"] is True
    assert metadata["size_bytes"] == 3
    assert metadata["regular_file_count"] == 1
    assert metadata["symlink_count"] == 1
    assert metadata["allocated_bytes"] >= 3


def test_trace_metadata_does_not_follow_root_symlink(tmp_path):
    target = tmp_path / "real.gputrace"
    target.mkdir()
    (target / "payload").write_bytes(b"abc")
    trace = tmp_path / "capture.gputrace"
    os.symlink(target, trace)

    metadata = capture._trace_metadata(trace)

    assert metadata["exists"] is True
    assert metadata["size_bytes"] == 0
    assert metadata["regular_file_count"] == 0
    assert metadata["symlink_count"] == 1
