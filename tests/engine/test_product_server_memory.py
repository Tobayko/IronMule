"""Pure contract tests for PROD12; no model, MLX, or native load."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import product_server_memory as harness


def test_plan_is_fixed_three_by_four_plus_one_four_client_burst():
    plan = harness.request_plan()
    assert len(plan) == 16
    assert [(row["case"], row["repeat"], row["warmup"]) for row in plan[:4]] == [
        ("long_8", 0, True), ("long_8", 1, False),
        ("long_8", 2, False), ("long_8", 3, False),
    ]
    assert [row["case"] for row in plan[:12]] == ["long_8"] * 4 + ["short_32"] * 4 + ["long_32"] * 4
    assert {row["batch"] for row in plan[12:]} == {"burst-4"}
    assert [row["slot"] for row in plan[12:]] == [0, 1, 2, 3]


def test_reference_constants_bind_exact_stock_outputs():
    assert harness.REFERENCE["long_8"]["prompt_tokens"] == 1077
    assert harness.REFERENCE["short_32"]["completion_tokens"] == 13
    assert harness.REFERENCE["long_32"]["finish_reason"] == "stop"
    assert all(len(row[key]) == 64 for row in harness.REFERENCE.values()
               for key in ("output_sha256", "token_sha256", "text_sha256"))


def test_http_gate_rejects_any_reference_field_change():
    expected = harness.REFERENCE["short_32"]
    actual = {key: expected[key] for key in ("text_sha256", "prompt_tokens", "completion_tokens", "finish_reason")}
    harness._assert_http(actual, expected)
    actual["completion_tokens"] += 1
    with pytest.raises(harness.ValidationFailure, match="http_reference_mismatch"):
        harness._assert_http(actual, expected)


def test_controller_module_has_no_mlx_import_and_native_run_is_opt_in():
    source = Path(harness.__file__).read_text()
    tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not any(name == "mlx" or name.startswith("mlx.") or name == "mlx_lm" or name.startswith("mlx_lm.") for name in imports)
    assert '"--execute"' in source
    assert '[sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--server-child"]' in source
    assert "spec_from_file_location" in source
    assert 'sys.path.insert(0, str(ROOT))' not in source
    assert "start_new_session=True" in source
    assert "deadline=time.monotonic() + 10" in source
    assert '"spec": spec.as_dict()' not in source


def test_isolated_help_is_inert_and_available():
    result = subprocess.run(
        [sys.executable, "-I", str(Path(harness.__file__)), "--help"],
        text=True, capture_output=True, timeout=5,
    )
    assert result.returncode == 0
    assert "--execute" in result.stdout
    assert "--server-child" in result.stdout


def test_ready_protocol_names_only_public_model_binding_fields():
    source = Path(harness.__file__).read_text()
    assert '"model_id": spec.model_id' in source
    assert '"revision": spec.revision' in source
    assert '"weight_bytes": spec.weight_bytes' in source
    assert '"bindings_before": bindings_before' in source
    assert '"model_observations": rows' not in source
    assert 'interval_seconds=1.0' in source


def test_protocol_reader_streams_observation_before_terminal_frame():
    program = (
        "import json; "
        "print(json.dumps({'type':'model_observation','observation':{'pid':123}}), flush=True); "
        "print(json.dumps({'type':'ready','server_pid':456}), flush=True)"
    )
    process = subprocess.Popen([sys.executable, "-I", "-c", program],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    wrapper = harness.ServerProcess.__new__(harness.ServerProcess)
    wrapper.process, wrapper.buffer, wrapper.last_event, wrapper.pending = process, bytearray(), None, []
    wrapper.on_event = None
    rows = []
    try:
        terminal = wrapper.read(deadline=harness.time.monotonic() + 5, on_event=rows.append)
        assert rows == [{"pid": 123}]
        assert terminal == {"type": "ready", "server_pid": 456}
    finally:
        process.wait(timeout=5)


def test_removed_artificial_limits_are_explicit():
    assert harness.POLICY["required_pause_s"] == 0
    assert harness.POLICY["automatic_retry"] is False
    for key in ("request_timeout_s", "startup_timeout_s", "generation_limit", "work_budget_s",
                "rss_stop_bytes", "swap_stop_bytes", "duty_cycle_limit"):
        assert harness.POLICY[key] is None
