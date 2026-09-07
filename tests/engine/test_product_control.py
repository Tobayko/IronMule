"""Product configuration and text/protocol contracts, without model fixtures."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from ironmule_product.errors import BackendUnavailable, InvalidRequest
from ironmule_product.service import ProductService, StopFilter
from ironmule_product.state import ProductStore
from ironmule_product.types import GenerationRequest, ModelSpec


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("chunks,stops,expected", [
    (["red<ST", "OP>hidden"], ("<STOP>",), "red"),
    (["red<ST", "raw"], ("<STOP>",), "red<STraw"),
    (["plain", " text"], (), "plain text"),
    (["abcXtail"], ("X", "bc"), "a"),
    (["a", "b", "a", "b", "x"], ("abab",), ""),
])
def test_stop_filter_across_delivery_boundaries(chunks, stops, expected):
    output = StopFilter(stops)
    assert "".join(output.feed(chunk) for chunk in chunks) + output.finish() == expected


@pytest.mark.parametrize("change", [
    {"max_tokens": True}, {"temperature": float("nan")}, {"n": 2},
    {"temperature": 0.5}, {"stop": [""]}, {"stream": 1},
    {"messages": [{"role": "user", "content": "\ud800"}]},
])
def test_exact_request_rejects_unsupported_or_malformed_values(change):
    payload = {"model": "local/model", "messages": [{"role": "user", "content": "hello"}]}
    with pytest.raises(InvalidRequest):
        GenerationRequest.from_payload({**payload, **change})


def test_control_plane_reports_registered_but_unloaded_model(tmp_path):
    store = ProductStore(tmp_path / "state")
    store.setup()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    store.register_model(ModelSpec("local/model", "revision", str(snapshot.resolve()), 1))
    service = ProductService(store)
    try:
        assert service.health()["ready"] is False
        assert service.models()[0]["loaded"] is False
        with pytest.raises(BackendUnavailable):
            service.stream(GenerationRequest("local/model", (("user", "hello"),)))
        assert service.health()["queued_requests"] == 0
    finally:
        service.close()
    assert not service._thread.is_alive()


def test_cli_setup_and_product_status_without_native_dependencies(tmp_path):
    state = tmp_path / "state"
    for command in (["setup", "--mode", "server"], ["status", "--product"]):
        result = subprocess.run([sys.executable, "-S", "-m", "ironmule_cli", *command,
                                 "--state-dir", str(state)], cwd=ROOT,
                                text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["settings"]["mode"] == "server"


def test_cli_serve_help_does_not_initialize_native_backend():
    result = subprocess.run([sys.executable, "-S", "-m", "ironmule_cli", "serve", "--help"],
                            cwd=ROOT, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--no-model" in result.stdout
