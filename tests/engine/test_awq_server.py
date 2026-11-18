import hashlib
import sys
import time
from types import SimpleNamespace

import pytest

from lab.portable_gpu import awq_server, fetch_awq


def _line(model, tokenizer, method):
    return (f"Initializing a V1 LLM engine (v0.11.0) with config: model={str(model)!r}, "
            f"tokenizer={str(tokenizer)!r}, dtype=torch.float16, quantization={method}, device_config=cuda,")


@pytest.mark.parametrize("method", ["awq", "awq_marlin"])
def test_only_the_ready_engines_actual_backend_is_attributed(tmp_path, method):
    log = tmp_path / "engine.log"
    model, tok = tmp_path / "awq weights", tmp_path / "base tokenizer"
    log.write_text("The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.\n")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        awq_server._engine_config(log, model, tok)
    log.write_text(_line(model, tok, method) + "\n")
    assert awq_server._engine_config(log, model, tok) == (method, "float16")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        awq_server._engine_config(log, model, tok.parent)
    log.write_text(_line(model, tok, "gguf"))
    with pytest.raises(ValueError, match="unexpected quantization"):
        awq_server._engine_config(log, model, tok)


@pytest.fixture
def config(tmp_path, monkeypatch):
    # Tiny pinned bytes exercise both manifests locally. The child emulates only the release's HTTP and log formats.
    model, tok = tmp_path / "weights", tmp_path / "tokenizer"
    model.mkdir()
    tok.mkdir()
    weight = b"test native weight"
    (model / "test.safetensors").write_bytes(weight)
    monkeypatch.setattr(fetch_awq, "FILES", {"test.safetensors": (len(weight), hashlib.sha256(weight).hexdigest())})
    hashes = {}
    for name in ("tokenizer.json", "config.json"):
        (tok / name).write_text(name)
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    monkeypatch.setattr(awq_server.compare_tokenizer, "_QWEN_FILES", {"tokenizer.json": hashes["tokenizer.json"]})
    monkeypatch.setitem(awq_server.compare_tokenizer.TOKENIZERS, "qwen3-32b", ("repo", "revision", hashes["config.json"]))

    def device(env, engine):
        assert engine == "vllm"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        return "GPU-test", "test board"

    monkeypatch.setattr(awq_server.compare_servers, "_device", device)
    exe = tmp_path / "vllm"
    exe.write_text("#!" + sys.executable + "\n" + """
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
def opt(k): return args[args.index(k)+1]
model, tokenizer = args[1], opt('--tokenizer')
assert args[0] == 'serve' and os.environ['CUDA_VISIBLE_DEVICES'] == '0' and os.environ['HF_HUB_OFFLINE'] == '1'
assert os.environ['TRANSFORMERS_OFFLINE'] == '1' and 'VLLM_API_KEY' not in os.environ
print(f"Initializing a V1 LLM engine (v0.11.0) with config: model={model!r}, tokenizer={tokenizer!r}, dtype=torch.float16, quantization=awq_marlin, device_config=cuda,", flush=True)
class S(BaseHTTPRequestHandler):
    def do_GET(self):
        data = {'/health': {}, '/version': {'version': '0.11.0'},
                '/v1/models': {'data': [{'id': 'compare','root': os.environ.get('TEST_MODEL_ROOT',model)}]}}[self.path]
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(data).encode())
HTTPServer(('127.0.0.1',int(opt('--port'))),S).serve_forever()
""")
    exe.chmod(0o700)
    return SimpleNamespace(awq_dir=model, awq_manifest=fetch_awq.verify(model), tokenizer=tok,
                           vllm=exe, startup_timeout=5, concurrency="1,4", context=2048, log_dir=tmp_path)


def test_native_server_uses_its_own_weights_and_observed_exact_backend(config, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "local-fixture")
    with awq_server.open_awq_server(config, 0, time.monotonic() + 6) as server:
        assert server["loaded_quantization"] == "awq_marlin" and server["loaded_dtype"] == "float16"
        assert server["weight_format"] == "awq" and server["model_root"] == str(config.awq_dir)
        assert server["awq_manifest_sha256"] == config.awq_manifest["manifest_sha256"]
        assert "gguf_sha256" not in server and server["placement_verified"]
        assert not {"--hf-config-path", "--quantization", "--enforce-eager", "--dtype"} & set(server["argv"])
        assert server["argv"][server["argv"].index("--tokenizer") + 1] == str(config.tokenizer)


def test_rejects_changed_weights_after_first_request_and_wrong_served_model(config, monkeypatch):
    monkeypatch.setenv("TEST_MODEL_ROOT", "/an/unrelated/model")
    with pytest.raises(ValueError, match="model root"), awq_server.open_awq_server(config, 0, time.monotonic() + 6):
        pytest.fail("server served a different snapshot")
    monkeypatch.delenv("TEST_MODEL_ROOT")
    with (pytest.raises(ValueError, match="snapshot differs"),
          awq_server.open_awq_server(config, 1, time.monotonic() + 6)):
        (config.awq_dir / "test.safetensors").write_bytes(b"changed")


def test_awq_and_gguf_must_use_the_same_pinned_base_tokenizer(config):
    config.tokenizer_sha256 = "f" * 64
    with pytest.raises(ValueError, match="manifests differ"):
        awq_server.check_snapshot(config)
    del config.tokenizer_sha256
    (config.tokenizer / "config.json").write_text('{"model_type":"other"}')
    with pytest.raises(ValueError, match="invalid Qwen3"):
        awq_server.check_snapshot(config)
