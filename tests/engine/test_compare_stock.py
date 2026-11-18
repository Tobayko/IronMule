import json
from types import SimpleNamespace

import pytest

from ironmule_product import compare_servers, compare_stock


def test_caller_pins_source_and_every_loaded_native_image(tmp_path, monkeypatch):
    revision = 'a' * 40
    exe = '1' * 64
    libs = {'libllama.so': '2' * 64, 'libggml-cpu.so': '3' * 64}
    monkeypatch.setattr(compare_stock, '_mapped_images', lambda pid: (exe, libs))
    manifest = {**compare_stock.BUNDLED, 'commit': revision, 'executable_sha256': exe, 'library_sha256': libs}
    path = tmp_path / 'reference.json'
    version = 'version: 12 (aaaaaaaa)'

    def check(expected=manifest, ver=version, env=None):
        path.write_text(json.dumps(expected))
        return compare_stock.stock_identity(123, env or {}, ver, path)

    assert check()['provenance'] == 'caller-pinned'
    assert check()['verified_upstream_commit'] == revision
    for changes in ({'patches': ['local.patch']}, {'source_tree_clean': False}, {'commit': 'b' * 40},
                    {'library_sha256': {'libllama.so': libs['libllama.so']}}, {'executable_sha256': '4' * 64}):
        assert not check({**manifest, **changes})['baseline_stock']
    assert not check(ver='version: 12 (aaaaaaaa-dirty)')['baseline_stock']
    assert not check(env={'LD_PRELOAD': '/tmp/plugin.so'})['baseline_stock']
    assert not compare_stock.stock_identity(123, {}, version)['baseline_stock']


def test_artifacts_cached_only_while_inputs_are_unchanged(tmp_path, monkeypatch):
    model, tok = tmp_path / 'model.gguf', tmp_path / 'tokenizer'
    model.write_bytes(b'model')
    tok.mkdir()
    (tok / 'tokenizer_config.json').write_text('{}')
    cfg = SimpleNamespace(gguf=model, tokenizer=tok)
    first = compare_servers.prepare_artifacts(cfg)
    monkeypatch.setattr(compare_servers, '_sha', lambda path: pytest.fail('rehash of unchanged artifacts'))
    assert compare_servers.prepare_artifacts(cfg) == first
    (tok / 'chat_template.jinja').write_text('new template')
    with pytest.raises(ValueError, match='changed'):
        compare_servers.prepare_artifacts(cfg)


def test_vllm_cuda_and_rocm_placements_remain_distinguishable():
    assert compare_servers._placement('vllm','GPU-123',{},'device_config=cuda')[0]
    assert not compare_servers._placement('vllm','GPU-123',{},'device_config=rocm')[0]
    assert not compare_servers._placement('vllm','rocm:machine:0',{},'device_config=cuda')[0]
    assert compare_servers._placement('vllm','rocm:machine:0',{},'device_config=cuda. Detected platform rocm')[0]


def test_startup_diagnostics_only_return_known_codes(tmp_path):
    path = tmp_path / 'engine.log'
    path.write_text('private-token-and-user-prompt\n{"gguf_model":"private id"}\n'
                    'ModuleNotFoundError: No module named \'mlx.core\'\n')
    assert compare_servers._startup_detail(path) == ('gguf_identified', 'missing_mlx')
    path.write_text('Initializing a V1 LLM engine. Free memory is less than desired GPU memory utilization\n'
                    'Engine core initialization failed. private-token')
    assert compare_servers._startup_detail(path) == ('vllm_engine', 'insufficient_free_gpu_memory')
    path.write_text('secret other data received from arbitrary log')
    assert compare_servers._startup_detail(path) == ('startup', 'unclassified')
