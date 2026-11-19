import hashlib
import io
import time
from types import SimpleNamespace

import pytest

from ironmule_product import compare_tokenizer as ct


def conversation(text):
    return [{"role": "user", "content": text}]


def test_live_tokenizers_hash_full_ordered_ids(tmp_path, monkeypatch):
    args = SimpleNamespace(gguf=tmp_path / "a.gguf", tokenizer=tmp_path / "tokenizer")
    server = {"argv": ["local", str(args.gguf), "--tokenizer", str(args.tokenizer)], "model_id": "a",
              "gguf_sha256": "a" * 64, "base_url": "http://127.0.0.1:8080"}
    responses = {"first": [1, 2, 3], "second": [1, 3, 2]}
    chat_calls = []

    def endpoint(url, deadline, body=None):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "a"}]}
        if url.endswith("/apply-template"):
            return {"prompt": body["messages"][0]["content"]}
        assert url.endswith("/tokenize")
        if "messages" in body:
            assert body["add_generation_prompt"] and not body["add_special_tokens"]
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            text = body["messages"][0]["content"]
            chat_calls.append(text)
            return {"tokens": responses[text], "count": len(responses[text])}
        assert body["parse_special"] and body["add_special"]
        return {"tokens": responses[body["content"]]}

    monkeypatch.setattr(ct, "_json", endpoint)
    messages = [conversation(value) for value in ("first", "second", "first")]
    deadline = time.monotonic() + 60
    baseline = ct.preflight("llama_cpp", server, args, messages, deadline)
    vllm = ct.preflight("vllm", server, args, messages, deadline)
    assert vllm["verified"] and vllm == baseline
    assert chat_calls == ["first", "second"]
    assert vllm["prompt_tokens"] == [3, 3, 3]
    changed = ct.preflight("vllm", server, args, messages[1:] + messages[:1], deadline)
    assert changed["digest"] != vllm["digest"]
    responses["first"] = [True]
    assert not ct.preflight("vllm", server, args, messages, deadline)["verified"]
    server["argv"].remove(str(args.gguf))
    assert not ct.preflight("vllm", server, args, messages, deadline)["verified"]


def test_tokenizer_snapshot_rechecks_cache_and_handles_failed_download(tmp_path, monkeypatch):
    inputs = {"config.json": b"config", "tokenizer.json": b"vocab"}
    monkeypatch.setattr(ct, "TOKENIZERS", {"test": ("repo", "revision", hashlib.sha256(inputs["config.json"]).hexdigest())})
    monkeypatch.setattr(ct, "_QWEN_FILES", {"tokenizer.json": hashlib.sha256(inputs["tokenizer.json"]).hexdigest()})
    monkeypatch.setattr(ct.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(inputs[url.rsplit("/", 1)[-1]]))
    folder = ct.fetch_hf_tokenizer("test", tmp_path)
    assert ct.fetch_hf_tokenizer("test", tmp_path) == folder
    (folder / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="unexpected tokenizer contents"):
        ct.fetch_hf_tokenizer("test", tmp_path)
    inputs["tokenizer.json"] = b"corrupt download"
    fresh = tmp_path / "other"
    with pytest.raises(ValueError, match="unexpected contents for"):
        ct.fetch_hf_tokenizer("test", fresh)
    assert list(fresh.iterdir()) == []
