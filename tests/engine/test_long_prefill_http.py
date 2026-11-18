import json

import pytest

from ironmule_product import compare
from lab.portable_gpu import long_prefill_http as followup, prefill_http


def cli(tmp_path, *flags):
    args = []
    for key in ("gguf", "tokenizer", "state-dir", "output"):
        args += ["--" + key, str(tmp_path / key)]
    return prefill_http.arguments(args + list(flags))


def test_receipt_is_required_before_hardware_or_server_work(tmp_path, monkeypatch):
    receipt = tmp_path / "refused.json"
    receipt.write_text('{"status":"incomplete"}')
    default = cli(tmp_path)
    assert default.arms == ["stock", "batch", "shared"] and default.concurrency == [1, 8, 32]
    with pytest.raises(SystemExit):
        cli(tmp_path, "--arms", "stock", "long-444")
    args = cli(tmp_path, "--long-native-receipt", str(receipt))
    assert (args.arms, args.concurrency, args.repeats, args.budget_seconds) == (["stock", "long-444"], [8, 32], 2, 900)
    monkeypatch.setattr(prefill_http.compare_servers, "prepare_artifacts", lambda _: pytest.fail("hardware work"))
    with pytest.raises(compare.Unverified, match="did not qualify"):
        prefill_http.run(args)
    assert json.loads(args.output.read_text())["status"] == "incomplete"


@pytest.mark.parametrize("arm, change", [("stock", {"2": 1, "4": 0}), ("long-444", {"2": 0, "4": 0}),
                                         ("long-444", {"2": 0, "4": 2})])
def test_fallback_and_disagreement_reject_a_measured_wave(arm, change):
    before = {"calls": 1, "rows": 888, "prompts": 2, "long_shape_hits": {"2": 1, "4": 0}}
    after = {**before, "calls": 2, "rows": 1776, "prompts": 4,
             "long_shape_hits": {"2": 1 + change["2"], "4": change["4"]}}
    with pytest.raises(compare.Unverified):
        followup.wave_hits(arm, after, before, {"calls": 1, "rows": 888, "prompts": 2})
    after["long_shape_hits"] = {"2": 2, "4": 0}
    assert followup.wave_hits("long-444", after, before, {"calls": 1, "rows": 888, "prompts": 2}) == {"2": 1, "4": 0}


def test_scoped_followup_binds_workers_and_pairs_only_matching_heldout_inputs(tmp_path, monkeypatch):
    import os

    gguf, path = tmp_path / "gguf", tmp_path / "native.json"
    gguf.write_bytes(b"x")
    model = "a"*64
    uuid = "abcdef01-1234-1234-1234-123456789abc"
    device = "GPU-"+uuid
    settings = {"flash_attn": True, "kv_unified": False}
    shape = {"seqs": 32, "seq_ctx": 2048, "threads": 4, "settings": settings}
    path.write_text(json.dumps(shape))
    ids = {"cuda_uuid": uuid.replace("-", ""), "native_runtime_sha256": "b"*64, "implementation_sha256": "c"*64}
    monkeypatch.setattr("ironmule.long_prefill_study.local_identity", lambda: ids)
    monkeypatch.setattr(followup, "_qualified", lambda rc, **scope: rc == shape and
                        (not scope or scope == {"gguf_sha256": model, **ids}))
    def prepare(args):
        args.gguf_sha256 = model
        return {"tokenizer_files": {**prefill_http.compare_tokenizer._QWEN_FILES, "config.json":
                                     prefill_http.compare_tokenizer.TOKENIZERS["qwen3-32b"][2]}}
    monkeypatch.setattr(prefill_http.compare_servers, "prepare_artifacts", prepare)
    monkeypatch.setitem(prefill_http.fetch.MODELS, "qwen3-32b", ("repo", "rev", "file", 1, model))
    expected = []
    def measure(args, name, rep, data, report, save, deadline, *, profile, label):
        candidate = label == "long-444"
        assert os.getenv("IRONMULE_GGML_PREFILL_RECEIPT") == (str(path) if candidate else None)
        assert os.getenv("IRONMULE_GGML_PREFILL_BATCH") == ("long-444" if candidate else "0")
        expected.append(label)
        server = {"device": device, "gguf_sha256": model}
        report["deployments"].append({"deployment": label, "repeat": rep, "prompt_ids_verified": True, "identity": server})
        layout = {**shape, "provider": "cuda", "cuda_uuid": ids["cuda_uuid"], "machine": label,
                  "implementation": ids["implementation_sha256"], "study_native_runtime_sha256": ids["native_runtime_sha256"],
                  "host_sha256": "host", "shared": False, "gguf_sha256": model}
        scope = prefill_http._scope(layout, server)
        stats = {"mode": "long-444" if candidate else "off", "enabled": candidate}
        ggml = {"experiment_binding": layout, "prefill_batch": stats, "gate": {"passes": True},
                "stream_qualification": {"passes": True}}
        followup.verify_worker(shape, label, server, ggml, scope)
        with pytest.raises(compare.Unverified):
            followup.verify_worker(shape, label, server, {**ggml, "stream_qualification": {"passes": False}}, scope)
        for width in data:
            report["measurements"].append({"deployment": label, "repeat": rep, "concurrency": width,
                "prompt_ids_sha256": ["key"]*width, "request_id_prefix": f"r{rep}-{width}",
                "quality": {"answers": {"fact": True}}, "tokens_per_second": 100, "requests_per_second": 10,
                "ttft_p95_ms": 100, "generation": [[1,"stop"]]*width, "profile": {"scope": scope,
                "outputs_sha256": "out", "batch_delta": {"long_shape_hits": {"2": int(candidate), "4": 0}}}})
    monkeypatch.setattr(prefill_http.production_compare, "measure", measure)
    args = cli(tmp_path, "--long-native-receipt", str(path))
    monkeypatch.setenv("IRONMULE_GGML_PREFILL_RECEIPT", "prior")
    assert prefill_http.run(args) == 0
    assert os.environ["IRONMULE_GGML_PREFILL_RECEIPT"] == "prior"
    result = json.loads(args.output.read_text())
    assert result["status"] == "measured" and len(result["pairs"]) == 4
    assert expected == ["stock", "long-444", "long-444", "stock"]
    result["measurements"][0]["prompt_ids_sha256"] = ["wrong"]
    with pytest.raises(compare.Unverified, match="paired HTTP"):
        followup.pairs(result)
