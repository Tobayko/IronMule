"""Regression coverage for controlled three-engine comparisons and fail-closed reports."""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import ironmule_cli
from ironmule_product import compare, compare_servers, compare_tokenizer


def test_stream_measures_visible_token_and_requires_done_and_completion_usage():
    from time import monotonic

    requests = []
    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            self.send_response(200)
            self.end_headers()
            for chunk in ({"choices": [{"delta": {"role": "assistant"}}]},
                          {"choices": [{"delta": {"content": "Rome"}}]},
                          {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                          {"choices": [], "usage": {"prompt_tokens": 6, "completion_tokens": 1}}):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            if body["messages"][0]["content"] != "cut":
                self.wfile.write(b"data: [DONE]\n\n")

    http = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = {"base_url": f"http://127.0.0.1:{http.server_port}", "model_id": "sample"}
        out = compare.stream("vllm", endpoint, "city", 10, monotonic() + 1)
        assert out["text"] == "Rome" and out["tokens"] == 1 and out["ttft_ms"] > 0
        assert requests[-1]["chat_template_kwargs"] == {"enable_thinking": False}
        with pytest.raises(RuntimeError, match="without"):
            compare.stream("ironmule", endpoint, "cut", 10, monotonic() + 1)
        assert "stream_options" not in requests[-1]
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_controlled_cli_rotates_arms_and_keeps_real_partial_evidence(tmp_path, monkeypatch):
    calls = []
    artifact = {"gguf_sha256": "a"*64, "tokenizer_sha256": "b"*64}
    def prepare(args):
        vars(args).update(artifact)
        return artifact

    @contextmanager
    def server(engine, args, repeat, deadline):
        record = {**artifact, "version": "fixture", "device": "cpu:fixture", "baseline_stock": True, "placement_verified": True, "autonomous": engine == "ironmule"}
        calls.append((repeat, engine))
        yield record

    def preflight(engine, server, args, messages, deadline):
        return {"verified": True, "digest": "1"*64, "prompt_tokens": [10]*len(messages)}

    def batch(engine, server, prompts, width, limit, deadline):
        from ironmule.factual_gate import cases
        # The fixture returns equal correct answers, but the reference at width two is faster.
        refs = {q.prompt: q.expected for rep in range(3) for q in cases(123 + 1009 * rep + width)}
        return [{"text": refs.get(prompt, "description"), "tokens": 2, "finish_reason": "length", "prompt_tokens": 10, "ttft_ms": 5.0}
                for prompt in prompts], (2 if width == 2 and engine == "ironmule" else 1)

    monkeypatch.setattr(compare_servers, "prepare_artifacts", prepare)
    monkeypatch.setattr(compare_servers, "open_server", server)
    monkeypatch.setattr(compare_servers, "refresh_identity", lambda *args: None)
    monkeypatch.setattr(compare_tokenizer, "preflight", preflight)
    monkeypatch.setattr(compare, "batch", batch)
    monkeypatch.setattr(compare, "stream", lambda *args: {"prompt_tokens": 10})
    report_file = tmp_path / "progress.json"
    argv = ["compare", "--gguf", "sample.gguf", "--tokenizer", ".", "--vllm", "vllm", "--llama-server", "llama",
            "--seed", "123", "--concurrency", "1,2", "--repeats", "3", "--output", str(report_file)]
    assert ironmule_cli.main(argv) == 0
    report = json.loads(report_file.read_text())
    assert [e for rep, e in calls if rep == 2] == ["vllm", "ironmule", "llama_cpp"]
    assert report["status"] == "complete" and len(report["runs"]) == 18
    assert report["verdict"]["status"] == "does_not_qualify"
    assert "throughput_margin" in report["verdict"]["reasons"]
    assert all(r["quality"]["passed"] == 18 for r in report["runs"])
    # A server accepting the same weight file can still tokenize the actual request differently.
    def mismatch(engine, server, args, messages, deadline):
        row = preflight(engine, server, args, messages, deadline)
        if engine == "llama_cpp":
            row["digest"] = "2"*64
        return row
    monkeypatch.setattr(compare_tokenizer, "preflight", mismatch)
    assert ironmule_cli.main(argv) == 0
    partial = json.loads(report_file.read_text())
    assert partial["verdict"]["status"] == "incomplete"
    assert "prompt_ids_unmatched" in partial["verdict"]["reasons"]
    assert len(partial["runs"]) == 18

    @contextmanager
    def broken(engine, args, repeat, deadline):
        if engine == "llama_cpp":
            raise RuntimeError("missing competitor executable")
        with server(engine, args, repeat, deadline) as info:
            yield info
    monkeypatch.setattr(compare_servers, "open_server", broken)
    assert ironmule_cli.main(argv) == 1
    partial = json.loads(report_file.read_text())
    assert partial["error_code"] == "launch_failed" and len(partial["failures"]) == 3
    assert {r["engine"] for r in partial["runs"]} == {"ironmule", "vllm"}
    assert len(partial["runs"]) == 12
    assert partial["verdict"]["by_opponent"]["llama_cpp"]["status"] == "incomplete"
    assert partial["verdict"]["by_opponent"]["vllm"]["status"] == "does_not_qualify"

    with pytest.raises(compare.Unverified, match="expected"):
        compare.check_usage(["same prompt"], [{"prompt_tokens": 8}], {"same prompt": 7})


def test_gate_rejects_lost_answer_and_p95_even_if_tok_s_improve():
    protocol = {"engines": list(compare.ENGINES), "repeats": 3, "concurrency": [1, 4]}
    cases = {"1:facts-0": True, "1:arithmetic-0": True, "1:extraction-0": True}
    runs = []
    for engine in compare.ENGINES:
        for rep in range(3):
            for width in protocol["concurrency"]:
                runs.append({"engine": engine, "repeat": rep, "concurrency": width,
                             "tokens_per_second": 20 if engine == "ironmule" else 10,
                             "requests_per_second": 3, "ttft_p95_ms": 100, "performance_prompt_tokens": [8],
                             "generation": [[3, "stop"]],
                             "quality": {"answers": dict(cases), "prompt_tokens": [8]}})
    arms = [{"engine": e, "repeat": r, "identity": {"baseline_stock": True, "placement_verified": True},
             "tokenization": {"digest": "a" * 64}} for e in compare.ENGINES for r in range(3)]
    report = {"protocol": protocol, "runs": runs, "arms": arms}
    assert compare.evaluate(report)["status"] == "qualifies"
    arms[0]["identity"]["placement_verified"] = False
    assert compare.evaluate(report)["status"] == "incomplete"
    arms[0]["identity"]["placement_verified"] = True
    runs[0]["ttft_p95_ms"] = 110
    runs[0]["quality"]["answers"]["1:extraction-0"] = False
    assert set(compare.evaluate(report)["reasons"]) == {"quality_regressed", "first_token_regressed"}


def test_pair_scope_does_not_require_omitted_vllm(tmp_path):
    opts = compare.arguments(["--engines", "ironmule,llama_cpp", "--gguf", "m.gguf", "--tokenizer", ".",
                              "--llama-server", "llama-server", "--state-dir", str(tmp_path)])
    assert opts.engines == ("ironmule", "llama_cpp") and opts.vllm is None


def test_ctrl_c_persists_report_and_stops_watcher(tmp_path, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(compare, "run", interrupt)
    out = tmp_path / "interrupted.json"
    assert compare.main(["--engines", "ironmule,llama_cpp", "--gguf", "m.gguf", "--tokenizer", ".",
                         "--llama-server", "llama-server", "--output", str(out)]) == 130
    assert json.loads(out.read_text())["error_code"] == "interrupted"
