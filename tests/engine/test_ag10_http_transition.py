import io
import json

import pytest

from lab.ag10 import http_transition as bench


def _sse(chunks):
    events = ("data: " + (json.dumps(chunk) if isinstance(chunk, dict) else chunk) + "\n\n" for chunk in chunks)
    return io.BytesIO("".join(events).encode())


def test_stream_must_finish_and_report_usage(monkeypatch):
    content = {"choices": [{"delta": {"content": "Paris"}}]}
    usage = {"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 14}}
    for incomplete in ([content, usage], [content, "[DONE]"], [usage, "[DONE]"]):
        monkeypatch.setattr(bench, "urlopen", lambda *_args, **_kwargs: _sse(incomplete))
        with pytest.raises(ValueError, match="incomplete streamed response"):
            bench.chat("http://localhost", "model", "capital?", "Paris", 8)
    monkeypatch.setattr(bench, "urlopen", lambda *_args, **_kwargs: _sse([content, usage, "[DONE]"]))
    row = bench.chat("http://localhost", "model", "capital?", "Paris", 8)
    assert row["correct"] and row["tokens"] == 2 and row["prompt_tokens"] == 14


def test_pair_requires_matching_starting_evidence():
    from lab.ag10.run_pair import compare

    row = {"completion_tok_s_active": 2.0, "ttft_p95_s": 1.0, "requests": 2, "correct": 1, "outputs": ["a", "b"]}
    data = {"protocol": {}, "model": "test", "phases": {p: row for p in bench.PHASES},
            "starting_ggml": {"sequences": {"sequences": 8}, "gate": {"passes": True}}}
    agent = {**data, "arm": "agent"}
    fixed = {**data, "arm": "fixed"}
    assert compare(agent, fixed)["documents"]["same_answers"] == 2
    with pytest.raises(ValueError, match="startup measurement"):
        compare(agent, {**fixed, "starting_ggml": {"sequences": {"sequences": 16}, "gate": {"passes": True}}})


def test_city_score_separates_content_from_requested_format():
    assert bench.score("Paris", "Paris") == (True, True)
    assert bench.score("The capital of France is Paris.", "Paris") == (True, False)
    assert bench.score("The capital of France is Rome.", "Paris") == (False, False)
