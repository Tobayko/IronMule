import json
import threading
import time

import pytest

from ironmule_product import compare
from lab.portable_gpu import mixed_traffic


def test_interleaved_tool_histories_are_matched_and_whole_workflows_are_graded(monkeypatch):
    seed = 17
    docs, short, agents = mixed_traffic.requests(seed)
    plan = docs + short + [r for pair in agents for r in pair]
    by_messages = {json.dumps(r.messages): r for r in plan}
    assert len(plan) == len(by_messages) == 18
    assert agents[0][1].messages[:1] == agents[0][0].messages
    assert mixed_traffic.requests(seed + 1) != (docs, short, agents)
    issued_short = threading.Event()

    def tokenizer(engine, server, conversations, deadline):
        assert conversations == [r.messages for r in plan]
        return {"digest": "a" * 64, "prompt_tokens": [10] * len(plan)}

    def stream(engine, server, messages, limit, deadline, *, request_id, on_first_visible):
        req = by_messages[json.dumps(messages)]
        assert req.limit == limit and request_id == f"mixed-{seed}-{req.key}"
        if req.workload == "long_document":
            on_first_visible()
            assert issued_short.wait(1)
        elif req.workload == "short_lookup":
            issued_short.set()
        return {"text": "wrong" if req.key == agents[0][0].key else req.expected,
                "tokens": 2, "prompt_tokens": 10, "visible_ttft_ms": 1, "finish_reason": "stop"}

    monkeypatch.setattr(mixed_traffic.compare_tokenizer, "rendered_prompt_ids", tokenizer)
    monkeypatch.setattr(compare, "stream", stream)
    out = mixed_traffic.measure("vllm", {}, seed, time.monotonic() + 2)
    assert out["fully_correct_workflows"] == 3 and out["classes"]["agent_result"]["correct"] == 4
    assert out["short_starts_during_long"] >= 1
    rows = {r["id"]: r for r in out["requests"]}
    assert all(rows[second.key]["issued_s"] >= rows[first.key]["ended_s"] for first, second in agents)
    assert mixed_traffic.pair(out, out)["status"] == "measured"
    assert mixed_traffic.pair(out, dict(out, short_starts_during_long=0))["status"] == "no_overlap"
    with pytest.raises(compare.Unverified):
        mixed_traffic.pair(out, dict(out, prompt_id_digest="wrong"))
