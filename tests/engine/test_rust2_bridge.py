"""RUST2-SRV: the rust2 bridge speaks the binary's serve protocol. Model-free: a fake binary echoes it."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from ironmule_product.rust2_bridge import Rust2Bridge

FAKE = r'''#!{python}
import json, sys
spec = json.load(open(sys.argv[2]))
print(json.dumps({{"ready": True, "max_positions": 1024, "speculative": spec["draft"] is not None}}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    if len(r["prompt"]) + r["max_tokens"] > 1024:
        print(json.dumps({{"id": r["id"], "error": "too long"}}), flush=True)
        continue
    out = [t + 100 for t in r["prompt"]][: r["max_tokens"]]
    stop = next((i for i, t in enumerate(out) if t in r["stop"]), None)
    out = out if stop is None else out[: stop + 1]
    for t in out:
        print(json.dumps({{"id": r["id"], "token": t}}), flush=True)
    finish = "stop" if out and out[-1] in r["stop"] else "length"
    print(json.dumps({{"id": r["id"], "done": True, "finish": finish, "tokens": len(out)}}), flush=True)
'''


@pytest.fixture
def binary(tmp_path: Path) -> str:
    path = tmp_path / "rust2"
    path.write_text(FAKE.format(python=sys.executable))
    os.chmod(path, 0o755)
    return str(path)


def test_tokens_stream_in_order_and_the_finish_reason_follows_the_stop_set(binary):
    bridge = Rust2Bridge(binary, "/models/qwen3", eos_ids=[105])
    try:
        seen = []
        result = bridge.generate([1, 2, 3], 2, on_token=seen.append)
        assert (result.tokens, seen, result.finish_reason, result.token_count) == ([101, 102], [101, 102], "length", 2)
        result = bridge.generate([1, 5, 9], 8)
        assert (result.tokens, result.finish_reason) == ([101, 105], "stop")
        assert result.metadata["engine"] == "rust2" and result.metadata["speculative"] is False
    finally:
        bridge.close()


def test_a_refused_request_raises_and_the_next_one_still_answers(binary):
    bridge = Rust2Bridge(binary, "/models/qwen3", eos_ids=[], draft_dir="/drafts/dspark")
    try:
        with pytest.raises(ValueError, match="refused"):
            bridge.generate([1] * 1000, 100)
        assert bridge.generate([7], 1).tokens == [107]
        with pytest.raises(ValueError, match="chat turns"):
            bridge.generate([7], 1, boundaries=[1])
    finally:
        bridge.close()


def test_a_missing_binary_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="unavailable"):
        Rust2Bridge(str(tmp_path / "absent"), "/models/qwen3", eos_ids=[])


def test_a_silent_binary_times_out_instead_of_hanging(tmp_path):
    path = tmp_path / "silent"
    path.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    os.chmod(path, 0o755)
    with pytest.raises(RuntimeError, match="sent nothing"):
        Rust2Bridge(str(path), "/models/qwen3", eos_ids=[], ready_timeout_s=0.5)
