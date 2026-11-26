import json

import pytest

from training.code_author import PendingAuthor, hosted_proposal, respond


def test_reply_is_bound_to_order_and_source_and_survives_restart(tmp_path):
    order = {"target": {"function": "count"}}
    src = "def count(items): return sum(1 for _ in items)"
    with pytest.raises(PendingAuthor):
        hosted_proposal(tmp_path, order, src)
    request = json.loads((tmp_path / "author" / "request.json").read_text())
    reply = {"definition": "def count(items): return len(items)", "rationale": "fixture"}
    with pytest.raises(ValueError):
        respond(tmp_path, "wrong", reply, "task-receipt", "fixture-model")
    respond(tmp_path, request["request_id"], reply, "task-receipt", "fixture-model")
    assert hosted_proposal(tmp_path, order, src) == reply
    with pytest.raises(ValueError, match="conflicting record"):
        hosted_proposal(tmp_path, order, src + "\n")
    new = {**reply, "rationale": "another attempt"}
    with pytest.raises(ValueError, match="conflicting record"):
        respond(tmp_path, request["request_id"], new, "task-receipt", "fixture-model")
