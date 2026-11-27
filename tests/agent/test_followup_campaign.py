import json

import pytest

from lab.rp import followup_campaign
from training import code_author


def test_author_can_decline_once_without_creating_or_testing_a_changed_source(tmp_path):
    base = tmp_path / "checkout"
    source = base / "ironmule/ggml_engine.py"
    source.parent.mkdir(parents=True)
    source.write_text("class GgmlEngine:\n    def __init__(self):\n        pass\n    def prices(self):\n        return {}\n")
    order = {"source_sha256": followup_campaign.checksum(source), "order_id": "sample",
             "target": {"class": "GgmlEngine", "function": "prices", "line": 4}}
    archive = tmp_path / "archive.jsonl"
    archive.touch()
    with pytest.raises(code_author.PendingAuthor):
        followup_campaign.author(order, base, tmp_path, "hosted", archive)
    state = tmp_path / "authoring"
    key, request = code_author.read_request(state / "author")
    code_author.respond(state, key, {"definition": request["source"], "rationale": "No measured change."}, "worker", "model")
    for _ in range(2):
        assert followup_campaign.author(order, base, tmp_path, "hosted", archive) is None
    assert json.loads((state / "unchanged.json").read_text()) == {"order": "sample", "reason": "No measured change."}
    assert not (state / "candidate.py").exists()
