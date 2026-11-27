import io
import json
import re
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from lab.rp import cpu_product_release as product
from training.serving_trial import fixture


def prompts(seed=9821, phase="short"):
    return [p for p, _ in fixture(seed, 4, phase)]


def test_scope_accepts_only_measured_cpu_product_shape():
    assert product.scope(prompts(), 3) and product.scope(prompts(9821, "document"), 3)
    assert not product.scope(prompts()[:2], 3) and not product.scope(prompts(), 4)
    assert not product.scope([prompts()[0], *prompts(1, "document")[1:]], 3)
    assert not product.scope(["unrelated"] * 4, 3)


def test_first_visible_text_is_timed_in_arrival_order_and_checked_against_the_answer():
    def packets(*rows):
        return SimpleNamespace(stdout=io.StringIO("".join(json.dumps(r) + "\n" for r in rows)))

    first = {"event": "first_text", "lane": 1, "text": "blue"}
    second = {"event": "first_text", "lane": 0, "text": "red"}
    end = {"answers": [{"text": "red!"}, {"text": "blue!"}], "ttfts_s": [0.0, 0.0]}
    row = product.collect_reply(packets(first, second, end), time.monotonic(), 2)
    assert 0 < row["ttfts_s"][1] <= row["ttfts_s"][0] <= row["broker_wall_s"]
    assert row["worker_ttfts_s"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="duplicate"):
        product.collect_reply(packets(first, first, end), time.monotonic(), 2)
    with pytest.raises(ValueError, match="completed answer"):
        product.collect_reply(packets(first, {**second, "text": "green"}, end), time.monotonic(), 2)
    empty = product.collect_reply(packets({"event": "empty_answer", "lane": 0},
                                          {"answers": [{"text": ""}], "ttfts_s": [None]}), time.monotonic(), 1)
    assert empty["ttfts_s"] == [None]


def test_regression_returns_completed_product_answer_then_persistently_uses_reference(tmp_path, monkeypatch):
    args = SimpleNamespace(state=tmp_path, receipt_sha256="sha")
    (tmp_path / "deployment").mkdir()
    pin = {"receipt": "hash"}
    (tmp_path / "deployment/product-gate.json").write_text(json.dumps(
        {"passed": True, "binding": pin, "reference_limits": {"short": {"wall_s": 1, "ttft_s": .5}}}))
    selected = {"name": "candidate"}
    monkeypatch.setattr(product.release, "selected", lambda *args: selected["name"])
    monkeypatch.setattr(product.release, "rollback", lambda *args: selected.update(name="reference"))
    monkeypatch.setattr(product, "binding", lambda *args: pin)

    @contextmanager
    def opened(_a, _r, arm):
        def run(batch, limit):
            return {"answers": [{"text": re.search(r"Color: (\w+)\nCopy:$", p)[1]} for p in batch],
                    "ttfts_s": [1 if arm == "candidate" else .1] * len(batch)}
        yield run, {"qualified": True}
    monkeypatch.setattr(product, "arm", opened)
    with product.service(args, {}) as dispatch:
        assert dispatch(prompts()[:2], 3)["arm"] == "reference"
        assert dispatch(prompts(), 3)["arm"] == "candidate"
        assert dispatch(prompts(), 3)["arm"] == "reference"
    with product.service(args, {}) as dispatch:
        assert dispatch(prompts(), 3)["arm"] == "reference"
