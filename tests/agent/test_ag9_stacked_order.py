"""AG9 step S's simulation (`training/ag9_stacked_order.py`): its engine model and its fixed rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ag9_stacked_order", ROOT / "training" / "ag9_stacked_order.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
PRICES = tool.Prices(tool.CPU6A2_PRICES)


def _req(k, arrive, limit, length=None, prompt=tool.SHORT_PROMPT):
    return {"id": k, "arrive": arrive, "prompt": prompt, "limit": limit, "length": length or limit}


def test_one_request_costs_its_prefill_then_one_step_per_further_token():
    out = tool.simulate([_req(0, 0.0, 5)], "fifo", PRICES)
    expected = PRICES.prefill(tool.SHORT_PROMPT) + 4 * PRICES.step(1, 0.0)
    assert abs(out["span_s"] - expected) < 1e-12 and abs(out["mean_first_s"] - PRICES.prefill(11)) < 1e-12


def test_every_order_makes_the_same_tokens_and_orders_differ_only_in_time():
    trace = [_req(k, 0.0, 256 if k < 8 else 32) for k in range(16)]  # the long ones arrived first
    runs = {order: tool.simulate(trace, order, PRICES) for order in tool.ORDERS}
    tokens = {round(r["tokens_per_s"] * r["span_s"]) for r in runs.values()}
    assert tokens == {8 * 288}
    assert runs["sjf"]["mean_first_s"] < runs["fifo"]["mean_first_s"] == runs["ljf"]["mean_first_s"]


def test_a_freed_slot_goes_to_a_waiting_request_by_the_order_and_ties_by_arrival():
    key = tool._key("sjf")
    waiting = [_req(0, 0.1, 64), _req(1, 0.0, 64), _req(2, 0.2, 32)]
    assert [r["id"] for r in sorted(waiting, key=key)] == [2, 1, 0]
    assert [r["id"] for r in sorted(waiting, key=tool._key("ljf"))] == [1, 0, 2]


def test_traces_are_fixed_by_their_seed():
    rate = tool.capacity(PRICES)
    assert tool.trace("spike", "early", 3, rate) == tool.trace("spike", "early", 3, rate)
    assert tool.trace("quiet", "to_limit", 0, rate) != tool.trace("quiet", "to_limit", 1, rate)
    assert all(r["prompt"] == tool.SHORT_PROMPT for r in tool.trace("quiet", "early", 0, rate))


def _cell(tps, first):
    return {"tokens_per_s": tps, "mean_first": first}


def test_the_rule_fixed_before_the_data_decides():
    flat = {o: _cell((1.0, 0.99, 1.01), (1.0, 0.99, 1.01)) for o in tool.ORDERS[1:]}
    report = {"cells": {"a": {"orders": flat}, "b": {"orders": flat}}}
    assert tool.analyse(report)["decision"].startswith("no headroom")
    report["cells"]["a"]["orders"] = {**flat, "ljf": _cell((1.06, 1.04, 1.08), (1.0, 0.99, 1.01))}
    assert tool.analyse(report)["decision"].startswith("fixed rule: ljf")
    report["cells"]["b"]["orders"] = {**flat, "ljf": _cell((0.95, 0.94, 0.96), (1.0, 0.99, 1.01))}
    assert tool.analyse(report)["decision"].startswith("states want different orders")
