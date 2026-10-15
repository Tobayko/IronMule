"""RSI4: the serving policy, its log, the dreamer's simulation, and the scheduler's use of it.
Model-free: no MLX, no timing claim."""
from __future__ import annotations

import json

from ironmule_product.serving_policy import (CostModel, ServingLog, ServingPolicy, dream, fit, load_policy,
                                             save_policy, simulate)
from tests.engine.test_product_group_scheduler import _Session, _StreamingTransport, _request, _service, _watch

COST = CostModel(prefill_s_per_token=0.002, round_s={1: 0.010, 2: 0.012, 4: 0.016, 8: 0.024})


def test_policy_round_trips_and_a_malformed_file_falls_back_to_serve8(tmp_path):
    assert load_policy(tmp_path) == ServingPolicy()
    save_policy(tmp_path, ServingPolicy(4, False, 150, "dream"), {"records": 9})
    assert load_policy(tmp_path) == ServingPolicy(4, False, 150, "dream")
    (tmp_path / "serving-policy.json").write_text(json.dumps({"max_width": 99, "join": True, "gather_ms": 0,
                                                             "source": "x"}))
    assert load_policy(tmp_path) == ServingPolicy()


def test_joining_cuts_the_late_arrivals_wait_in_the_simulation():
    arrivals = [(0.0, 50, 128), (0.5, 50, 128), (1.0, 50, 128), (1.5, 50, 128)]
    joined = simulate(arrivals, ServingPolicy(8, True, 0), COST)
    queued = simulate(arrivals, ServingPolicy(8, False, 0), COST)
    assert joined["max_ttft_s"] < queued["max_ttft_s"]
    assert joined["requests"] == queued["requests"] == 4


def test_width_one_serves_one_chat_at_a_time():
    arrivals = [(0.0, 10, 10), (0.0, 10, 10)]
    alone = simulate(arrivals, ServingPolicy(1, True, 0), COST)
    # second chat waits for the first's prefill and nine rounds, then its own prefill
    assert abs(alone["max_ttft_s"] - (0.02 + 9 * 0.010 + 0.02)) < 1e-9


def test_fit_reads_prefill_from_lone_requests_and_rounds_per_width():
    records = [{"arrive": 0, "start": 0, "first": 0.1, "end": 1.1, "prompt_tokens": 50, "new_tokens": 101, "width": 1},
               {"arrive": 0, "start": 0, "first": 0.3, "end": 2.3, "prompt_tokens": 50, "new_tokens": 101, "width": 4}]
    cost = fit(records)
    assert abs(cost.prefill_s_per_token - 0.002) < 1e-12
    assert abs(cost.round_s[1] - 0.01) < 1e-12 and abs(cost.round_s[4] - 0.02) < 1e-12


def test_dream_keeps_the_policy_without_enough_log_and_never_picks_a_worse_p95():
    current = ServingPolicy()
    assert dream([], current)[0] == current
    records = []
    for i in range(12):
        t = i * 0.7
        records.append({"arrive": t, "start": t, "first": t + 0.1, "end": t + 1.5, "prompt_tokens": 50,
                        "new_tokens": 128, "width": 1 + i % 4})
    policy, evidence = dream(records, current)
    assert evidence["chosen"]["p95_ttft_s"] <= evidence["current"]["p95_ttft_s"]
    assert evidence["chosen"]["tok_s"] >= 0.97 * evidence["current"]["tok_s"]


def test_the_log_skips_bad_lines(tmp_path):
    log = ServingLog(tmp_path)
    log.append({"arrive": 1, "start": 1, "first": 2, "end": 3, "prompt_tokens": 4, "new_tokens": 5, "width": 1})
    with (tmp_path / "serving-log.jsonl").open("a") as stream:
        stream.write("not json\n{\"arrive\": \"x\"}\n")
    assert len(log.read()) == 1


def test_the_log_stays_under_its_cap_and_keeps_the_newest_requests(tmp_path):
    log = ServingLog(tmp_path, max_bytes=2000)
    for i in range(200):
        log.append({"arrive": i, "start": i, "first": i, "end": i, "prompt_tokens": 1, "new_tokens": 1, "width": 1})
    assert (tmp_path / "serving-log.jsonl").stat().st_size <= 2000
    rows = log.read()
    assert rows[-1]["arrive"] == 199 and [r["arrive"] for r in rows] == sorted(r["arrive"] for r in rows)


def test_the_scheduler_caps_width_and_turns_joining_off_when_the_policy_says_so():
    backend = _StreamingTransport()
    service = _service(backend)
    service.policy = ServingPolicy(2, True, 0)
    assert service._batch_capacity() == 2  # backend offers 4
    service.policy = ServingPolicy(1, True, 0)
    assert service._batch_capacity() == 1
    service.policy = ServingPolicy(8, False, 0)
    first, waiting = _Session(_request("a"), None), _Session(_request("c"), None)
    service._pending.put(waiting)
    backend.delivered_to = lambda _rid: _watch(service, first)
    service._perform_batch([first])
    assert backend.joined == []  # the newcomer stays queued for the next group
    assert not service._pending.empty()


def test_a_completed_request_is_logged_with_its_group_width(tmp_path):
    backend = _StreamingTransport()
    service = _service(backend)
    service.serving_log = ServingLog(tmp_path)
    first, second = _Session(_request("a"), None), _Session(_request("b"), None)
    first.width = second.width = 2
    backend.delivered_to = lambda _rid: _watch(service, first)
    service._perform_batch([first, second])
    rows = ServingLog(tmp_path).read()
    assert len(rows) == 2 and {r["width"] for r in rows} == {2}
    assert all(r["arrive"] <= r["start"] <= r["first"] <= r["end"] for r in rows)
