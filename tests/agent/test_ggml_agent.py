"""AG10: the ggml engine's agent -- world model, simulation, dreaming, switch and revert, persistence. No llama.cpp."""

import math
import random

from ironmule.ggml_agent import GgmlAgent, StepCost, configs_from_measurement, dream, simulate

# Shaped after SEQ1's measurement on a 4-vCPU Xeon (Qwen3-4B Q8_0): one row about 90 ms, 16 rows 129 ms, 32 rows 185 ms.
NARROW = {"seqs": 16, "settings": {"flash_attn": True}, "seq_ctx": 2048, "step_one": 0.090, "step_full": 0.129,
          "prefill_per_1k": 2.5}
WIDE = {"seqs": 32, "settings": {"flash_attn": True, "kv_q8": True}, "seq_ctx": 2048, "step_one": 0.092,
        "step_full": 0.185, "prefill_per_1k": 2.6}


def heavy(n=400, gap=0.02, seed=1):
    rng = random.Random(seed)
    return [(rng.expovariate(1 / gap), rng.randint(20, 120), rng.randint(40, 200)) for _ in range(n)]


def light(n=400, gap=8.0, seed=2):
    rng = random.Random(seed)
    return [(rng.expovariate(1 / gap), rng.randint(20, 120), rng.randint(40, 200)) for _ in range(n)]


def test_step_cost_learns_a_changed_machine():
    cost = StepCost.from_measurement(16, 0.09, 0.129, 2.5)
    rng = random.Random(0)
    for _ in range(600):  # the true machine is twice as slow per row as measured
        rows, prompt, kv = rng.randint(1, 16), rng.choice([0, 0, 0, 80]), rng.randint(0, 3000)
        cost.update(rows, prompt, kv, 0.085 + 0.0052 * rows + 0.0025 * prompt + 0.00001 * kv)
    assert abs(cost.theta[1] - 0.0052) < 5e-4 and abs(cost.predict(8, 0, 1000) - (0.085 + 0.0416 + 0.01)) < 0.005


def test_simulation_counts_every_token_and_serves_in_order():
    cost = StepCost([0.1, 0.0, 0.0, 0.0])
    one = simulate(cost, 1, [(0.0, 10, 3), (0.0, 10, 3)])
    assert one["tokens"] == 6 and abs(one["seconds"] - 0.6) < 1e-9  # one at a time: 3 steps each
    two = simulate(cost, 2, [(0.0, 10, 3), (0.0, 10, 3)])
    assert two["tokens"] == 6 and abs(two["seconds"] - 0.3) < 1e-9 and two["tok_s"] > one["tok_s"]


def test_dreaming_prefers_the_wider_configuration_under_heavy_traffic_only():
    configs = {"w16": NARROW, "w32-kv8": WIDE}
    costs = {k: StepCost.from_measurement(v["seqs"], v["step_one"], v["step_full"], v["prefill_per_1k"])
             for k, v in configs.items()}
    busy = dream(configs, costs, "w16", heavy(), random.Random(0))["candidates"]["w32-kv8"]
    quiet = dream(configs, costs, "w16", light(), random.Random(0))["candidates"]["w32-kv8"]
    assert busy["low"] > 0.2 and busy["ttft_ratio"] < 1.0
    assert abs(quiet["gain"]) < 0.03  # a quiet server gains nothing from width


def make_agent(tmp_path, memory):
    agent = GgmlAgent(tmp_path, {"machine": "test"}, {"w16": NARROW, "w32-kv8": WIDE}, "w16")
    t = 0.0
    for gap, prompt, answer in memory:
        t += gap
        agent.observe_request(t, prompt, answer)
    return agent


def test_agent_switches_under_heavy_traffic_and_keeps_a_switch_the_live_steps_confirm(tmp_path):
    agent = make_agent(tmp_path, heavy(200))
    assert agent.idle() == "w32-kv8"
    agent.switched("w32-kv8")
    assert agent.current == "w32-kv8" and agent.trial is not None
    for rows in range(1, 33):  # live steps as measured
        agent.observe_step(rows, 0, 2000, 0.089 + 0.003 * rows)
    for gap, prompt, answer in heavy(100, seed=5):
        agent.observe_request(agent._last_arrival + gap, prompt, answer)
    assert agent.idle() is None and agent.trial is None and agent.current == "w32-kv8"


def test_agent_reverts_when_live_steps_show_the_wider_one_is_slow(tmp_path):
    agent = make_agent(tmp_path, heavy(200))
    for _ in range(20):  # the incumbent served as measured
        for rows in range(1, 17):
            agent.observe_step(rows, 0, rows * 100, 0.0874 + 0.0026 * rows)
    assert agent.idle() == "w32-kv8"
    agent.switched("w32-kv8")
    assert agent.idle() is None and agent.history[-1]["action"] == "watch"  # no traffic since the switch yet
    for _ in range(40):  # on this machine the 8-bit cache turns out to cost far more per row
        for rows in (4, 16, 32):
            agent.observe_step(rows, 0, 2000, 0.09 + 0.012 * rows)
    for gap, prompt, answer in heavy(100, seed=6):
        agent.observe_request(agent._last_arrival + gap, prompt, answer)
    assert agent.idle() == "w16" and "w32-kv8" in agent.rejected
    agent.switched("w16")
    assert agent.idle() is None  # rejected: not tried again before RETRY_AFTER requests


def test_agent_stays_on_a_quiet_server_and_its_state_survives_a_restart(tmp_path):
    agent = make_agent(tmp_path, light(120))
    assert agent.idle() is None and agent.history[-1]["action"] == "stay"
    again = GgmlAgent(tmp_path, {"machine": "test"}, {"w16": NARROW, "w32-kv8": WIDE}, "w16")
    assert again.requests == 120 and len(again.memory) == 120
    other = GgmlAgent(tmp_path, {"machine": "other"}, {"w16": NARROW, "w32-kv8": WIDE}, "w16")
    assert other.requests == 0  # another binding starts fresh


def test_decisions_are_microseconds():
    import time

    agent = GgmlAgent(None, {"m": 1}, {"w16": NARROW, "w32-kv8": WIDE}, "w16")
    started = time.perf_counter()
    for _ in range(10000):
        _ = agent.current
        agent.observe_step(16, 0, 1000, 0.1)
    assert (time.perf_counter() - started) / 10000 < 20e-6


def test_configs_from_measurement():
    rows = [{"settings": {"flash_attn": True}, "step_s": {"8": 0.11, "16": 0.13}, "step_one_s": {"8": 0.09, "16": 0.09}},
            {"settings": {"flash_attn": True, "kv_q8": True}, "step_s": {"32": 0.19}}]
    configs = configs_from_measurement(rows, 2048)
    assert set(configs) == {"w8-fa", "w16-fa", "w32-fa-kv8"} and configs["w32-fa-kv8"]["step_one"] == 0.19
    long = [{"settings": {"flash_attn": False}, "step_s": {"16": 0.21}, "step_long_s": {"16": 0.29}, "long_ctx": 512,
             "short_ctx": 64}]
    assert abs(configs_from_measurement(long, 2048)["w16"]["kv_per_1k"] - 0.08 / (448 * 16 / 1000)) < 1e-9


def test_dreaming_chooses_attention_by_the_traffic_context():
    """Shaped after KV1 on a 4-vCPU Xeon (Qwen3-4B, 16 sequences): flash attention is cheaper per step at short
    contexts, llama.cpp's matmul attention cheaper per attended token."""
    fa = {"seqs": 16, "settings": {"flash_attn": True}, "seq_ctx": 2048, "step_one": 0.093, "step_full": 0.157,
          "kv_per_1k": 0.021}
    mm = {"seqs": 16, "settings": {}, "seq_ctx": 2048, "step_one": 0.116, "step_full": 0.213, "kv_per_1k": 0.013}
    configs = {"w16-fa": fa, "w16": mm}
    costs = {k: StepCost.from_measurement(v["seqs"], v["step_one"], v["step_full"], 0.0, v["kv_per_1k"])
             for k, v in configs.items()}
    rng = random.Random(3)
    short = [(rng.expovariate(1 / 0.02), rng.randint(20, 60), rng.randint(40, 120)) for _ in range(300)]
    long = [(rng.expovariate(1 / 0.02), rng.randint(1500, 1800), rng.randint(40, 120)) for _ in range(300)]
    assert dream(configs, costs, "w16-fa", short, random.Random(0))["candidates"]["w16"]["gain"] < 0
    assert dream(configs, costs, "w16-fa", long, random.Random(0))["candidates"]["w16"]["low"] > 0.03


def test_the_bridge_feeds_the_agent_and_switches_the_engine_between_runs(tmp_path, monkeypatch):
    """The ggml bridge with a stand-in engine: every served request enters the traffic memory, every step the world
    model, and the agent's switch rebuilds the engine's context under the run lock."""
    import threading
    import time

    from ironmule_product.ggml_bridge import GgmlBridge

    class Engine:
        seqs, settings, gpu_layers, observer = 16, {"flash_attn": True}, 0, None

        def __init__(self):
            self.reconfigured = []

        def run(self, jobs, admit=None):
            for job in jobs:
                self.observer(len(jobs), len(job.ids), 100, 0.1)
                job.tokens, job.reason = [5] * job.limit, "length"
            return jobs

        def reconfigure(self, seqs, settings):
            self.reconfigured.append((seqs, dict(settings)))
            self.seqs, self.settings = seqs, dict(settings)

        def summary(self):
            return {"steps": 0, "speculation": "off"}

    class Tokenizer:
        def decode(self, ids):
            return "x" * len(ids)

    monkeypatch.setattr(threading.Thread, "start", lambda self: None)  # no idle thread: the test calls the agent
    bridge = GgmlBridge.__new__(GgmlBridge)
    bridge._engine, bridge._tokenizer, bridge._eos = Engine(), Tokenizer(), (0,)
    bridge._gguf_sha256, bridge._speculate = "0" * 64, False
    bridge._run_lock, bridge._last_run, bridge._agent, bridge._stop = threading.Lock(), time.monotonic(), None, None
    bridge._gate = {"agent": {"configs": {"w16-fa": NARROW, "w32-fa-kv8": WIDE}, "incumbent": "w16-fa",
                              "root": str(tmp_path)}}
    bridge._stop = threading.Event()
    bridge._start_agent()
    assert bridge._engine.observer is not None and bridge._agent.current == "w16-fa"
    for _ in range(3):
        bridge.generate_many([[1, 2, 3]] * 20, 50)
    assert bridge._agent.requests == 60 and len(bridge._agent._steps) == 60
    bridge._switch("w32-fa-kv8")
    assert bridge._engine.reconfigured == [(32, {"flash_attn": True, "kv_q8": True})]
    assert bridge.counters()["ggml_agent"]["serving"] == "w32-fa-kv8"


def test_a_configuration_that_never_served_does_not_dream_its_prefill_free(tmp_path):
    """Seen live on 2026-11-13: the served configuration learned what a prompt costs (3.3 s per 1,000 tokens on that
    CPU) and the others, never served, priced it at zero, so the dream preferred any of them. With the startup
    measurement's prefill as an anchor, equal configurations stay equal."""
    same = {"seqs": 16, "settings": {"flash_attn": True}, "seq_ctx": 2048, "step_one": 0.093, "step_full": 0.134,
            "kv_per_1k": 0.01, "prefill_per_1k": 3.3}
    other = {**same, "settings": {}}
    agent = GgmlAgent(tmp_path, {"m": "live"}, {"w16-fa": same, "w16": other}, "w16-fa")
    rng = random.Random(4)
    t = 0.0
    for wave in range(10):  # waves of 12 requests with a long system prompt, all at once
        for k in range(12):
            agent.observe_request(t, 520, 48, 520 if wave == 0 else 30)
        for rows in range(1, 13):
            agent.observe_step(rows, 0, rows * 560, 0.09 + 0.003 * rows + 0.01 * rows * 0.56)
        agent.observe_step(0, 6000, 0, 3.3 * 6 * math.exp(rng.gauss(0, 0.03)))
        t += 10.0
    assert agent.idle() is None and agent.history[-1]["action"] == "stay"


def record_busy_window(agent, *, prompt, step, count=16):
    start = (agent._last_arrival or 0.0) + 10
    for i in range(count):
        arrival = start + i * 0.001
        agent.observe_request(arrival, prompt, 8, first=arrival + step, finished=arrival + 4 * step)


def test_a_real_regression_with_matching_traffic_withdraws_a_trial_early(tmp_path):
    agent = make_agent(tmp_path, heavy(47))
    assert agent.idle() is None
    record_busy_window(agent, prompt=1024, step=0.1)
    assert agent.idle() == "w32-kv8"
    agent.switched("w32-kv8")
    record_busy_window(agent, prompt=1018, step=0.3)
    assert agent.idle() == "w16"  # 16 requests after switching rather than the old 64
    assert agent.history[-1]["observed"]["rate_ratio"] < 0.5
    scope = agent.history[-1]["observed"]["scope"]
    assert agent.idle() == "w16"  # the engine was busy at the previous reconfiguration attempt
    agent.switched("w16")
    again = GgmlAgent(tmp_path, {"machine": "test"}, agent.configs, "w16")
    assert again._observed_regression(scope, "w32-kv8", "w16")
    assert not again._observed_regression(scope, "w16", "w32-kv8")
    assert not again.rejected  # other workloads remain eligible


def test_trial_cannot_regress_on_unmatched_or_untimed_traffic(tmp_path):
    agent = make_agent(tmp_path, heavy(47))
    assert agent.idle() is None
    record_busy_window(agent, prompt=50, step=0.1)
    assert agent.idle() == "w32-kv8"
    agent.switched("w32-kv8")
    record_busy_window(agent, prompt=1024, step=0.5)
    assert agent.idle() is None and agent.trial is not None
    for i in range(16):
        agent.observe_request(agent._last_arrival + 0.1, 50, 8)  # old or synthetic caller
    assert agent.idle() is None and agent.trial is not None
    assert len(agent.observations) == 2


def test_rollback_waits_for_a_new_consistent_trace_before_further_exploration(tmp_path, monkeypatch):
    import ironmule.ggml_agent as module

    agent = make_agent(tmp_path, heavy(47))
    assert agent.idle() is None
    record_busy_window(agent, prompt=1024, step=0.1)
    assert agent.idle() == "w32-kv8"
    agent.switched("w32-kv8")
    record_busy_window(agent, prompt=1024, step=0.3)
    assert agent.idle() == "w16"
    # A restart between deciding to revert and physically switching restores the qualified incumbent.
    agent = GgmlAgent(tmp_path, {"machine": "test"}, agent.configs, "w16")
    assert agent.current == "w16" and agent.recovery["matching"] == 0
    seen = []
    def consistent(_configs, _prices, _current, memory, _rng):
        seen.append([m[1] for m in memory])
        return {"candidates": {"w32-kv8": {"low": 0.10, "ttft_ratio": 0.90}}}
    monkeypatch.setattr(module, "dream", consistent)
    for prompt in [1024, 50, 50, 50]:
        record_busy_window(agent, prompt=prompt, step=0.1)
        assert agent.idle() is None and not seen
    record_busy_window(agent, prompt=50, step=0.1)
    assert agent.idle() == "w32-kv8"
    assert seen == [[50] * 64]  # prior documents cannot qualify a switch under the new chat traffic


def test_low_volume_after_rollback_supplies_fresh_arrivals_to_the_dream(tmp_path, monkeypatch):
    import ironmule.ggml_agent as module

    agent = make_agent(tmp_path, [])
    agent.recovery = {"serving": "w16", "since": 0, "scope": None, "matching": 0}
    agent.requests = 48
    agent.memory.extend(heavy(48))  # existing pre-recovery experience
    agent.recovery["since"] = agent.requests
    seen = []
    def fresh(_configs, _prices, _current, memory, _rng):
        seen.append(len(memory))
        return {"candidates": {}}
    monkeypatch.setattr(module, "dream", fresh)
    for i in range(64):
        agent.observe_request(1 + i * 10, 20 + i, 8)
        assert agent.idle() is None
    assert seen == [64]
    history = list(agent.history)
    assert agent.idle() is None and agent.history == history
    assert not agent.observations  # quiet traffic cannot directly promote or withdraw a configuration


def test_continually_changing_load_eventually_leaves_recovery_without_old_evidence(tmp_path, monkeypatch):
    import ironmule.ggml_agent as module

    agent = make_agent(tmp_path, heavy(80))
    agent.idle()
    agent.trial = agent.pending_switch = None
    agent.recovery = {"serving": agent.current, "since": agent.requests, "scope": None, "matching": 0}
    seen = []
    def fresh(_configs, _prices, _current, memory, _rng):
        seen.append(len(memory))
        return {"candidates": {}}
    monkeypatch.setattr(module, "dream", fresh)
    for i in range(module.RETRY_AFTER // 16):
        record_busy_window(agent, prompt=40 if i % 2 else 1200, step=0.1)
        agent.idle()
    assert seen == [module.RETRY_AFTER]
    assert agent.recovery is None


def test_gpu_burst_must_repay_actual_rebuilds_within_the_observed_window(tmp_path):
    # Four requests can run faster side by side after widening. Restoring the
    # incumbent consumes that saving for a small burst; 32 requests repay it.
    configs = {
        "one": {"seqs": 1, "step_one": .01, "step_full": .01, "context_build_s": .109, "context_release_s": .001},
        "four": {"seqs": 4, "step_one": .01, "step_full": .01, "context_build_s": .009, "context_release_s": .001},
    }
    agent = GgmlAgent(tmp_path, {"machine": "gpu", "gpu_layers": 999}, configs, "one", price_switches=True)
    for batch in range(12):
        for _ in range(4):
            agent.observe_request(1, 8, 3)
        assert agent.idle() is None
    forecast = agent.history[-1]["dream"]
    assert forecast["window_requests"] == 4
    assert math.isclose(forecast["switch_costs_s"]["four"]["enter"], .01)
    assert math.isclose(forecast["switch_costs_s"]["four"]["return"], .11)
    assert forecast["candidates"]["four"]["low"] < 0
    for _ in range(32):
        agent.observe_request(1, 8, 3)
    assert agent.idle() == "four"
    agent.switched("four", seconds=.08)
    restored = GgmlAgent(tmp_path, agent.binding, configs, "one", price_switches=True)
    assert restored.current == "one"  # restore the incumbent after an unconfirmed trial
    assert restored._switch_cost("one", "four") == .08
    assert math.isclose(restored._switch_cost("four", "one"), .11)
    pooled = GgmlAgent(tmp_path, agent.binding, configs, "one", shared_cap=16, price_switches=True)
    assert pooled.requests == restored.requests
    assert pooled._switch_cost("one", "four") is None
    assert pooled._switch_cost("four", "one") is None


def test_gpu_existing_unpriced_plans_retain_their_dream_and_unknown_costs(tmp_path):
    configs = {"one": {"seqs": 1, "step_one": .01, "step_full": .01},
               "four": {"seqs": 4, "step_one": .01, "step_full": .01}}
    agent = GgmlAgent(tmp_path, {"machine": "gpu", "gpu_layers": 999}, configs, "one", price_switches=True)
    for _ in range(48):
        agent.observe_request(1, 8, 3)
    assert agent.idle() == "four"
    forecast = agent.history[-1]["dream"]
    assert forecast["switch_costs_s"]["four"] == {"enter": None, "return": None}
    assert forecast["window_requests"] == 32
    agent.switched("four", seconds=0)
    assert agent._switch_cost("one", "four") is None
    agent.switched("one", seconds=.04)
    assert agent._switch_cost("four", "one") == .04
    assert agent._switch_cost("one", "four") is None

def test_product_gpu_uses_the_standard_window_with_measured_rebuild_costs(tmp_path):
    from types import SimpleNamespace

    from ironmule_product.ggml_bridge import GgmlBridge

    configs = {name: {"seqs": width, "step_one": .01, "step_full": .01,
                      "context_build_s": 10.0, "context_release_s": 1.0}
               for name, width in (("one", 1), ("four", 4))}
    bridge = GgmlBridge.__new__(GgmlBridge)
    bridge._engine = SimpleNamespace(gpu_layers=999, seqs=1, settings={})
    bridge._gguf_sha256 = "a" * 64
    bridge._gate = {"agent": {"configs": configs, "incumbent": "one", "root": str(tmp_path)}}
    bridge._start_agent()
    agent = bridge._agent
    assert agent.status()["price_switches"] is False
    for batch in range(12):
        for _ in range(4):
            agent.observe_request(1, 8, 3)
        decision = agent.idle()
    assert decision == "four"
    forecast = agent.history[-1]["dream"]
    assert forecast["window_requests"] == 32 and "switch_costs_s" not in forecast
    agent.switched("four", seconds=.25)
    restored = GgmlAgent(tmp_path, agent.binding, configs, "one")
    assert restored.switch_seconds == {"one": {"four": .25}}
    challenger = GgmlAgent(tmp_path, agent.binding, configs, "one", price_switches=True)
    assert challenger.path != agent.path and challenger.requests == 0
