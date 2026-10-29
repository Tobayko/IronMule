"""RL4: a model-based RL agent that places and orders each request on the HW1 pool (BACKLOG RL4).

RL2-H measured fixed arms. A policy that decides per request, from that request's output limit and
prompt length and from the work outstanding on each card, is not bounded by that measurement. Training
it on hardware would cost GPU hours per episode, so it learns in a discrete-event model of the pool,
fitted to the per-request times `tools/rl2_headroom.py` records on the T4s, and goes back to hardware
only through the gates BACKLOG RL4 fixes.

The model follows the product, not an idealised server:

  - each card runs one group at a time and takes the next group when the last one ended; a group holds
    up to `capacity` waiting requests (the scheduler's width, 8 at the reference); nothing joins a
    running group (SERVE8 is off);
  - the members are prefilled one after another, each first token streams at the end of its own
    prefill (`prefill_s + prefill_s_per_token x prompt tokens`);
  - decode runs in rounds of at most four sessions (`ironmule.executor.MAX_GROUP_WIDTH`), rotating
    served sessions to the back; a round of k sessions takes `step_s + step_s_per_session x (k - 1)`;
  - every member's last token and completion are released when the whole group ends (the worker
    checks the streamed tokens against the run's result, BACKLOG HOL1).

The agent is a softmax policy over two decisions: the card for each arriving request, and `fifo` or `sjf`
at each group start. REINFORCE with a per-state running-mean baseline; the return is minus the mean of
the live learner's request cost (`tools/rl2_headroom.py:request_cost`). Std only.

    python tools/rl4_agent.py fit   --report H3.json [--report H2.json] --out model.json
    python tools/rl4_agent.py gate  --model model.json --report H3.json
    python tools/rl4_agent.py train --model model.json --out policy.json
    python tools/rl4_agent.py judge --model model.json --policy policy.json --noise 0.035
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl2_headroom as rl2  # noqa: E402  (the harness this agent is judged by)

ENGINE_WIDTH = 4                 # ironmule.executor.MAX_GROUP_WIDTH
TRAIN_SEED0 = 10_000             # training traffic seeds; the hardware traces use indices 0-34 (rl2_headroom)
HELD_OUT_SEED0 = 50_000          # the simulator gate's held-out traffic
FIXED_ARMS = ("shortest/w8/g0", "shortest/w8/g0/sjf", "segregate/w8/g0", "segregate/w8/g0/sjf",
              "spill/w8/g0", "spill/w8/g0/sjf")
SIMULATOR_ONLY_ARMS = ("leastwork/w8/g0", "leastwork/w8/g0/sjf")  # compared in the model, never on hardware
# RL6: every fixed width a width-learning policy competes with (all run on hardware too)
WIDTH_ARMS = tuple(f"shortest/w{w}/g0{o}" for w in (1, 2, 4) for o in ("", "/sjf"))


@dataclass(frozen=True)
class PoolModel:
    prefill_s: float = 0.05
    prefill_s_per_token: float = 2e-4
    step_s: float = 0.025
    step_s_per_session: float = 0.004
    cards: int = 2
    short_prompt_tokens: int = 24
    long_prompt_tokens: int = 1030

    def prefill(self, long_prompt: bool) -> float:
        tokens = self.long_prompt_tokens if long_prompt else self.short_prompt_tokens
        return self.prefill_s + self.prefill_s_per_token * tokens

    def step(self, sessions: int) -> float:
        return self.step_s + self.step_s_per_session * (sessions - 1)


@dataclass
class _Request:
    index: int
    arrive: float
    max_tokens: int
    long_prompt: bool
    first: float | None = None
    end: float | None = None


def group_timeline(model: PoolModel, members: list[_Request], start: float) -> float:
    """Fill in each member's first-token time; return when the group ends (all completions released)."""
    t = start
    remaining = {}
    for member in members:
        t += model.prefill(member.long_prompt)
        member.first = t
        remaining[member.index] = member.max_tokens - 1
    active = [m for m in members if remaining[m.index] > 0]
    while active:
        served = active[:ENGINE_WIDTH]
        t += model.step(len(served))
        for member in served:
            remaining[member.index] -= 1
        active = active[len(served):] + [m for m in served if remaining[m.index] > 0]
    for member in members:
        member.end = t
    return t


class _Card:
    def __init__(self) -> None:
        self.waiting: list[_Request] = []
        self.running: list[_Request] = []
        self.free_at = 0.0

    def outstanding(self) -> list[_Request]:
        return self.waiting + self.running


Placer = Callable[[_Request, list[_Card], float], int]
Orderer = Callable[[list[_Request], float], str]


def simulate(model: PoolModel, items: list[dict[str, Any]], place: Placer, order: Orderer,
             capacity: int = 8, width: Callable[[list[_Request], float], int] | None = None) -> list[_Request]:
    """Replay one trace (rl2_headroom's item format) through the pool model under `place` and `order`;
    `width`, when given, picks each group's width (at most `capacity`) from the ordered queue (RL6)."""
    requests = [_Request(i, item["offset_s"], item["max_tokens"], bool(item.get("long")))
                for i, item in enumerate(items)]
    cards = [_Card() for _ in range(model.cards)]
    pending = sorted(requests, key=lambda r: (r.arrive, r.index))

    def start(card: _Card, now: float) -> None:
        if card.running or not card.waiting:
            return
        if order(card.waiting, now) == "sjf":  # as the harness's OrderedService: aged requests by arrival
            card.waiting.sort(key=lambda r: (now - r.arrive <= rl2.SJF_AGE_S,
                                             r.max_tokens if now - r.arrive <= rl2.SJF_AGE_S else 0, r.arrive))
        size = capacity if width is None else min(capacity, width(card.waiting, now))
        card.running, card.waiting = card.waiting[:size], card.waiting[size:]
        card.free_at = group_timeline(model, card.running, now)

    while pending or any(card.running for card in cards):
        busy = [card for card in cards if card.running]
        next_free = min((card.free_at for card in busy), default=math.inf)
        if pending and pending[0].arrive <= next_free:
            request = pending.pop(0)
            now = request.arrive
            card = cards[place(request, cards, now)]
            card.waiting.append(request)
            start(card, now)
        else:
            card = min(busy, key=lambda c: c.free_at)
            now = card.free_at
            card.running = []
            start(card, now)
    return requests


def cost(requests: list[_Request], deadline_s: float = 120.0) -> float:
    return statistics.fmean(rl2.request_cost({"outcome": "ok", "first_s": r.first - r.arrive,
                                              "end_s": r.end - r.arrive}, deadline_s)
                            for r in requests)


# ---------------------------------------------------------------- fixed arms in the model

def _is_long(request: _Request) -> bool:
    return request.max_tokens >= rl2.LONG_TOKENS


def fixed_placer(placement: str, width: int = 8) -> Placer:
    def place(request: _Request, cards: list[_Card], now: float) -> int:
        loads = [len(card.outstanding()) for card in cards]
        shortest = min(range(len(cards)), key=lambda i: (loads[i], i))
        if placement == "leastwork":  # a simulator-only rule: the fewest output tokens still owed
            return min(range(len(cards)), key=lambda i: (sum(r.max_tokens for r in cards[i].outstanding()), i))
        if placement == "shortest" or len(cards) < 2:
            return shortest
        if placement == "fill":  # as the harness: the lowest-index card with room, else the shortest
            return next((i for i in range(len(cards)) if loads[i] < width), shortest)
        preferred = len(cards) - 1 if _is_long(request) else 0
        if placement == "spill":
            other = min((i for i in range(len(cards)) if i != preferred), key=lambda i: (loads[i], i))
            if loads[preferred] - loads[other] >= rl2.SPILL_GAP:
                return other
        return preferred
    return place


def fixed_orderer(order: str) -> Orderer:
    return lambda waiting, now: order


def arm_policy(name: str) -> tuple[Placer, Orderer, int]:
    placement, width, gather = rl2.parse_arm(name)
    if gather:
        raise ValueError(f"the pool model has no gather wait: {name}")
    return fixed_placer(placement, width), fixed_orderer(rl2.arm_order(name)), width


# ---------------------------------------------------------------- the learned policy

# The features and the policy live in the harness, which the Kaggle kernel embeds whole.
placement_features = rl2.placement_features
order_features = rl2.order_features
PLACEMENT_SIZE = rl2.PLACEMENT_FEATURES
ORDER_SIZE = rl2.ORDER_FEATURES
Policy = rl2.LearnedPolicy


def traffic(state: int, seed: int, capacity_rps: float) -> list[dict[str, Any]]:
    """RL2-H2's state generators at a seed outside the hardware traces."""
    return rl2.trace_h2(state, seed, capacity_rps)


def train(model: PoolModel, capacity_rps: float, *, episodes: int = 6000, lr: float = 0.05,
          seed: int = 0, widths: bool = False) -> tuple[Policy, list[float]]:
    """REINFORCE with a running-mean baseline per state; returns the policy and the learning curve.
    With `widths` (RL6) the policy also learns each group's width."""
    rng = random.Random(seed)
    policy = Policy.zero(widths)
    baseline = [None] * len(rl2.H2_STATES)
    curve = []
    for episode in range(episodes):
        state = episode % len(rl2.H2_STATES)
        items = traffic(state, TRAIN_SEED0 + episode, capacity_rps)
        grad_p = [0.0] * PLACEMENT_SIZE
        grad_o = [0.0] * ORDER_SIZE
        grad_w = [0.0] * (len(rl2.WIDTH_CHOICES) * ORDER_SIZE)

        def place(request: _Request, cards: list[_Card], now: float) -> int:
            p = policy.card_probabilities(request, cards)
            choice = rng.choices(range(len(p)), weights=p)[0]
            feats = [placement_features(request, cards, c) for c in range(len(cards))]
            for c, f in enumerate(feats):  # d log softmax = f(choice) - sum_c p_c f(c)
                for k in range(PLACEMENT_SIZE):
                    grad_p[k] += ((1.0 if c == choice else 0.0) - p[c]) * f[k]
            return choice

        def order(waiting: list[_Request], now: float) -> str:
            q = policy.sjf_probability(waiting)
            sjf = rng.random() < q
            f = order_features(waiting)
            for k in range(ORDER_SIZE):
                grad_o[k] += ((1.0 if sjf else 0.0) - q) * f[k]
            return "sjf" if sjf else "fifo"

        def width(waiting: list[_Request], now: float) -> int:
            p = policy.width_probabilities(waiting)
            choice = rng.choices(range(len(p)), weights=p)[0]
            f = order_features(waiting)
            for c in range(len(p)):  # d log softmax over widths, one weight row per width
                for k in range(ORDER_SIZE):
                    grad_w[c * ORDER_SIZE + k] += ((1.0 if c == choice else 0.0) - p[c]) * f[k]
            return rl2.WIDTH_CHOICES[choice]

        value = -cost(simulate(model, items, place, order, width=width if widths else None))
        if baseline[state] is None:
            baseline[state] = value
        advantage = (value - baseline[state]) / max(1e-6, abs(baseline[state]))
        baseline[state] += 0.05 * (value - baseline[state])
        for k in range(PLACEMENT_SIZE):
            policy.placement[k] += lr * advantage * grad_p[k]
        for k in range(ORDER_SIZE):
            policy.order[k] += lr * advantage * grad_o[k]
        if widths:
            for k in range(len(grad_w)):
                policy.width[k] += lr * advantage * grad_w[k]
        curve.append(-value)
    return policy, curve


VALIDATION_SEED0 = 30_000        # picks among training seeds; disjoint from training and held-out traffic


def train_best(model: PoolModel, capacity_rps: float, *, seeds: int = 4, episodes: int = 12000,
               lr: float = 0.05, validation_traces: int = 60, widths: bool = False) -> tuple[Policy, dict[str, Any]]:
    """Train one policy per seed and keep the one with the lowest weighted cost on validation traffic."""
    runs = []
    for seed in range(seeds):
        policy, curve = train(model, capacity_rps, episodes=episodes, lr=lr, seed=seed, widths=widths)
        costs = evaluate(model, capacity_rps, {"policy": _learned(policy)}, validation_traces,
                         seed0=VALIDATION_SEED0)["policy"]
        runs.append((sum(w * c for w, c in zip(rl2.WEIGHTS, costs)), seed, policy, curve))
    best = min(runs, key=lambda run: (run[0], run[1]))
    return best[2], {"validation_cost_per_seed": [run[0] for run in runs], "chosen_seed": best[1],
                     "first_tenth_cost": statistics.fmean(best[3][:max(1, len(best[3]) // 10)]),
                     "last_tenth_cost": statistics.fmean(best[3][-max(1, len(best[3]) // 10):])}


def _learned(policy: Policy) -> tuple[Placer, Orderer, int, Any]:
    place, order = policy.greedy()
    return place, order, 8, policy.greedy_width()


def evaluate(model: PoolModel, capacity_rps: float, policies: dict[str, tuple],
             traces: int = 200, seed0: int = HELD_OUT_SEED0) -> dict[str, list[float]]:
    """Mean cost per state for each policy (place, order, capacity[, width]) on the same held-out traffic."""
    out: dict[str, list[float]] = {name: [] for name in policies}
    for state in range(len(rl2.H2_STATES)):
        batch = [traffic(state, seed0 + t, capacity_rps) for t in range(traces)]
        for name, (place, order, capacity, *width) in policies.items():
            out[name].append(statistics.fmean(cost(simulate(model, items, place, order, capacity, *width))
                                              for items in batch))
    return out


def judge(model: PoolModel, capacity_rps: float, policy: Policy, noise: float, traces: int = 200) -> dict[str, Any]:
    """The simulator gate: the learned policy against every fixed arm on held-out traffic; a policy that
    picks widths (RL6) also faces every fixed width."""
    fixed = FIXED_ARMS + (WIDTH_ARMS if policy.width is not None else ())
    arms = {name: arm_policy(name) for name in fixed + SIMULATOR_ONLY_ARMS}
    arms["learned"] = _learned(policy)
    per_state = evaluate(model, capacity_rps, arms, traces)
    weighted = {name: sum(w * c for w, c in zip(rl2.WEIGHTS, costs)) for name, costs in per_state.items()}
    best_fixed = min(fixed, key=lambda name: (weighted[name], name))
    gain = 1 - weighted["learned"] / weighted[best_fixed]
    best_rule = min(SIMULATOR_ONLY_ARMS, key=lambda name: (weighted[name], name))
    return {"per_state": per_state, "weighted": weighted, "best_fixed": best_fixed, "gain_vs_best_fixed": gain,
            "noise": noise, "passes": gain > noise,
            # information only, not the gate: does learning add to the simplest rule it resembles?
            "best_simulator_only_rule": best_rule, "gain_vs_that_rule": 1 - weighted["learned"] / weighted[best_rule]}


# ---------------------------------------------------------------- fitting and the fidelity gate

def _cells(report: dict[str, Any]) -> dict[tuple[int, str], list[list[dict[str, Any]]]]:
    """Measured first runs per (state, arm): each a list of request records with their trace items."""
    run = report.get("rl2h", report)
    plan = run.get("plan") or ("h1" if run.get("rates") else None)  # run 1's report predates `plan`
    if plan not in ("h1", "h2", "h3", "h4"):
        raise ValueError("the pool model is fitted to rl2_headroom's runs (plans h1 to h4)")
    capacity = run["calibration"]["capacity_rps"]
    out: dict[tuple[int, str], list[list[dict[str, Any]]]] = {}
    for row in run["rows"]:
        if row["run"] != 0 or rl2.parse_arm(row["arm"])[2] or rl2.LEARNED == row["arm"]:  # no gather wait in the model
            continue
        items = ([{**item, "long": False} for item in rl2.trace(row["level"], row["trace"], run["rates"][row["level"]])]
                 if plan == "h1" else rl2.trace_h2(row["level"], row["trace"], capacity))
        out.setdefault((row["level"], row["arm"]), []).append(
            [{**item, **request} for item, request in zip(items, row["requests"])])
    return out


def _measured_cost(cell: list[list[dict[str, Any]]], deadline_s: float = 120.0) -> float:
    return statistics.fmean(statistics.fmean(rl2.request_cost(r, deadline_s) for r in trace) for trace in cell)


def _measured_times(cell: list[list[dict[str, Any]]]) -> tuple[float, float]:
    """Mean first-content and mean end time of the cell's completed requests."""
    done = [r for trace in cell for r in trace if r.get("outcome") == "ok" and r.get("first_s") is not None]
    return statistics.fmean(r["first_s"] for r in done), statistics.fmean(r["end_s"] for r in done)


def _simulated_times(model: PoolModel, arm: str, cell: list[list[dict[str, Any]]]) -> tuple[float, float]:
    place, order, capacity = arm_policy(arm)
    done = [r for trace in cell for r in simulate(model, trace, place, order, capacity)]
    return statistics.fmean(r.first - r.arrive for r in done), statistics.fmean(r.end - r.arrive for r in done)


def _simulated_cost(model: PoolModel, arm: str, cell: list[list[dict[str, Any]]]) -> float:
    place, order, capacity = arm_policy(arm)
    return statistics.fmean(cost(simulate(model, trace, place, order, capacity)) for trace in cell)


def _nelder_mead(loss: Callable[[list[float]], float], x0: list[float], scale: float = 0.4,
                 evaluations: int = 400) -> tuple[list[float], float]:
    """Plain Nelder-Mead (reflection 1, expansion 2, contraction 0.5, shrink 0.5)."""
    simplex = [list(x0)] + [[v + (scale if i == j else 0.0) for j, v in enumerate(x0)] for i in range(len(x0))]
    values = [loss(x) for x in simplex]
    used = len(simplex)
    while used < evaluations:
        order = sorted(range(len(simplex)), key=lambda i: values[i])
        simplex, values = [simplex[i] for i in order], [values[i] for i in order]
        centroid = [statistics.fmean(x[k] for x in simplex[:-1]) for k in range(len(x0))]
        worst = simplex[-1]
        reflected = [c + (c - w) for c, w in zip(centroid, worst)]
        fr = loss(reflected)
        used += 1
        if fr < values[0]:
            expanded = [c + 2 * (c - w) for c, w in zip(centroid, worst)]
            fe = loss(expanded)
            used += 1
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = [c + 0.5 * (w - c) for c, w in zip(centroid, worst)]
            fc = loss(contracted)
            used += 1
            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:
                simplex = [simplex[0]] + [[b + 0.5 * (v - b) for b, v in zip(simplex[0], x)] for x in simplex[1:]]
                values = [values[0]] + [loss(x) for x in simplex[1:]]
                used += len(simplex) - 1
    best = min(range(len(simplex)), key=lambda i: values[i])
    return simplex[best], values[best]


FIT_PARAMETERS = ("prefill_s", "prefill_s_per_token", "step_s", "step_s_per_session")


def fit(reports: Iterable[dict[str, Any]], start: PoolModel | None = None, evaluations: int = 400,
        restarts: int = 2) -> tuple[PoolModel, float]:
    """Least squared log error between simulated and measured mean first-content and end times per cell,
    by Nelder-Mead over the four timing parameters in log space, restarted from its own best."""
    cells = {}
    for report in reports:
        for key, cell in _cells(report).items():
            cells[(id(report),) + key] = (key[1], cell, _measured_times(cell))
    base = start or PoolModel()

    def model_at(x: list[float]) -> PoolModel:
        return replace(base, **{name: math.exp(v) for name, v in zip(FIT_PARAMETERS, x)})

    def loss(x: list[float]) -> float:
        m = model_at(x)
        errors = []
        for arm, cell, measured in cells.values():
            simulated = _simulated_times(m, arm, cell)
            errors += [(math.log(s) - math.log(t)) ** 2 for s, t in zip(simulated, measured)]
        return statistics.fmean(errors)

    x = [math.log(getattr(base, name)) for name in FIT_PARAMETERS]
    best = math.inf
    for _ in range(restarts + 1):
        x, value = _nelder_mead(loss, x, evaluations=evaluations)
        if value >= best - 1e-12:
            break
        best = value
    return model_at(x), best


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return 1.0
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def fidelity(model: PoolModel, report: dict[str, Any], concordance_min: float = 0.9,
             error_max: float = 0.25) -> dict[str, Any]:
    """BACKLOG RL4's fidelity gate on one report: every cell's mean cost within `error_max`, and in each
    state the simulator orders at least `concordance_min` of the arm pairs that the hardware separated
    by more than the report's A/A noise (pairs closer than that have no measured order to agree with)."""
    analysis = rl2.analyse(report.get("rl2h", report), resamples=20)  # only its A/A noise is used
    noise = analysis.get("aa_noise", 0.0) if analysis.get("valid") else 0.0
    cells = _cells(report)
    states = sorted({state for state, _ in cells})
    result: dict[str, Any] = {"noise": noise, "states": {}, "passes": True}
    for state in states:
        arms = sorted(arm for s, arm in cells if s == state)
        measured = [_measured_cost(cells[state, arm]) for arm in arms]
        simulated = [_simulated_cost(model, arm, cells[state, arm]) for arm in arms]
        errors = [abs(s - m) / m for s, m in zip(simulated, measured)]
        resolved = [(i, j) for i in range(len(arms)) for j in range(i + 1, len(arms))
                    if abs(measured[i] - measured[j]) / min(measured[i], measured[j]) > noise]
        agree = sum((measured[i] < measured[j]) == (simulated[i] < simulated[j]) for i, j in resolved)
        concordance = agree / len(resolved) if resolved else 1.0
        ok = concordance >= concordance_min and max(errors) <= error_max
        result["states"][str(state)] = {"arms": arms, "measured": measured, "simulated": simulated,
                                        "resolved_pairs": len(resolved), "concordance": concordance,
                                        "rho": _spearman(simulated, measured), "max_error": max(errors),
                                        "passes": ok}
        result["passes"] = result["passes"] and ok
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--report", type=Path, action="append", required=True)
    p_fit.add_argument("--out", type=Path, required=True)
    p_gate = sub.add_parser("gate")
    p_gate.add_argument("--model", type=Path, required=True)
    p_gate.add_argument("--report", type=Path, required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--model", type=Path, required=True)
    p_train.add_argument("--out", type=Path, required=True)
    p_train.add_argument("--episodes", type=int, default=12000)
    p_train.add_argument("--seeds", type=int, default=4)
    p_train.add_argument("--widths", action="store_true", help="RL6: also learn each group's width")
    p_judge = sub.add_parser("judge")
    p_judge.add_argument("--model", type=Path, required=True)
    p_judge.add_argument("--policy", type=Path, required=True)
    p_judge.add_argument("--noise", type=float, required=True)
    args = parser.parse_args(None if argv is None else list(argv))

    def load_model(path: Path) -> tuple[PoolModel, float]:
        data = json.loads(path.read_text())
        return PoolModel(**data["model"]), float(data["capacity_rps"])

    if args.command == "fit":
        reports = [json.loads(path.read_text()) for path in args.report]
        model, loss = fit(reports)
        capacity = reports[0].get("rl2h", reports[0])["calibration"]["capacity_rps"]
        args.out.write_text(json.dumps({"model": asdict(model), "loss": loss, "capacity_rps": capacity,
                                        "reports": [str(p.name) for p in args.report]}, indent=1) + "\n")
        print(json.dumps({"model": asdict(model), "loss": loss}))
    elif args.command == "gate":
        model, _ = load_model(args.model)
        print(json.dumps(fidelity(model, json.loads(args.report.read_text())), indent=1))
    elif args.command == "train":
        model, capacity = load_model(args.model)
        policy, summary = train_best(model, capacity, seeds=args.seeds, episodes=args.episodes, widths=args.widths)
        args.out.write_text(json.dumps(policy.to_json(), indent=1) + "\n")
        print(json.dumps({**summary, "policy": policy.to_json()}))
    elif args.command == "judge":
        model, capacity = load_model(args.model)
        policy = Policy.from_json(json.loads(args.policy.read_text()))
        print(json.dumps(judge(model, capacity, policy, args.noise), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
