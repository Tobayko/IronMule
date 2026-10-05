"""Which hardware test to run next, learned from every test this machine has run (RSI1).

Modelled on Zheng et al., *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*
(arXiv 2609.14858): an exploration policy runs online and logs what it found; the log becomes a
replay simulator; the policy is improved by replaying it, then runs online again. Here:

* **Experience history.** Every configuration a tune measures is a node of a discovery tree
  (its parent, the one knob it changes, the measured time ratio against the stock engine,
  token identity, determinism, what the test cost in seconds). Trees are stored per device
  class and model and never edited (``STORE/dream/trees``).
* **Online learning, in Rust.** ``native/experiment_planner`` keeps one conjugate normal
  posterior per knob change (its log effect) and the mean cost of testing it. Each real
  measurement updates it incrementally and the state is checkpointed after every update
  (``STORE/dream/planner-<class>.json``, with a log of which test changed what). The next
  test is the one with the largest expected improvement per expected second; the planner
  stops when no test is worth ``min_gain_per_s``.
* **Replay (dreaming).** Stored trees are replayed deterministically: a policy may open only
  recorded configurations and sees an outcome only after opening it. Replay picks the
  planner's settings and compares it, held out tree by tree, with the fixed order of
  ``tune.SEARCH`` and with random order. Replay never writes to the planner state and never
  stands in for a hardware test: a profile still needs the tune's paired confirmation.
* **The simple rule wins ties.** If the planner does not beat the fixed order and random
  order on held-out replay, it is disabled for that machine and tune keeps coordinate descent.

Correctness is outside the planner: a configuration counts only with tokens identical to the
stock engine and deterministic; whether the winner is faster is decided by the paired A/B.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .hw import STORE

SCHEMA_TREE = "ironmule.dream_tree.v1"
SCHEMA_STATE = "ironmule.dream_planner.v1"
SCHEMA_CONFIG = "ironmule.dream_config.v1"
# Wider than tune.SEARCH: more readback and slack values; the planner decides what is worth it.
SPACE: list[tuple[str, list[Any]]] = [
    ("compiled_fixed_cache", [True]),
    ("fused_argmax", [True]),
    ("head_skip_prefill", [True]),
    ("prefill_into_fixed", [True]),
    ("readback_every", [2, 4, 8, 16]),
    ("speculate_k", [4]),
    ("capacity_slack", [64, 128, 256]),
    ("wired_fraction", [0.6]),
    ("fuse_projections", [True]),
]
ARMS = [f"{knob}={value}" for knob, values in SPACE for value in values]
DEFAULT_CONFIG = {"prior_sd": 0.05, "noise_sd": 0.02, "min_gain_per_s": 0.02, "default_cost_s": 20.0}
GRID = [{**DEFAULT_CONFIG, "min_gain_per_s": g, "noise_sd": n}
        for g in (0.005, 0.01, 0.02, 0.05, 0.1) for n in (0.01, 0.02, 0.04)]
#: Utility of a search: gain points found minus this many points per minute it took. Declared
#: before any tree was recorded (RSI1); the gain lasts, the minutes are paid once.
POINTS_PER_MINUTE = 1.0
KEEP_IF_RATIO_BELOW = 0.995   # as tune: a winner has to pay for itself


def _dir() -> Path:
    return STORE / "dream"


def tree_dir() -> Path:
    return _dir() / "trees"


def _safe(device_class: str | None) -> str:
    return "".join(c if c.isalnum() else "_" for c in (device_class or "unknown"))


def state_path(device_class: str | None) -> Path:
    return _dir() / f"planner-{_safe(device_class)}.json"


def config_path(device_class: str | None) -> Path:
    return _dir() / f"config-{_safe(device_class)}.json"


def _key(knobs: Mapping[str, Any]) -> str:
    return "|".join(f"{k}={knobs[k]}" for k in sorted(knobs))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


# -- the Rust core -------------------------------------------------------------------------

class _Config(C.Structure):
    _fields_ = [("prior_sd", C.c_double), ("noise_sd", C.c_double),
                ("min_gain_per_s", C.c_double), ("default_cost_s", C.c_double)]


class _ArmStatus(C.Structure):
    _fields_ = [("observations", C.c_uint64), ("failures", C.c_uint64), ("posterior_mean", C.c_double),
                ("posterior_sd", C.c_double), ("cost_s", C.c_double)]


def library_path() -> Path:
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return (Path(__file__).resolve().parents[1] / "native" / "experiment_planner" / "target" / "release"
            / f"libironmule_experiment_planner{suffix}")


_LIB: Any = None


def _lib():
    """The planner library, or None where it was not built (then tune keeps coordinate descent)."""
    global _LIB
    if _LIB is None:
        path = library_path()
        if not path.is_file():
            return None
        lib = C.CDLL(str(path))
        for name, args, result in (
                ("create", [C.POINTER(_Config)], C.c_void_p), ("free", [C.c_void_p], None),
                ("observe", [C.c_void_p, C.c_uint32, C.c_double, C.c_double], C.c_int32),
                ("fail", [C.c_void_p, C.c_uint32, C.c_double], C.c_int32),
                ("choose", [C.c_void_p, C.c_size_t, C.POINTER(C.c_uint32), C.POINTER(C.c_double), C.c_double,
                            C.POINTER(C.c_int64), C.POINTER(C.c_double)], C.c_int32),
                ("status", [C.c_void_p, C.c_uint32, C.POINTER(_ArmStatus)], C.c_int32),
                ("checkpoint", [C.c_void_p, C.POINTER(C.c_uint8), C.c_size_t, C.POINTER(C.c_size_t)], C.c_int32),
                ("restore", [C.POINTER(_Config), C.POINTER(C.c_uint8), C.c_size_t], C.c_void_p)):
            function = getattr(lib, "iep_" + name)
            function.argtypes, function.restype = args, result
        _LIB = lib
    return _LIB


class Planner:
    """A handle on the Rust planner; one per device class."""

    def __init__(self, config: Mapping[str, float], state: bytes | None = None):
        self.handle = None
        lib = _lib()
        if lib is None:
            raise RuntimeError("the experiment planner library is not built")
        self.lib, self.config = lib, dict(config)
        native = _Config(*(float(config[f]) for f, _ in _Config._fields_))
        if state is None:
            self.handle = lib.iep_create(C.byref(native))
        else:
            buffer = (C.c_uint8 * len(state)).from_buffer_copy(state)
            self.handle = lib.iep_restore(C.byref(native), buffer, len(state))
        if not self.handle:
            raise ValueError("planner configuration or state rejected")

    def close(self) -> None:
        if self.handle:
            self.lib.iep_free(self.handle)
            self.handle = None

    def __del__(self) -> None:
        self.close()

    def observe(self, arm: str, log_effect: float, seconds: float) -> None:
        if self.lib.iep_observe(self.handle, ARMS.index(arm), log_effect, seconds) != 0:
            raise ValueError("planner rejected an observation")

    def fail(self, arm: str, seconds: float) -> None:
        if self.lib.iep_fail(self.handle, ARMS.index(arm), seconds) != 0:
            raise ValueError("planner rejected a failure")

    def choose(self, arms: Sequence[str], parent_logs: Sequence[float], best_log: float) -> tuple[int, float]:
        count = len(arms)
        ids = (C.c_uint32 * max(1, count))(*(ARMS.index(a) for a in arms))
        parents = (C.c_double * max(1, count))(*parent_logs)
        index, score = C.c_int64(), C.c_double()
        if self.lib.iep_choose(self.handle, count, ids, parents, best_log, C.byref(index), C.byref(score)) != 0:
            raise ValueError("planner rejected a choice")
        return index.value, score.value

    def status(self, arm: str) -> dict[str, float]:
        out = _ArmStatus()
        if self.lib.iep_status(self.handle, ARMS.index(arm), C.byref(out)) != 0:
            raise ValueError("planner rejected a status query")
        return {f: getattr(out, f) for f, _ in _ArmStatus._fields_}

    def checkpoint(self) -> bytes:
        needed = C.c_size_t()
        self.lib.iep_checkpoint(self.handle, None, 0, C.byref(needed))
        buffer = (C.c_uint8 * needed.value)()
        if self.lib.iep_checkpoint(self.handle, buffer, needed.value, C.byref(needed)) != 0:
            raise RuntimeError("planner checkpoint failed")
        return bytes(buffer)


def load_config(device_class: str | None) -> dict[str, Any]:
    try:
        value = json.loads(config_path(device_class).read_text())
        if value.get("schema") == SCHEMA_CONFIG:
            return value
    except (OSError, ValueError):
        pass
    return {"schema": SCHEMA_CONFIG, "config": dict(DEFAULT_CONFIG), "enabled": True, "source": "default"}


def load_planner(device_class: str | None, config: Mapping[str, float]) -> tuple[Planner, dict[str, Any]]:
    """The persisted planner for this class; a missing, corrupt or foreign state starts fresh."""
    envelope: dict[str, Any] = {"schema": SCHEMA_STATE, "device_class": device_class, "arms": ARMS,
                                "observations": 0, "log": []}
    try:
        stored = json.loads(state_path(device_class).read_text())
        state = bytes.fromhex(stored["state_hex"])
        if (stored.get("schema") == SCHEMA_STATE and stored.get("arms") == ARMS
                and stored.get("device_class") == device_class
                and hashlib.sha256(state).hexdigest() == stored.get("state_sha256")):
            return Planner(config, state), stored
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return Planner(config), envelope


def save_planner(planner: Planner, envelope: dict[str, Any]) -> None:
    state = planner.checkpoint()
    envelope.update(state_hex=state.hex(), state_sha256=hashlib.sha256(state).hexdigest(),
                    learned={arm: planner.status(arm) for arm in ARMS}, saved_at=time.time())
    _write(state_path(envelope["device_class"]), envelope)


# -- the discovery tree --------------------------------------------------------------------

def candidates(nodes: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, str, Any, dict[str, Any]]]:
    """Unopened one-knob children of valid nodes, each from its best parent: (parent, knob, value, knobs)."""
    out, seen = [], set()
    for parent_key in sorted((k for k, n in nodes.items() if n["status"] == "ok"), key=lambda k: nodes[k]["ratio"]):
        for knob, values in SPACE:
            for value in values:
                if nodes[parent_key]["knobs"][knob] == value:
                    continue
                child = dict(nodes[parent_key]["knobs"], **{knob: value})
                if (key := _key(child)) not in nodes and key not in seen:
                    seen.add(key)
                    out.append((parent_key, knob, value, child))
    return out


def best_key(nodes: Mapping[str, Mapping[str, Any]]) -> str:
    return min((k for k, n in nodes.items() if n["status"] == "ok"), key=lambda k: nodes[k]["ratio"])


def _learn_from(planner: Planner, nodes: Mapping[str, Mapping[str, Any]], parent_key: str, knob: str,
                value: Any, node: Mapping[str, Any]) -> dict[str, Any]:
    """Feed one test to the planner; returns what changed, for the log."""
    arm = f"{knob}={value}"
    before = planner.status(arm)
    if node["status"] == "ok":
        effect = math.log(node["ratio"] / nodes[parent_key]["ratio"])
        planner.observe(arm, effect, node["seconds"])
    else:
        effect = None
        planner.fail(arm, node["seconds"])
    after = planner.status(arm)
    return {"arm": arm, "status": node["status"], "log_effect": effect, "seconds": node["seconds"],
            "posterior_mean": [before["posterior_mean"], after["posterior_mean"]],
            "posterior_sd": [before["posterior_sd"], after["posterior_sd"]]}


def pick(planner: Planner | None, nodes, rng: random.Random | None = None,
         allowed: Mapping[str, Any] | None = None):
    """The next test: the planner's choice, a random one (``planner=None``), or None to stop."""
    pool = [c for c in candidates(nodes) if allowed is None or _key(c[3]) in allowed]
    if not pool:
        return None
    if planner is None:
        return (rng or random.Random(0)).choice(pool)
    best_log = math.log(nodes[best_key(nodes)]["ratio"])
    index, _ = planner.choose([f"{k}={v}" for _, k, v, _ in pool],
                              [math.log(nodes[p]["ratio"]) for p, _, _, _ in pool], best_log)
    return None if index < 0 else pool[index]


# -- replay ---------------------------------------------------------------------------------

def replay(tree: Mapping[str, Any], explorer: str, *, config: Mapping[str, float] | None = None,
           experience: Sequence[Mapping[str, Any]] = (), seed: int = 0,
           budget_s: float = math.inf) -> dict[str, Any]:
    """Replay one recorded tree with ``planner`` (taught by other trees first, learning as it goes,
    in a throwaway copy), ``coordinate`` (tune.SEARCH order) or ``random``. Only recorded
    configurations can be opened, outcomes are revealed only when opened, and time is the
    recorded cost of each test."""
    recorded = tree["nodes"]
    root = next(k for k, n in recorded.items() if n["parent"] is None)
    nodes, curve, spent = {root: recorded[root]}, [(0.0, 1.0)], 0.0
    planner = None
    if explorer == "planner":
        planner = Planner(config or DEFAULT_CONFIG)
        for other in experience:
            teach(planner, other)
    rng = random.Random(seed)

    def reveal(key: str) -> None:
        nonlocal spent
        nodes[key] = recorded[key]
        spent += recorded[key]["seconds"]
        curve.append((spent, recorded[best_key(nodes)]["ratio"]))

    if explorer == "coordinate":
        from .tune import SEARCH
        best = root
        for knob, values in SEARCH:
            for value in values:
                child = dict(recorded[best]["knobs"], **{knob: value})
                key = _key(child)
                if child == recorded[best]["knobs"] or key not in recorded or key in nodes:
                    continue
                reveal(key)
                node = recorded[key]
                if node["status"] == "ok" and node["ratio"] < recorded[best]["ratio"] * KEEP_IF_RATIO_BELOW:
                    best = key
    else:
        while spent < budget_s:
            chosen = pick(planner, nodes, rng, allowed=recorded)
            if chosen is None:
                break
            parent_key, knob, value, child = chosen
            if planner is not None:
                _learn_from(planner, nodes, parent_key, knob, value, recorded[_key(child)])
            reveal(_key(child))
    if planner is not None:
        planner.close()
    best_ratio = recorded[best_key(nodes)]["ratio"]
    found = best_ratio if best_ratio < KEEP_IF_RATIO_BELOW else 1.0
    tree_best = min(n["ratio"] for n in recorded.values() if n["status"] == "ok")
    near = next((t for t, r in curve if r <= tree_best * 1.01), None)
    return {"explorer": explorer, "opened": len(nodes) - 1, "seconds": round(spent, 2), "best_ratio": found,
            "seconds_to_within_1pct": None if near is None else round(near, 2),
            "utility": 100.0 * (1.0 - found) - POINTS_PER_MINUTE * spent / 60.0, "curve": curve}


def teach(planner: Planner, tree: Mapping[str, Any]) -> None:
    """Experience from a stored tree, in the order it was measured."""
    nodes = tree["nodes"]
    for node in sorted((n for n in nodes.values() if n["parent"] is not None), key=lambda n: n.get("order", 0)):
        if node["parent"] in nodes and nodes[node["parent"]]["status"] == "ok":
            _learn_from(planner, nodes, node["parent"], *node["change"], node)


def held_out(trees: Sequence[Mapping[str, Any]], config: Mapping[str, float], random_seeds: int = 20) -> dict[str, Any]:
    """Each tree replayed by the planner taught by all the others, by the fixed order, and by random
    order for as long as the planner took (averaged over seeds)."""
    rows = []
    for index, tree in enumerate(trees):
        others = list(trees[:index]) + list(trees[index + 1:])
        planned = replay(tree, "planner", config=config, experience=others)
        fixed = replay(tree, "coordinate")
        randoms = [replay(tree, "random", seed=s, budget_s=planned["seconds"]) for s in range(random_seeds)]
        rows.append({"tree": tree.get("name", str(index)), "model_id": tree.get("model_id"),
                     "planner": _brief(planned), "coordinate": _brief(fixed),
                     "random": {"utility": sum(r["utility"] for r in randoms) / len(randoms),
                                "best_ratio": sum(r["best_ratio"] for r in randoms) / len(randoms),
                                "seconds": sum(r["seconds"] for r in randoms) / len(randoms)}})
    mean = {name: sum(r[name]["utility"] for r in rows) / len(rows) for name in ("planner", "coordinate", "random")}
    return {"rows": rows, "mean_utility": mean}


def _brief(result: Mapping[str, Any]) -> dict[str, Any]:
    return {k: result[k] for k in ("utility", "best_ratio", "seconds", "opened", "seconds_to_within_1pct")}


def load_trees(device_class: str | None = None) -> list[dict[str, Any]]:
    trees = []
    for path in sorted(tree_dir().glob("*.json")):
        try:
            tree = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (tree.get("schema") == SCHEMA_TREE and len(tree.get("nodes", {})) > 1
                and (device_class is None or tree.get("device_class") == device_class)):
            tree["name"] = path.stem
            trees.append(tree)
    return trees


def dream(device_class: str | None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Replay every stored tree of this class under each planner setting; keep the best setting,
    enabled only if it beats the fixed order and random order held out."""
    current = load_config(device_class)
    if _lib() is None:
        return current
    trees = load_trees(device_class)
    if len(trees) < 2:
        log(f"dream: {len(trees)} stored tree(s) for this machine; replay needs two")
        return current
    started = time.perf_counter()
    scored = [(held_out(trees, config, random_seeds=1)["mean_utility"]["planner"], -i) for i, config in enumerate(GRID)]
    best = GRID[-max(scored)[1]]
    report = held_out(trees, best)
    mean = report["mean_utility"]
    enabled = mean["planner"] > max(mean["coordinate"], mean["random"])
    value = {"schema": SCHEMA_CONFIG, "config": best, "enabled": enabled, "source": "replay",
             "trees": len(trees), "held_out": report, "replay_seconds": round(time.perf_counter() - started, 2),
             "dreamed_at": time.time()}
    _write(config_path(device_class), value)
    log("dream: held-out utility " + ", ".join(f"{k} {v:.2f}" for k, v in mean.items())
        + f"; planner {'enabled' if enabled else 'disabled, the simple order is better here'}"
        + f" ({value['replay_seconds']} s of replay)")
    return value


def available(device_class: str | None) -> bool:
    """Whether tune may use the planner here: built, and not beaten by the simple orders in replay."""
    return _lib() is not None and load_config(device_class).get("enabled", True) is True


# -- online exploration (a drop-in for tune._screen) ---------------------------------------

def explore(model_id: str, prompt: str, max_tokens: int, repeats: int, compute_dtype: str | None,
            known: Mapping[str, Any], budget_s: float, explorer: str = "planner",
            seed: int | None = None) -> dict[str, Any]:
    """Run hardware tests chosen by the planner (or at random) until it stops or the budget ends.

    Every test updates the persisted planner state at once; the tree is stored at the end.
    Returns what ``tune._screen`` returns, so the paired confirmation and the profile are unchanged.
    """
    from .runtime import BASELINE, Engine, Knobs
    from .tune import (_check_compute_dtype, _close_engine, _eos_ids, _is_unsupported_candidate,
                       _release_device_memory, load_engine, measure, probe_too_slow, prompt_ids,
                       resolve_local_model)

    started = time.monotonic()
    device_class = known.get("class")
    settings = load_config(device_class)
    planner, envelope = load_planner(device_class, settings["config"])
    rng = random.Random(seed if seed is not None else time.time_ns())
    resolved = resolve_local_model(model_id)
    name = f"{_safe(device_class)}__{resolved.identity.identity_sha256[:16]}__{time.time_ns()}"
    engine = None
    try:
        engine, tokenizer = load_engine(model_id, BASELINE, resolved_source=resolved,
                                        compute_dtype=_check_compute_dtype(compute_dtype))
        ids, eos = prompt_ids(tokenizer, prompt), _eos_ids(tokenizer)
        if (elapsed := probe_too_slow(engine, ids, eos)) is not None:
            raise RuntimeError(f"tuning would take hours here: 20 tokens took {elapsed:.0f} s warm (TUNE1)")
        base = measure(engine, ids, max_tokens, eos, repeats=repeats)
        if not base["deterministic"]:
            raise RuntimeError("baseline is not deterministic, cannot gate on token identity")
        reference = base["logical_tokens"]
        root = BASELINE.as_dict()
        nodes = {_key(root): {"knobs": root, "parent": None, "change": None, "depth": 0, "status": "ok",
                              "ratio": 1.0, "seconds": 0.0, "order": 0,
                              "result": {k: base[k] for k in ("total_ns", "prefill_ns", "decode_ns")}}}
        current, results, trials, decisions_ns = Knobs(**root), {_key(root): base}, [], []
        print(f"dream: baseline {base['total_ns']/1e6:.2f} ms; explorer {explorer}, planner state "
              f"{envelope['observations']} observations, budget {budget_s:.0f} s", flush=True)
        while time.monotonic() - started < budget_s:
            decided = time.perf_counter_ns()
            chosen = pick(planner if explorer == "planner" else None, nodes, rng)
            decisions_ns.append(time.perf_counter_ns() - decided)
            if chosen is None:
                print("dream: no remaining test is worth its time", flush=True)
                break
            parent_key, knob, value, child = chosen
            candidate = Knobs(**child)
            node = {"knobs": child, "parent": parent_key, "change": [knob, value],
                    "depth": nodes[parent_key]["depth"] + 1, "order": len(nodes)}
            test_started = time.monotonic()
            try:
                if engine is None or Engine.needs_reload(current, candidate):
                    _close_engine(engine)
                    engine = None
                    engine, tokenizer = load_engine(model_id, candidate, resolved_source=resolved,
                                                    compute_dtype=compute_dtype)
                else:
                    engine.knobs = candidate
                    engine._compiled = None
                current = candidate
                result = measure(engine, ids, max_tokens, eos, repeats=repeats)
            except (ValueError, RuntimeError, TypeError) as exc:
                if not _is_unsupported_candidate(exc):
                    raise
                node.update(status="unsupported", ratio=None, reason=f"{type(exc).__name__}: {exc}")
                _close_engine(engine)  # the next test reloads from a known state
                engine = None
            else:
                node.update(status=("differs" if result["logical_tokens"] != reference
                                    else "nondeterministic" if not result["deterministic"] else "ok"),
                            ratio=result["total_ns"] / base["total_ns"],
                            result={k: result[k] for k in ("total_ns", "prefill_ns", "decode_ns")})
                results[_key(child)] = result
            node["seconds"] = round(time.monotonic() - test_started, 3)
            nodes[_key(child)] = node
            change = _learn_from(planner, nodes, parent_key, knob, value, node)
            envelope["observations"] += 1
            envelope["log"].append({"tree": name, "explorer": explorer, "at": time.time(), **change})
            save_planner(planner, envelope)  # a crash keeps every completed test
            shown = node["status"] if node["status"] != "ok" else f"ratio {node['ratio']:.4f}"
            print(f"  {knob}={value!r:>6} (depth {node['depth']}, {node['seconds']:.1f} s): {shown}", flush=True)
            trials.append({"knob": knob, "value": value, "ratio": node["ratio"], "depth": node["depth"],
                           "disposition": "unsupported" if node["status"] == "unsupported" else "measured",
                           "verdict": node["status"]})
        winner = best_key(nodes)
        if nodes[winner]["ratio"] >= KEEP_IF_RATIO_BELOW:
            winner = _key(root)
        for trial in trials:  # knowledge counts a knob as kept when the winner carries it
            if trial["disposition"] == "measured":
                kept = winner != _key(root) and nodes[winner]["knobs"][trial["knob"]] == trial["value"]
                trial["disposition"] = "accepted" if kept and trial["verdict"] == "ok" else "rejected"
        tree = {"schema": SCHEMA_TREE, "device_class": device_class, "model_id": model_id,
                "model_identity_sha256": resolved.identity.identity_sha256, "compute_dtype": compute_dtype,
                "prompt_tokens": len(ids), "max_tokens": max_tokens, "repeats": repeats, "explorer": explorer,
                "planner_config": settings["config"], "planner_enabled": settings.get("enabled", True),
                "planner_observations_before": envelope["observations"] - len(trials),
                "budget_s": budget_s, "elapsed_s": round(time.monotonic() - started, 2),
                "decision_ns": decisions_ns, "nodes": nodes, "created_at": time.time()}
        _write(tree_dir() / f"{name}.json", tree)
        print(f"dream: {len(nodes) - 1} tests, best ratio {nodes[winner]['ratio']:.4f}, "
              f"decisions {sum(decisions_ns) / 1e6:.2f} ms in total; tree {name}", flush=True)
        return {"prompt_tokens": len(ids), "base": base, "best": Knobs(**nodes[winner]["knobs"]).as_dict(),
                "best_result": results[winner], "trials": trials, "tree": name}
    finally:
        planner.close()
        _close_engine(engine)
        _release_device_memory()


# -- the hardware session ------------------------------------------------------------------

def learn(models: Sequence[str], minutes: float = 12.0, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Within ``minutes``, per model: planner-chosen tests, the paired confirmation, a replay round.
    Each round can hand the next model a better-informed planner (recursive self-improvement)."""
    from .tune import tune
    deadline = time.monotonic() + minutes * 60
    rows = []
    for index, model in enumerate(models):
        share = (deadline - time.monotonic()) / (len(models) - index)
        if share < 90:
            log(f"learn: no time left for {len(models) - index} model(s)")
            break
        began = time.monotonic()
        log(f"learn: {model}, about {share / 60:.1f} min")
        try:
            profile = tune(model, explorer="planner", budget_s=0.6 * share, screen_in_child=True)
        except Exception as exc:  # noqa: BLE001 - one model failing must not end the session
            rows.append({"model": model, "error": f"{type(exc).__name__}: {exc}"})
            log(f"learn: {model} failed ({rows[-1]['error']})")
            continue
        rows.append({"model": model, "explorer": profile.get("explorer"), "gain": profile["gain"],
                     "confirmation": profile.get("confirmation"), "knobs": profile["knobs"],
                     "tests": len(profile.get("trials", [])), "seconds": round(time.monotonic() - began, 1)})
        log("learn: " + json.dumps(rows[-1], default=str))
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="ironmule learn",
        description="A bounded hardware session (default 12 min): per cached model the planner picks "
                    "the tests, the winner is confirmed against stock, and replay tunes the planner (RSI1).")
    parser.add_argument("--model", action="append", default=[],
                        help="model to learn (repeatable); default: every cached MLX model, smallest first")
    parser.add_argument("--minutes", type=float, default=12.0)
    parser.add_argument("--report", action="store_true",
                        help="print what the planner learned and the held-out replay; measures nothing")
    args = parser.parse_args(argv)
    from . import knowledge
    from .hw import probe
    device_class = knowledge.device_class(probe())
    if args.report:
        try:
            state = json.loads(state_path(device_class).read_text())
            state.pop("state_hex", None)
        except (OSError, ValueError):
            state = {}
        print(json.dumps({"device_class": device_class, "trees": len(load_trees(device_class)),
                          "config": load_config(device_class), "planner": state}, indent=1, default=str))
        return 0
    if args.minutes <= 0:
        parser.error("--minutes must be positive")
    models = list(dict.fromkeys(args.model))
    if not models:
        from ironmule_inventory import discover_models
        rows = [r for r in discover_models(loader="mlx_lm") if r.get("status") == "available"]
        models = list(dict.fromkeys(r["model_id"] for r in sorted(rows, key=lambda r: r.get("weight_bytes") or 0)))
    if not models:
        print("ironmule learn: no cached model; download one first", file=sys.stderr)
        return 1
    rows = learn(models, args.minutes)
    print(json.dumps({"learned": rows, "config": load_config(device_class)}, indent=1, default=str))
    return 0 if any("gain" in row for row in rows) else 1


__all__ = ["ARMS", "DEFAULT_CONFIG", "GRID", "Planner", "SPACE", "available", "candidates", "dream", "explore",
           "held_out", "learn", "load_config", "load_planner", "load_trees", "main", "pick", "replay", "teach"]
