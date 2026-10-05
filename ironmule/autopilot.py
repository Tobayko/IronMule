"""Autopilot: one command, and this machine and model choose their own configuration.

Each step is reused when it was already done for this exact machine and model:

1. measure the hardware once (`hw.probe`, cached per hardware fingerprint);
2. bind the exact model identity, quantised or dense (`model_identity`);
3. tune the engine knobs once (`tune.tune`: exact tokens, paired confirmation);
4. serve through the online controller, which keeps comparing serving profiles and
   adopts a faster one only after sequential qualification (`sequential_noise_v4`). A bounded
   self-training burst collects its labels with directed exploration (`CTRL13`).

Nothing here changes a token: every candidate must reproduce the reference output,
and the learned state persists per machine and model under `~/.ironmule/autopilot`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

from .hw import STORE

STATE = STORE / "autopilot"
# A stored winner older than this is re-checked against the untuned engine before use.
REVALIDATE_AFTER_S = 7 * 86400
# Short, fixed and private-data-free; they only drive label collection and comparisons.
TRAINING_PROMPTS = (
    "Explain how rain forms.",
    "Give three tips for writing clear code.",
    "What is the difference between a list and a tuple in Python?",
    "Summarise why the sky is blue in two sentences.",
)


def controller_config(max_tokens: int, learn_speculation: bool = False, **overrides: Any):
    """Noise-robust sequential qualification (CTRL19) and directed labels; buffered answers.

    A buffered answer becomes visible only when it is complete, so its first visible
    content arrives at completion: both limits are the completion limit.
    """
    from .online_controller import ControllerConfig
    limit = max(30.0, 0.5 * max_tokens)
    return ControllerConfig(qualification_protocol="sequential_noise_v4", directed_training=True,
                            latency_view="delivered", exploration_ppm=50_000,
                            ttft_limit_s=limit, latency_limit_s=limit,
                            # CTRL23: offer speculation as a profile; strict-plan arms then compare
                            # terminal offsets, because speculation changes low cache bits.
                            speculative_profiles=learn_speculation,
                            state_signature="offset" if learn_speculation else "hash", **overrides)


def checkpoint_path(fingerprint: str, identity_sha256: str, plan: str | None = None) -> Path:
    return STATE / f"{fingerprint}-{identity_sha256[:16]}{'-' + plan if plan else ''}.json"


def prepare(model_id: str, *, tune_missing: bool = True, log=print,
            allow_numeric_plans: bool = False, learn_minutes: float | None = None) -> dict[str, Any]:
    """Measure the machine, choose a numeric plan if allowed, and tune unless already known."""
    from .hw import probe
    from .tune import load_profile, resolve_local_model, revalidate

    hardware = _probe_out_of_process(probe)
    resolved = resolve_local_model(model_id)
    plan = None
    if allow_numeric_plans:
        from .numeric_choice import choose_out_of_process
        decision = choose_out_of_process(model_id, hardware["fingerprint"], resolved.identity.identity_sha256,
                                         log=log)
        plan = decision["plan"]
        log(f"autopilot: numeric plan {plan or 'none (checkpoint dtype)'}: {decision['reason']}")
    profile = load_profile(model_id, model_identity=resolved.identity, compute_dtype=plan)
    tuned_now = False
    checked = _last_checked(resolved.identity.identity_sha256, profile)
    if profile is not None and tune_missing and time.time() - checked > REVALIDATE_AFTER_S:
        try:
            verdict = revalidate(model_id, compute_dtype=plan)["verdict"]
        except Exception as exc:  # noqa: BLE001 - keep the stored settings; re-check next start
            verdict = f"check_failed ({type(exc).__name__})"
        log(f"autopilot: stored settings re-checked after {(time.time() - checked) / 86400:.0f} days: {verdict}")
        if verdict == "retune_required":
            profile = None
        else:
            _mark_checked(resolved.identity.identity_sha256)
    untunable = f"untunable-{resolved.identity.identity_sha256}-{plan}"
    if profile is None and tune_missing and time.time() - _last_checked(untunable, None) < REVALIDATE_AFTER_S:
        log("autopilot: tuning was too slow here within the last week (TUNE1); serving untuned")
    elif profile is None and tune_missing:
        log("autopilot: no tuned profile for this machine and model, tuning once ...")
        for attempt in (1, 2):  # one retry: a first CUDA start lost a confirmation child (AUTO2)
            try:
                profile = _tune_out_of_process(model_id, plan, learn_minutes)
                tuned_now = True
                break
            except TuneTooSlow as exc:  # the same answer twice costs minutes; remember it instead
                log(f"autopilot: {exc}")
                _mark_checked(untunable)
                break
            except Exception as exc:  # noqa: BLE001 - serve untuned; the reference path stays exact
                log(f"autopilot: tuning attempt {attempt} failed ({type(exc).__name__}: {exc})")
        if profile is None:
            log("autopilot: serving the untuned engine; the online controller still learns")
    return {"fingerprint": hardware["fingerprint"], "model_identity": resolved.identity,
            "profile": profile, "tuned_now": tuned_now, "compute_dtype": plan}


class TuneTooSlow(RuntimeError):
    """Tuning on this machine would take hours; serve untuned and do not retry for a week."""


def _tune_out_of_process(model_id: str, plan: str | None, learn_minutes: float | None = None) -> dict[str, Any]:
    """Tune in a child, so a fault there cannot poison the process that serves.

    On a T4 the native plan with fused projections hit an illegal memory access during
    screening, which left this process's CUDA context unusable and the following load
    aborted (NUM1 attempt 2). The autopilot checked `gpu_busy` at start; the child must not
    mistake this parent for another model process.
    """
    import subprocess

    from .tune import load_profile, resolve_local_model
    code = ("import sys; from ironmule.tune import tune\n"
            "budget = float(sys.argv[3]) * 60 if sys.argv[3] else None\n"
            "try:\n    tune(sys.argv[1], compute_dtype=sys.argv[2] or None, force=True, screen_in_child=True,\n"
            "         explorer='planner' if budget else 'coordinate', budget_s=budget)\n"
            "except RuntimeError as exc:\n    sys.exit(3 if 'TUNE1' in str(exc) else 1)")
    status = subprocess.run([sys.executable, "-c", code, model_id, plan or "", str(learn_minutes or "")],
                            timeout=7200).returncode
    if status == 3:
        raise TuneTooSlow("a 20-token probe took over 10 s here; tuning would take hours (TUNE1)")
    if status:
        raise RuntimeError(f"the tune child exited with status {status}")
    profile = load_profile(model_id, model_identity=resolve_local_model(model_id).identity, compute_dtype=plan)
    if profile is None:
        raise RuntimeError("the tune child stored no profile for this machine and model")
    return profile


def numeric_consent(given: bool | None) -> bool:
    """The caller's consent to numeric plans: given or withdrawn now, else as last given."""
    path = STATE / "consent.json"
    if given is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"numeric_plans": given}) + "\n")
        return given
    try:
        return json.loads(path.read_text()).get("numeric_plans") is True
    except (OSError, ValueError, AttributeError):
        return False


def _probe_out_of_process(probe) -> dict[str, Any]:
    """Measure the hardware in a child so this process holds no GPU state before tuning.

    A first CUDA start, the only one that measured in this process, repeatedly lost a
    tune confirmation child (AUTO2, AUTO4, AUTO6, AUTO7); warm starts did not.
    """
    import subprocess

    from .hw import apply_cuda_graph_defaults
    try:
        subprocess.run([sys.executable, "-c", "from ironmule.hw import probe; probe()"],
                       check=True, timeout=900)
        # The record binds the CUDA graph environment the child applied (AUTO8).
        apply_cuda_graph_defaults()
        return probe(allow_measure=False)
    except Exception:  # noqa: BLE001 - measure here as before; the cache stays the single source
        return probe()


def _last_checked(identity_sha256: str, profile: dict | None) -> float:
    try:
        marks = json.loads((STATE / "revalidated.json").read_text())
    except (OSError, ValueError):
        marks = {}
    return max(float(marks.get(identity_sha256, 0)), float((profile or {}).get("tuned_at", 0)))


def _mark_checked(identity_sha256: str) -> None:
    path = STATE / "revalidated.json"
    try:
        marks = json.loads(path.read_text())
    except (OSError, ValueError):
        marks = {}
    marks[identity_sha256] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marks, indent=1, sort_keys=True) + "\n")


def load(model_id: str, prepared: dict[str, Any], max_tokens: int, log=print, learn_speculation: bool = False):
    """Load with this machine and model's learned state, or start it fresh.

    A checkpoint bound to a different runtime identity (library, source, EOS set,
    configuration) is rejected by the controller. It is kept, renamed, never deleted,
    and learning restarts from nothing rather than from foreign evidence.
    """
    from .service import Runtime
    plan = prepared.get("compute_dtype")
    path = checkpoint_path(prepared["fingerprint"], prepared["model_identity"].identity_sha256, plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = controller_config(max_tokens, learn_speculation)
    runtime = Runtime.load(model_id, learning_checkpoint=path, controller_config=config, compute_dtype=plan)
    if not runtime.online_controller.snapshot().get("checkpoint_rejected"):
        return runtime
    runtime.close()
    kept = path.with_name(f"{path.stem}.rejected-{int(time.time())}.json")
    path.rename(kept)
    log(f"autopilot: learned state belongs to another runtime identity, kept as {kept.name}; relearning")
    return Runtime.load(model_id, learning_checkpoint=path, controller_config=config, compute_dtype=plan)


def self_train(runtime, *, max_tokens: int, seconds: float, budget: int = 256,
               prompts: Sequence[str] = TRAINING_PROMPTS, log=print) -> dict[str, Any]:
    """Run a bounded burst of real requests until a decision is reached or time is up.

    The controller pays every extra execution from this burst's explicit grant; the
    grant expires with the burst and is never persisted.
    """
    controller = runtime.online_controller
    started, calls, converged = time.monotonic(), 0, False
    first = controller.snapshot()
    cell = 1 if max_tokens > 32 else 0  # the single-request cell this burst trains
    with controller.offline_training(max_extra_comparison_executions=budget):
        while time.monotonic() - started < seconds:
            runtime.generate(prompts[calls % len(prompts)], max_tokens=max_tokens)
            controller.flush()
            calls += 1
            state = controller.snapshot()
            if state["promoted"] > first["promoted"] or state["rejected"] > first["rejected"]:
                break
            if state["offline_training"]["remaining"] < 2:
                break
            if _converged(controller, state, first, cell):
                converged = True
                break
    state = controller.snapshot()
    result = {"calls": calls, "seconds": round(time.monotonic() - started, 2),
              "converged": converged, "promoted": state["promoted"] - first["promoted"],
              "rejected": state["rejected"] - first["rejected"],
              "extra_executions": state["offline_training"]["used"]}
    log("autopilot: self-training " + json.dumps(result))
    return result


def _converged(controller, state: dict, first: dict, cell: int) -> bool:
    """Nothing left to learn here: every observed profile has its labels, two freeze
    rounds have passed and no candidate beat the serving profile by the minimum gain."""
    config = controller.config
    if state["candidate_version"] or state["updates"] - first["updates"] < 2 * config.freeze_every:
        return False
    counts = [a["observations"] for a in controller.hardware_knowledge()["cells"][cell]["actions"]]
    observed = [count for count in counts if count]
    return len(observed) >= 2 and min(observed) >= config.min_train


class IdleLearner:
    """Keeps learning from the user's own recent prompts while nobody is waiting.

    Prompts stay in memory only. A training call holds `lock`; a user request takes the
    same lock, so it waits at most for the one call in flight. The offline grant is
    opened and closed in this thread, as the controller requires.
    """

    def __init__(self, runtime, *, max_tokens: int, idle_s: float = 5.0, budget: int = 64,
                 keep: int = 16, log=print):
        self.runtime, self.max_tokens, self.idle_s, self.budget, self.log = runtime, max_tokens, idle_s, budget, log
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.recent: deque[str] = deque(maxlen=keep)
        self.last_activity = time.monotonic()
        self.calls = 0
        self.thread = threading.Thread(target=self._run, name="ironmule-idle-learner", daemon=True)

    def seen(self, prompt: str) -> None:
        self.recent.append(prompt)
        self.last_activity = time.monotonic()

    def _idle(self) -> bool:
        return not self.stop.is_set() and bool(self.recent) and time.monotonic() - self.last_activity >= self.idle_s

    def _run(self) -> None:
        controller = self.runtime.online_controller
        while not self.stop.wait(0.5):
            if not self._idle():
                continue
            try:
                with controller.offline_training(max_extra_comparison_executions=self.budget):
                    while self._idle() and controller.snapshot()["offline_training"]["remaining"] >= 2:
                        with self.lock:
                            if not self._idle():
                                break
                            prompt = self.recent[self.calls % len(self.recent)]
                            self.runtime.generate(prompt, max_tokens=self.max_tokens)
                        self.calls += 1
            except RuntimeError as exc:  # a killed or quarantined controller: stop learning, keep serving
                self.log(f"autopilot: idle learning stopped ({exc})")
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop.set()
        self.thread.join(timeout=60)


def decisions(runtime) -> dict[str, Any]:
    """What the system currently runs, per workload cell, and why."""
    knowledge = runtime.online_controller.hardware_knowledge()
    profiles = runtime.online_controller.action_contract["profiles"]
    cells = []
    for cell in knowledge["cells"]:
        observed = {a["profile_id"]: round(a["ew_cost_seconds_per_request"], 4)
                    for a in cell["actions"] if a["ew_cost_seconds_per_request"] is not None}
        if observed:
            cells.append({"requests": cell["request_count"], "tokens": cell["requested_tokens"],
                          "serving": profiles[cell["active_action"]], "cost_s": observed})
    state = runtime.online_controller.snapshot()
    from .runtime import BASELINE
    baseline = BASELINE.as_dict()
    return {"knobs": {k: v for k, v in runtime.engine.knobs.as_dict().items() if baseline.get(k) != v},
            "cells": cells, "promoted": state["promoted"], "updates": state["updates"],
            "qualifying": state["candidate_version"] != 0, "windows": state["pairs_completed"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ironmule autopilot",
        description="Measure, tune and keep optimising this machine and model; then answer prompts.")
    parser.add_argument("--model", default=None, help="cached model id (quantised or dense)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--train-seconds", type=float, default=300.0,
                        help="bound for the self-training burst at startup (0 skips it)")
    parser.add_argument("--no-tune", action="store_true", help="never run the offline knob search")
    parser.add_argument("--learn-minutes", type=float, default=5.0,
                        help="first tune: minutes the learned test planner may spend on hardware tests before "
                             "the paired confirmation (RSI1b); 0 uses the fixed order. Without the planner "
                             "library, or where replay found the fixed order better, the fixed order runs")
    parser.add_argument("--learn-speculation", action="store_true",
                        help="let the controller qualify draft-gated speculation per workload (CTRL23)")
    consent = parser.add_mutually_exclusive_group()
    consent.add_argument("--allow-numeric-plans", dest="numeric_plans", action="store_const", const=True,
                         help="consent, remembered, to output-changing numeric plans; one is used only "
                              "where its quality is measured inside the bound and it is faster (NUM1)")
    consent.add_argument("--no-numeric-plans", dest="numeric_plans", action="store_const", const=False,
                         help="withdraw that consent")
    parser.add_argument("--training-prompts", type=Path,
                        help="file with one typical prompt per line for the self-training burst")
    parser.add_argument("--no-idle-learning", dest="idle_learning", action="store_false",
                        help="interactive sessions: do not keep learning from recent prompts while idle")
    parser.add_argument("--prompt", action="append", default=[],
                        help="answer this prompt and exit (repeatable); otherwise read stdin lines")
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.train_seconds < 0 or args.learn_minutes < 0:
        parser.error("--max-tokens must be positive, --train-seconds and --learn-minutes non-negative")
    from ironmule_controller import default_library_path
    if not default_library_path().is_file():
        print("ironmule autopilot: the online controller (Rust) is not built. From a source checkout run\n"
              "  cargo build --release --manifest-path native/online_controller/Cargo.toml\n"
              "and optionally the planner: --manifest-path native/experiment_planner/Cargo.toml", file=sys.stderr)
        return 1
    from .tune import DEFAULT_MODEL, gpu_busy
    busy = gpu_busy()
    if busy:
        print(f"ironmule autopilot: another model process holds the GPU ({busy})", file=sys.stderr)
        return 1
    model = args.model or DEFAULT_MODEL
    prepared = prepare(model, tune_missing=not args.no_tune, allow_numeric_plans=numeric_consent(args.numeric_plans),
                       learn_minutes=args.learn_minutes or None)
    profile = prepared["profile"]
    print("autopilot: hardware", prepared["fingerprint"], "| model", prepared["model_identity"].model_id,
          "| tuned gain", f"{profile['gain'] * 100:.1f} %" if profile else "none")
    training = TRAINING_PROMPTS
    if args.training_prompts:
        training = tuple(line.strip() for line in args.training_prompts.read_text().splitlines() if line.strip())
        if not training:
            parser.error("--training-prompts holds no prompt")
    with load(model, prepared, args.max_tokens, learn_speculation=args.learn_speculation) as runtime:
        if args.train_seconds:
            self_train(runtime, max_tokens=args.max_tokens, seconds=args.train_seconds, prompts=training)
        print("autopilot: decisions " + json.dumps(decisions(runtime)), flush=True)
        prompts = args.prompt or (line.strip() for line in sys.stdin)
        # Interactive sessions keep learning from their own prompts between questions.
        idle = IdleLearner(runtime, max_tokens=args.max_tokens) if not args.prompt and args.idle_learning else None
        with idle or contextlib.nullcontext():
            for prompt in prompts:
                if not prompt:
                    continue
                with idle.lock if idle else contextlib.nullcontext():
                    result = runtime.generate(prompt, max_tokens=args.max_tokens)
                if idle:
                    idle.seen(prompt)
                self_report(runtime, result)
        if idle:
            print(f"autopilot: idle learning made {idle.calls} calls", flush=True)
        print("autopilot: decisions " + json.dumps(decisions(runtime)), flush=True)
    return 0


def self_report(runtime, result) -> None:
    metrics = result.metrics
    print(result.text, flush=True)
    print(f"[{metrics['generated_tokens']} tokens, {metrics['latency_ms']:.0f} ms, "
          f"{metrics['decode_tokens_per_second'] or 0:.1f} tok/s, "
          f"profile {runtime.online_controller.last_decision.get('actual_action')}]", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
