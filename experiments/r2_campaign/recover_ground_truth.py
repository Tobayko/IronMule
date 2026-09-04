"""Does the R0 to R1 pipeline recover a truth that was planted in it?

Offline end-to-end check of the chain that will consume the most expensive
resource this project has: five approved measurement blocks. A campaign is
drawn, synthetic outcomes with a *known* per-action effect are written into a
real Optimization Memory, and the real off-policy estimators are asked to
recover that effect from the log alone.

Synthetic rewards exercise the estimators; they are not evidence and support
no performance claim. What is being validated is the machinery, not the model.
"""

from __future__ import annotations

import random
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from friday_optimizer.campaign import CampaignPlan  # noqa: E402
from friday_optimizer.candidates import CandidateRegistry  # noqa: E402
from friday_optimizer.decisions import OutcomeEvent, SelectionPolicy  # noqa: E402
from friday_optimizer.memory import OptimizationMemoryV2  # noqa: E402
from friday_optimizer.replay import (  # noqa: E402
    ReplayEnv,
    ips,
    load_steps,
    replayer,
    snips,
)
from test_optimizer_decisions import make_fingerprint  # noqa: E402

HINT = "head_skip_prefill"

#: The planted truth: a ratio per action, candidate over baseline. Lower is
#: faster. These mirror the project's own confirmed magnitudes so the check
#: runs in a realistic range, but they are invented for this dry run.
TRUE_RATIO = {
    "baseline": 1.000,
    "head_skip_prefill": 0.870,
    "persistent_process": 0.930,
    "fixed_compiled_cache": 0.985,
    "readback_every_2": 0.995,
}
NOISE_SD = 0.010
#: One measurement in twenty is censored, as real gated runs are.
CENSOR_RATE = 0.05


def build_corpus(points: int, epsilon: float, seed: int, memory_path: Path) -> tuple[int, int]:
    fingerprint = make_fingerprint()
    registry = CandidateRegistry()
    policy = SelectionPolicy("log-v1", rule="epsilon_greedy", epsilon=epsilon)
    plan = CampaignPlan(campaign_id="dryrun", policy=policy, seed_base=seed,
                        points=points, hints=(HINT,))
    events = plan.decisions(fingerprint, registry=registry)
    rng = random.Random(seed)
    censored = 0
    with OptimizationMemoryV2(memory_path) as memory:
        for event in events:
            memory.append(event.as_record())
            if rng.random() < CENSOR_RATE:
                memory.append(OutcomeEvent(event.decision_id, "censored_timeout").as_record())
                censored += 1
                continue
            ratio = rng.gauss(TRUE_RATIO[event.chosen], NOISE_SD)
            memory.append(OutcomeEvent(event.decision_id, "observed", reward=max(ratio, 1e-3)).as_record())
        chain_ok = memory.verify_chain()
    if not chain_ok:
        raise SystemExit("memory hash chain broke during the dry run")
    return len(events), censored


def evaluate(memory_path: Path, target_action: str, min_samples: int) -> dict:
    with OptimizationMemoryV2(memory_path) as memory:
        steps = load_steps(memory)
    environment = ReplayEnv(steps)
    target = SelectionPolicy(f"target-{target_action.replace('_', '-')}")
    # The target concentrates on one action via its own hints. The corpus is
    # untouched; only the policy being priced changes.
    hints = () if target_action == "baseline" else (target_action,)
    shared = {"min_samples": min_samples, "resamples": 400, "target_hints": hints}
    return {
        "ips": ips(environment, target, **shared),
        "snips": snips(environment, target, **shared),
        "replayer": replayer(environment, target, **shared),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    points = int(arguments[0]) if arguments else 400
    epsilon, seed, min_samples = 0.6, 20260902, 30
    with tempfile.TemporaryDirectory() as directory:
        # macOS puts temp dirs behind a /var symlink and the memory refuses a
        # symlinked ancestor, which is the guard doing its job.
        path = Path(directory).resolve() / "dryrun.sqlite3"
        written, censored = build_corpus(points, epsilon, seed, path)
        print(f"corpus: {written} decisions, {censored} censored ({100*censored/written:.1f} %),"
              f" epsilon {epsilon}, hash chain verified")
        print(f"planted truth (ratio, lower is faster): "
              + ", ".join(f"{k}={v:.3f}" for k, v in TRUE_RATIO.items()))
        print()
        print(f"{'target action':24s} {'true gain':>10s} {'snips':>10s} {'ips':>10s}"
              f" {'replayer':>10s} {'ESS':>7s} {'status':>18s}")
        errors = []
        for action, ratio in TRUE_RATIO.items():
            estimates = evaluate(path, action, min_samples)
            true_gain = 1.0 - ratio
            row = estimates["snips"]
            print(f"{action:24s} {true_gain*100:9.2f} % {estimates['snips'].value*100:9.2f} %"
                  f" {estimates['ips'].value*100:9.2f} % {estimates['replayer'].value*100:9.2f} %"
                  f" {row.effective_samples:7.1f} {row.status:>18s}")
            if row.status == "ok":
                errors.append(abs(row.value - true_gain))
        print()
        if errors:
            print(f"conclusive estimates: {len(errors)}/{len(TRUE_RATIO)},"
                  f" median absolute error {statistics.median(errors)*100:.2f} points,"
                  f" worst {max(errors)*100:.2f} points")
        else:
            print("no estimate reached the sample floor")
        # The ordering is what a tuner would act on, so check it directly.
        ranked = []
        for action in TRUE_RATIO:
            value = evaluate(path, action, min_samples)["snips"]
            ranked.append((value.value if value.value is not None else float("-inf"), action))
        ranked.sort(reverse=True)
        truth = sorted(((1 - r, a) for a, r in TRUE_RATIO.items()), reverse=True)
        print()
        print("recovered ranking:", " > ".join(a for _, a in ranked))
        print("true ranking:     ", " > ".join(a for _, a in truth))
        print("ranking recovered:", [a for _, a in ranked] == [a for _, a in truth])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
