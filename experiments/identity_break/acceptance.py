"""The engine's own acceptance counter — the replayer's independent check.

`Engine.generate` returns `"acceptance"` (accepted / drafted). It does not travel
through `friday_calibrate.runner.Sample`, but a direct call needs no Sample and
touches nothing that is frozen. If the engine reports `0.0` on this workload, the
replay's "no draft was ever accepted" is confirmed by the engine rather than
reconstructed from its output.

Run: ``python experiments/identity_break/acceptance.py --execute``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
IRONMULE = ROOT / ".worktrees" / "friday-optimizer-ironmule"

STUDY_ID = "identity-break-acceptance-20260902-01"
OUTPUT = Path(__file__).resolve().parent
COMBINED = {"head_skip_prefill": True, "compiled_fixed_cache": True, "readback_every": 8}
TOKENS = 128
WIDTHS = (1, 2, 3)


def self_check() -> int:
    print(json.dumps({"state": "self_check", "study_id": STUDY_ID,
                      "measures": "Engine.generate()['acceptance'] on the sealed workload",
                      "widths": list(WIDTHS), "tokens": TOKENS,
                      "formal_claim": False, "no_activation": True}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from _bench import (BudgetGuard, enforce_offline, release_gate, require_ac_power,
                        resolve_local_model_snapshot, study_provenance)

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate

    require_ac_power()
    guard = BudgetGuard()
    enforce_offline()
    sys.path.insert(0, str(IRONMULE))
    from ironmule import Engine, Knobs  # noqa: E402
    from mlx_lm import load  # noqa: E402

    model_id = "mlx-community/gemma-3-4b-it-4bit"
    snapshot = resolve_local_model_snapshot(model_id)
    model, tokenizer = load(str(snapshot.path))
    filler = ("You are a careful engineering assistant working in a Python repository. "
              "Follow the existing style and explain your reasoning briefly. ") * 40
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": filler + "\n\n" + "Why is false sharing slow?"}],
        add_generation_prompt=True)
    ids = list(templated if isinstance(templated, list) else tokenizer.encode(templated))
    eos_ids = tuple(sorted({int(v) for v in (getattr(tokenizer, "eos_token_id", None),
                                             *(getattr(tokenizer, "eos_token_ids", None) or ()))
                            if isinstance(v, int)}))

    debt = 0.0

    def charge(seconds: float) -> None:
        nonlocal debt
        guard.record_gpu(seconds)
        debt += seconds * (1 - 0.15) / 0.15
        while debt >= 4.0:
            guard.required_break()
            debt -= 4.0

    started = time.time()
    rows = []
    for width in WIDTHS:
        engine = Engine(model, tokenizer, Knobs(**dict(COMBINED, speculate_k=width,
                                                       speculate_ngram=3)))
        at = time.perf_counter()
        out = engine.generate(ids, TOKENS, eos_ids)
        charge(time.perf_counter() - at)
        rows.append({"speculate_k": width, "acceptance": out["acceptance"],
                     "tokens": len(out["logical_tokens"]),
                     "decode_seconds": out["decode_ns"] / 1e9})
    report = {
        "study_id": STUDY_ID, "state": "measured", "model_id": model_id,
        "model_revision": snapshot.revision, "prompt_tokens": len(ids),
        "arm": COMBINED, "rows": rows,
        "all_zero_acceptance": all(row["acceptance"] == 0.0 for row in rows),
        "wall_seconds": round(time.time() - started, 1), "budget": guard.summary(),
        "provenance": study_provenance([Path(__file__), IRONMULE / "ironmule" / "runtime.py"],
                                       extra={"model_id": model_id}),
        "formal_claim": False, "no_activation": True,
    }
    (args.out or OUTPUT / "acceptance.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
