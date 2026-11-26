"""Price measured first-use opportunities against the complete original product contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
import subprocess
from pathlib import Path

from training import calibration_portfolio as portfolio
from training import portfolio_shadow as shadow
from training.calibration_stability import sha
from training.code_author import write_once
from training.code_trial import read
from training.trial_release import require

FIELDS = {"padding": "pad_to", "passes": "pass_costs", "pilot": "costs"}


def source_line(path, anchor):
    hits = [
        i for i, text in enumerate(path.read_text().splitlines(), 1) if anchor in text
    ]
    require(len(hits) == 1, f"missing or ambiguous product contract {anchor!r}")
    return hits[0]


def consumer_contract(checkout, raw_arm, protocol):
    engine = checkout / "ironmule/ggml_engine.py"
    require(sha(engine) == protocol["source_sha256"], "different PR60 engine")
    members = (
        "ironmule/ggml_engine.py",
        "ironmule_product/ggml_bridge.py",
        "ironmule_product/worker.py",
        "ironmule_product/service.py",
        "ironmule_product/cli.py",
    )
    for member in members:
        pinned = subprocess.check_output(
            ["git", "show", f"{protocol['base_commit']}:{member}"], cwd=checkout
        )
        require(
            hashlib.sha256(pinned).hexdigest() == sha(checkout / member),
            "different product source",
        )
    harness = Path(__file__).resolve().parents[1] / "lab/pk1/restart_arm.py"
    require(sha(harness) == raw_arm["harness_sha256"], "different measured harness")
    engine_text = engine.read_text()
    engine_ast = ast.parse(engine_text)
    owner = next(
        n
        for n in engine_ast.body
        if isinstance(n, ast.ClassDef) and n.name == "GgmlEngine"
    )
    reuse = next(n for n in owner.body if getattr(n, "name", None) == "_even_reuse")
    guard = next(
        n
        for n in reuse.body
        if isinstance(n, ast.If)
        and '"IRONMULE_PS1"' in ast.get_source_segment(engine_text, n.test)
    )
    require(
        isinstance(guard.body[-1], ast.Return)
        and ast.unparse(guard.body[-1].value) == "group"
        and guard.end_lineno
        < source_line(engine, 'costs = getattr(self, "pass_costs", None)'),
        "unknown scheduling dependency",
    )
    bridge, worker, service, cli = (checkout / member for member in members[1:])
    return {
        "source_sha256": {p: sha(checkout / p) for p in members},
        "harness_sha256": sha(harness),
        "flags": raw_arm["binding"]["flags"],
        "ordinary_serve": {
            "cli_product_service": source_line(
                cli, "service = ProductService(store, backend=backend, spec=spec)"
            ),
            "cli_listener_health": source_line(
                cli, '"ready": service.health()["ready"]'
            ),
            "worker_ready_uses_metadata": source_line(
                worker, '**({"engine": engine_bridge.metadata()}'
            ),
            "bridge_static_metadata": source_line(bridge, "def metadata(self)"),
            "health_reads_ready_engine": source_line(
                service, 'or {}).get("engine") or {}'
            ),
            "done_requires_first_terminal_metadata": source_line(
                worker, '"engine": engine_section(computed)'
            ),
            "done_calls_counters": source_line(
                worker, 'getattr(engine_bridge, "counters", dict)()'
            ),
            "bridge_calls_counters_after_serving": source_line(
                bridge, "metadata = {**self.metadata(), **self.counters()} if served"
            ),
            "bridge_counter_reads_whole_summary": source_line(
                bridge, "summary = self._engine.summary()"
            ),
            "summary_exports_pack_pass_ms": source_line(
                engine, "for r, t in self.pass_costs.points"
            ),
            "summary_exports_padding_prices": source_line(
                engine, "in self.pad_prices.items()"
            ),
            "summary_exports_pilot_status": source_line(engine, "self.pilot.status()"),
            "scope": "Ready/health use metadata or cached status. A streamed terminal builds current counters. The complete pack.pass_ms table is required at that point on this packed profile. The bridge also builds a summary after served jobs.",
        },
        "unmodified_harness_collects_full_prices_line": source_line(
            harness, '"passes": eng.pass_costs.points'
        ),
        "ps1_returns_before_scheduling_pass_reads": [guard.lineno, guard.end_lineno],
    }


def assess(data, control, weights, checkout):
    folder = data / "a100-portfolio-shadow"
    paths = [folder / f"{s}.json" for s in read(shadow.CONTRACT)["seeds"]]
    gr = shadow.grade(data, control, weights, folder / "prediction.json", paths)
    require(gr == read(folder / "result.json"), "unrelated observations")
    originals = [read(path) for path in paths]
    assumptions = consumer_contract(
        checkout, originals[0]["arm"], read(shadow.CONTRACT)
    )
    accounting = read(data / "a100-calibration-sampling/target-accounting.json")
    experiment_s = accounting["cost"]["seconds"]["target_trial_elapsed_s"]
    products = control.parent / "product-v3"
    failure_path = products / "http-transport-diagnostic.json"
    failure = read(failure_path)
    require(
        read(products / "manifest.json")["files"][failure_path.name]["sha256"]
        == sha(failure_path),
        "unrecognized product evidence",
    )
    require(
        failure["complete_arm_receipts"] == 0,
        "new complete ordinary comparison requires review",
    )
    ranking = []
    for family, field in FIELDS.items():
        uses = [
            r["serving_field_reads"][phase].get(field, {}).get("count", 0)
            for r in originals
            for phase in ("cold", "warm", "factual")
        ]
        seconds = [
            next(
                p["elapsed_s"]
                for p in r["calibrations"]
                if p["method"] == portfolio.METHODS[family]
            )
            for r in originals
        ]
        # A rank is for information only. Moving a measurement after readiness does not remove
        # the work from first complete serving and the public telemetry contract.
        ranking.append(
            {
                "family": family,
                "field": field,
                "measured_calibration_s": seconds,
                "six_measured_serving_reads": uses,
                "observed_read_count": sum(uses),
                "whole_wave_without_read": sum(v == 0 for v in uses),
                "upper_before_first_exclusive_waves_s": min(seconds)
                if not any(uses)
                else 0.0,
                "verified_avoidable_full_contract_s": 0.0,
            }
        )
    ranking.sort(
        key=lambda p: (-p["upper_before_first_exclusive_waves_s"], p["family"])
    )
    leader = ranking[0]
    bound = leader["upper_before_first_exclusive_waves_s"]
    return {
        "schema": "ironmule.calibration-consumer-replan.v1",
        "portfolio_observations_sha256": [sha(path) for path in paths],
        "portfolio_result_sha256": sha(folder / "result.json"),
        "product_diagnostic_sha256": sha(failure_path),
        "product_terminal_comparisons": failure["complete_arm_receipts"],
        "consumer_contract": assumptions,
        "ranked_from_direct_engine_waves": ranking,
        "question": f"Can the {leader['family']} state be produced more cheaply while preserving every current price, the first streamed terminal frame, and the original native state?",
        "comparator": "Active PR60 original full calibrations and fixed qualified PS1/PK6 scheduling; its current measurements, digests, allocations and telemetry are charged.",
        "projection": {
            "previous_22_process_trial_s": experiment_s,
            "previous_22_process_trial_usd": accounting["cost"][
                "attributable_trial_only_usd"
            ],
            "gross_256_starts_upper_s": 256 * bound,
            "full_work_break_even_starts": None,
            "ideal_start_count_at_full_phase_elimination_before_public_work": math.ceil(
                experiment_s / bound
            )
            if bound
            else None,
            "physical_current_price_table_required_by_first_product_completion": True,
        },
        "decision": "retain current measured complete profile",
        "source_order": None,
        "new_paid_action": None,
        "reason": "The scheduling loop bypasses the leading field on this exact PS1 profile. The first ordinary streamed terminal still publishes the fresh measured table and the fixed-work harness collects it. Moving full measurement into that first consumer cannot save verified full-path work and clearing a live context can affect serving. An isolated after-service sentinel also drifted outside the frozen admission limit. There is no qualified substitute for the first complete, current six-width table nor a measured positive overlap after paying shared device and memory contention.",
    }


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "control", "weights", "checkout", "out"):
        cmd.add_argument("--" + flag, type=Path, required=True)
    args = cmd.parse_args()
    write_once(args.out, assess(args.data, args.control, args.weights, args.checkout))
