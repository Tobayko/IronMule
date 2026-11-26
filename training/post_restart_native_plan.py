"""Rank genuinely overlapping startup setup and locate the measured native interval below the source wrapper."""
from __future__ import annotations

import argparse
import ast
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from lab.pk1 import native_order, post_restart_native
from lab.pk1.followup_contract import contract_digest
from training.code_author import write_once
from training.code_trial import read

SOURCE = "ironmule/ggml_engine.py"


def source_methods(path):
    tree = ast.parse(path.read_bytes())
    engine = next(cls for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "GgmlEngine")
    methods = {}
    for fn in engine.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        loops = set()
        for node in ast.walk(fn):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                loops.update(range(node.lineno, node.end_lineno + 1))
        methods["GgmlEngine." + fn.name] = {"line": fn.lineno, "loops": loops, "decorated": bool(fn.decorator_list)}
    return methods


def setup_credits(records, methods):
    """One conservative potential duplicate per signature. Same-owner repetitions provide no credit."""
    grouped = defaultdict(dict)
    modes = Counter()
    for idx, row in enumerate(records):
        owner = row["owner"]
        if row["phase"] != "startup":
            continue
        if (not row["known_logical_kv"] or not row["input_sha256"] or row["native_statuses"] != [0]
                or row["activity_sampled"]):
            modes["unknown state, retry or instrumented sample"] += 1
            continue
        if (owner not in methods or owner == "GgmlEngine.__init__" or methods[owner]["decorated"]
                or any("check_" in n or "qualif" in n or "verify" in n or "measure_kg" in n for n in row["source_callers"])):
            modes["protected path or indirect timed helper"] += 1
            continue
        if row["source_lines"][0][1] in methods[owner]["loops"]:
            modes["loop or timing repetition"] += 1
            continue
        key = (row["input_sha256"], row["output_sha256"])
        secs = row["decode_s"] + row["readback_s"]
        loc = row["source_lines"][0][1]
        options = grouped[key]
        # Repeats at a source site never receive independent simulated savings.
        if owner not in options or secs < options[owner]["abi_s"]:
            options[owner] = {"owner": owner, "site_line": loc, "abi_s": secs, "record": idx}
        modes["known non-loop setup"] += 1
    credits = {}
    for key, ops in grouped.items():
        if len(ops) < 2:
            continue
        ordered = sorted(ops.values(), key=lambda x: (x["abi_s"], x["owner"]))
        first, rest = ordered[0], ordered[1:]
        # Keep one shared execution. Across sites the lower observed cost is a conservative bound.
        # Keep at most one pair per signature so the observed setup cannot be counted twice.
        other = min(rest, key=lambda x: x["owner"])
        pair = sorted([first, other], key=lambda x: x["owner"])
        names = tuple((r["owner"], r["site_line"]) for r in pair)
        credits[(*names, *key)] = {"matched": pair, "potential_s": first["abi_s"]}
    return credits, dict(modes)


def inventory(rows):
    phases = defaultdict(lambda: {"calls": 0, "decode_s": 0., "readback_s": 0., "rows": 0})
    for item in rows:
        summary = phases[item["phase"], item["owner"]]
        summary["calls"] += 1
        for key in ("rows", "decode_s", "readback_s"):
            summary[key] += item[key]
    return [{"phase": phase, "owner": owner, **summary} for (phase, owner), summary in
            sorted(phases.items(), key=lambda p: (-p[1]["decode_s"] - p[1]["readback_s"], p[0]))]


def device_evidence(events, checkout):
    kernel_src = (checkout / native_order.SOURCE).read_text()
    patches = defaultdict(list)
    for path in sorted((checkout / "native/ggml_patches").rglob("*.diff")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            definition = re.match(r"^[+>]\s*.*?__global__\s+void\s+(ironmule_\w+)\s*\(", line)
            if definition:
                patches[definition[1]].append({"path": str(path.relative_to(checkout)), "line": lineno})
    result = []
    for event in events:
        sample = event["activity"]
        names = []
        for symbol in sample["events"]:
            match = native_order.DEVICE_SYMBOL.search(symbol["symbol"])
            source = None
            if match and (line := next((idx for idx, s in enumerate(kernel_src.splitlines(), 1)
                                        if s.startswith("__global__ void " + match[1] + "(")), None)):
                source = {"path": native_order.SOURCE, "function": match[1], "line": line}
            project_symbol = re.search(r"\b(ironmule_\w+)(?:<[^()]*>)?\(", symbol["symbol"])
            definitions = patches.get(project_symbol[1], []) if project_symbol else []
            names.append({**symbol, "matching_owned_definition": source, "matching_ggml_patch_definitions": definitions,
                          "interpretation": "symbol matches bound W4 source; a changed binary requires fresh qualification" if source
                          else "symbol matches project GGML patch; changed GGML image invalidates the active native receipt" if definitions
                          else "unattributed device image or external library"})
        result.append({"cell": event["cell"], "record_index": event["record_index"], "status": sample["status"],
                       "detail": sample["detail"], "device_s": sample["summed_device_s"], "symbols": names})
    return result


def plan(observations, control, weights, contract, checkout, hourly, incident):
    from lab.pk1.post_restart_native_verify import verify_pair

    if not math.isfinite(hourly) or hourly <= 0:
        raise ValueError("missing positive actual lease hourly cost")
    incident_hash = hashlib.sha256(incident.read_bytes()).hexdigest()
    if incident_hash != "76c9fe067c28a21d0eec98bf136c545f8fc8237b222c9d7f1333277656947016":
        raise ValueError("missing or foreign timing incident receipt")
    proof = verify_pair(observations, control, weights, contract, checkout)
    protocol = read(contract)
    rows = sorted((read(p) for p in observations), key=lambda r: r["seed"])
    methods = source_methods(checkout / SOURCE)
    inventories = [inventory(r["records"]) for r in rows]
    collected = [setup_credits(r["records"], methods) for r in rows]
    diagnostics = sum(r["broker_elapsed_s"] for r in rows)
    qualifying = sum(protocol["qualification_workers"].values()) * max(r["broker_elapsed_s"] for r in rows)
    threshold = max(r["arm"]["cold_s"] for r in rows) * (1 - 1 / 1.02)
    # Use only duplicates whose *whole* signature and both source locations recur across starts.
    prior, confirmation = (x[0] for x in collected)
    ranked = []
    for key in sorted(set(prior) & set(confirmation)):
        times = [prior[key]["potential_s"], confirmation[key]["potential_s"]]
        predicted = .25 * min(times) - abs(times[0] - times[1])
        ranked.append({"sites": list(key[:2]), "input_sha256": key[2], "output_sha256": key[3],
                       "records": [prior[key]["matched"], confirmation[key]["matched"]],
                       "measured_lower_duplicate_abi_s": times, "conditional_saved_s": max(0., predicted),
                       "eligible": predicted > threshold and 256 * predicted > qualifying + diagnostics})
    ranked.sort(key=lambda c: (-c["conditional_saved_s"], c["sites"]))
    common = {"base_commit": protocol["base_commit"], "parent_order": protocol["parent_order"],
              "weights_sha256": protocol["weights_sha256"], "control_receipt_sha256": protocol["control_receipt_sha256"],
              "source_sha256": protocol["source_sha256"], "contract": contract_digest(), "verification": proof,
              "observations": proof["observation_sha256"], "candidates": ranked, "excluded_counts": [x[1] for x in collected],
              "native_abi_inventory": inventories, "activity_selection": post_restart_native.select_cells(rows[0]["records"]),
              "device_activity": device_evidence(rows[1]["events"], checkout), "simulation_kind": "simulated",
              "required_cold_saving_s": threshold, "future_starts_for_price_sensitivity": 256,
              "measured_discovery_s": diagnostics, "discovery_usd": diagnostics * hourly / 3600,
              "estimated_qualification_s": qualifying, "estimated_qualification_usd": qualifying * hourly / 3600,
              "lease_rate_usd_per_hour": hourly,
              "scope": "Real ABI timings can include GPU waits. Device times can overlap. Logical setup equality does not imply transferable CUDA graph state. Qualification cost includes 22 representative worker lifetimes."}
    # The operator independently recorded a concurrent nested CUDA allocation attempt during
    # the first start. Its effects on timing cannot be reconstructed from in-process traces.
    # This exact historical pair must remain ineligible even if a caller omits a warning flag.
    return {**common, "schema": "ironmule.post-restart-native-deferred.v1", "selected": None,
            "timing_eligible": False, "training_eligible": False, "authoring_eligible": False,
            "confounder": {"evidence": incident.name, "sha256": incident_hash, "reason": "Independent nested tokenizer process attempted a CUDA allocation during the first inventory. The dependent second sample was selected from these timings."},
            "reason": "Observational source, call identities and kernel symbols retained. Foreground scientific timing and price labels are ineligible. A changed GGML or W4 binary would require fresh calibration."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observation", type=Path, action="append", required=True)
    for k in ("control", "weights", "contract", "checkout", "incident", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--hourly-usd", type=float, required=True)
    a = p.parse_args()
    write_once(a.out, plan(a.observation, a.control, a.weights, a.contract, a.checkout, a.hourly_usd, a.incident))


if __name__ == "__main__":
    main()
