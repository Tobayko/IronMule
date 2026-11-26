"""Independently replay source ownership, native accounting and activity selection for two A100 starts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from itertools import groupby
from pathlib import Path

from lab.pk1 import post_restart_activity, post_restart_native, restart_trial
from lab.pk1.post_restart_verify import SOURCE, check_selection, finite, sha
from training import scoped_release
from training.trial_release import require

FROZEN = (
    Path(__file__).resolve().parents[2] / "training/post_restart_native_contract.json"
)
PHASES = ("startup", "cold", "warm", "factual")
ABI = ("llama_decode", "llama_get_logits_ith")


def source_locations(source):
    """Locate frames in the exact merged source across CPython versions."""
    result = defaultdict(set)
    (engine,) = [
        n
        for n in ast.parse(source.read_bytes()).body
        if isinstance(n, ast.ClassDef) and n.name == "GgmlEngine"
    ]

    def visit(node, scope):
        if isinstance(node, ast.FunctionDef):
            name = scope + "." + node.name
        elif isinstance(
            node,
            (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        ):
            display = {
                ast.Lambda: "lambda",
                ast.ListComp: "listcomp",
                ast.SetComp: "setcomp",
                ast.DictComp: "dictcomp",
                ast.GeneratorExp: "genexpr",
            }[type(node)]
            name = scope + ".<" + display + ">"
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, scope)
            return
        result[name].update(range(node.lineno, node.end_lineno + 1))
        for child in ast.iter_child_nodes(node):
            visit(child, name + ".<locals>")

    for node in engine.body:
        visit(node, engine.name)
    return result


def hash_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def check_activity(raw, record, ordinal, devices):
    status = raw["status"]
    require(
        raw["schema"] == "ironmule.native-activity.v1"
        and raw["sample"] == ordinal
        and status in ("measured", "no_cuda_events", "unsupported"),
        "unregistered activity sample",
    )
    require(
        finite(raw["native_call_wall_s"])
        and finite(raw["capture_wall_s"])
        and raw["native_call_wall_s"]
        <= raw["capture_wall_s"]
        <= record["wrapper_s"] + 1e-5,
        "activity duration outside enclosing call",
    )
    events = raw["events"]
    if status == "unsupported":
        require(
            raw["device_events"] == 0
            and raw["summed_device_s"] is None
            and events == []
            and isinstance(raw["detail"], str)
            and raw["detail"],
            "unsupported activity credited as device work",
        )
        return
    require(
        raw["visible_devices"] == devices and raw["detail"] is None,
        "device visibility differs",
    )
    maps = raw["mapped_after"]
    require(
        set(maps) == {"libcudart", "libcublas", "libcublasLt"}
        and all(len(v) == 1 for v in maps.values())
        and raw["mapped_after_import"] == maps
        and all(maps.get(k) == v for k, v in raw["mapped_before"].items()),
        "vendor mappings changed in capture",
    )
    count, total = raw["device_events"], raw["summed_device_s"]
    require(
        integer(count)
        and finite(total, allow_zero=True)
        and isinstance(events, list)
        and len(events) <= 48
        and integer(raw["omitted_calls"])
        and finite(raw["omitted_device_s"], allow_zero=True),
        "invalid device event totals",
    )
    names = set()
    for event in events:
        symbol, symbol_hash = event["symbol"], event["symbol_sha256"]
        require(
            isinstance(symbol, str)
            and 0 < len(symbol) <= 512
            and hash_string(symbol_hash)
            and symbol_hash not in names
            and integer(event["calls"], 1)
            and finite(event["device_s"]),
            "invalid device symbol or time",
        )
        if len(symbol) < 512:
            require(
                hashlib.sha256(symbol.encode()).hexdigest() == symbol_hash,
                "device symbol identity differs",
            )
        names.add(symbol_hash)
    require(
        sum(e["calls"] for e in events) + raw["omitted_calls"] == count
        and math.isclose(
            math.fsum([e["device_s"] for e in events] + [raw["omitted_device_s"]]),
            total,
            rel_tol=1e-8,
            abs_tol=1e-9,
        )
        and (
            (status == "measured" and count > 0 and total > 0)
            or (status == "no_cuda_events" and count == 0 and total == 0)
        ),
        "device event accounting differs",
    )


def check_records(row, locations):
    arm, records = row["arm"], row["records"]
    require(
        [k for k, _ in groupby(records, lambda r: r["phase"])] == list(PHASES),
        "missing or reordered call phase",
    )
    llama_sha = arm["binding"]["native"]["sha256"]["llama_backend_init"]
    require(
        row["abi_images"] == dict.fromkeys(ABI, llama_sha)
        and hash_string(llama_sha)
        and set((*ABI, "llama_memory_clear", "llama_memory_seq_rm"))
        <= set(row["abi_symbols"]),
        "unbound ABI symbols",
    )
    durations = defaultdict(float)
    for r in records:
        owners, lines = r["source_callers"], r["source_lines"]
        require(
            owners == [name for name, _ in lines]
            and len(owners) <= 12
            and r["owner"] == (owners[0] if owners else "outside-engine")
            and all(lineno in locations.get(name, ()) for name, lineno in lines),
            "unknown source caller or location",
        )
        n, seqs, logits = r["rows"], r["sequences"], r["logits"]
        require(
            integer(n, 1)
            and integer(seqs, 1)
            and seqs <= n
            and integer(logits)
            and logits <= n
            and r["row_bucket"] == n.bit_length()
            and integer(r["threads"], 1)
            and len(r["mask_bits"]) == 2
            and all(integer(b) and b <= 0xFFFFFFFF for b in r["mask_bits"])
            and type(r["argmax_on_device"]) is bool
            and type(r["activity_sampled"]) is bool,
            "invalid decode geometry or flags",
        )
        statuses = r["native_statuses"]
        require(
            statuses in ([0], [1, 0])
            and all(type(s) is int for s in statuses)
            and integer(r["decode_calls"], 1)
            and r["decode_calls"] == len(statuses)
            and integer(r["readback_calls"])
            and r["readback_calls"] == logits,
            "incomplete successful ABI calls",
        )
        require(
            finite(r["decode_s"])
            and finite(r["readback_s"], allow_zero=True)
            and finite(r["wrapper_s"])
            and ((r["readback_s"] > 0) == (logits > 0))
            and r["decode_s"] + r["readback_s"] <= r["wrapper_s"] + 1e-6,
            "overlapping ABI intervals",
        )
        known = r["known_logical_kv"]
        require(
            type(known) is bool
            and hash_string(r["output_sha256"])
            and hash_string(r["row_sha256"])
            and (hash_string(r["kv_sha256"]) if known else r["kv_sha256"] is None),
            "unbound input or return",
        )
        fingerprint = [
            r[k]
            for k in (
                "row_sha256",
                "kv_sha256",
                "mask_bits",
                "threads",
                "argmax_on_device",
            )
        ]
        require(
            r["input_sha256"]
            == (post_restart_native.digest(fingerprint) if known else None),
            "logical input digest differs",
        )
        if r["phase"] != "startup":
            require(
                r["mask_bits"] == arm["mask_bits"],
                "serving flags differ from released profile",
            )
        durations[r["phase"]] += r["wrapper_s"]
    envelopes = [
        arm["initialized_s"],
        arm["cold"]["wall_s"],
        arm["warm"]["wall_s"],
        arm["factual_timing"]["wall_s"],
    ]
    require(
        all(
            durations[k] <= wall + 1e-4
            for k, wall in zip(PHASES, envelopes, strict=True)
        ),
        "nested call durations exceed workload wall time",
    )


def check_samples(row, selected):
    records = row["records"]
    require(
        row["selected_cells"] == selected,
        "second capture differs from first measured inventory",
    )
    indices = {}
    missing = []
    for option in selected:
        key = option["cell"]
        i = next(
            (i for i, r in enumerate(records) if post_restart_native.cell(r) == key),
            None,
        )
        if i is None:
            missing.append(key)
        else:
            indices[i] = key
    require(
        row["not_observed"] == missing
        and sorted(indices)
        == [i for i, r in enumerate(records) if r["activity_sampled"]]
        and [e["record_index"] for e in row["events"]] == sorted(indices),
        "sampled wrong or repeated call",
    )
    for ordinal, event in enumerate(row["events"], 1):
        i = event["record_index"]
        require(event["cell"] == indices[i], "device sample with wrong selected cell")
        check_activity(
            event["activity"],
            records[i],
            ordinal,
            row["arm"]["binding"]["fp"]["devices"],
        )


def verify_pair(observations, control, weights, contract, checkout):
    limits = json.loads(contract.read_bytes())
    require(sha(contract) == sha(FROZEN), "unregistered native discovery")
    receipt = json.loads(control.read_bytes())
    q = receipt["qualification"]
    scoped_release.validate_order("qwen-restart", receipt["order"])
    scoped_release.validate(q)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{commit}"], cwd=checkout, text=True
    ).strip()
    source = checkout / SOURCE
    bound = {
        "base_commit": actual,
        "parent_order": receipt["order"]["order_id"],
        "control_receipt_sha256": sha(control),
        "weights_sha256": sha(weights),
        "source_sha256": sha(source),
    }
    require(
        all(limits[k] == v for k, v in bound.items())
        and receipt["adapter"] == "qwen-restart"
        and receipt["publication"]["pr"] == 60
        and receipt["publication"]["candidate"] == actual
        and q["candidate_sha256"] == bound["source_sha256"],
        "unrelated control, checkpoint or merged source",
    )
    locations = source_locations(source)
    rows = sorted(
        ((p, json.loads(p.read_bytes())) for p in observations),
        key=lambda pair: pair[1]["seed"],
    )
    require(
        [r["seed"] for _, r in rows] == limits["seeds"],
        "expected both preregistered seeds",
    )
    previous = None
    selected = []
    for path, row in rows:
        require(
            row["schema"] == limits["observation_schema"]
            and all(row[k] == v for k, v in bound.items())
            and row["contract_sha256"] == sha(contract)
            and row["profiler_sha256"] == sha(post_restart_native.__file__)
            and row["activity_helper_sha256"] == sha(post_restart_activity.__file__)
            and row["prior_observation_sha256"] == previous
            and row["selection_before"] == row["selection_after"] == "candidate",
            "changed native observation",
        )
        check_selection(row["selection"], weights, receipt, limits)
        arm = row["arm"]
        restart_trial.validate(arm, row["seed"], expected_source=bound["source_sha256"])
        restart_trial.verify_binding(arm, receipt["order"])
        binding = q["binding"]
        require(
            arm["binding"] == binding["runtime"]
            and arm["reused"] is True
            and arm["factual_first"] == []
            and arm["cuda_vendor"] == binding["vendor"]
            and arm["mask_bits"] == binding["mask_bits"]
            and arm["pilot_sha256"] == binding["pilot_sha256"]
            and finite(row["broker_elapsed_s"])
            and arm["broker_elapsed_s"] <= row["broker_elapsed_s"] + 1e-6,
            "incomplete calibrated worker",
        )
        check_records(row, locations)
        check_samples(row, selected)
        if previous is None:
            selected = post_restart_native.select_cells(row["records"])
            require(len(selected) == 2, "no regular work following startup")
        previous = sha(path)
    return {
        **bound,
        "contract_sha256": sha(contract),
        "seeds": limits["seeds"],
        "observation_sha256": [sha(p) for p, _ in rows],
        "selected_cells": selected,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observation", type=Path, action="append", required=True)
    for field in ("control", "weights", "contract", "checkout"):
        p.add_argument("--" + field, type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            verify_pair(a.observation, a.control, a.weights, a.contract, a.checkout),
            sort_keys=True,
        )
    )
