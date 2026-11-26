"""Sensitivity tests for the CPU replay of nested hardware observations."""

import copy
import hashlib

import pytest

from lab.pk1 import post_restart_native as trace
from lab.pk1 import post_restart_native_verify as verify


def record(phase, decoded=0.04):
    raw = {
        "phase": phase,
        "owner": "GgmlEngine.price",
        "source_callers": ["GgmlEngine.price"],
        "source_lines": [["GgmlEngine.price", 14]],
        "rows": 2,
        "sequences": 1,
        "logits": 1,
        "row_bucket": 2,
        "mask_bits": [3, 7],
        "threads": 2,
        "argmax_on_device": True,
        "known_logical_kv": True,
        "row_sha256": "0" * 64,
        "kv_sha256": "1" * 64,
        "output_sha256": "2" * 64,
        "decode_s": decoded,
        "readback_s": 0.01,
        "wrapper_s": 0.06,
        "native_statuses": [0],
        "decode_calls": 1,
        "readback_calls": 1,
        "activity_sampled": False,
    }
    raw["input_sha256"] = trace.digest(
        [
            raw[k]
            for k in (
                "row_sha256",
                "kv_sha256",
                "mask_bits",
                "threads",
                "argmax_on_device",
            )
        ]
    )
    return raw


def test_rejects_false_native_identity_time_and_kv_overlap():
    sha = "3" * 64
    arm = {
        "mask_bits": [3, 7],
        "initialized_s": 0.5,
        "binding": {"native": {"sha256": {"llama_backend_init": sha}}},
        **{key: {"wall_s": 0.5} for key in ("cold", "warm", "factual_timing")},
    }
    row = {
        "records": [record(phase) for phase in verify.PHASES],
        "arm": arm,
        "abi_images": dict.fromkeys(verify.ABI, sha),
        "abi_symbols": [*verify.ABI, "llama_memory_clear", "llama_memory_seq_rm"],
    }
    locations = {"GgmlEngine.price": {14}}
    verify.check_records(row, locations)
    for k, v, reason in [
        ("source_lines", [["GgmlEngine.price", 90]], "location"),
        ("decode_s", 0.055, "overlapping"),
        ("known_logical_kv", False, "unbound"),
        ("readback_calls", 0, "incomplete"),
    ]:
        altered = copy.deepcopy(row)
        altered["records"][0][k] = v
        with pytest.raises(ValueError, match=reason):
            verify.check_records(altered, locations)
    row["abi_images"]["llama_decode"] = "f" * 64
    with pytest.raises(ValueError, match="unbound ABI"):
        verify.check_records(row, locations)


def test_activity_cannot_create_credit_for_unobserved_or_repeated_cell():
    records = [record("startup"), record("startup"), record("warm")]
    selected = trace.select_cells(records)
    device = [{"name": "local", "cc": "80", "uuid": "aa"}]
    empty = {
        "schema": "ironmule.native-activity.v1",
        "sample": 1,
        "status": "unsupported",
        "events": [],
        "device_events": 0,
        "summed_device_s": None,
        "detail": "unavailable",
        "capture_wall_s": 0.055,
        "native_call_wall_s": 0.05,
    }
    row = {
        "records": records,
        "selected_cells": selected,
        "not_observed": [],
        "events": [],
        "arm": {"binding": {"fp": {"devices": device}}},
    }
    for ordinal, i in enumerate((0, 2), 1):
        records[i]["activity_sampled"] = True
        row["events"].append(
            {
                "cell": trace.cell(records[i]),
                "record_index": i,
                "activity": {**empty, "sample": ordinal},
            }
        )
    verify.check_samples(row, selected)
    row["events"][0]["record_index"] = 1
    with pytest.raises(ValueError, match="wrong or repeated"):
        verify.check_samples(row, selected)
    row["events"][0]["record_index"] = 0
    row["events"][0]["activity"]["device_events"] = 1
    with pytest.raises(ValueError, match="unsupported activity"):
        verify.check_samples(row, selected)


def test_device_time_conservation_includes_omitted_symbols():
    maps = {
        tag: [[tag, "aa:01", "1234"]]
        for tag in ("libcudart", "libcublas", "libcublasLt")
    }
    name = "a::kernel"
    raw = {
        "schema": "ironmule.native-activity.v1",
        "sample": 1,
        "status": "measured",
        "visible_devices": ["a100"],
        "events": [
            {
                "symbol": name,
                "symbol_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "calls": 25,
                "device_s": 0.013,
            }
        ],
        "device_events": 29,
        "summed_device_s": 0.014,
        "omitted_calls": 4,
        "omitted_device_s": 0.001,
        "detail": None,
        "capture_wall_s": 0.05,
        "native_call_wall_s": 0.045,
        "mapped_after": maps,
        "mapped_after_import": maps,
        "mapped_before": {},
    }
    verify.check_activity(raw, record("startup"), 1, ["a100"])
    raw["summed_device_s"] += 0.001
    with pytest.raises(ValueError, match="accounting"):
        verify.check_activity(raw, record("startup"), 1, ["a100"])
