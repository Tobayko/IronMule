import json
from pathlib import Path

import pytest

from lab.pk6.native_dispatch.live_policy import choose
from lab.pk6.native_dispatch.score import exclusion, wave_energy
from lab.pk6.native_dispatch.tile_order import order
from training.code_author import read_request


def test_native_order_is_reproducible_from_only_counted_requests(tmp_path):
    root = (
        Path(__file__).resolve().parents[2]
        / "lab/autonomous/results/l40s-native-dispatch"
    )
    result = order(root / "http/primary", tmp_path)
    archived_id, original = read_request(root / "tile-order/author")
    request_id, replay = read_request(tmp_path / "author")
    assert result["status"] == "awaiting_codex" and archived_id == request_id
    assert replay == original and replay["order"]["target"]["ggml_type"] == 12


def test_power_estimate_requires_two_bracketing_samples_and_rejects_an_outage():
    wave = {"unix_start": 1.25, "unix_end": 1.75}
    assert wave_energy([(1.0, 100.0, 20.0), (2.0, 100.0, 40.0)], wave) == pytest.approx(
        15.0
    )
    assert wave_energy([(1.5, 100.0, 20.0), (2.0, 100.0, 40.0)], wave) is None
    assert wave_energy([(1.0, 100.0, 20.0), (5.0, 100.0, 40.0)], wave) is None


def test_observed_kernel_fallback_is_ineligible_even_with_matching_outputs():
    assert exclusion(
        "F1",
        {
            "prompt_plugin_dispatches_in_http_by_width": {"32": 100},
            "prompt_plugin_dispatches_in_http_quality": 0,
        },
    )
    assert exclusion("N1", {"shared_pool": True, "streams_pass": False})
    assert exclusion("P2", {})


def test_foreign_pilot_scope_falls_back_before_loading_any_weights(tmp_path):
    from lab.pk6.warm_repeat import GGUF

    (tmp_path / "fit.json").write_text(
        json.dumps(
            {
                "target": {
                    "hardware": {"name": "original"},
                    "gguf": GGUF,
                    "ggml_cuda": "one",
                }
            }
        )
    )
    assert choose(tmp_path, {"name": "other"}, "one")["profile"] == "stock"


def test_grid_keeps_only_cells_confirmed_on_an_independent_seed(tmp_path):
    from lab.pk6.native_dispatch.grid_screen import MODES, SEEDS, WIDTHS, select

    protocol = {
        "hardware": "local",
        "reference_cuda": "native",
        "gguf": "weights",
        "backend": "isolated",
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    for n in range(1, 6):
        for seed in SEEDS:
            cells = {}
            for b in WIDTHS:
                entries = {
                    m: {"us": 12, "graphs": 8, "dispatches": 8, "relative_rms": 0}
                    for m in MODES
                }
                entries["stock"]["us"] = 10
                entries["xy"]["us"] = 6 if (n == 1 or seed == SEEDS[0]) else 11
                if b != 448:
                    entries["xy"]["dispatches"] = 0
                cells[str(b)] = entries
            (tmp_path / f"shape-{n}-{seed}.json").write_text(
                json.dumps({"n": n, "k": 32, "seed": seed, "cells": cells})
            )
    assert select(tmp_path)["ranges"] == [[1, 32, 448, 448, 1]]
    (tmp_path / f"shape-1-{SEEDS[1]}.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        select(tmp_path)


def test_compact_capacity_is_checked_against_actual_worker_allocation():
    assert exclusion("H1", {"verified_compact_pool": False})
    assert exclusion("H1", {"verified_compact_pool": True}) is None


def test_packed_plan_preserves_unmeasured_widths_and_free_headroom():
    from lab.pk6.native_dispatch.build import packed_routes

    hardware = {"memory_mib": 46068}
    plan = {
        "hardware": hardware,
        "ranges": [[1024, 5120, 143, 256]],
        "memory": {
            "hardware": hardware,
            "reserve_mib": 6144,
            "measured_serving_peak_mib": 23834,
            "cache_gib": 15,
        },
    }
    text = packed_routes(plan)
    assert "{1024,5120,143}" in text and "{1024,5120,256}" in text
    assert "{1024,5120,144}" not in text
    plan["memory"]["cache_gib"] = 16
    with pytest.raises(ValueError, match="reservation"):
        packed_routes(plan)
    assert exclusion("W1", {"verified_compact_pool": False})


def test_vllm_reuse_retains_scope_and_raw_reference(tmp_path):
    from lab.pk6.native_dispatch import reference, score

    root = reference.ARCHIVE
    target = score.read(root / "http/primary/target.json")
    target["vllm_installation"] = score.read(root / "vllm-reference.json")[
        "installation"
    ]
    measured = reference.lookup(root, target)
    assert measured["scope"] == "same_device" and len(measured["factual_correct"]) == 2
    assert measured["metrics"]["32"]["tok_s"]["median"] == pytest.approx(
        266.55, abs=0.01
    )
    assert reference.lookup(root, target, artifact="identical_gguf") is None
    assert reference.lookup(root, {**target, "vllm_version": "changed"}) is None
    foreign = {
        **target,
        "hardware": {**target["hardware"], "device_sha256": "another device"},
    }
    assert reference.lookup(root, foreign)["scope"] == "historical_target"
    assert (
        reference.lookup(root, {**target, "vllm_installation": {}})["scope"]
        == "historical_target"
    )
    (tmp_path / "target.json").write_text(json.dumps(target))
    receipt = tmp_path / "reused-vllm.json"
    receipt.write_text(json.dumps(measured))
    assert list(score.run(tmp_path)["arms"]) == ["vllm"]
    receipt.write_text(json.dumps({**measured, "original_start": 0}))
    with pytest.raises(ValueError, match="verify"):
        score.run(tmp_path)


def test_runner_reuses_vllm_without_launching_the_comparator(tmp_path, monkeypatch):
    import sys
    import time
    from lab.pk6.native_dispatch import reference, run, score

    archived = score.read(reference.ARCHIVE / "http/primary/target.json")
    output = tmp_path / "http"
    monkeypatch.setattr(sys, "argv", ["run", "--arms", "vllm", "--out", str(output)])
    monkeypatch.setenv("RP_DEADLINE", str(int(time.time()) + 120))
    monkeypatch.setattr(run.trial, "board", lambda: archived["hardware"])
    monkeypatch.setattr(
        reference,
        "installation",
        lambda *args: score.read(reference.ARCHIVE / "vllm-reference.json")[
            "installation"
        ],
    )
    monkeypatch.setattr(run, "checksum", lambda _: archived["gguf_sha256"])
    monkeypatch.setattr(
        run, "verify", lambda _: {"manifest_sha256": archived["awq_manifest_sha256"]}
    )
    monkeypatch.setattr(
        run.subprocess,
        "check_output",
        lambda argv, **kwargs: (
            archived["source"] if argv[0] == "git" else archived["vllm_version"]
        ),
    )
    monkeypatch.setattr(
        run.trial,
        "measure",
        lambda *args, **kwargs: pytest.fail("unexpected vLLM launch"),
    )
    monkeypatch.setattr(run.trial, "SEEDS", run.trial.SEEDS)
    run.main()
    assert not (output / "start-vllm.json").exists()
    completed = score.read(output / "completed.json")
    assert completed["order"] == [] and completed["reused"] == ["vllm"]
    assert (
        score.run(output)["arms"]["vllm"]["start"]
        == score.read(output / "reused-vllm.json")["original_start"]
    )


def test_detached_comparator_requires_the_existing_archive(tmp_path, monkeypatch):
    import sys
    from lab.pk6.native_dispatch import run

    out = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--arms",
            "vllm",
            "--out",
            str(out),
            "--reference-archive",
            str(tmp_path / "missing"),
        ],
    )
    monkeypatch.setattr(run.trial, "board", lambda: pytest.fail("unexpected GPU probe"))
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2 and not out.exists()


def test_native_only_rerun_needs_no_external_weights_or_runtime(tmp_path, monkeypatch):
    import os
    import sys
    import time
    from lab.pk6.native_dispatch import reference, run, score

    external = reference.lookup(reference.ARCHIVE)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run", "--arms", "R1", "--out", str(output)])
    monkeypatch.setattr(
        run.os, "environ", {**os.environ, "RP_DEADLINE": str(time.time() + 180)}
    )
    monkeypatch.setattr(run.trial, "board", lambda: external["hardware"])
    monkeypatch.setattr(run.trial, "SEEDS", run.trial.SEEDS)
    monkeypatch.setattr(run, "checksum", lambda _: run.GGUF)
    monkeypatch.setattr(
        run, "verify", lambda _: pytest.fail("unexpected AWQ verification")
    )
    monkeypatch.setattr(
        reference, "installation", lambda *_: pytest.fail("unexpected vLLM probe")
    )

    def commit(argv, **kwargs):
        assert argv == ["git", "rev-parse", "HEAD"]
        return "a" * 40

    measured = []
    monkeypatch.setattr(run.subprocess, "check_output", commit)
    monkeypatch.setattr(
        run.trial, "measure", lambda args, arm, *rest: measured.append(arm)
    )
    run.main()
    assert measured == ["R1"]
    target = score.read(output / "target.json")
    assert target["external_reference_waves_sha256"] == external["waves_sha256"]
    assert target["awq_manifest_sha256"] == external["quantized_model"]
    assert "vllm_installation" not in target


def test_bounded_w4_order_uses_two_complete_fixed_work_pairs():
    from lab.pk6.native_dispatch.reference import ARCHIVE
    from lab.pk6.native_dispatch.w4_profile import select

    plan = select(ARCHIVE / "bounded-w4/followup")
    assert plan == json.loads((ARCHIVE / "bounded-w4/worker-plan.json").read_bytes())
    assert plan["selected"] and len(plan["sources"]) == 4
    assert all(row["8"] > 1.05 for row in plan["worker_ratios"])


def test_w4_counter_forwards_both_native_calls(tmp_path):
    import ctypes
    import shutil
    import subprocess
    import sys

    if sys.platform != "linux" or shutil.which("c++") is None:
        pytest.skip("requires the Linux C++ loader")
    source = tmp_path / "original.cpp"
    source.write_text("""
#include <cstdint>
extern "C" int other_entry() { return 7; }
extern "C" int ironmule_kg_mul_mat(int t, const void*, const float*, float*, int64_t, int64_t, int64_t, void*) { return t; }
extern "C" int ironmule_kg_mul_mat_group(int n, const int*, const void*const*, const float*, float*const*, const int64_t*, int64_t, int64_t, void*) { return n-2; }
""")
    counter = (
        Path(__file__).resolve().parents[2] / "lab/pk6/native_dispatch/w4_count.cpp"
    )
    original, wrapper = tmp_path / "w4-original.so", tmp_path / "w4-count.so"
    subprocess.run(["c++", "-fPIC", "-shared", source, "-o", original], check=True)
    subprocess.run(
        [
            "c++",
            "-fPIC",
            "-shared",
            counter,
            "-L",
            tmp_path,
            "-Wl,--no-as-needed",
            "-l:w4-original.so",
            "-Wl,-rpath,$ORIGIN",
            "-ldl",
            "-o",
            wrapper,
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(wrapper))
    ptr, count = ctypes.c_void_p, ctypes.c_int64
    lib.ironmule_kg_mul_mat.argtypes = [
        ctypes.c_int,
        ptr,
        ptr,
        ptr,
        count,
        count,
        count,
        ptr,
    ]
    lib.ironmule_kg_mul_mat_group.argtypes = [
        ctypes.c_int,
        ptr,
        ptr,
        ptr,
        ptr,
        ptr,
        count,
        count,
        ptr,
    ]
    lib.im_prompt_dispatches.restype = ctypes.c_uint64
    assert lib.other_entry() == 7 and lib.im_prompt_dispatches() == 0
    assert lib.ironmule_kg_mul_mat(3, None, None, None, 1, 1, 1, None) == 3
    assert lib.im_prompt_dispatches() == 0
    assert lib.ironmule_kg_mul_mat(0, None, None, None, 1, 1, 1, None) == 0
    assert (
        lib.ironmule_kg_mul_mat_group(2, None, None, None, None, None, 1, 1, None) == 0
    )
    assert lib.im_prompt_dispatches() == 2


def test_repeated_worker_seed_pruned_only_for_a_measured_numeric_rejection(tmp_path):
    from lab.pk6.native_dispatch.reference import ARCHIVE
    from lab.pk6.native_dispatch.run import rejected_first_seed

    source = next((ARCHIVE / "http").rglob("ready-M1.json"))
    ready = tmp_path / source.name
    ready.write_bytes(source.read_bytes())
    assert rejected_first_seed(tmp_path, "M2") is None
    (tmp_path / "end-M1.json").write_text("{}")
    assert rejected_first_seed(tmp_path, "M2")["first"] == "M1"
    doc = json.loads(ready.read_bytes())
    for attempt in doc["health"]["kernel_fallback"]["attempts"]:
        if attempt.get("diagnostic"):
            attempt["diagnostic"]["code"] = "out_of_memory"
    ready.write_text(json.dumps(doc))
    assert rejected_first_seed(tmp_path, "M2") is None


def test_draft_slots_prioritize_the_cheapest_available_context():
    from lab.pk6.native_dispatch.draft_load import prioritize

    wants = {0: ([8] * 128, 1), 1: ([1, 2, 3], 2), 2: ([4] * 32, 1)}
    held = {0: [4] * 31, 1: [1, 2]}
    assert list(prioritize(wants, {1: 1}, held)) == [1, 2, 0]
    assert prioritize(wants, {0: 0, 1: 1}, held) is wants
    assert exclusion("C1", {"verified_at_wider_draft_load": 0})


def test_draft_load_order_replays_from_measured_requests():
    from lab.pk6.native_dispatch.reference import ARCHIVE
    from lab.pk6.native_dispatch.draft_load import propose

    plan = propose(ARCHIVE / "http", ARCHIVE / "draft-metadata.json")
    assert plan == json.loads((ARCHIVE / "draft-load-plan.json").read_bytes())
    assert plan["selected"] and plan["additional_kv_bytes_estimate"] == 448 * 2**20


def test_pool_profile_rejects_foreign_device_and_mixed_traffic(tmp_path):
    from lab.pk6.native_dispatch import pool_pilot, score

    root = (
        Path(__file__).resolve().parents[2]
        / "lab/autonomous/results/l40s-native-dispatch"
    )
    proof = pool_pilot.fit(root, tmp_path)
    assert {r["server_seed_index"] for r in proof["scores"] if r["split"] == "fit"} == {
        2
    }
    assert {
        r["server_seed_index"] for r in proof["scores"] if r["split"] == "heldout"
    } == {1}
    prior = score.waves(root / "http/primary", "R1")
    assert (
        pool_pilot.select(tmp_path, prior, 32, proof["hardware"], proof["native"])[
            "profile"
        ]
        == "stock"
    )
    (long_file,) = (root / "controls/capacity").rglob("waves-R1.jsonl")
    long_rows = score.waves(long_file.parent, "R1")
    assert (
        pool_pilot.select(tmp_path, long_rows, 32, {"name": "other"}, proof["native"])[
            "profile"
        ]
        == "stock"
    )
    assert (
        pool_pilot.select(tmp_path, long_rows, 32, proof["hardware"], "different")[
            "profile"
        ]
        == "stock"
    )
    expected = "compact" if 32 in proof["qualified_widths"] else "stock"
    assert (
        pool_pilot.select(tmp_path, long_rows, 32, proof["hardware"], proof["native"])[
            "profile"
        ]
        == expected
    )


def test_detached_native_client_can_read_only_the_historical_target(monkeypatch):
    from lab.pk6.native_dispatch import reference, score

    target = score.read(reference.ARCHIVE / "http/primary/target.json")
    monkeypatch.delattr(reference.trial, "vllm_command")
    saved = reference.lookup(reference.ARCHIVE)
    assert saved["scope"] == "historical_target"
    assert saved["metrics"]["32"]["tok_s"]["median"] == pytest.approx(266.55, abs=0.01)
    assert reference.lookup(reference.ARCHIVE, target) is None


def test_compact_verification_counts_cpu_cost_in_the_measured_gain():
    from lab.pk6.native_dispatch import pool_pilot, score

    root = (
        Path(__file__).resolve().parents[2]
        / "lab/autonomous/results/l40s-native-dispatch/controls/capacity/capacity-confirm"
    )
    stock = score.waves(root / "R1", "R1")
    assert pool_pilot.verify_trial(root / "H1", "H1", stock, 32)["retained"]
    assert not pool_pilot.verify_trial(root / "H1", "H1", stock, 32, controller_s=100)[
        "retained"
    ]


def test_fresh_pool_history_needs_the_served_workload_and_stock_trace():
    from lab.pk6.native_dispatch import pool_pilot, reference, score, stress
    from lab.pk6.replay_http import checksum

    root = reference.ARCHIVE
    proof = score.read(root / "pool-pilot/fit.json")
    stock = root / "controls/pool-restart/pool-live/restored/http"
    target = {
        **score.read(stock / "target.json"),
        "workload_sha256": checksum(Path(stress.__file__)),
        "backend_sha256": proof["native"],
    }
    assert pool_pilot.verified_stock_history(
        stock, "R1", target, proof, proof["native"]
    )
    for field in ("workload_sha256", "backend_sha256", "engine_sha256"):
        changed = {**target, field: "changed"}
        assert not pool_pilot.verified_stock_history(
            stock, "R1", changed, proof, proof["native"]
        )
    wrong_arm = {**target, "order": ["H1"]}
    assert not pool_pilot.verified_stock_history(
        stock, "R1", wrong_arm, proof, proof["native"]
    )
