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
