from pathlib import Path

import pytest

from lab.pk1.native_order import ranked


def test_native_order_requires_symbol_in_two_device_windows():
    source = (Path(__file__).resolve().parents[2] / "native/kernels/w4mma.cu").read_text()
    gate = {"symbol": "void (anonymous namespace)::w4d<4, 1, 4, 2, 1>(uint8_t*, ...)", "device_s": .15, "calls": 2048}
    down = {"symbol": "void (anonymous namespace)::w4f6<4, 4, 1>(uint8_t*, ...)", "device_s": .05, "calls": 512}
    other = {"symbol": "third_party_gemm", "device_s": .35, "calls": 10}
    result = ranked([[gate, other, down], [down, other, gate]], source, 16)
    assert [r["function"] for r in result] == ["w4d", "w4f6"]
    assert result[0]["calls_per_step"] == [128, 128]
    assert result[0]["simulated_31_step_saving_at_10_percent_s"] == pytest.approx(.15 / 16 * 31 * .1)
    assert source.splitlines()[result[0]["line"] - 1].startswith("w4d(")
    assert len(ranked([[other, down], [down, gate]], source, 16)) == 1
    with pytest.raises(ValueError):
        ranked([[down], [gate]], source, 16)


def test_native_gate_verifies_the_separately_mounted_kernel_and_all_quality_cases():
    from lab.pk1 import native_trial as trial

    devices = [{"uuid": "device-a"}]
    native = {"sha256": {"backend/CUDA": "ggml"}}
    sources = {"ironmule/pilot.py": "cpu-controller"}
    bound = {"native": native, "fp": {"devices": devices}, "gguf": "weights", "serving_sources": sources}
    treatment = {"binary": "candidate", "table": "incumbent-selection"}
    row = {"schema": trial.SCHEMA, "evaluator_sha256": trial.checksum(trial.__file__), "treatment": treatment,
           "binding": {"fp": {"devices": devices}, "native": native, "model_sha256": "weights", "runtime_sources": sources},
           "completed": 32, "pilot_decisions": 50, "facts": [True] * 18,
           "workload": {"clients": 32, "tokens": 32, "context": 2048, "quality_cases": 18, "warmup_waves": 1, "seed": 47}}
    trial.validate_arm(row, bound, treatment, 47)
    for changes in ({"treatment": dict(treatment, table="changed")}, {"facts": [True] * 17 + [False]},
                    {"completed": 31}, {"pilot_decisions": 0}):
        with pytest.raises(ValueError):
            trial.validate_arm({**row, **changes}, bound, treatment, 47)


def test_native_worker_imports_only_the_single_expected_regular_output(tmp_path):
    import os

    from lab.pk1.native_trial import copy_result

    raw = tmp_path / "candidate.json"
    raw.write_text("measured")
    retained = tmp_path / "retained.json"
    copy_result(raw, retained, os.getuid())
    assert retained.read_text() == "measured"
    linked = tmp_path / "linked"
    linked.symlink_to(raw)
    with pytest.raises(OSError):
        copy_result(linked, retained, os.getuid())
    os.link(raw, tmp_path / "alias")
    with pytest.raises(ValueError):
        copy_result(raw, retained, os.getuid())


def test_next_experiment_requires_new_source_and_realistic_whole_wave_headroom():
    from lab.pk1.native_order import next_target

    options = [{"function": name, "kernel_seconds_per_step": [sec, sec]}
               for name, sec in [("declined", .035), ("plausible", .030), ("too_small", .0001)]]
    assert next_target(options, {"declined"}, 3.3, .1)["function"] == "plausible"
    assert next_target(options, {"declined", "plausible"}, 3.3, .1) is None
    assert next_target(options, {"declined"}, 3.3, .025) is None
