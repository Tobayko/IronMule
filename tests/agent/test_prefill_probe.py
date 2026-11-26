import hashlib
import json

import pytest

from lab.pk1 import prefill_probe


@pytest.fixture
def files(tmp_path):
    profile = {"kind": "measured_native_diagnostic", "base_commit": "a" * 40, "binding": {"fp": {}},
               "prompt_tokens_per_client": 238, "width": 32,
               "dispatch_rows_type_n_k_batch_variant_ksplit": [[12, 25600, 5120, 32, 184, 1], [14, 5120, 25600, 32, 266, 4]]}
    path = tmp_path / "order.json"
    profile_path = path.with_suffix(".profile.json")
    profile_path.write_text(json.dumps(profile))
    order = {"base_commit": profile["base_commit"], "binding": profile["binding"],
             "profile_sha256": prefill_probe.checksum(profile_path)}
    identity = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps({**order, "order_id": identity}))
    trace = tmp_path / "trace.log"
    trace.write_text(f"""loader: general.architecture str = qwen3
print_info: n_layer = 64
print_info: n_embd = 5120
print_info: n_ff = 25600
prefill 7616 rows in 2.369 s
== prefill: 2338.4 ms of kernels
 1036.75 ms 44.3% x126 ampere_s16816gemm_fp16_x
 527.10 ms 22.5% x63 ampere_s16816gemm_fp16_y
 0.12 ms 0.0% x63 Memset (Device)
decode 16 steps x 32 rows
 13.00 ms 0.0% x63 ampere_s16816gemm_fp16_z
Native order {identity} for w4d
""")
    return trace, path


def test_real_work_dimensions_and_cost_order(files):
    result = prefill_probe.plan(*files)
    assert result["kind"] == "information_experiment"
    shapes = [(row["family"], row["m"], row["n"], row["k"], row["calls"]) for row in result["targets"]]
    assert shapes == [("gate_up", 25600, 7616, 5120, 126), ("down", 5120, 7616, 25600, 63)]


@pytest.mark.parametrize(("old", "new"), [("x126", "x125"), ("prefill 7616", "prefill 8192"),
                                         ("Native order", "Other order"), ("= qwen3", "= qwen2")])
def test_reject_unmatched_trace(files, old, new):
    trace, order = files
    trace.write_text(trace.read_text().replace(old, new))
    with pytest.raises(ValueError):
        prefill_probe.plan(trace, order)


def test_profile_cannot_be_substituted(files):
    trace, order = files
    profile = order.with_suffix(".profile.json")
    profile.write_text(profile.read_text() + "\n")
    with pytest.raises(ValueError):
        prefill_probe.plan(trace, order)


def test_cached_probe_needs_no_gpu(files, tmp_path, monkeypatch):
    import ironmule.kernel_gen as kg

    order = prefill_probe.plan(*files)
    output = tmp_path / "screen.json"
    result = {"request_id": prefill_probe.digest(order), "status": "completed"}
    output.write_text(json.dumps(result))
    monkeypatch.setattr(kg, "fingerprint", lambda: pytest.fail("no GPU work during replay"))
    assert prefill_probe.run(order, output) == result
    output.write_text(json.dumps({**result, "status": "failed", "reason": "short allocation"}))
    with pytest.raises(RuntimeError, match="short allocation"):
        prefill_probe.run(order, output)
