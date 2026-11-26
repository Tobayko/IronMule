"""Historical decisions must retain their hardware and comparison boundaries."""

import json
from pathlib import Path

import pytest

from training import experience_archive as archive


def test_report_provenance_and_exclusions():
    rows = list(archive.build())
    assert len({row["id"] for row in rows}) == len(rows)
    placement = [r for r in rows if r["kind"] == "placement_pair"]
    good = [r for r in placement if r["label_eligible"]]
    assert len(good) == 80
    assert all(r["quality"]["response_and_stop_digests"][0] == r["quality"]["response_and_stop_digests"][1]
               for r in good)
    assert {r["reason"] for r in placement if not r["label_eligible"]} == {
        "response_digests_unavailable", "single_worker"}
    assert len({r["group"] for r in good}) == 2

    k1 = next(r for r in rows if r["locator"] == "./http_K1_32.jsonl:2")
    assert k1["metrics"][0]["tok_s"] == 297.06
    assert k1["base_commit"] == "02b285b"
    assert k1["label_eligible"] is False
    assert any(r["reason"] == "different_quantization" for r in rows)
    forecast = [r for r in rows if r["kind"] == "traffic_forecast"]
    assert forecast and all(r["provenance"] == "simulated" and not r["label_eligible"] for r in forecast)
    assert any(r["reason"] == "cross_pod_reference_changed" for r in rows)
    scratch = next(r for r in rows if r["source"] == "evidence/runtime/cpu-pilot-four-slot-discovery.json")
    assert scratch["decision"] is False and scratch["label_eligible"] is False
    assert scratch["quality"]["correct"] == scratch["quality"]["completed"] == 16
    [metric] = scratch["metrics"]
    assert metric["head_proposals"] == 5 and metric["scratch_measured_decisions"] == 0
    assert metric["search_and_warmup_s"] == pytest.approx(83.03, abs=.001)

    head = next(row for row in rows if row["kind"] == "pilot_w4_profile_holdout")
    assert head["reason"] == "unresolved model margin" and not head["label_eligible"]
    assert head["workload"]["geometry"]["sequences"] == 1 and head["workload"]["held_out_widths"] == [32]
    assert head["metrics"][0]["kernel_rejections"] == 196


def test_failed_rebuild_keeps_previous_index(tmp_path):
    out = tmp_path / "index.jsonl"
    archive.write_archive(out, iter([{"kind": "test", "name": "previous"}]))

    def interrupted():
        yield {"kind": "test", "name": "replacement"}
        raise ValueError("bad row")

    with pytest.raises(ValueError, match="bad row"):
        archive.write_archive(out, interrupted())
    assert list(archive.search(out, "previous", 1)) == [{"kind": "test", "name": "previous"}]
    assert list(tmp_path.iterdir()) == [out]


def test_mismatched_output_is_preserved_without_a_reward():
    path = Path("lab/placement/results/smollm2-cpu-reward-burst-v3.json")
    trace = json.loads(path.read_text())
    trace["trials"][0]["outputs"][1] = "0" * 64
    row = next(archive.placement(str(path), "test-only", trace))
    assert row["label_eligible"] is False
    assert row["reason"] == "different_completions"
    assert len(row["metrics"]) == 2


def test_published_code_change_rejects_and_retains_codex_rationale(tmp_path):
    root = Path(__file__).resolve().parents[2]
    raw = "lab/autonomous/results/cpu-tokenizer-01/result.json"
    index = tmp_path / "index.jsonl"
    rows = list(archive.code_experiment(root, raw, archive.canonical_sha256((root / raw).read_bytes())))
    archive.write_archive(index, rows)
    [rejected] = list(archive.search(index, "rejected GgufTokenizer.encode ctypes", 1))
    assert rejected["decision"] is False and rejected["label_eligible"] is False
    assert "4/6 pairs faster" in rejected["note"]
    assert "Codex" in rejected["authorship"] and "IronMule" in rejected["authorship"]
    assert len({r["group"] for r in rows}) == 1
    assert len([r for r in rows if r["kind"] == "code_trial_pair"]) == 6


def test_native_trial_preserves_calibration_failure_and_dream_boundaries():
    from training.native_evidence import verify_manifest

    root = Path(__file__).resolve().parents[2]
    scope = root / "lab/autonomous/results/a100-native"
    seen = set()
    manifests = sorted(scope.glob("*/manifest.json"))
    assert manifests
    for manifest in manifests:
        verify_manifest(manifest.parent)
        rows = list(archive.native_experiment(root, str(manifest.relative_to(root)),
                    archive.canonical_sha256(manifest.read_bytes()), seen))
        assert all(not row["label_eligible"] for row in rows)
        profile = next(r for r in rows if r["kind"] == "native_device_profile")
        assert profile["workload"]["width"] == 32 and min(profile["metrics"][0]["kernel_seconds_per_step"]) > 0
        assert next(r for r in rows if r["kind"] == "native_dream")["provenance"] == "simulated"
    rows = list(archive.build())
    calibrations = [r for r in rows if r["kind"] == "native_calibration"]
    assert calibrations and len({r["group"] for r in calibrations}) == len(calibrations)
    assert any(len(r["metrics"][0]["counts_by_outcome"]) > 1 for r in calibrations)
    assert any(r["kind"] == "native_failed_attribution" and not r["decision"] for r in rows)
    restart = next(r for r in rows if r["kind"] == "native_coordinator_restart")
    assert restart["metrics"][0]["interruption_s"] > 0 and restart["context_sha256"] and not restart["label_eligible"]
    repair = next(r for r in rows if r["kind"] == "native_startup_repair")
    assert "ModuleNotFoundError" in repair["reason"] and "/opt/" not in repair["reason"]


def test_partial_native_trial_preserves_ttft_without_trusting_pass(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from training import native_evidence

    scope = Path(__file__).resolve().parents[2] / "lab/autonomous/results/a100-native"
    order = next(scope.glob("*/request.json")).with_name("order.json")
    state = tmp_path / "campaign"
    trial = state / "trial"
    trial.mkdir(parents=True)
    (trial / "context.json").write_text(json.dumps({"order": order.parent.name}))
    monkeypatch.setattr(native_evidence, "source_at", lambda base, path: None)
    monkeypatch.setattr(native_evidence, "ROOT", tmp_path)  # Only the archived, hash-bound request survives.
    (trial / "ref-0.json").write_text(json.dumps({"ttft_s": 0.13}))
    (trial / "result.json").write_text(json.dumps({"passed": True, "reason": "claimed"}))
    args = SimpleNamespace(state=state, order=order, destination=tmp_path / "export",
                           tuning=None, tuner=None, table=None, prior_failure_log=None)
    manifest = native_evidence.export(args)
    rows = list(archive.native_experiment(tmp_path, str(manifest.relative_to(tmp_path)),
                archive.canonical_sha256(manifest.read_bytes()), set()))
    pair = next(r for r in rows if r["kind"] == "native_code_pair")
    assert pair["reason"] == "unpaired_or_confounded" and pair["metrics"][0]["ttft_s"] == .13
    result = next(r for r in rows if r["kind"] == "native_code_decision")
    assert result["decision"] is None and result["reported_qualification"] is True and not result["label_eligible"]
    assert result["coverage"]["comparable_serving_pairs"] == 0 and result["workload"]["decode_rows"] == 32
    (state / "outcome.json").write_text(json.dumps({"order": order.parent.name, "authorization": "private"}))
    with pytest.raises(ValueError, match="private fields"):
        native_evidence.export(args)
    assert not (manifest.parent / "outcome.json").exists()
    (manifest.parent / "order.profile.json").write_text("{}")
    with pytest.raises(ValueError, match="changed portable evidence"):
        native_evidence.verify_manifest(manifest.parent)


def test_native_export_checks_worker_bytes_even_if_local_hash_was_rewritten(tmp_path):
    import hashlib
    import shutil
    from types import SimpleNamespace

    from training import native_evidence

    scope = Path(__file__).resolve().parents[2] / "lab/autonomous/results/a100-native"
    folder = next(scope.glob("*/candidate.diff")).parent
    author = tmp_path / "authoring/author"
    author.mkdir(parents=True)
    for name in ("request.json", "response.json"):
        shutil.copyfile(folder / name, author / name)
    unbound = b"unrelated source"
    (author.parent / "candidate.cu").write_bytes(unbound)
    local_record = json.loads((folder / "authorship.json").read_text())
    local_record["candidate_sha256"] = hashlib.sha256(unbound).hexdigest()
    (author.parent / "authorship.json").write_text(json.dumps(local_record))
    args = SimpleNamespace(state=tmp_path, order=folder / "order.json", destination=tmp_path / "export",
                           tuning=None, tuner=None, table=None, prior_failure_log=None)
    with pytest.raises(ValueError, match="worker's replacement"):
        native_evidence.export(args)


def test_cpu_heldout_comparators_and_no_change_stay_out_of_training():
    rows = list(archive.build())
    epyc = next(x for x in rows if x["kind"] == "pilot_heldout_replay" and "EPYC" in x["hardware"])
    assert all(r["seconds"]["candidate"] == r["seconds"]["all_threads"] for r in epyc["metrics"])
    assert epyc["parameters"] == 20232 and not epyc["label_eligible"] and epyc["changed_training_decisions"] == []
    failures = [r for r in rows if r["kind"] == "cpu_campaign_attempt" and r["note"]]
    assert any("no CUDA-capable device" in row["note"] for row in failures)
    conflict = next(x for x in rows if x["kind"] == "cpu_confounded_code_pair")
    assert conflict["profiles"]["reference"] != conflict["profiles"]["candidate"] and not conflict["label_eligible"]
    assert conflict["quality"]["outputs_match"] and conflict["decision"] is False
    noop = next(x for x in rows if x["kind"] == "cpu_serving_author_no_change")
    assert noop["decision"] is False and "list-copy" in noop["note"] and noop["quality"]["producer_correct"] == 32
    defer = [x for x in rows if x["kind"] == "cpu_deferred_experiment"]
    assert len(defer) >= 2 and all(not row["label_eligible"] and row["provenance"] == "simulated" for row in defer)


def test_prefill_microtimings_and_output_mismatches_remain_separate_from_planned_savings():
    rows = list(archive.build())
    probes = [x for x in rows if x["kind"] == "prefill_gemm_microprobe"]
    assert len(probes) == 7 and sum(x["reason"] == "changed_output" for x in probes) == 2
    assert all(x["provenance"] == "measured" and not x["label_eligible"] for x in probes)
    [plan] = [x for x in rows if x["kind"] == "prefill_gemm_planning"]
    assert plan["provenance"] == "simulated" and plan["reason"] == "defer source order" and not plan["label_eligible"]
    assert plan["metrics"][0]["optimistic_best_sum_ms"] == pytest.approx(24.826, abs=.001)


def test_qwen_startup_order_binds_both_observations_and_worker_replacement():
    from training import restart_evidence as startup

    root = Path(__file__).resolve().parents[2]
    scope = root / "lab/autonomous/results/a100-restart"
    folder = next(scope.glob("*/manifest.json")).parent
    manifest = startup.manifest(folder)
    order = json.loads((folder / "order.json").read_bytes())
    ids = [json.loads((folder / f"observation-{i}.json").read_bytes())["order_id"] for i in range(2)]
    assert ids == order["evidence"]["local_profile_ids"] and len(set(ids)) == 2
    request = json.loads((folder / "request.json").read_bytes())
    response = json.loads((folder / "response.json").read_bytes())
    rebuilt = startup.candidate_bytes((folder / "reference-source.txt").read_text(), request, order, response["proposal"])
    assert startup.sha(rebuilt) == manifest["candidate_sha256"]
    rows = list(startup.experiences(root, str((folder / "manifest.json").relative_to(root))))
    assert all(not r["label_eligible"] for r in rows)
    assert next(r for r in rows if r["kind"] == "restart_experiment_plan")["provenance"] == "simulated"
    head = next(r for r in rows if r["kind"] == "restart_pilot_heldout")
    assert head["weights_sha256"] == startup.sha((folder / "startup-head.npz").read_bytes())
    scored = head["metrics"][0]
    assert scored["train_family"] != scored["holdout_family"] and scored["holdout_correct_profile"]
    ablation = scored["matched_eligibility_ablation"][scored["holdout_family"]]
    assert ablation["random_initialization"]["profile"] == "reference"
    assert ablation["after_mixed_observations"]["profile"] == "candidate"
    observed = next(r for r in rows if r["kind"] == "restart_cost_observation")
    assert observed["cost_usd"] is None and observed["metrics"][0]["retained_tail_seconds"] == 600
    assert observed["metrics"][0]["campaign_reserved_usd"] <= 15
    if "result.json" not in manifest["files"]:
        assert next(r for r in rows if r["kind"] == "restart_code_decision")["decision"] is None


def test_qwen_product_partial_and_rejected_evidence_preserve_bound_measurements(monkeypatch, tmp_path):
    from training import restart_evidence as startup, scoped_release

    product = tmp_path / "product"
    product.mkdir()
    assets = {"control.json": b"pin", "startup-head.npz": b"checkpoint"}
    order = {"order_id": "order"}
    verified = {"reference_sha256": "ref"}
    monkeypatch.setattr(scoped_release, "pinned", lambda *_: {"order": order, "qualification": verified})
    raw = {"seed": 95111, "source_sha256": "ref", "startup_decision": {"weights_sha256": startup.sha(b"checkpoint")}}
    arm = product / "reference-95111.json"
    arm.write_text(json.dumps(raw))
    for name in ("decision.json", "product-qualify.json"):
        (product / name).write_text('{"passed": false, "reason": "gate failed"}')
    startup.product_assets(tmp_path, order, verified, assets)
    assert json.loads(assets["http-product-qualify.json"])["passed"] is False
    assert json.loads(assets["http-reference-95111.json"]) == raw
    raw["startup_decision"]["weights_sha256"] = "unrelated head"
    arm.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unbound ordinary product arm"):
        startup.product_assets(tmp_path, order, verified, assets)
