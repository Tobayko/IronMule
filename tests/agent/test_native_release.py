import hashlib
from types import SimpleNamespace

import pytest

from lab.pk1 import native_release as release


def test_target_selects_checks_and_remembers_a_regression(tmp_path, monkeypatch):
    """Exercise the real atomic target state with a deterministic substitute for the expensive hardware arm."""
    receipt = {"order": {"order_id": "test", "base_commit": "base"}, "qualification": {"passed": True},
               "publication": {"order_id": "test", "pr": 12, "base": "base"}, "wall_limit": 11., "ttft_limit": 1.,
               "rss_limit": 1150, "device_limit": 150}
    manifest = tmp_path / "receipt.json"
    release.save(manifest, receipt)
    args = SimpleNamespace(state=tmp_path, receipt=manifest, command="activate", seed=9,
                           receipt_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    calls, slowed = [], {}
    def measure(args, receipt, cand, seed, dest):
        calls.append(cand)
        return {"wall_s": slowed.get(cand, 9 if cand else 10), "ttft_s": .9 if cand else 1,
                "tokens": 1024, "output_sha256": "matched", "fact_digest": "equal", "completed": 32,
                "peak_rss_kib": 1024, "device_used_bytes_after_phases": [128] * 4}

    monkeypatch.setattr(release, "measured", measure)
    assert release.target(args)["arm"] == "cand" and calls == [True, False]
    calls.clear()
    args.command = "run"
    assert release.target(args)["wall_s"] == 9 and calls == [True]
    slowed[True] = 12
    assert release.target(args)["wall_s"] == 10 and calls[-2:] == [True, False]
    assert release.read(tmp_path / "deployment/selection.json")["withdrawn"]
    calls.clear()
    assert release.target(args)["wall_s"] == 10 and calls == [False]
    args.command = "activate"
    assert release.target(args)["withdrawn"]
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="untrusted"):
        release.target(args)


def test_incomplete_trial_or_corrupt_numeric_screen_never_submits(tmp_path, monkeypatch):
    args = SimpleNamespace(state=tmp_path, trial=tmp_path, order=tmp_path / "order.json", publish=True)
    monkeypatch.setattr(release, "submit_candidate", lambda *a: pytest.fail("unauthorized publication"))
    with pytest.raises(FileNotFoundError):
        release.control(args)
    monkeypatch.setattr(release.trial, "validate_screen", lambda *a: None)
    row = {"measurements": [{"arms": [{"relative_error": 0, "microseconds": 3},
                                     {"relative_error": float("nan"), "microseconds": 4}]}]}
    with pytest.raises(ValueError, match="numerical"):
        release.score_screen(row, {"binding": {}}, "", None)


def test_complete_native_control_dry_run_uses_measured_bytes_only(tmp_path, monkeypatch):
    """Synthetic fixture for receipts and real Git trees. No CUDA claims or network publication."""
    import json
    from ironmule.factual_gate import cases
    from training import code_author, code_trial
    from training.serving_trial import digest
    from lab.pk1 import native_trial as trial

    repo, state, samples = (tmp_path / name for name in ("src", "state", "trial"))
    for path in (repo, state, samples):
        path.mkdir()
    source = repo / release.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("__global__ void synthetic() {\n    return;\n}\n")
    names = ("ironmule/ggml_engine.py", "ironmule/gguf_tokenizer.py", "ironmule/pilot.py",
             "ironmule/w4_kernels.py", "ironmule/kernel_gen.py")
    (repo / "ironmule").mkdir()
    for name in names:
        (repo / name).write_text("# fixture\n")
    for argv in (("init", "-q"), ("config", "user.name", "Fixture"), ("config", "user.email", "fixture@invalid"),
                 ("add", "."), ("commit", "-qm", "fixture")):
        release.git(*argv, cwd=repo)
    monkeypatch.setattr(code_trial, "ROOT", repo)
    monkeypatch.setattr(release, "submit_candidate", lambda *a: pytest.fail("dry run cannot publish"))
    table = tmp_path / "table"
    table.write_text("12 512 256 32 1 1\n")
    runtime = {name: trial.checksum(repo / name) for name in names}
    binding = {"fp": {"devices": ["fixture"]}, "native": {"sha256": "native"}, "engine": {},
               "gguf": release.QWEN32, "source_sha256": trial.checksum(source), "table_sha256": trial.checksum(table),
               "w4_sha256": "reference", "serving_sources": runtime}
    base = release.git("rev-parse", "HEAD", cwd=repo)
    profile = tmp_path / "order.profile.json"
    release.save(profile, {"binding": binding, "base_commit": base})
    order = {"schema": "ironmule.native-code-experiment.v1", "base_commit": base,
             "binding": binding, "target": {"path": release.SOURCE, "function": "synthetic", "line": 1},
             "allowed_paths": [release.SOURCE], "evaluator": trial.SCHEMA, "profile_sha256": trial.checksum(profile),
             "workload": {"decode_rows": 32}}
    order["order_id"] = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
    order_path = tmp_path / "order.json"
    release.save(order_path, order)
    authored = tmp_path / "authoring"
    authored.mkdir()
    proposal = authored / "candidate.cu"
    author_args = SimpleNamespace(order=order_path, checkout=repo, output=proposal)
    with pytest.raises(code_author.PendingAuthor):
        trial.author(author_args)
    request = release.read(authored / "author/request.json")
    code_author.respond(authored, request["request_id"], {"definition": source.read_text().replace(
        "return;", "return; // synthetic fixture"), "rationale": "synthetic fixture"}, "test", "test")
    trial.author(author_args)
    (samples / "candidate.so").write_bytes(b"SYNTHETIC FIXTURE")
    cand = trial.checksum(samples / "candidate.so")
    context = {"order": order["order_id"], "base_commit": base, "order_sha256": trial.checksum(order_path),
               "profile_sha256": trial.checksum(profile), "candidate_source": trial.checksum(proposal),
               "harness": trial.checksum(trial.__file__), "reference": "reference", "table": trial.checksum(table)}
    release.save(samples / "context.json", context)
    release.save(samples / "build.json", {"source": trial.checksum(proposal), "binary": cand})
    release.save(samples / "screen.json", {"schema": trial.SCHEMA, "stage": "screen", "all_elements_identical": True,
                 "model": release.QWEN32, "reference": "reference", "candidate": cand, "table": trial.checksum(table),
                 "evaluator_sha256": trial.checksum(trial.__file__), "measurements": [{"shape": [12, 512, 256, 32],
                 "variant": 1, "ksplit": 1, "arms": [{"relative_error": 0., "microseconds": 1.}] * 2}]})
    rows = []
    for i in range(6):
        seed = 92741 + i
        factual = [{"output_ids": [1], "answer": case.expected, "stop": "eos"} for case in cases(seed)]
        answers = [{"output_ids": [8], "answer": "fixture", "stop": "eos"}] * 32
        common = {"schema": trial.SCHEMA, "profiled": False, "pilot_decisions": 7, "facts": [True] * 18,
                  "binding": {"native": binding["native"], "fp": binding["fp"], "model_sha256": release.QWEN32,
                              "flags": trial.FLAGS, "engine": {}, "runtime_sources": runtime},
                  "workload": {"seed": seed, "tokens": 32, "clients": 32, "context": 2048,
                               "warmup_waves": 1, "quality_cases": 18},
                  "quality_answers": factual, "answers": answers, "fact_digest": digest([[1]] * 18),
                  "output_sha256": digest(answers), "completed": 32, "tokens": 32,
                  "peak_rss_kib": 20, "device_used_bytes_after_phases": [400] * 4,
                  "evaluator_sha256": context["harness"], "ttft_s": 1.}
        pair = []
        for label, binary, wall in (("ref", "reference", 2.), ("cand", cand, 1.9)):
            row = {**common, "treatment": {"binary": binary, "table": context["table"]}, "wall_s": wall}
            release.save(samples / f"{label}-{i}.json", row)
            pair.append(row)
        rows.append(pair)
    signed = ["build.json", "screen.json", *(f"{arm}-{i}.json" for i in range(6) for arm in ("ref", "cand"))]
    release.save(samples / "result.json", {"schema": trial.SCHEMA, **trial.judge_native(rows), "context": context,
                 "candidate": cand, "incumbent": "reference",
                 "measurements_sha256": {k: trial.checksum(samples / k) for k in signed}})
    args = SimpleNamespace(state=state, trial=samples, order=order_path, table=table,
                           candidate_source=proposal, publish=False)
    assert release.control(args)["status"] == "qualified, awaiting publication"
    assert not (state / "deployment/control.json").exists()
    (samples / "cand-5.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt mismatch"):
        release.control(args)
