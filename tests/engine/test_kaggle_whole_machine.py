"""The Kaggle U1/HW1/HW2 harness keeps no prompt or answer text and compares answers by digest."""

from __future__ import annotations

import json

from tools import kaggle_whole_machine as harness


def test_a_level_reports_rates_and_digests_but_no_text(monkeypatch):
    calls = []

    def fake_chat(port, model, prompt, max_tokens):
        calls.append(prompt)
        return {"status": 200, "first_s": 0.1 * (len(calls)), "total_s": 1.0, "completion_tokens": 4,
                "answer_sha256": "d" * 64}

    monkeypatch.setattr(harness, "_chat", fake_chat)
    level = harness._level(1, "m", clients=2, requests=6, max_tokens=4)
    assert level["ok"] == 6 and level["completion_tokens"] == 24 and level["tokens_per_s"] > 0
    assert level["first_p50_s"] is not None and level["first_p95_s"] >= level["first_p50_s"]
    assert set(level["answers"]) == {str(index) for index in range(6)}
    assert not any(prompt in json.dumps(level) for prompt in harness.PROMPTS)


def test_answers_must_agree_for_a_prompt_across_every_arm(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "_download", lambda model: "rev")
    monkeypatch.setattr(harness, "_ironmule", lambda *a, **k: {"returncode": 0})
    answers = iter(["a", "a", "b"])

    def fake_serve(model, state, gpus, levels, requests_per_client, max_tokens):
        return {"gpus": gpus, "levels": [{"answers": {"0": next(answers)}}]}

    monkeypatch.setattr(harness, "_serve", fake_serve)
    result = harness.hw1("m", arms=(None, "all", None), levels=(1,), requests_per_client=1, max_tokens=4,
                         work=tmp_path)
    assert result["order"] == ["one", "all", "one"] and result["answers_identical"] is False
    assert all("answers" not in level for run in result["runs"] for level in run["levels"])


def test_the_environment_phase_alone_writes_a_report(tmp_path):
    out = tmp_path / "report.json"
    assert harness.main(["--out", str(out), "--phases", "env"]) == 0
    report = json.loads(out.read_text())
    assert report["schema"] == "ironmule.kaggle_whole_machine.v1" and "env" in report and "u1" not in report


def test_a_kernel_folder_pins_one_commit_and_runs_the_same_phases(tmp_path):
    commit = "0123456789abcdef0123456789abcdef01234567"
    folder = harness.write_kernel(tmp_path / "kernel", "someone", commit, "env,hw1")
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["id"] == "someone/ironmule-whole-machine" and metadata["is_private"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4" and metadata["code_file"] == "run.py"
    script = (folder / "run.py").read_text()
    assert f"@{commit}" in script and "'--phases', 'env,hw1'" in script
    compile(script, "run.py", "exec")
    for bad in ("main", "0123", commit.upper()):
        try:
            harness.write_kernel(tmp_path / "bad", "someone", bad, "env")
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
