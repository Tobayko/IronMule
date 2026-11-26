"""Exercise local hosted-response handoff through the campaign command entry points."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from lab.rp import cpu_campaign, followup_campaign, native_campaign, restart_campaign
from training import code_author


def deliver_after_dispatch(directory, dispatched, *, unchanged=False):
    assert dispatched.wait(timeout=3)
    key, request = code_author.read_request(directory / "author")
    # A hosted reply is published atomically *after* the campaign reports the
    # request. It arrives during the final polling interval for --wait-seconds 1.
    time.sleep(0.04)
    proposal = {
        "definition": request["source"]
        if unchanged
        else "def score(x):\n    return x + 1\n",
        "rationale": "CPU-only test fixture",
    }
    code_author.respond(directory, key, proposal, "local-test-receipt", "fixture")
    return key, proposal


def invoke(monkeypatch, capsys, module, args, seconds):
    monkeypatch.setattr(
        sys, "argv", [module.__name__, *map(str, args), "--wait-seconds", str(seconds)]
    )
    module.main()
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


@pytest.mark.parametrize(
    ("module", "advance", "pending", "extra"),
    [
        (
            native_campaign,
            "cycle",
            "awaiting_author",
            ("--config", "--order", "--checkout"),
        ),
        (
            restart_campaign,
            "cycle",
            "awaiting author",
            ("--config", "--order", "--checkout"),
        ),
        (cpu_campaign, "advance", "awaiting Codex", ("--target", "--head")),
    ],
)
def test_bounded_campaign_waits_for_real_response_and_resumes_once(
    module, advance, pending, extra, monkeypatch, tmp_path, capsys
):
    args = ["--state", tmp_path]
    for flag in extra:
        args.extend([flag, tmp_path / flag[2:]])
    source = "def score(x):\n    return x\n"
    order = {"order_id": code_author.digest(source)}
    directory = tmp_path / "author"
    dispatched = Event()
    measured = tmp_path / "dry-qualification.json"
    result = {"stage": "qualified", "mode": "dry transport"}
    calls = []

    def checkpoint(_args):
        # Only the target-facing stage is simulated; the request/response
        # transport, durability, digest validation and CLI wait loop are real.
        if measured.exists():
            return json.loads(measured.read_bytes())
        try:
            proposal = code_author.hosted_proposal(tmp_path, order, source)
        except code_author.PendingAuthor:
            dispatched.set()
            return {"stage": pending, "request": str(directory / "request.json")}
        calls.append(proposal)
        code_author.write_once(measured, result)
        return result

    monkeypatch.setattr(module, advance, checkpoint)
    assert invoke(monkeypatch, capsys, module, args, 0)[-1]["stage"] == pending
    req = directory / "request.json"
    original = req.read_bytes()
    assert not measured.exists() and not calls
    dispatched.clear()
    with ThreadPoolExecutor(max_workers=1) as executor:
        reply = executor.submit(deliver_after_dispatch, tmp_path, dispatched)
        assert invoke(monkeypatch, capsys, module, args, 1)[-1] == result
        key, proposal = reply.result(timeout=3)
    assert code_author.read_request(directory)[0] == key
    assert code_author.read_response(directory, key) == proposal
    assert invoke(monkeypatch, capsys, module, args, 0) == [result]
    assert req.read_bytes() == original and calls == [proposal]


def test_followup_actual_author_receives_delayed_unchanged_reply(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / followup_campaign.trial.PATH
    source.parent.mkdir(parents=True)
    source.write_text(
        "class GgmlEngine:\n    def __init__(self):\n        pass\n"
        "    def prices(self):\n        return {}\n"
    )
    target = {"class": "GgmlEngine", "function": "prices", "line": 4}
    order = {"source_sha256": followup_campaign.checksum(source), "target": target}
    order["order_id"] = code_author.digest(order)
    archive = tmp_path / "archive.jsonl"
    archive.touch()
    state = tmp_path / "campaign"
    args = [
        "--config",
        "unused",
        "--checkout",
        tmp_path,
        "--state",
        state,
        "--prior-dir",
        tmp_path,
        "--forecast",
        "unused",
        "--control",
        "unused",
        "--weights",
        "unused",
        "--archive",
        archive,
        "--remote-checkout",
        "/offline",
        "--observations",
        "a",
        "b",
    ]
    monkeypatch.setattr(
        followup_campaign.calibration_sampling, "reconstruct", lambda *_: order
    )
    monkeypatch.setattr(
        followup_campaign.followup_release, "verified_sampling", lambda *_: None
    )
    monkeypatch.setattr(
        followup_campaign,
        "cycle",
        lambda *_: pytest.fail("unchanged source reached target"),
    )
    dispatched = Event()
    original_wait = followup_campaign.wait_for_author_response

    def observe(request, deadline):
        dispatched.set()
        return original_wait(request, deadline)

    monkeypatch.setattr(followup_campaign, "wait_for_author_response", observe)
    with ThreadPoolExecutor(max_workers=1) as executor:
        reply = executor.submit(
            deliver_after_dispatch, state / "authoring", dispatched, unchanged=True
        )
        assert (
            invoke(monkeypatch, capsys, followup_campaign, args, 1)[-1]["stage"]
            == "unchanged"
        )
        key, _ = reply.result(timeout=3)
    outcome = json.loads((state / "outcome.json").read_bytes())
    assert outcome == {
        "stage": "unchanged",
        "order": order["order_id"],
        "reason": "CPU-only test fixture",
    }
    assert invoke(monkeypatch, capsys, followup_campaign, args, 0) == [outcome]
    assert code_author.read_request(state / "authoring/author")[0] == key
    assert not (state / "authoring/candidate.py").exists()


def test_wait_expires_and_later_resume_requires_the_original_request_digest(
    tmp_path,
):
    order, source = {"order_id": "bound-fixture"}, "def score(x):\n    return x\n"
    with pytest.raises(code_author.PendingAuthor):
        code_author.hosted_proposal(tmp_path, order, source)
    request = tmp_path / "author/request.json"
    deadline = time.monotonic() + 0.04
    assert not native_campaign.wait_for_author_response(request, deadline)
    assert time.monotonic() >= deadline
    proposal = {"definition": source, "rationale": "unchanged"}
    with pytest.raises(ValueError, match="request digest mismatch"):
        code_author.respond(tmp_path, "foreign", proposal, "local", "fixture")
    assert not request.with_name("response.json").exists()
    key = code_author.read_request(request.parent)[0]
    code_author.respond(tmp_path, key, proposal, "local", "fixture")
    assert code_author.hosted_proposal(tmp_path, order, source) == proposal
