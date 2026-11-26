import json

import pytest

from training.code_author import PendingAuthor, hosted_proposal, respond


def test_reply_is_bound_to_order_and_source_and_survives_restart(tmp_path):
    order = {"target": {"function": "count"}}
    src = "def count(items): return sum(1 for _ in items)"
    with pytest.raises(PendingAuthor):
        hosted_proposal(tmp_path, order, src)
    request = json.loads((tmp_path / "author" / "request.json").read_text())
    reply = {"definition": "def count(items): return len(items)", "rationale": "fixture"}
    with pytest.raises(ValueError):
        respond(tmp_path, "wrong", reply, "task-receipt", "fixture-model")
    respond(tmp_path, request["request_id"], reply, "task-receipt", "fixture-model")
    assert hosted_proposal(tmp_path, order, src) == reply
    with pytest.raises(ValueError, match="conflicting record"):
        hosted_proposal(tmp_path, order, src + "\n")
    new = {**reply, "rationale": "another attempt"}
    with pytest.raises(ValueError, match="conflicting record"):
        respond(tmp_path, request["request_id"], new, "task-receipt", "fixture-model")


def test_subscription_consumer_is_bounded_and_recovers_a_successful_reply(tmp_path, monkeypatch):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from training.code_author import consume_cli

    state = tmp_path / "state"
    state.mkdir()
    order = {"order_id": "fixture", "target": {"path": "native/kernels/w4mma.cu", "function": "load"}}
    with pytest.raises(PendingAuthor):
        hosted_proposal(state, order, "void load() {}", knowledge=[{"id": "measured"}])
    native_reply = {"definition": "void load() {}", "rationale": "no faster correct variant"}
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")

    def invoke(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == "codex" and kwargs["check"] and "OPENAI_API_KEY" not in kwargs["env"]
        if argv[1] == "login":
            return subprocess.CompletedProcess(argv, 0, "Logged in using ChatGPT")
        if argv[1] == "debug":
            return subprocess.CompletedProcess(argv, 0, '{"models":[{"slug":"gpt-6.2-sol","visibility":"list"}]}')
        assert argv[1] == "exec" and argv[argv.index("-s") + 1] == "read-only"
        assert "model_reasoning_effort=high" in argv and "--ignore-user-config" in argv
        assert {"shell_tool", "unified_exec", "apps", "multi_agent_v2"} <= set(argv)
        assert "void load() {}" in kwargs["input"] and "measured" in kwargs["input"]
        Path(argv[argv.index("-o") + 1]).write_text(json.dumps(native_reply))
        return subprocess.CompletedProcess(argv, 0, "")

    monkeypatch.setattr(subprocess, "run", invoke)
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(lambda _: consume_cli(state), range(2))) == [native_reply, native_reply]
    (state / "author/response.json").unlink()  # successful exit was persisted first
    assert consume_cli(state) == native_reply
    assert [cmd[1] for cmd in calls] == ["login", "debug", "exec"]
    assert hosted_proposal(state, order, "void load() {}", knowledge=[{"id": "measured"}]) == native_reply


def test_cli_preflight_and_nonzero_exit_do_not_spend_another_attempt(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path

    from training.code_author import consume_cli

    order = {"target": {"path": "ironmule/ggml_engine.py"}}
    with pytest.raises(PendingAuthor):
        hosted_proposal(tmp_path, order, "def run(): return 0")
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv[1])
        if argv[1] == "login":
            auth = "API key login" if len(calls) == 1 else "Logged in using ChatGPT"
            return subprocess.CompletedProcess(argv, 0, auth)
        if argv[1] == "debug":
            return subprocess.CompletedProcess(argv, 0, '{"models":[{"slug":"gpt-6-sol","visibility":"list"}]}')
        Path(argv[argv.index("-o") + 1]).write_text('{"definition":"def run(): return 0","rationale":"unused"}')
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(subprocess, "run", invoke)
    with pytest.raises(RuntimeError, match="ChatGPT"):
        consume_cli(tmp_path)
    with pytest.raises(RuntimeError, match="retry preflight"):
        consume_cli(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        consume_cli(tmp_path, retry_preflight=True)
    with pytest.raises(RuntimeError, match="already attempted"):
        consume_cli(tmp_path, retry_preflight=True)
    assert calls == ["login", "login", "debug", "exec"]
    assert not (tmp_path / "author/response.json").exists()


def test_updates_to_the_lab_notebook_leave_dispatched_knowledge_frozen(tmp_path):
    order, source = {"target": {"function": "op"}}, "void op() {}"
    with pytest.raises(PendingAuthor):
        hosted_proposal(tmp_path, order, source, knowledge=[{"evidence": "initial notebook"}])
    path = tmp_path / "author/request.json"
    frozen = path.read_bytes()
    with pytest.raises(PendingAuthor):
        hosted_proposal(tmp_path, order, source, knowledge=[{"evidence": "updated notebook"}])
    assert path.read_bytes() == frozen
    response = {"definition": source, "rationale": "preserve the bound order"}
    respond(tmp_path, json.loads(frozen)["request_id"], response, "worker", "model")
    assert hosted_proposal(tmp_path, order, source, knowledge=[]) == response
