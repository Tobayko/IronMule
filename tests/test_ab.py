import inspect
import json
import subprocess
from types import SimpleNamespace

import pytest

from ironmule import ab
from ironmule.runtime import Knobs


def _child_record(*, order=("baseline", "candidate"), pid=1):
    arm = {
        "total_ns": [1.0], "prefill_ns": [0.5], "decode_ns": [0.5],
        "logical_tokens": [7], "logical_tokens_per_repeat": [[7]],
        "physical_tokens_per_repeat": [[7]],
        "token_counts": [{"logical": 1, "physical": 1}],
        "stop_reasons": ["length"], "capacities": [64],
        "deterministic": True, "decode_steps": 0,
        "prompt_tokens": 1, "mlx_peak_bytes": 10,
    }
    return {"pid": pid, "arms": {"baseline": dict(arm), "candidate": dict(arm)},
            "order": list(order), "mlx_peak_bytes": 10}


class _FakeProcess:
    pid = 123

    def __init__(self, *, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.communicate_timeout = None
        self.actions = []
        self.wait_calls = 0

    def communicate(self, timeout=None):
        self.communicate_timeout = timeout
        return self.stdout, self.stderr

    def terminate(self):
        self.actions.append("terminate")

    def kill(self):
        self.actions.append("kill")

    def wait(self, timeout=None):
        self.wait_calls += 1
        return self.returncode

    def poll(self):
        return self.returncode


def _run_setup(monkeypatch, payload=None):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    calls = []

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        spec = json.loads(args[-1])
        child = payload or _child_record(order=spec["order"], pid=100 + len(calls))
        return _FakeProcess(stdout="@@" + json.dumps(child) + "\n")

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)
    return calls


def _arms():
    return {"baseline": Knobs(), "candidate": Knobs(readback_every=2)}


def test_run_callbacks_receive_each_index_and_defensive_json_copy(monkeypatch):
    calls = _run_setup(monkeypatch)
    before = []
    completed = []

    def before_child(index, order):
        before.append((index, order))

    def on_child(index, record):
        completed.append((index, record))
        record["arms"]["baseline"]["logical_tokens"][0] = 999

    result = ab.run(_arms(), processes=2, repeats=7, warmup=2,
                    before_child=before_child, on_child=on_child)

    assert before == [(0, ["baseline", "candidate"]),
                      (1, ["candidate", "baseline"])]
    assert [index for index, _ in completed] == [0, 1]
    assert completed[0][1] is not result["raw"][0]
    assert result["raw"][0]["arms"]["baseline"]["logical_tokens"] == [7]
    assert len(calls) == 2
    assert "start_new_session" not in calls[0][1]
    assert result["token_count_identity"] is True
    assert result["stop_reason_identity"] is True


def test_run_uses_only_real_popen_compatible_pipe_kwargs(monkeypatch):
    """Keep the A/B launch contract valid for subprocess.Popen itself."""
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    real_signature = inspect.signature(subprocess.Popen)
    observed = []

    def strict_popen(args, *, stdout, stderr, text, cwd, env):
        # bind() is the regression guard: an unsupported Popen kwarg raises here.
        real_signature.bind(args, stdout=stdout, stderr=stderr, text=text,
                            cwd=cwd, env=env)
        observed.append({"stdout": stdout, "stderr": stderr, "text": text,
                         "cwd": cwd, "env": env})
        spec = json.loads(args[-1])
        child = _child_record(order=spec["order"], pid=100 + len(observed))
        return _FakeProcess(stdout="@@" + json.dumps(child) + "\n")

    monkeypatch.setattr(ab.subprocess, "Popen", strict_popen)
    result = ab.run(_arms(), processes=1, repeats=7, warmup=2)

    assert len(observed) == 1
    assert observed[0]["stdout"] is subprocess.PIPE
    assert observed[0]["stderr"] is subprocess.PIPE
    assert observed[0]["text"] is True
    assert "start_new_session" not in observed[0]
    assert result["token_identity"] is True


def test_run_timeout_is_passed_to_subprocess_and_is_loud(monkeypatch):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    observed = {}
    process = _FakeProcess()

    def communicate(timeout=None):
        observed["timeout"] = timeout
        raise subprocess.TimeoutExpired(["secret", "prompt"], 0.5)

    process.communicate = communicate
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(ab, "_terminate_child", lambda _process: None)
    with pytest.raises(RuntimeError, match="child 0 timed out") as error:
        ab.run(_arms(), processes=1, child_timeout_seconds=0.5)
    assert observed["timeout"] == 0.5
    assert "secret" not in str(error.value)
    assert error.value.__cause__ is None

    with pytest.raises(ValueError, match="finite and positive"):
        ab.run(_arms(), child_timeout_seconds=0)
    with pytest.raises(ValueError, match="finite and positive"):
        ab.run(_arms(), child_timeout_seconds=10**1000)


@pytest.mark.parametrize("returncode,stdout,needle", [
    (7, "@@{}\n", "exited with status 7"),
    (0, "no marker\n", "no result marker"),
    (0, "@@{\"value\":NaN}\n", "invalid JSON"),
])
def test_run_rejects_nonzero_or_missing_child_marker(monkeypatch, returncode, stdout, needle):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(ab.subprocess, "Popen",
                        lambda *_args, **_kwargs: _FakeProcess(
                            returncode=returncode, stdout=stdout, stderr="secret"))
    with pytest.raises(RuntimeError, match=needle) as error:
        ab.run(_arms(), processes=1)
    assert "secret" not in str(error.value)


def test_run_communication_error_reaps_child_before_raising(monkeypatch):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    process = _FakeProcess()
    process.communicate = lambda timeout=None: (_ for _ in ()).throw(OSError("pipe secret"))

    def terminate_child(child):
        child.actions.append("terminate")

    monkeypatch.setattr(ab.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(ab, "_terminate_child", terminate_child)
    with pytest.raises(ab.ABRunError, match="communication failed") as error:
        ab.run(_arms(), processes=1)
    assert process.actions == ["terminate"]
    assert "pipe secret" not in str(error.value)


def test_direct_child_cleanup_escalates_only_when_terminate_does_not_reap(monkeypatch):
    process = _FakeProcess()
    process.returncode = None
    wait_results = iter([subprocess.TimeoutExpired(["child"], 2), 9])

    def wait(timeout=None):
        process.wait_calls += 1
        result = next(wait_results)
        if isinstance(result, BaseException):
            raise result
        process.returncode = result
        return result

    process.wait = wait
    process.communicate = lambda timeout=None: ("", "")
    ab._terminate_child(process)
    assert process.actions == ["terminate", "kill"]
    assert process.wait_calls == 2


def test_direct_child_cleanup_kills_after_wait_oserror_while_child_is_alive():
    process = _FakeProcess()
    process.returncode = None
    wait_calls = 0

    def wait(timeout=None):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            raise OSError("wait unavailable")
        process.returncode = 9
        return 9

    def kill():
        process.actions.append("kill")

    process.wait = wait
    process.kill = kill
    process.communicate = lambda timeout=None: ("", "")
    with pytest.raises(RuntimeError, match="wait: OSError"):
        ab._terminate_child(process)
    assert process.actions == ["terminate", "kill"]
    assert process.poll() == 9, "wait OSError must not leave a live child behind"


def test_run_rejects_incomplete_child_before_aggregation(monkeypatch):
    import importlib

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(ab.subprocess, "Popen", lambda *_args, **_kwargs:
                        _FakeProcess(stdout="@@{}\n"))
    with pytest.raises(ab.ABRunError, match="incomplete result") as error:
        ab.run(_arms(), processes=1)
    assert error.value.child_index == 0
    assert error.value.partial_children == []
